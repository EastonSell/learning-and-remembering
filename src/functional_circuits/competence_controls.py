"""Validation-only checkpoint pairing, with independently mapped held-out IIA."""
import argparse
import json
from pathlib import Path
import platform
import numpy as np
from .data import generate, memory_counterfactual
from .experiment import digest
from .metrics import accuracy, choose_mapping
from .models import train
from .role_controls import make_control

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT/'docs/protocols/competence_controls_v1.json'


def load_protocol(path=PROTOCOL):
    return json.loads(Path(path).read_text())


def select_pair(aligned, mixed, overall_tolerance, context_tolerance, floor):
    """Select using prediction validation scores only; None means no match."""
    eligible = []
    for i, a in enumerate(aligned):
        for j, b in enumerate(mixed):
            lower = min(a['accuracy'], b['accuracy'])
            gap = abs(a['accuracy']-b['accuracy'])
            context_gap = max(abs(x-y) for x, y in zip(a['context_accuracy'], b['context_accuracy']))
            if lower >= floor and gap <= overall_tolerance and context_gap <= context_tolerance:
                eligible.append(((lower, -gap, -a['epoch'], -b['epoch']), i, j))
    if not eligible:
        return None
    _, i, j = max(eligible)
    return i, j


def profile(logits, data):
    return dict(accuracy=accuracy(logits, data.y),
                context_accuracy=[accuracy(logits[data.context == c], data.y[data.context == c]) for c in (0,1)])


def explicit_counterfactual(base, donors):
    x = base.x.copy()
    x[:, 6:12] = donors.x[:, 6:12]
    return x


def shared_subset_metrics(y, cf, factual, explicit, patched):
    shared = np.logical_and.reduce([p == target for predictions, target in
                                   [(factual, y), (explicit, cf)] for p in predictions])
    changed = shared & (cf != y)
    rate = lambda mask: [float(np.mean(p[mask] == cf[mask])) if mask.any() else None for p in patched]
    return dict(examples=int(shared.sum()), changed_examples=int(changed.sum()),
                iia=rate(shared), changed_iia=rate(changed))


def run(protocol_path, output):
    p = load_protocol(protocol_path)
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output must be empty or absent')
    for key in ('seeds', 'epochs'):
        values = p[key]
        if not values or len(set(values)) != len(values) or any(type(v) is not int or v < (0 if key == 'seeds' else 1) for v in values):
            raise ValueError('Use unique nonnegative seeds and positive epochs')
    for key in ('train_size', 'selection_size', 'mapping_size', 'test_size'):
        if type(p[key]) is not int or p[key] < 2:
            raise ValueError('Split sizes must be integers of at least two')
    for key in ('overall_accuracy_tolerance', 'context_accuracy_tolerance', 'minimum_accuracy'):
        if not 0 <= p[key] <= 1:
            raise ValueError('Accuracy thresholds must lie in [0, 1]')
    if p['swaps'] != [0,1,2,3]:
        raise ValueError('This study compares swaps 0 against 1, 2 and 3')
    # Test examples are deliberately not generated until every selection is saved.
    def dataset(seed, name, size):
        data = generate(size, 100*seed+p['seed_offsets'][name],
                        p['train_agreement'] if name == 'train' else p['evaluation_agreement'])
        if name in ('selection', 'mapping', 'test') and np.unique(data.context).size != 2:
            raise ValueError(f'{name} seed {seed} lacks both contexts; increase split size')
        hashes.setdefault(str(seed), {})[name] = data.fingerprint()
        return data
    def save(name, obj):
        (out/name).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')
    hashes, candidates, selections, models = {}, [], [], {}
    out.mkdir(parents=True, exist_ok=True)
    for seed in p['seeds']:
        training = dataset(seed, 'train', p['train_size'])
        selection = dataset(seed, 'selection', p['selection_size'])
        scores = {}
        for swaps in p['swaps']:
            scores[swaps] = []
            for epoch in p['epochs']:
                model = make_control(seed, swaps)
                train(model, training, seed+10000, epochs=epoch)
                row = dict(seed=seed, swaps=swaps, epoch=epoch, **profile(model.forward(selection.x)[0],selection))
                scores[swaps].append(row)
                candidates.append(row)
                models[seed, swaps, epoch] = model
        for swaps in p['swaps'][1:]:
            pair = select_pair(scores[0], scores[swaps], p['overall_accuracy_tolerance'],
                               p['context_accuracy_tolerance'], p['minimum_accuracy'])
            selections.append(dict(seed=seed, swaps=swaps, matched=pair is not None,
                                   selected=[scores[0][pair[0]], scores[swaps][pair[1]]] if pair else None))
        print(f'Validation selection finished for seed {seed}', flush=True)
    save('candidates.json', candidates)
    save('selections.json', selections)
    locked_selection_hash = digest(out/'selections.json')
    rows = []
    for selection in selections:
        seed, swaps = selection['seed'], selection['swaps']
        if not selection['matched']:
            rows.append(dict(seed=seed, swaps=swaps, matched=False))
            continue
        mapping_data = dataset(seed, 'mapping', p['mapping_size'])
        mapping_donors = dataset(seed, 'mapping_donors', p['mapping_size'])
        selected_models = [models[seed, condition, chosen['epoch']]
                           for condition, chosen in zip((0, swaps), selection['selected'])]
        mappings = [choose_mapping(m, mapping_data, mapping_donors)[0] for m in selected_models]
        test = dataset(seed, 'test', p['test_size'])
        donors = dataset(seed, 'test_donors', p['test_size'])
        cf = memory_counterfactual(test, donors)
        cf_x = explicit_counterfactual(test, donors)
        logits = [m.forward(test.x)[0] for m in selected_models]
        patched = [m.patch(test.x, donors.x, mapping) for m, mapping in zip(selected_models, mappings)]
        profiles = [profile(z, test) for z in logits]
        gap = profiles[0]['accuracy']-profiles[1]['accuracy']
        context_gaps = [a-b for a,b in zip(profiles[0]['context_accuracy'], profiles[1]['context_accuracy'])]
        changed = cf != test.y
        iia = [accuracy(z, cf) for z in patched]
        changed_iia = [accuracy(z[changed], cf[changed]) if changed.any() else None for z in patched]
        rows.append(dict(seed=seed, swaps=swaps, matched=True,
                         epochs=[x['epoch'] for x in selection['selected']], parameters=[m.parameter_count for m in selected_models],
                         mappings=mappings, test_profiles=profiles, accuracy_gap_aligned_minus_mixed=gap,
                         context_accuracy_gaps=context_gaps,
                         test_match_transferred=abs(gap)<=p['overall_accuracy_tolerance'] and
                         max(map(abs,context_gaps))<=p['context_accuracy_tolerance'] and
                         min(x['accuracy'] for x in profiles)>=p['minimum_accuracy'],
                         memory_iia=iia, iia_gap_aligned_minus_mixed=iia[0]-iia[1], changed_memory_iia=changed_iia,
                         shared_correct=shared_subset_metrics(test.y,cf,[z>=0 for z in logits],
                             [m.forward(cf_x)[0]>=0 for m in selected_models],[z>=0 for z in patched])))
    assert digest(out/'selections.json') == locked_selection_hash
    summaries = []
    for swaps in p['swaps'][1:]:
        matched = [r for r in rows if r['swaps']==swaps and r['matched']]
        def mean(values):
            return float(np.mean(values)) if values else None
        summaries.append(dict(swaps=swaps, attempted_pairs=len(p['seeds']), validation_matched_pairs=len(matched),
                              test_match_transferred_pairs=sum(r['test_match_transferred'] for r in matched),
                              mean_accuracy_gap=mean([r['accuracy_gap_aligned_minus_mixed'] for r in matched]),
                              mean_iia_gap=mean([r['iia_gap_aligned_minus_mixed'] for r in matched]),
                              shared_correct_examples=[r['shared_correct']['examples'] for r in matched],
                              mean_shared_correct_iia_gap=mean([r['shared_correct']['iia'][0]-r['shared_correct']['iia'][1]
                                  for r in matched if r['shared_correct']['examples']]),
                              paired_iia_gaps=[r['iia_gap_aligned_minus_mixed'] for r in matched]))
    save('metrics.json', rows)
    save('summary.json', summaries)
    sources = sorted(path for folder in ('src','tests','scripts') for path in (ROOT/folder).rglob('*.py'))
    save('manifest.json', dict(study=p['study'], protocol=p, protocol_sha256=digest(Path(protocol_path)),
                              selections_locked_before_test_sha256=locked_selection_hash,
                              python=platform.python_version(), numpy=np.__version__, dataset_sha256=hashes,
                              source_sha256={str(path.relative_to(ROOT)):digest(path) for path in sources},
                              artifact_sha256={path.name:digest(path) for path in sorted(out.iterdir()) if path.is_file()}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=PROTOCOL)
    parser.add_argument('--output', type=Path, default=Path('results/competence_controls'))
    args=parser.parse_args()
    run(args.protocol,args.output)


if __name__=='__main__':
    main()
