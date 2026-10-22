"""Matched-capacity sensitivity to supplied role masks, independent of model names."""
import argparse
import copy
import json
from pathlib import Path
import platform
import numpy as np
from .data import generate
from .experiment import digest
from .models import Model, train
from .metrics import choose_mapping, evaluate


def make_control(seed, swaps):
    """Nested, outcome-independent exchanges between two six-input masks."""
    if not isinstance(swaps, int) or not 0 <= swaps <= 3:
        raise ValueError('swaps must be an integer in [0, 3]')
    model = Model('functional', seed)
    rng = np.random.default_rng(seed + 888)
    left, right = rng.permutation(6), rng.permutation(6)
    for a, b in zip(left[:swaps], right[:swaps]):
        model.groups[0][a], model.groups[1][b] = 6+b, a
    return model


def rename_branches(model):
    """Exact reparameterization, not another training run or a biological control."""
    renamed = copy.deepcopy(model)
    renamed.groups = [model.groups[1].copy(), model.groups[0].copy()]
    for suffix in ('W', 'b', 'v', 'c'):
        renamed.params['a_'+suffix] = model.params['b_'+suffix].copy()
        renamed.params['b_'+suffix] = model.params['a_'+suffix].copy()
    renamed.params['gate_w'] *= -1
    renamed.params['gate_b'] *= -1
    return renamed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='results/role_controls')
    parser.add_argument('--seeds', nargs='+', type=int, default=[11, 23, 37, 53, 71])
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--train-size', type=int, default=2048)
    parser.add_argument('--test-size', type=int, default=2048)
    parser.add_argument('--validation-size', type=int, default=512)
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds) or min(args.epochs, args.train_size, args.test_size, args.validation_size) < 1:
        parser.error('Use unique seeds and positive epochs/split sizes')
    # Evaluation reports per-context lesion/gate metrics. Reject an undefined
    # subgroup before training any condition or creating partial output files.
    for seed in args.seeds:
        for name, offset, agreement in [('id', 4, .8), ('balanced', 5, .5), ('conflict', 6, 0.)]:
            data = generate(args.test_size, seed*100+offset, agreement)
            if np.unique(data.context).size != 2:
                parser.error(f'Test split {name} for seed {seed} lacks both contexts; increase --test-size')
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        parser.error('Output must be empty or absent; existing results are protected')
    out.mkdir(parents=True, exist_ok=True)
    rows, hashes, curves = [], {}, {}
    for seed in args.seeds:
        training = generate(args.train_size, seed*100+1, .8)
        validation = generate(args.validation_size, seed*100+2, .5)
        vd = generate(args.validation_size, seed*100+3, .5)
        tests = {name: generate(args.test_size, seed*100+offset, agreement)
                 for name, offset, agreement in [('id', 4, .8), ('balanced', 5, .5), ('conflict', 6, 0.)]}
        td = generate(args.test_size, seed*100+7, .5)
        hashes[str(seed)] = {name: data.fingerprint() for name, data in
                            dict(train=training, validation=validation, validation_donors=vd,
                                 test_donors=td, **tests).items()}
        for swaps in range(4):
            model = make_control(seed, swaps)
            curves[f'{seed}/{swaps}'] = train(model, training, seed+10000, args.epochs)
            mapping, metadata = choose_mapping(model, validation, vd)
            renamed = rename_branches(model)
            for split, data in tests.items():
                logits = model.forward(data.x)[0]
                patch = model.patch(data.x, td.x, mapping)
                rename_error = max(float(np.max(np.abs(logits-renamed.forward(data.x)[0]))),
                                   float(np.max(np.abs(patch-renamed.patch(data.x, td.x, [1-mapping[0]])))))
                rows.append(dict(seed=seed, swaps=swaps, split=split, parameters=model.parameter_count,
                                 groups=[g.tolist() for g in model.groups], mapping=mapping,
                                 rename_max_logit_error=rename_error, **metadata,
                                 **evaluate(model, data, td, mapping)))
            print(f'seed={seed} swaps={swaps} balanced_accuracy={rows[-2]["accuracy"]:.4f}', flush=True)
    summaries = []
    for split in tests:
        for swaps in range(4):
            group = [r for r in rows if r['split'] == split and r['swaps'] == swaps]
            baseline = {r['seed']: r for r in rows if r['split'] == split and r['swaps'] == 0}
            summary = dict(split=split, swaps=swaps, seeds=len(group))
            for metric in ('accuracy', 'memory_iia', 'memory_iia_changed'):
                values = [r[metric] for r in group if r[metric] is not None]
                differences = [r[metric]-baseline[r['seed']][metric] for r in group
                               if r[metric] is not None and baseline[r['seed']][metric] is not None]
                summary[metric] = dict(mean=float(np.mean(values)) if values else None,
                                      sample_sd=float(np.std(values, ddof=1)) if len(values)>1 else None,
                                      paired_differences_from_aligned=differences)
            summaries.append(summary)
    for name, obj in [('metrics.json', rows), ('summary.json', summaries), ('learning_curves.json', curves)]:
        (out/name).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')
    root = Path(__file__).resolve().parents[2]
    source = sorted(p for folder in ('src', 'tests', 'scripts') for p in (root/folder).rglob('*.py'))
    manifest = dict(experiment='role-mask-controls-v1', baseline_commit='b2bdba110d8f21e27f5c2b4701c2f8d1864a1514',
                    config=vars(args), python=platform.python_version(), numpy=np.__version__,
                    dataset_sha256=hashes, source_sha256={str(p.relative_to(root)): digest(p) for p in source},
                    artifact_sha256={p.name: digest(p) for p in sorted(out.iterdir())},
                    design='Fixed width 8, 132 parameters, 120 default epochs, Adam lr .01, batch 128; two mapping candidates. No outcome-based mask selection.',
                    limitations='One nested mask draw per seed; roles and targets remain engineered. Capacity matched, not accuracy matched. No biological or real-world claim.')
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
