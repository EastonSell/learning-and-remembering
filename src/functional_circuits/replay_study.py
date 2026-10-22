"""Separate validation-selection and sealed-test stages for replay_digits_v1.

Invoke through scripts/run_replay_digits.py so thread limits precede NumPy import.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import resource
import signal
import time
import numpy as np
from . import replay_core as core
from . import replay_data as data

ROOT = data.ROOT
AUDIT = ROOT / 'results/replay_data_validation/report.json'
AUDIT_SHA = 'f2bc38875ad3cd4f8b549e1b7f4fed72bba48d4039bf88cbdd2bb8c9fb40cd3f'
ARCHIVE_SHA = '0d7b054fea010270e9b3f06411c654c5e59547732ad626381980baffe0a23fb0'
PROTOCOL_SHA = 'c0fde2c0a8a187d1d88248e7fd4127094bc5ea54a8840fca31bcd2bf42dd1250'
THREAD_VARIABLES = ('OPENBLAS_NUM_THREADS', 'OPENBLAS_DEFAULT_NUM_THREADS', 'OMP_NUM_THREADS',
                    'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
CONFIG = dict(protocol='replay_digits_v1', rates=[.03, .1, .3], development_seeds=[3, 7],
              evaluation_seeds=[211,223,227,229,233,239,241,251,257,263],
              arms=list(core.ARMS), batch_size=16, updates_per_batch=4, memory_capacity=100,
              network=[64,32,10], dtype='float64', maximum_threads=1)
RUNTIME_FILES = ['src/functional_circuits/replay_data.py', 'src/functional_circuits/replay_core.py',
                 'src/functional_circuits/replay_study.py', 'scripts/run_replay_digits.py',
                 'docs/protocols/replay_digits_v1.md']


def source_hashes():
    if data.digest(data.PROTOCOL.read_bytes()) != PROTOCOL_SHA:
        raise ValueError('Frozen protocol changed; execution requires a reviewed new version')
    return {name:data.digest((ROOT/name).read_bytes()) for name in RUNTIME_FILES}


def versions():
    return dict(python=platform.python_version(), numpy=np.__version__)


def thread_guard():
    if any(os.environ.get(name) != '1' for name in THREAD_VARIABLES):
        raise ValueError('Launch scripts/run_replay_digits.py in a fresh Python process')


def load_verified_data(archive):
    """Integrity inspection may read test bytes; no model or score is used here."""
    if data.digest(AUDIT.read_bytes()) != AUDIT_SHA:
        raise ValueError('Published data audit was modified')
    expected = json.loads(AUDIT.read_text())
    arrays, meta = data.inspect_archive(archive)
    if meta['archive_sha256'] != ARCHIVE_SHA:
        raise ValueError('Archive does not match the verified official source')
    observed = data.validate(archive)
    for key in ('archive_sha256','members','protocol_sha256','split_ids','split_id_sha256',
                'split_counts','duplicate_audit','status'):
        if observed[key] != expected[key]:
            raise ValueError(f'Data gate changed: {key}')
    if observed['status'] != 'passed':
        raise ValueError('Data gate did not pass')
    train, val = [np.asarray(expected['split_ids'][name], dtype=int) for name in ('train','validation')]
    x, y = arrays['optdigits.tra']; xt, yt = arrays['optdigits.tes']
    identity = dict(archive_sha256=ARCHIVE_SHA, audit_sha256=AUDIT_SHA,
                    split_id_sha256=expected['split_id_sha256'], members=expected['members'])
    return dict(train=(x[train]/16., y[train], train), validation=(x[val]/16., y[val]),
                test=(xt/16., yt), identity=identity)


def save(out, name, value):
    """Exclusive output creation plus a 250 MB per-stage artifact ceiling."""
    encoded = json.dumps(value, indent=2, allow_nan=False) + '\n'
    used = sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    if used + len(encoded.encode()) > 250_000_000:
        raise RuntimeError('Artifact budget exceeded')
    with (out/name).open('x') as f:
        f.write(encoded)


@contextmanager
def limits(seconds):
    """Linux-only hard wall timer and conservative 1 GiB virtual-memory cap.

    The address-space cap is stricter than the protocol RSS ceiling. Restore
    process-local limits/handler on exit; never alter global system settings.
    """
    if platform.system() != 'Linux':
        raise RuntimeError('Resource enforcement currently requires Linux')
    if signal.getitimer(signal.ITIMER_REAL)[0] != 0:
        raise RuntimeError('Another process-local alarm is active')
    previous = signal.getsignal(signal.SIGALRM)
    old = resource.getrlimit(resource.RLIMIT_AS)
    bound = min(v for v in (old[0], old[1], 1 << 30) if v != resource.RLIM_INFINITY)
    def expired(signum, frame):
        raise TimeoutError('Stage wall-time budget exceeded; no retuning permitted')
    try:
        resource.setrlimit(resource.RLIMIT_AS, (bound, old[1]))
        signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, seconds)
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        resource.setrlimit(resource.RLIMIT_AS, old)


def peak_rss():
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024
    if value > 1 << 30:
        raise RuntimeError('Peak RSS exceeded 1 GiB')
    return value


def finish_manifest(out, sources):
    if source_hashes() != sources:
        raise ValueError('Runtime source changed during stage')
    save(out, 'manifest.json', dict(source_sha256=sources,
         artifact_sha256={p.name:data.digest(p.read_bytes()) for p in sorted(out.iterdir()) if p.is_file()}))


def develop(archive, output):
    thread_guard()
    sources = source_hashes()
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    with limits(900):
        d = load_verified_data(archive)
        rows, candidates = [], {rate:[] for rate in CONFIG['rates']}
        for rate in CONFIG['rates']:
            for seed in CONFIG['development_seeds']:
                for arm in CONFIG['arms']:
                    before = time.monotonic()
                    run = core.train_stream(*d['train'], seed, rate, arm)
                    profile = core.score(core.predict(run['snapshots'][-1], d['validation'][0]), d['validation'][1])
                    candidates[rate].append(profile['macro_accuracy'])
                    rows.append(dict(rate=rate, seed=seed, arm=arm, validation=profile,
                                     training_and_validation_seconds=time.monotonic()-before,
                                     peak_rss_bytes=peak_rss(), counts=run['counts']))
        if source_hashes() != sources:
            raise ValueError('Runtime source changed during development')
        selection = dict(schema_version=1, stage='validation_selection_complete',
            created_utc=datetime.now(timezone.utc).isoformat(), config=CONFIG, versions=versions(),
            source_sha256=sources, data_identity=d['identity'], selected_rate=core.choose_rate(candidates),
            development_scores=rows, test_model_evaluations=0,
            freeze_scope='Selected common rate, scores, configuration, source/data versions; no model weights persisted.')
        save(out, 'selection.json', selection)
        finish_manifest(out, sources)
    return data.digest((out/'selection.json').read_bytes())


def read_selection(path, expected_sha):
    path = Path(path)
    if (path.parent/'incomplete.json').exists() or not (path.parent/'manifest.json').is_file():
        raise ValueError('Development stage is incomplete or lacks its completion manifest')
    manifest = json.loads((path.parent/'manifest.json').read_text())
    raw = path.read_bytes()
    if manifest.get('artifact_sha256',{}).get(path.name) != expected_sha:
        raise ValueError('Selection is not pinned by the completed development manifest')
    if data.digest(raw) != expected_sha:
        raise ValueError('Selection digest mismatch')
    selection = json.loads(raw)
    if (selection.get('stage') != 'validation_selection_complete'
            or selection.get('config') != CONFIG or selection.get('versions') != versions()
            or selection.get('source_sha256') != source_hashes()
            or manifest.get('source_sha256') != selection.get('source_sha256')
            or selection.get('test_model_evaluations') != 0):
        raise ValueError('Selection configuration/source/version mismatch')
    rows = selection['development_scores']
    expected = {(r,s,a) for r in CONFIG['rates'] for s in CONFIG['development_seeds'] for a in CONFIG['arms']}
    if len(rows) != len(expected) or {(r['rate'],r['seed'],r['arm']) for r in rows} != expected:
        raise ValueError('Incomplete or duplicated development comparisons')
    candidates = {rate:[r['validation']['macro_accuracy'] for r in rows if r['rate']==rate] for rate in CONFIG['rates']}
    if selection['selected_rate'] != core.choose_rate(candidates):
        raise ValueError('Selected rate does not match the locked validation rule')
    return selection


def evaluate_run(run, x, y):
    profiles, matrix = [], []
    for parameters in run['snapshots']:
        guesses = core.predict(parameters, x)
        profiles.append(core.score(guesses, y))
        matrix.append([float(np.mean(guesses[np.isin(y, run['label_order'][2*b:2*b+2])] ==
                                         y[np.isin(y, run['label_order'][2*b:2*b+2])])) for b in range(5)])
    return dict(final=profiles[-1], boundary_profiles=profiles, block_accuracy=matrix,
                forgetting=core.forgetting(matrix), mean_immediate_acquisition=float(np.diag(matrix).mean()))


def summarize(rows):
    lookup = {(row['seed'],row['model']):row for row in rows}
    primary = [lookup[s,'reservoir']['test']['final']['macro_accuracy']-
               lookup[s,'fifo']['test']['final']['macro_accuracy'] for s in CONFIG['evaluation_seeds']]
    secondary = [lookup[s,'reservoir']['test']['final']['macro_accuracy']-
                 lookup[s,'current_only']['test']['final']['macro_accuracy'] for s in CONFIG['evaluation_seeds']]
    acquisition = {a:float(np.mean([lookup[s,a]['test']['mean_immediate_acquisition']
                                  for s in CONFIG['evaluation_seeds']])) for a in CONFIG['arms']}
    return dict(primary_reservoir_minus_fifo=core.paired_summary(primary),
                exploratory_reservoir_minus_current_only=dict(mean_difference=float(np.mean(secondary)), per_seed=secondary),
                primary_per_seed=primary, mean_immediate_acquisition=acquisition,
                acquisition_collapse=any(v < .5 for v in acquisition.values()),
                claim_scope='Algorithmic retention on this setup, not biological fidelity or novelty.')


def evaluate(archive, selection_path, selection_sha, output):
    thread_guard()
    selection = read_selection(selection_path, selection_sha)  # Before model access to test.
    sources = source_hashes()
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    with limits(1800):
        d = load_verified_data(archive)
        if d['identity'] != selection['data_identity']:
            raise ValueError('Data/split identity differs from the frozen selection')
        save(out, 'evaluation_lock.json', dict(selection_sha256=selection_sha,
             selection=selection, data_identity=d['identity'], source_sha256=sources,
             created_utc=datetime.now(timezone.utc).isoformat()))
        rows = []
        for seed in CONFIG['evaluation_seeds']:
            paired = []
            for arm in CONFIG['arms']:
                before = time.monotonic()
                run = core.train_stream(*d['train'], seed, selection['selected_rate'], arm)
                train_seconds = time.monotonic()-before
                profile = evaluate_run(run, *d['test'])
                row = dict(model=arm, seed=seed, split='official_test', test=profile,
                           training_seconds=train_seconds, peak_rss_bytes=peak_rss(),
                           diagnostics={k:v for k,v in run.items() if k != 'snapshots'})
                rows.append(row); paired.append(run)
            if (len({r['initial_sha256'] for r in paired}) != 1 or
                    len({json.dumps(r['counts'],sort_keys=True) for r in paired}) != 1 or
                    len({str(r['label_order']) for r in paired}) != 1):
                raise AssertionError('Paired capacity/initialization/compute contract failed')
        save(out, 'metrics.json', rows)
        save(out, 'summary.json', summarize(rows))
        finish_manifest(out, sources)
    return summarize(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='stage', required=True)
    for name in ('develop','evaluate'):
        parser = sub.add_parser(name)
        parser.add_argument('--archive', type=Path, required=True)
        parser.add_argument('--output', type=Path, required=True)
        if name == 'evaluate':
            parser.add_argument('--selection', type=Path, required=True)
            parser.add_argument('--selection-sha256', required=True)
    a = p.parse_args()
    existed = a.output.exists()
    try:
        result = (develop(a.archive,a.output) if a.stage=='develop' else
                  evaluate(a.archive,a.selection,a.selection_sha256,a.output))
    except (ValueError, RuntimeError, TimeoutError, MemoryError, AssertionError) as exc:
        if not existed and a.output.exists():
            save(a.output, 'incomplete.json', dict(stage=a.stage,status='incomplete',error=str(exc)))
        raise
    print(result if isinstance(result,str) else json.dumps(result))
