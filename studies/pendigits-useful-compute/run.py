"""Admission-gated executable. Import and --selftest never read real data."""
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import contextlib
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import resource
import sys
import time
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import engine

ROOT = Path(__file__).resolve().parent
REQUIRED_FREEZE_FILES = ('PROTOCOL.md','protocol.json','engine.py','run.py','test_engine.py','test_runner.py')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    # A temporary in the same directory makes each ledger update atomic.
    path = Path(path)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def verify_admission(root, admission_path):
    root, admission_path = Path(root), Path(admission_path)
    if not admission_path.is_file():
        raise ValueError('independent admission is missing')
    admission = json.loads(admission_path.read_text())
    manifest_path = root/'PRE_DATA_MANIFEST.json'
    if not manifest_path.is_file():
        raise ValueError('frozen manifest missing')
    manifest = json.loads(manifest_path.read_text())
    digest = sha256(manifest_path)
    if manifest.get('status') != 'FROZEN_PENDING_INDEPENDENT_REVIEW':
        raise ValueError('invalid frozen status')
    if not set(REQUIRED_FREEZE_FILES).issubset(manifest.get('files', {})):
        raise ValueError('manifest missing required files')
    if admission.get('status') != 'APPROVED_FOR_EXECUTION' or admission.get('manifest_sha256') != digest:
        raise ValueError('admission does not bind the exact frozen manifest')
    for name, expected in manifest['files'].items():
        p = root/name
        if p.resolve().parent != root.resolve() or not p.is_file() or sha256(p) != expected:
            raise ValueError('frozen file hash mismatch: '+name)
    review_path = Path(admission.get('review_path', ''))
    if not review_path.is_file() or sha256(review_path) != admission.get('review_sha256'):
        raise ValueError('independent review artifact missing or hash mismatch')
    review = json.loads(review_path.read_text())
    if review.get('status') != 'APPROVED' or review.get('manifest_sha256') != digest or not review.get('reviewer'):
        raise ValueError('review does not approve this exact manifest')
    return {'manifest_sha256': digest, 'admission_sha256': sha256(admission_path),
            'review_sha256': sha256(review_path), 'reviewer': review['reviewer']}


def read_member(path, expected_hash, expected_shape):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_hash:
        raise ValueError('raw member hash mismatch')
    a = np.loadtxt(io.BytesIO(raw), delimiter=',')
    if a.shape != tuple(expected_shape) or not np.isfinite(a).all():
        raise ValueError('data shape or finite-value mismatch')
    if (a[:, :16] < 0).any() or (a[:, :16] > 100).any():
        raise ValueError('coordinate range mismatch')
    if not np.equal(a[:, 16], np.floor(a[:, 16])).all() or set(a[:, 16]) != set(range(10)):
        raise ValueError('label range or integrality mismatch')
    return a[:, :16]/100., a[:, 16].astype(int)


def check_leakage(x, tx, train, dev):
    tr, de, te = set(map(tuple, x[train])), set(map(tuple, x[dev])), set(map(tuple, tx))
    if tr.intersection(de) or tr.intersection(te) or de.intersection(te):
        raise ValueError('cross-split duplicate feature rows')


def environment():
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        np.show_config()
    return {'python': platform.python_version(), 'numpy': np.__version__,
            'sklearn': sklearn.__version__, 'platform': platform.platform(),
            'processor': platform.processor(), 'numpy_configuration': stream.getvalue(),
            'threads': {k: os.environ.get(k) for k in
                        ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')}}


def planned_cases(config):
    return [{'seed': seed, 'arm': arm, 'position': position, 'status': 'not_started', 'boundaries': []}
            for i, seed in enumerate(config['seeds'])
            for position, arm in enumerate(engine.case_order(i))]


def execute_arrays(config, arrays, out, check, provenance, run_started=None):
    """Core case loop, also exercised end to end with explicitly synthetic arrays.

    The command-line real-data path verifies admission before calling this.
    Caller-provided synthetic provenance must never be relabeled empirical.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    (out/'INCOMPLETE').write_text('No completed scientific result until COMPLETE.json.\n')
    rows = planned_cases(config)
    write_json(out/'cases.json', rows)
    write_json(out/'provenance.json', provenance)
    active = None
    start = time.perf_counter()
    if run_started is None:
        run_started = time.monotonic()
    try:
        check()
        x, y, tx, ty, train, dev = arrays
        check_leakage(x, tx, train, dev)
        np.savez_compressed(out/'split.npz', train=train, development=dev)
        np.savez_compressed(out/'test-inputs.npz', x=tx, y=ty)
        write_json(out/'environment.json', environment())
        # The only warmup is fixed synthetic arithmetic, independent of data.
        rng = np.random.default_rng(0)
        engine.step(engine.init(0), rng.normal(size=(32,16)), rng.integers(0,10,32))
        check()
        for i, seed in enumerate(config['seeds']):
            phases = engine.build_phases(y, train, seed)
            write_json(out/f'{seed}-phases.json', [{'pair': list(pair), 'ids': ids.tolist()} for pair,ids in phases])
            for arm in engine.case_order(i):
                active = next(r for r in rows if r['seed'] == seed and r['arm'] == arm)
                active['status'] = 'running'
                write_json(out/'cases.json', rows)
                def boundary(phase, pair, weights, partial):
                    check()
                    m, pred, probs = engine.metrics(weights, tx, ty)
                    m.update(phase=phase, current_pair=list(pair),
                             acquisition=float(np.mean([m['recalls'][c] for c in pair])))
                    active['boundaries'].append(m)
                    active['partial_accounting'] = partial
                    np.savez_compressed(out/f'{seed}-{arm}-{phase}.npz',
                                        **{f'w{k}': v for k,v in enumerate(weights)},
                                        predictions=pred, probabilities=probs)
                    write_json(out/'cases.json', rows)
                    check()
                result = engine.train_case(x, y, phases, seed, arm, callback=boundary, check=check)
                active.update({k: v for k,v in result.items() if k != 'weights'})
                active.update(status='complete', final_macro=active['boundaries'][-1]['macro'],
                              mean_acquisition=float(np.mean([b['acquisition'] for b in active['boundaries']])))
                active.pop('partial_accounting', None)
                write_json(out/'cases.json', rows)
                check()
                active = None
        engine.validate_accounting(rows, config['seeds'])
        summary = engine.analyze(rows, config['seeds'], config['bootstrap_seed'], config['bootstrap_replicates'])
        summary['case_loop_seconds'] = time.perf_counter()-start
        write_json(out/'summary.json', summary)
        check()
        write_json(out/'resources.json', {'wall_seconds': time.monotonic()-run_started,
                   'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024})
        check()
        write_json(out/'COMPLETE.json', {'completed_cases': len(rows), 'checkpoints': 360,
                                        'summary_sha256': sha256(out/'summary.json'),
                                        'cases_sha256': sha256(out/'cases.json'),
                                        'resources_sha256': sha256(out/'resources.json'),
                                        'provenance': provenance})
        check()
        (out/'INCOMPLETE').unlink()
        return summary
    except BaseException as error:
        if active is not None:
            active['status'] = 'failed'
            active['error'] = type(error).__name__+': '+str(error)
        write_json(out/'cases.json', rows)
        write_json(out/'FAILURE.json', {'type': type(error).__name__, 'message': str(error),
                                        'case_loop_seconds': time.perf_counter()-start})
        # If a final cap check fails after writing COMPLETE, it is not complete.
        if (out/'COMPLETE.json').exists():
            (out/'COMPLETE.json').rename(out/'REJECTED_COMPLETE.json')
        raise


def main(argv=None):
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selftest', action='store_true')
    parser.add_argument('--execute-admitted', action='store_true')
    parser.add_argument('--admission', type=Path)
    parser.add_argument('--training', type=Path)
    parser.add_argument('--test', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.selftest:
        import unittest
        suite = unittest.defaultTestLoader.discover(str(ROOT), pattern='test_*.py')
        return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
    if not args.execute_admitted or any(v is None for v in (args.admission,args.training,args.test,args.output)):
        parser.error('real execution requires --execute-admitted, --admission, --training, --test and --output')
    provenance = verify_admission(ROOT, args.admission)
    config = json.loads((ROOT/'protocol.json').read_text())
    def check():
        if time.monotonic()-started > config['max_wall_seconds']:
            raise TimeoutError('600-second frozen overall cap')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 > config['max_rss_bytes']:
            raise MemoryError('1GiB frozen peak RSS cap')
    if args.output.exists():
        raise FileExistsError('output directory already exists')
    try:
        check()
        if platform.system() != 'Linux':
            raise ValueError('Linux-only RSS convention frozen; re-review other environments')
        for key, actual in [('python',platform.python_version()),('numpy',np.__version__),('sklearn',sklearn.__version__)]:
            if actual != config[key]:
                raise ValueError('runtime version differs from frozen '+key)
        # Admission and versions are checked before either real member is opened.
        x,y = read_member(args.training,config['training_sha256'],config['training_shape'])
        tx,ty = read_member(args.test,config['test_sha256'],config['test_shape'])
        train,dev = train_test_split(np.arange(len(y)),test_size=.2,random_state=config['split_seed'],stratify=y)
        check()
    except BaseException as error:
        args.output.mkdir(parents=True,exist_ok=False)
        (args.output/'INCOMPLETE').write_text('Input or resource checks failed before training.\n')
        write_json(args.output/'cases.json',planned_cases(config))
        write_json(args.output/'provenance.json',provenance)
        write_json(args.output/'FAILURE.json',{'type':type(error).__name__,'message':str(error),'stage':'before_training'})
        raise
    provenance.update(kind='PROSPECTIVE_NEW_SEED_EXTENSION_REUSED_DATASET',
                      training_sha256=config['training_sha256'],test_sha256=config['test_sha256'],
                      base_commit=config['base_commit'])
    execute_arrays(config,(x,y,tx,ty,train,dev),args.output,check,provenance,run_started=started)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
