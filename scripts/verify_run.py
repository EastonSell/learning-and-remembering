"""Audit recorded hashes and compare complete numerical/checkpoint coverage."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# This audit tool was recorded as source by three historical studies. Preserve
# their manifests and verify the exact archived tool, never substitute model code.
HISTORICAL_AUDIT_TOOL = {
    'b9be1d1fb77d531eeeabd97ff58caf7f8801be50a72589a21b3bcfa30ee1c34b':
        'docs/archive/verify_run_pre_coverage_fix.py.txt'
}
IGNORED_METRIC_FIELDS = {'training_seconds', 'peak_rss_bytes'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path):
    def reject(value):
        raise ValueError(f'Nonfinite JSON constant: {value}')
    return json.loads(path.read_text(), parse_constant=reject)


def metric_index(rows, fields=None):
    if not isinstance(rows, list) or not rows or not all(isinstance(r,dict) for r in rows):
        raise ValueError('Metrics must be a nonempty list of row objects')
    if fields is None:
        for candidate in (('model','seed','split'), ('seed','swaps','split'), ('seed','swaps')):
            if all(name in rows[0] for name in candidate):
                fields = candidate
                break
        if fields is None:
            raise ValueError('Unsupported metric key schema')
    indexed = {}
    for row in rows:
        if type(row.get('seed')) is not int or row['seed'] < 0:
            raise ValueError('Metric seeds must be nonnegative integers')
        try:
            key = tuple(row[name] for name in fields)
            if key in indexed:
                raise ValueError(f'Duplicate metric key: {key}')
            indexed[key] = row
        except (KeyError,TypeError) as exc:
            raise ValueError('Missing or invalid metric key') from exc
    return indexed, fields


def compare_runs(reference, candidate, manifest, seeds, allow_missing):
    expected, fields = metric_index(load_json(reference/'metrics.json'))
    observed, _ = metric_index(load_json(candidate/'metrics.json'), fields)
    if seeds is not None:
        available = {r['seed'] for r in expected.values()}
        if not seeds or len(seeds) != len(set(seeds)) or not set(seeds) <= available:
            raise ValueError('Comparison scope must contain unique existing reference seeds')
        expected = {k:r for k,r in expected.items() if r['seed'] in seeds}
    if set(observed) != set(expected):
        raise AssertionError(f'Metric key coverage differs: expected {len(expected)}, observed {len(observed)}; '
                             f'missing={set(expected)-set(observed)}, extra={set(observed)-set(expected)}')
    for key, row in observed.items():
        a = {k:v for k,v in expected[key].items() if k not in IGNORED_METRIC_FIELDS}
        b = {k:v for k,v in row.items() if k not in IGNORED_METRIC_FIELDS}
        if a != b:
            raise AssertionError(f'Numerical fields or values differ: {key}')
    checkpoint_names = {name for name in manifest['artifact_sha256']
                        if name.startswith('checkpoints/') and name.endswith('.npz')}
    if seeds is not None:
        def checkpoint_seed(name):
            try:
                return int(Path(name).stem.rsplit('_',1)[1])
            except (IndexError,ValueError) as exc:
                raise ValueError('Cannot resolve checkpoint seed for explicit partial comparison') from exc
        checkpoint_names = {name for name in checkpoint_names if checkpoint_seed(name) in seeds}
    actual_names = {str(path.relative_to(candidate)) for path in (candidate/'checkpoints').rglob('*') if path.is_file()}
    if actual_names != checkpoint_names:
        raise AssertionError(f'Checkpoint coverage differs: missing={checkpoint_names-actual_names}, '
                             f'extra={actual_names-checkpoint_names}')
    for name in sorted(checkpoint_names):
        old, new = reference/name, candidate/name
        if not old.exists() and allow_missing:
            if digest(new) != manifest['artifact_sha256'][name]:
                raise AssertionError(f'Regenerated checkpoint hash differs: {name}')
            continue
        with np.load(old, allow_pickle=False) as a, np.load(new, allow_pickle=False) as b:
            if a.files != b.files:
                raise AssertionError(f'Checkpoint fields differ: {name}')
            for field in a.files:
                if a[field].dtype != b[field].dtype:
                    raise AssertionError(f'Checkpoint dtype differs: {name}, {field}')
                np.testing.assert_array_equal(a[field], b[field])
    scope = 'all reference seeds' if seeds is None else f'explicit seeds {sorted(seeds)}'
    print(f'Exact numerical match for {len(observed)} metric rows and {len(checkpoint_names)} expected checkpoints ({scope}).')
    print('Only training_seconds and peak_rss_bytes are excluded from metric equality.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--compare', type=Path)
    parser.add_argument('--compare-seeds', nargs='+', type=int,
                        help='Explicit partial comparison; candidate must contain exactly these seeds')
    parser.add_argument('--allow-missing-checkpoints', action='store_true',
                        help='Allow omitted REFERENCE .npz files; never relax candidate coverage')
    args = parser.parse_args()
    if args.compare_seeds is not None and args.compare is None:
        parser.error('--compare-seeds requires --compare')
    manifest = load_json(args.run/'manifest.json')
    omitted = []
    for key, base in (('source_sha256', ROOT), ('artifact_sha256', args.run)):
        for name, expected in manifest[key].items():
            path = base/name
            if (not path.exists() and args.allow_missing_checkpoints and key=='artifact_sha256'
                    and name.startswith('checkpoints/') and name.endswith('.npz')):
                omitted.append(name)
                continue
            got = digest(path)
            if (got != expected and key=='source_sha256' and name=='scripts/verify_run.py'
                    and expected in HISTORICAL_AUDIT_TOOL):
                archived = ROOT/HISTORICAL_AUDIT_TOOL[expected]
                if digest(archived) == expected:
                    print('Verified recorded historical audit-tool source from its archived snapshot, not the current verifier.')
                    continue
            if got != expected:
                raise AssertionError(f'Hash mismatch: {name}')
    if args.compare:
        compare_runs(args.run,args.compare,manifest,args.compare_seeds,args.allow_missing_checkpoints)
    if omitted:
        print(f'Intentionally omitted reference checkpoints: {len(omitted)}. Regenerate with the recorded command.')
    print(f"Verified {len(manifest['source_sha256'])} source hashes and {len(manifest['artifact_sha256'])-len(omitted)} artifact hashes.")


if __name__ == '__main__':
    main()
