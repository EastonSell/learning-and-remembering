"""Pinned Fashion-MNIST integrity, prospective leakage audit, and split preparation.

This module does no training or prediction. Integrity preparation may read official
 test bytes; development loading never returns them. All generated artifacts are
write-once, and raw datasets must remain outside the repository/public root.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
from typing import Callable
import urllib.request

import numpy as np

REVISION = 'b2617bb6d3ffa2e429640350f613e3291e10b141'
BASE_URL = f'https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/{REVISION}/'
ASSETS = {
    'train-images-idx3-ubyte.gz': ('8d4fb7e6c68d591d4c3dfef9ec88bf0d', 60000, True),
    'train-labels-idx1-ubyte.gz': ('25c81989df183df01b3e8a0aad5dffbe', 60000, False),
    't10k-images-idx3-ubyte.gz': ('bef4ecab320f06d8554ea6380940ec79', 10000, True),
    't10k-labels-idx1-ubyte.gz': ('bb300cfdad3c16e7a12a480ee83cd310', 10000, False),
}
SCHEMA_VERSION = 'fashion-mnist-replay-data-v1'
SPLIT_VERSION = 'prospective-feature-grouped-v1'
DEFAULT_DATA_ROOT = Path('/workspace/shared/neuro-replay-selection-data')


class DataIntegrityError(ValueError):
    """A source or artifact fails an integrity check."""


class DataGateError(ValueError):
    """A prospective data/provenance gate is not satisfied."""


def file_hashes(path: Path | str) -> dict:
    hashes = {name: hashlib.new(name) for name in ('md5', 'sha256')}
    size = 0
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            size += len(block)
            for h in hashes.values():
                h.update(block)
    return {name: h.hexdigest() for name, h in hashes.items()} | {'bytes': size}


def verify_asset(path: Path | str, expected_md5: str) -> dict:
    result = file_hashes(path)
    if result['md5'] != expected_md5:
        raise DataIntegrityError(f'Published MD5 mismatch for {Path(path).name}; refuse replacement.')
    return result


def write_json_once(path: Path | str, value: dict) -> None:
    encoded = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf-8')
    with Path(path).open('xb') as f:
        f.write(encoded)


def _save_array_once(path: Path, array: np.ndarray) -> dict:
    with path.open('xb') as f:
        np.save(f, array, allow_pickle=False)
    return {'file': path.name, 'sha256': file_hashes(path)['sha256'],
            'dtype': array.dtype.str, 'shape': list(array.shape),
            'array_sha256': hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()}


def _outside_code(path: Path) -> None:
    code_root = Path(__file__).resolve().parents[1]
    if path.resolve().is_relative_to(code_root):
        raise DataGateError('Dataset/result bytes must be outside the code and public root.')


def _download_once(url: str, target: Path) -> None:
    # Only immutable, pinned official repository files are allowed here.
    allowed = {BASE_URL + 'LICENSE'} | {BASE_URL + 'data/fashion/' + n for n in ASSETS}
    if url not in allowed:
        raise DataIntegrityError('Unrecognized download URL.')
    if target.exists():
        raise FileExistsError(target)
    temp_name = None
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            if response.geturl() != url:
                raise DataIntegrityError('Unexpected download redirect; no alternate source permitted.')
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.download-', delete=False) as f:
                temp_name = f.name
                total = 0
                while block := response.read(1024 * 1024):
                    total += len(block)
                    if total > 60 * 1024 * 1024:
                        raise DataIntegrityError('Asset exceeds bounded download size.')
                    f.write(block)
        # Atomic no-overwrite publication of completed bytes, even with another writer.
        os.link(temp_name, target)
    finally:
        if temp_name is not None:
            Path(temp_name).unlink(missing_ok=True)


def download_verified_assets(data_root: Path | str = DEFAULT_DATA_ROOT) -> dict:
    root = Path(data_root)
    _outside_code(root)
    raw = root / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    result = {}
    for name, (md5, _, _) in ASSETS.items():
        path = raw / name
        url = BASE_URL + 'data/fashion/' + name
        if not path.exists():
            _download_once(url, path)
        result[name] = verify_asset(path, md5) | {'url': url, 'published_md5': md5}
    license_path = raw / 'LICENSE'
    if not license_path.exists():
        _download_once(BASE_URL + 'LICENSE', license_path)
    license_text = license_path.read_text(encoding='utf-8')
    if 'The MIT License (MIT)' not in license_text or '2017 Zalando SE' not in license_text:
        raise DataIntegrityError('Expected official MIT notice is absent.')
    result['LICENSE'] = file_hashes(license_path) | {'url': BASE_URL + 'LICENSE', 'license': 'MIT'}
    return result



def verify_source_manifest(data_root: Path | str, manifest: dict) -> dict:
    """Recheck all pinned raw sources against the frozen preparation manifest.

    This integrity-only check includes both train/test compressed bytes and the
    retained license; it never decompresses official-test images or predicts.
    Published MD5 checks remain additional upstream consistency gates, not a
    substitute for the manifest's SHA256 and exact byte count.
    """
    expected_names = set(ASSETS) | {'LICENSE'}
    entries = manifest.get('assets')
    if not isinstance(entries, dict) or set(entries) != expected_names:
        raise DataIntegrityError('Manifest must identify exactly four pinned assets and LICENSE.')
    raw = Path(data_root) / 'raw'
    verified = {}
    for name in sorted(expected_names):
        info = entries[name]
        expected_url = BASE_URL + ('LICENSE' if name == 'LICENSE' else 'data/fashion/' + name)
        if not isinstance(info, dict) or info.get('url') != expected_url:
            raise DataIntegrityError(f'Pinned source URL mismatch: {name}.')
        try:
            actual = file_hashes(raw / name)
        except OSError as exc:
            raise DataIntegrityError(f'Cannot verify required raw source: {name}.') from exc
        for field in ('sha256', 'bytes', 'md5'):
            if info.get(field) != actual[field]:
                raise DataIntegrityError(f'Raw source manifest {field} mismatch: {name}.')
        if name in ASSETS:
            published_md5 = ASSETS[name][0]
            if actual['md5'] != published_md5 or info.get('published_md5') != published_md5:
                raise DataIntegrityError(f'Published MD5 mismatch: {name}.')
        elif info.get('license') != 'MIT':
            raise DataIntegrityError('Manifest must retain the official MIT license designation.')
        verified[name] = actual
    return verified


def decode_idx(content: bytes, *, images: bool, expected_count: int) -> np.ndarray:
    header_len = 16 if images else 8
    if len(content) < header_len:
        raise DataIntegrityError('Truncated IDX header.')
    header = struct.unpack('>IIII' if images else '>II', content[:header_len])
    if header[:2] != ((2051 if images else 2049), expected_count):
        raise DataIntegrityError('IDX magic or sample count mismatch.')
    if images and header[2:] != (28, 28):
        raise DataIntegrityError('IDX dimensions are not 28 by 28.')
    expected_bytes = expected_count * (784 if images else 1)
    if len(content) != header_len + expected_bytes:
        raise DataIntegrityError('IDX payload length mismatch.')
    array = np.frombuffer(content, dtype=np.uint8, offset=header_len)
    if not images and np.any(array > 9):
        raise DataIntegrityError('Fashion-MNIST labels must be in 0..9.')
    return array.reshape(expected_count, 28, 28) if images else array


def _read_asset(raw: Path, name: str) -> np.ndarray:
    md5, count, images = ASSETS[name]
    verify_asset(raw / name, md5)
    with gzip.open(raw / name, 'rb') as f:
        content = f.read(16 + count * (784 if images else 1) + 1)
    return decode_idx(content, images=images, expected_count=count)


def load_verified_assets(data_root: Path | str, *, include_test: bool = False) -> dict:
    """Integrity-only helper; never call with include_test in a learner process."""
    raw = Path(data_root) / 'raw'
    result = {'train_images': _read_asset(raw, 'train-images-idx3-ubyte.gz'),
              'train_labels': _read_asset(raw, 'train-labels-idx1-ubyte.gz')}
    if include_test:
        result.update(test_images=_read_asset(raw, 't10k-images-idx3-ubyte.gz'),
                      test_labels=_read_asset(raw, 't10k-labels-idx1-ubyte.gz'))
    for split in ('train', 'test'):
        if split + '_labels' in result:
            expected = 6000 if split == 'train' else 1000
            if not np.array_equal(np.bincount(result[split + '_labels'], minlength=10), [expected] * 10):
                raise DataIntegrityError(f'Official {split} class counts are unexpected.')
    return result


def preprocess_images(images: np.ndarray) -> np.ndarray:
    images = np.asarray(images)
    if images.dtype != np.uint8 or images.ndim != 3 or images.shape[1:] != (28, 28):
        raise ValueError('Expected uint8 images of shape (n, 28, 28).')
    # Every four-pixel integer sum is exactly representable in float32.
    blocked = images.reshape(-1, 14, 2, 14, 2)
    return (blocked.mean(axis=(2, 4), dtype=np.float32).reshape(-1, 196) / np.float32(255)).astype(np.float32)


def feature_hashes(features: np.ndarray) -> np.ndarray:
    """SHA256 of each row's canonical contiguous little-endian feature bytes."""
    features = np.asarray(features)
    dtype = features.dtype.newbyteorder('<')
    canonical = np.ascontiguousarray(features, dtype=dtype)
    return np.array([hashlib.sha256(row.tobytes()).hexdigest().encode('ascii') for row in canonical], dtype='S64')


def _match_counts(left: np.ndarray, right: np.ndarray) -> dict:
    a, b = Counter(left.tolist()), Counter(right.tolist())
    common = a.keys() & b.keys()
    return {'groups': len(common), 'left_items': sum(a[h] for h in common),
            'right_items': sum(b[h] for h in common), 'pairs': sum(a[h] * b[h] for h in common)}


def make_split_indices(labels: np.ndarray, transformed_hashes: np.ndarray | None = None,
                       *, train_per_class: int = 1000, val_per_class: int = 500,
                       seed: int = 73102) -> dict:
    labels = np.asarray(labels)
    if labels.ndim != 1 or train_per_class < 1 or val_per_class < 1:
        raise ValueError('Expected one-dimensional labels and positive class quotas.')
    rng = np.random.Generator(np.random.PCG64(seed))
    permutations = [rng.permutation(np.flatnonzero(labels == c)) for c in np.unique(labels)]
    if any(len(p) < train_per_class + val_per_class for p in permutations):
        raise DataGateError('Insufficient official training examples for fixed class quotas.')
    train = np.concatenate([p[:train_per_class] for p in permutations]).astype('<i8')
    val = np.concatenate([p[train_per_class:train_per_class + val_per_class] for p in permutations]).astype('<i8')
    result = {'train_ids': train, 'val_ids': val, 'initial_train_ids': train.copy(),
              'initial_val_ids': val.copy(), 'amended': False}
    if transformed_hashes is None:
        return result
    hashes = np.asarray(transformed_hashes)
    if hashes.shape != labels.shape:
        raise ValueError('Each official training item needs one feature hash.')
    if not set(hashes[train]) & set(hashes[val]):
        return result
    # Training remains exactly the original predeclared 1000/class. Assign all
    # its full-dataset feature groups to training. Validation scans the original
    # class permutations from their start, skipping every training-owned group.
    # Unselected group members are ignored, never allocated to the other split.
    train_groups = set(hashes[train])
    new_val = []
    for permutation in permutations:
        eligible = [int(i) for i in permutation if hashes[i] not in train_groups]
        if len(eligible) < val_per_class:
            raise DataGateError('Feature-grouped validation cannot meet fixed class quotas.')
        new_val.extend(eligible[:val_per_class])
    result['val_ids'] = np.asarray(new_val, dtype='<i8')
    result['amended'] = True
    return result


def audit_cross_split_duplicates(raw_reference: np.ndarray, transformed_reference: np.ndarray,
                                 raw_test: np.ndarray, transformed_test: np.ndarray) -> dict:
    if len(raw_reference) != len(transformed_reference) or len(raw_test) != len(transformed_test):
        raise ValueError('Raw and transformed feature hashes must have matching lengths.')
    raw_set, transformed_set = set(raw_reference), set(transformed_reference)
    raw_matches = np.array([h in raw_set for h in raw_test], dtype=bool)
    transformed_matches = np.array([h in transformed_set for h in transformed_test], dtype=bool)
    exclude = raw_matches | transformed_matches
    return {'raw_match_items': int(raw_matches.sum()),
            'transformed_match_items': int(transformed_matches.sum()),
            'both_match_items': int((raw_matches & transformed_matches).sum()),
            'excluded_items': int(exclude.sum()), 'retained_items': int((~exclude).sum()),
            'raw': _match_counts(raw_reference, raw_test),
            'transformed': _match_counts(transformed_reference, transformed_test),
            'keep_ids': np.flatnonzero(~exclude).astype('<i8'),
            'excluded_ids': np.flatnonzero(exclude).astype('<i8')}


def check_retained_class_counts(labels: np.ndarray, keep_ids: np.ndarray, *, minimum: int = 500) -> list:
    counts = np.bincount(np.asarray(labels)[keep_ids], minlength=10).tolist()
    if len(counts) != 10 or min(counts) < minimum:
        raise DataGateError(f'Retained official test fails >= {minimum}/class gate: {counts}.')
    return counts


def prepare_dataset(data_root: Path | str = DEFAULT_DATA_ROOT, result_dir: Path | str | None = None) -> dict:
    root = Path(data_root)
    out = Path(result_dir) if result_dir is not None else root / 'prepared-v1'
    _outside_code(root)
    _outside_code(out)
    if out.exists():
        raise FileExistsError(f'Immutable preparation already exists: {out}')
    source_manifest = download_verified_assets(root)
    assets = load_verified_assets(root, include_test=True)
    train_raw = feature_hashes(assets['train_images'])
    train_transformed = feature_hashes(preprocess_images(assets['train_images']))
    test_raw = feature_hashes(assets['test_images'])
    test_transformed = feature_hashes(preprocess_images(assets['test_images']))
    split = make_split_indices(assets['train_labels'], train_transformed)
    audits = {}
    for prefix, train_ids, val_ids in [('initial', split['initial_train_ids'], split['initial_val_ids']),
                                       ('final', split['train_ids'], split['val_ids'])]:
        audits[prefix] = {'raw': _match_counts(train_raw[train_ids], train_raw[val_ids]),
                          'transformed': _match_counts(train_transformed[train_ids], train_transformed[val_ids])}
    if any(audits['final'][kind]['groups'] for kind in ('raw', 'transformed')):
        raise DataGateError('Train-validation leakage remains after prospective split amendment.')
    reference = np.concatenate([split['train_ids'], split['val_ids']])
    test_audit = audit_cross_split_duplicates(train_raw[reference], train_transformed[reference],
                                             test_raw, test_transformed)
    keep_ids, excluded_ids = test_audit.pop('keep_ids'), test_audit.pop('excluded_ids')
    counts = np.bincount(assets['test_labels'][keep_ids], minlength=10).tolist()
    class_gate = len(counts) == 10 and min(counts) >= 500
    # Saving occurs only after deterministic preprocessing and audits finish.
    out.mkdir(parents=True, exist_ok=False)
    arrays = {k: v for k, v in split.items() if isinstance(v, np.ndarray)}
    arrays.update(train_raw_sha256=train_raw, train_transformed_sha256=train_transformed,
                  test_raw_sha256=test_raw, test_transformed_sha256=test_transformed,
                  test_keep_ids=keep_ids, test_excluded_ids=excluded_ids)
    array_manifest = {name: _save_array_once(out / (name + '.npy'), arr) for name, arr in arrays.items()}
    doc_path = Path(__file__).resolve().parents[1] / 'docs' / 'data-amendment.md'
    manifest = {
        'schema_version': SCHEMA_VERSION,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_revision': REVISION, 'source_repository': 'https://github.com/zalandoresearch/fashion-mnist',
        'assets': source_manifest, 'numpy_version': np.__version__,
        'rng': {'algorithm': 'PCG64', 'seed': 73102, 'class_order': list(range(10))},
        'preprocessing': '28x28 uint8; nonoverlapping 2x2 mean float32; /float32(255); row-major 196',
        'feature_hash_rule': 'SHA256 of canonical contiguous little-endian raw uint8 or transformed float32 bytes',
        'split_version': SPLIT_VERSION if split['amended'] else 'original-pcg64-stratified-v1',
        'prospective_amendment_applied': split['amended'],
        'amendment_sha256': file_hashes(doc_path)['sha256'],
        'train_count': len(split['train_ids']), 'validation_count': len(split['val_ids']),
        'train_class_counts': np.bincount(assets['train_labels'][split['train_ids']], minlength=10).tolist(),
        'validation_class_counts': np.bincount(assets['train_labels'][split['val_ids']], minlength=10).tolist(),
        'train_validation_duplicate_audit': audits, 'test_duplicate_audit': test_audit,
        'test_exclusion_rule': 'raw OR transformed SHA256 matches any selected training/validation feature; no labels used',
        'retained_test_class_counts': counts, 'minimum_retained_per_class': 500,
        'class_count_gate_passed': class_gate,
        'prior_test_use_attestation': 'unresolved; explicit no-prior-selection attestation required at complete lock',
        'test_predictions_performed': False, 'arrays': array_manifest,
    }
    write_json_once(out / 'manifest.json', manifest)
    if not class_gate:
        raise DataGateError(f'Test retained-count gate failed; integrity report saved at {out / "manifest.json"}.')
    return manifest


def _verified_preparation(result_dir: Path | str) -> tuple[dict, dict]:
    out = Path(result_dir)
    with (out / 'manifest.json').open(encoding='utf-8') as f:
        manifest = json.load(f)
    if manifest.get('schema_version') != SCHEMA_VERSION or manifest.get('source_revision') != REVISION:
        raise DataIntegrityError('Unknown preparation schema or source revision.')
    arrays = {}
    for name, info in manifest['arrays'].items():
        path = out / info['file']
        if path.parent.resolve() != out.resolve() or file_hashes(path)['sha256'] != info['sha256']:
            raise DataIntegrityError(f'Prepared index/hash artifact mismatch: {name}.')
        array = np.load(path, allow_pickle=False)
        if list(array.shape) != info['shape'] or array.dtype.str != info['dtype']:
            raise DataIntegrityError(f'Prepared array schema mismatch: {name}.')
        if hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest() != info['array_sha256']:
            raise DataIntegrityError(f'Prepared array payload mismatch: {name}.')
        arrays[name] = array
    return manifest, arrays


def load_development(data_root: Path | str, result_dir: Path | str) -> dict:
    """Return only development features/labels/official source IDs, never test arrays."""
    manifest, indices = _verified_preparation(result_dir)
    if not manifest['class_count_gate_passed']:
        raise DataGateError('Preparation did not pass the retained test class-count gate.')
    verify_source_manifest(data_root, manifest)
    assets = load_verified_assets(data_root, include_test=False)
    train_ids, val_ids = indices['train_ids'], indices['val_ids']
    return {'x_train': preprocess_images(assets['train_images'][train_ids]),
            'y_train': assets['train_labels'][train_ids], 'train_ids': train_ids.copy(),
            'x_val': preprocess_images(assets['train_images'][val_ids]),
            'y_val': assets['train_labels'][val_ids], 'val_ids': val_ids.copy()}


def load_test_for_scoring(data_root: Path | str, result_dir: Path | str, *,
                          lock_path: Path | str | None,
                          verify_lock: Callable | None) -> dict:
    """Scorer-only entry. A complete lock verifier must return explicit evidence.

    The verifier is supplied by the orchestration module, and must validate all
    locked artifacts/configuration. Required returned keys: complete=True,
    no_prior_test_selection=True, and data_manifest_sha256 matching this exact
    preparation. A boolean or user assertion alone is deliberately insufficient.
    """
    if lock_path is None or verify_lock is None or not callable(verify_lock):
        raise DataGateError('Official test scoring requires a complete verified lock.')
    evidence = verify_lock(Path(lock_path))
    expected_hash = file_hashes(Path(result_dir) / 'manifest.json')['sha256']
    if (not isinstance(evidence, dict) or evidence.get('complete') is not True
            or evidence.get('no_prior_test_selection') is not True
            or evidence.get('data_manifest_sha256') != expected_hash):
        raise DataGateError('Lock verification lacks complete provenance and matching data evidence.')
    manifest, indices = _verified_preparation(result_dir)
    if manifest.get('class_count_gate_passed') is not True:
        raise DataGateError('Official test retained-count gate failed.')
    verify_source_manifest(data_root, manifest)
    raw = Path(data_root) / 'raw'
    images = _read_asset(raw, 't10k-images-idx3-ubyte.gz')
    labels = _read_asset(raw, 't10k-labels-idx1-ubyte.gz')
    keep = indices['test_keep_ids']
    check_retained_class_counts(labels, keep)
    return {'x_test': preprocess_images(images), 'y_test': labels, 'keep_ids': keep.copy()}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Integrity and split preparation only. No predictions.')
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument('--result-dir', type=Path)
    args = parser.parse_args()
    result = prepare_dataset(args.data_root, args.result_dir)
    print(json.dumps({'split_version': result['split_version'],
                      'train_validation_duplicate_audit': result['train_validation_duplicate_audit'],
                      'test_duplicate_audit': result['test_duplicate_audit'],
                      'retained_test_class_counts': result['retained_test_class_counts']}, indent=2))
