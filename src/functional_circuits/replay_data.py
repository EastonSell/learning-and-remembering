"""Offline UCI Optdigits integrity gate for the frozen, unrun replay protocol.

No downloads, archive extraction, fitting, or model selection occur here.
"""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import platform
import re
import stat
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / 'docs/protocols/replay_digits_v1.md'
SOURCE_URL = ('https://archive.ics.uci.edu/static/public/80/'
              'optical%2Brecognition%2Bof%2Bhandwritten%2Bdigits.zip')
COUNTS = {'optdigits.tra': 3823, 'optdigits.tes': 1797}
CLASS_COUNTS = {'optdigits.tra': [376,389,380,389,387,376,377,387,380,382],
                'optdigits.tes': [178,182,177,183,181,182,181,179,174,180]}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def parse_rows(raw, expected_rows):
    """Reject decimals, overflow, nonfinite values and shape errors before casting."""
    try:
        lines = raw.decode('ascii').splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError('Dataset must be ASCII') from exc
    if len(lines) != expected_rows:
        raise ValueError(f'Expected {expected_rows} rows; found {len(lines)}')
    rows = []
    for number, line in enumerate(lines):
        fields = line.split(',')
        if len(fields) != 65 or any(not re.fullmatch(r'[0-9]{1,2}', f.strip()) for f in fields):
            raise ValueError(f'Invalid integer row {number}')
        row = [int(f) for f in fields]
        if any(v > 16 for v in row[:64]) or row[64] > 9:
            raise ValueError(f'Out-of-range row {number}')
        rows.append(row)
    values = np.asarray(rows, dtype=np.int16)
    return values[:, :64], values[:, 64].astype(np.int64)


def split_training(labels):
    """Original file row IDs; one RNG, classes in order, no test-data access."""
    labels = np.asarray(labels)
    if set(labels.tolist()) != set(range(10)):
        raise ValueError('Training file must contain all ten labels')
    rng = np.random.default_rng(24601)
    train, validation = [], []
    for label in range(10):
        ids = rng.permutation(np.flatnonzero(labels == label))
        n = len(ids) // 5
        if n == 0 or n == len(ids):
            raise ValueError('Each class needs both training and validation rows')
        validation.extend(ids[:n].tolist())
        train.extend(ids[n:].tolist())
    return np.asarray(train, dtype=np.int64), np.asarray(validation, dtype=np.int64)


def audit_duplicates(splits):
    """Return all duplicate groups as IDs, not dataset feature values."""
    groups = {}
    for name, (features, labels, ids) in splits.items():
        if not (len(features) == len(labels) == len(ids)):
            raise ValueError('Split arrays have different lengths')
        for x, y, row_id in zip(features, labels, ids):
            key = np.asarray(x, dtype='<i2').tobytes()
            groups.setdefault(key, []).append((name, int(row_id), int(y)))
    duplicates, cross, conflicts = [], 0, 0
    for members in groups.values():
        if len(members) < 2:
            continue
        counts = Counter(row[0] for row in members)
        conflict = len({row[2] for row in members}) > 1
        cross += len(counts) > 1
        conflicts += conflict
        duplicates.append(dict(rows=len(members), split_counts=dict(counts),
                               conflicting_labels=conflict,
                               members=[dict(split=s, row_id=i, label=y) for s, i, y in members]))
    return dict(passed=not (cross or conflicts), cross_split_groups=cross,
                conflicting_label_groups=conflicts, duplicate_groups=duplicates)


def inspect_archive(path, max_compressed=10_000_000, max_expanded=50_000_000):
    """Read bounded members in memory, never extract arbitrary archive paths."""
    path = Path(path)
    if path.stat().st_size > max_compressed:
        raise ValueError('Compressed archive exceeds limit')
    raw = path.read_bytes()
    if len(raw) > max_compressed:
        raise ValueError('Compressed archive exceeds limit')
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        infos = z.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise ValueError('Duplicate archive member names')
        if sum(info.file_size for info in infos) > max_expanded:
            raise ValueError('Expanded archive exceeds limit')
        for info in infos:
            name = info.filename
            if (PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts
                    or '\\' in name or ':' in name or '\x00' in name
                    or stat.S_ISLNK(info.external_attr >> 16) or info.flag_bits & 1):
                raise ValueError('Unsafe archive member')
        arrays, metadata = {}, {}
        for name, count in COUNTS.items():
            if name not in names:
                raise ValueError(f'Missing required member: {name}')
            info = z.getinfo(name)
            with z.open(info) as f:
                member = f.read(min(info.file_size + 1, max_expanded + 1))
            if len(member) != info.file_size:
                raise ValueError('Member size mismatch')
            arrays[name] = parse_rows(member, count)
            observed = np.bincount(arrays[name][1], minlength=10).tolist()
            if observed != CLASS_COUNTS[name]:
                raise ValueError(f'Unexpected official class counts: {name}')
            metadata[name] = dict(bytes=len(member), sha256=digest(member),
                                  rows=count, class_counts=observed)
    return arrays, dict(archive_bytes=len(raw), archive_sha256=digest(raw),
                        members=metadata, archive_member_names=names)


def validate(path):
    arrays, metadata = inspect_archive(path)
    x, y = arrays['optdigits.tra']
    xt, yt = arrays['optdigits.tes']
    train, validation = split_training(y)
    ids = dict(train=train.tolist(), validation=validation.tolist(), test=list(range(len(yt))))
    audit = audit_duplicates(dict(train=(x[train], y[train], train),
                                  validation=(x[validation], y[validation], validation),
                                  test=(xt, yt, np.arange(len(yt)))))
    report = dict(schema_version=1, stage='data_integrity_only',
                  status='passed' if audit['passed'] else 'blocked_by_frozen_data_policy',
                  training_performed=False, source_url=SOURCE_URL,
                  provenance_note='Source URL identifies the official download; this offline validator does not authenticate download provenance.',
                  dataset_doi='10.24432/C50P49', license='CC-BY-4.0',
                  creators=['E. Alpaydin', 'C. Kaynak'],
                  python=platform.python_version(), numpy=np.__version__,
                  protocol_sha256=digest(PROTOCOL.read_bytes()),
                  source_sha256={str(Path(__file__).relative_to(ROOT)): digest(Path(__file__).read_bytes())},
                  split_ids=ids, split_id_namespace='Original zero-based row in .tra (train/validation) or .tes (test)',
                  split_id_sha256={k:digest(np.asarray(v,dtype='<i8').tobytes()) for k,v in ids.items()},
                  split_counts={k:len(v) for k,v in ids.items()}, duplicate_audit=audit,
                  **metadata)
    return report


def write_report(path, report):
    """Exclusive creation: never overwrite an existing result, even if empty."""
    encoded = json.dumps(report, indent=2, allow_nan=False) + '\n'
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        f.write(encoded)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; use a new path')
    report = validate(args.archive)
    write_report(args.output, report)
    print(json.dumps({k:report[k] for k in ('status','split_counts','archive_sha256','training_performed')}))
    if report['status'] != 'passed':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
