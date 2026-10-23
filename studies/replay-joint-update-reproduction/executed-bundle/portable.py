"""Standard-library-only portable integrity checks. Never decode arrays or images."""
import hashlib
import json
from pathlib import Path, PurePosixPath

ROOT=Path(__file__).resolve().parent

def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024**2),b''): h.update(chunk)
    return h.hexdigest()

def safe_file(root,relative):
    root=Path(root).resolve()
    path=PurePosixPath(relative)
    if not isinstance(relative,str) or not relative or path.is_absolute() or str(path)!=relative or any(p in ('..','.') for p in path.parts):
        raise ValueError('non-canonical relative input binding')
    out=root
    for part in path.parts:
        out=out/part
        if out.is_symlink(): raise ValueError('symlink input binding')
    if not out.is_file(): raise ValueError('missing regular input file: '+relative)
    return out

def _verified(path,info):
    if path.stat().st_size!=info['bytes'] or sha256_file(path)!=info['sha256']:
        raise ValueError('input byte/hash mismatch: '+path.name)
    return info['sha256']

def verify_packaged_inputs(root,catalog):
    root=Path(root)
    want=catalog['files']
    actual={str(p.relative_to(root)) for p in (root/'inputs').rglob('*') if p.is_file() or p.is_symlink()}
    if actual!=set(want): raise ValueError('packaged scientific input inventory mismatch')
    return {name:_verified(safe_file(root,name),info) for name,info in sorted(want.items())}

def verify_training(raw,assets):
    return {'training/'+name:_verified(safe_file(raw,name),info) for name,info in sorted(assets.items())}

def verify_frozen(frozen,root,raw):
    if not isinstance(frozen.get('files'),dict) or not frozen['files']: raise ValueError('missing portable freeze')
    for name,want in frozen['files'].items():
        path=safe_file(raw,name.removeprefix('training/')) if name.startswith('training/') else safe_file(root,name)
        if sha256_file(path)!=want: raise ValueError('portable freeze mismatch: '+name)

def verify_package(root=ROOT):
    root=Path(root)
    manifest=json.loads(safe_file(root,'PACKAGE-FILES.json').read_text())
    actual={str(p.relative_to(root)) for p in root.rglob('*')
        if (p.is_file() or p.is_symlink()) and '__pycache__' not in p.parts and p!=root/'PACKAGE-FILES.json'}
    if actual!=set(manifest['files']): raise ValueError('package file inventory mismatch')
    hashes={name:_verified(safe_file(root,name),info) for name,info in sorted(manifest['files'].items())}
    verify_packaged_inputs(root,json.loads(safe_file(root,'input-catalog.json').read_text()))
    return hashes

def expected_freeze(root=ROOT):
    """Exact schema and role identities required for this package, without decoding."""
    root=Path(root)
    files=verify_package(root)
    catalog=json.loads((root/'input-catalog.json').read_text())
    files.update({'training/'+name:info['sha256'] for name,info in sorted(catalog['external_training_assets'].items())})
    # Bind the exact allowlist too, without a self-referential manifest entry.
    files['PACKAGE-FILES.json']=sha256_file(root/'PACKAGE-FILES.json')
    return {'format':'portable-joint-reproduction-freeze-v1',
        'kind':'new-scientific-reproduction-input-freeze-not-historical',
        'scientific_execution_performed':False,'checkpoint_arrays_loaded':False,
        'image_arrays_decoded':False,'official_test_opened':False,
        'seeds':[2001,2002,2003,2004],'blocks':[0,1,2,3],
        'eta':0.3,'origin':'uniform','state_count':16,'cluster_count':4,'epsilon_nats':0.005,
        'caps':{'cpu_soft_seconds':180,'cpu_hard_backstop_seconds':181,'wall_signal_seconds':300,
            'address_space_limit_bytes':768*1024**2,'observed_rss_gate_bytes':768*1024**2,
            'artifact_output_limit_bytes':32*1024**2,'threads':1},
        'files':files}

def make_freeze(raw,root=ROOT):
    frozen=expected_freeze(root)
    catalog=json.loads((Path(root)/'input-catalog.json').read_text())
    for name,digest in verify_training(raw,catalog['external_training_assets']).items():
        if frozen['files'][name]!=digest: raise ValueError('training catalog binding mismatch')
    return frozen

def validate_run_freeze(frozen,root=ROOT):
    """Validate claims against this exact package and catalog, not observed execution."""
    want=expected_freeze(root)
    # Canonical JSON rejects differing numeric/boolean types as well as missing,
    # extra or changed schema fields, constants, caps, and every role binding.
    if json.dumps(frozen,sort_keys=True,allow_nan=False)!=json.dumps(want,sort_keys=True,allow_nan=False):
        raise ValueError('run input freeze schema/configuration/source/input identities mismatch')
    return len(want['files'])

