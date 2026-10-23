#!/usr/bin/env python3
"""Verify public saved numerical evidence using only the standard library."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import sys

ROOT=Path(__file__).resolve().parent
EXECUTED_MANIFEST='3cb7c99ff63c95d0a24cf8c77fa8fabbb3bed96d699ac26c0176bbba37aace0f'
RUN_FREEZE='fd604745e8986617d22c2e736fc44519000a07a753b51c3477538e3b09d2ed16'
RUN_COMPLETE='7c66754505b6efa6440ad5dfb4199a251836457f3223a01d7bb782820bddfc66'
SUMMARY_SCIENCE='fa9f4312192fc5289610d2d8c6d033878e74b4d462a0f8badd122b9388f1fbf6'

def identity(path):
    data=Path(path).read_bytes()
    return {'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),
        'git_blob_sha1':hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()}

def verify_inventory(root=ROOT):
    root=Path(root)
    manifest=json.loads((root/'PUBLIC-FILES.json').read_text())
    if any(p.is_symlink() for p in root.rglob('*')): raise ValueError('symlink public file')
    actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()
        and '__pycache__' not in p.parts and p!=root/'PUBLIC-FILES.json'}
    if actual!=set(manifest['files']): raise ValueError('public file inventory mismatch')
    for name,want in manifest['files'].items():
        relative=PurePosixPath(name)
        if relative.is_absolute() or str(relative)!=name or '..' in relative.parts:
            raise ValueError('unsafe public allowlist path')
        if identity(root/name)!=want: raise ValueError('public file identity mismatch: '+name)
    return len(actual)

def verify_saved(root=ROOT):
    root=Path(root)
    count=verify_inventory(root)
    source=root/'executed-bundle'
    run=root/'results/reproduction-001'
    if identity(source/'PACKAGE-FILES.json')['sha256']!=EXECUTED_MANIFEST:
        raise ValueError('executed source-bundle manifest changed')
    if identity(run/'run-input-freeze.json')['sha256']!=RUN_FREEZE:
        raise ValueError('accepted new-run freeze changed')
    if identity(run/'COMPLETE.json')['sha256']!=RUN_COMPLETE:
        raise ValueError('accepted new-run completion changed')
    # This module imports admission/portable only, never NumPy or numerical runner.
    sys.path.insert(0,str(source))
    spec=importlib.util.spec_from_file_location('retained_joint_comparator',source/'compare_reproduction.py')
    comparator=importlib.util.module_from_spec(spec);spec.loader.exec_module(comparator)
    report=comparator.compare_run(run,exit_code=0)
    saved=json.loads((root/'comparison.json').read_text())
    if report!=saved or not report['exact_scientific_json_match']:
        raise ValueError('saved numerical comparison changed or mismatched')
    rows=[f'seed-{s}-block-{b}.json' for s in (2001,2002,2003,2004) for b in (0,1,2,3)]
    if not all((run/n).read_bytes()==(source/'expected/diagnostic-v2'/n).read_bytes() for n in rows):
        raise ValueError('state files are not byte-identical')
    summary=json.loads((run/'summary.json').read_text())
    science={k:v for k,v in summary.items() if k!='resources_sampled_before_summary_write'}
    canonical=(json.dumps(science,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
    if hashlib.sha256(canonical).hexdigest()!=SUMMARY_SCIENCE:
        raise ValueError('scientific summary projection changed')
    if 'numpy' in sys.modules: raise ValueError('saved-only verifier unexpectedly imported NumPy')
    return {'format':'public-joint-numerical-reproduction-verification-v1',
        'public_manifest_listed_files_verified':count,'executed_source_bundle_files':80,
        'source_input_bindings':report['source_input_binding_count'],
        'all_sixteen_state_files_byte_identical':True,
        'scientific_summary_exact_except_measured_resource_sample':True,
        'scientific_summary_canonical_sha256':SUMMARY_SCIENCE,
        'new_complete_and_freeze_integrity_valid':True,
        'saved_comparison_recomputed_exactly':True,
        'model_execution_performed_by_verifier':False,'numpy_imported':False,
        'execution_independently_attested':False,
        'reported_scientific_exit':0,'reported_comparator_exit':0,
        'observation_boundary':'Independent saved numerical/integrity checks pass. Foreground completion is reported from the author-retained tool receipt; this verifier is not a live witness or independent OS/process attestation.'}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    report=verify_saved()
    encoded=json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+'\n'
    if args.output:
        with args.output.open('x') as f:f.write(encoded)
    print(encoded,end='')

if __name__=='__main__':main()

