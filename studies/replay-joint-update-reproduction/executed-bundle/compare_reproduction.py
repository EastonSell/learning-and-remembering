#!/usr/bin/env python3
"""Validate saved freeze/receipt claims and exact science JSON, not execution observation."""
import argparse
import json
from pathlib import Path
from jointaudit.admission import verify_completion
import portable

def same_scientific_json(a,b,*,summary=False):
    if summary:
        a={k:v for k,v in a.items() if k!='resources_sampled_before_summary_write'}
        b={k:v for k,v in b.items() if k!='resources_sampled_before_summary_write'}
    # Canonical JSON also distinguishes integer and floating-point representations.
    return json.dumps(a,sort_keys=True,allow_nan=False)==json.dumps(b,sort_keys=True,allow_nan=False)

def _unique_object(pairs):
    result={}
    for key,value in pairs:
        if key in result: raise ValueError('duplicate JSON key in run input freeze')
        result[key]=value
    return result

def compare_run(run,*,exit_code):
    run=Path(run)
    portable.verify_package()
    names=[f'seed-{seed}-block-{block}.json' for seed in (2001,2002,2003,2004) for block in (0,1,2,3)]
    required=set(names)|{'summary.json','run-input-freeze.json','COMPLETE.json'}
    if run.is_symlink() or not run.is_dir(): raise ValueError('regular run directory required')
    if any(x.is_symlink() or not x.is_file() for x in run.iterdir()): raise ValueError('nonregular run artifact')
    if {x.name for x in run.iterdir()}!=required: raise ValueError('exact 19-file run inventory required')
    receipt=verify_completion(run,exit_code=exit_code)
    receipt_fields={'status','states','seed_clusters','files','artifact_bytes_including_complete','resources','success_requires'}
    if set(receipt)!=receipt_fields or receipt['status']!='posthoc_diagnostic_complete': raise ValueError('wrong completion receipt schema/status')
    if set(receipt['files'])!=required-{'COMPLETE.json'}: raise ValueError('wrong receipt-bound run inventory')
    if type(receipt['states']) is not int or type(receipt['seed_clusters']) is not int or receipt['states']!=16 or receipt['seed_clusters']!=4:
        raise ValueError('wrong state/cluster inventory')
    frozen=json.loads((run/'run-input-freeze.json').read_text(),object_pairs_hook=_unique_object)
    binding_count=portable.validate_run_freeze(frozen)
    expected=portable.ROOT/'expected/diagnostic-v2'
    comparisons={name:same_scientific_json(json.loads((expected/name).read_text()),json.loads((run/name).read_text())) for name in names}
    comparisons['summary.json']=same_scientific_json(json.loads((expected/'summary.json').read_text()),json.loads((run/'summary.json').read_text()),summary=True)
    return {'format':'joint-reproduction-scientific-comparison-v2',
        'completion_receipt_integrity_valid':True,'exact_scientific_json_match':all(comparisons.values()),
        'supplied_exit_code':exit_code,'exit_code_source':'Caller-supplied; the comparator did not observe process execution.',
        'execution_observed_by_comparator':False,
        'run_input_freeze_matches_current_package_and_training_catalog':True,'source_input_binding_count':binding_count,
        'comparisons':comparisons,'historical_execution_freeze_reproduced':False,
        'resource_measurements_compared':False,'new_freeze_sha256':portable.sha256_file(run/'run-input-freeze.json'),
        'scope':'Exact freeze/receipt claim validation against the current package, manifest and two TRAIN catalog hashes; exact 16 state JSON payloads and summary except resource timing; not independent observation or proof that computation occurred. No numerical tolerance is silently applied.'}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--exit-code',type=int,required=True)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    result=compare_run(args.run,exit_code=args.exit_code)
    encoded=json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n'
    if args.output:
        with args.output.open('x') as f: f.write(encoded)
    print(encoded,end='')
    if not result['exact_scientific_json_match']: raise SystemExit(1)

if __name__=='__main__': main()
