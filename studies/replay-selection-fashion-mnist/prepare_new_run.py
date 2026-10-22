#!/usr/bin/env python3
"""Prepare a NEW freeze and operator-supplied review receipt; no research execution."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import argparse,json,hashlib,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'source'))
from replaystudy import stages
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--workspace',type=Path,required=True)
p.add_argument('--review-report',type=Path,required=True)
p.add_argument('--reviewer',required=True)
p.add_argument('--accept-reviewed-protocol',action='store_true',required=True)
p.add_argument('--attest-no-prior-test-selection',action='store_true',required=True,help='Explicitly attest the new design was not selected using test predictions; cannot be used after test-driven tuning')
p.add_argument('--no-prior-test-selection-basis',required=True)
a=p.parse_args()
if not a.review_report.is_file() or not a.review_report.read_bytes():p.error('A completed independent review report is required')
if not a.reviewer.strip() or not a.no_prior_test_selection_basis.strip():p.error('Review identity and truthful test-selection basis are required')
a.workspace.mkdir(parents=True,exist_ok=False)
f=a.workspace.resolve()/'source-freeze.json';stages.freeze(f)
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
receipt={'source_freeze_sha256':sha(f),'parent_go_ahead':True,'approved_stages':['development','lock','evaluation'],'independent_review':{'passed':True,'reviewer':a.reviewer,'report_path':str(a.review_report.resolve()),'report_sha256':sha(a.review_report)},'no_prior_test_selection':True,'test_provenance_basis':a.no_prior_test_selection_basis}
out=a.workspace.resolve()/'clearance.json';out.write_text(json.dumps(receipt,indent=2)+'\n')
stages.authorize(f,out,'development')
print('Created a new public-tree freeze and operator-supplied receipt. No experiment was run.')
