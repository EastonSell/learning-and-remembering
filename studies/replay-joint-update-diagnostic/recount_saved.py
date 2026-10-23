"""Saved JSON analysis only. No dataset/checkpoint/model imports or calls."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/diagnostic-v2'
DEST=ROOT
rows=[json.loads((OUT/f'seed-{seed}-block-{block}.json').read_text()) for seed in range(2001,2005) for block in range(4)]
saved=json.loads((OUT/'summary.json').read_text())

def summary(values):
    values=np.asarray(values,np.float64)
    means=np.array([values[i:i+4].mean() for i in range(0,16,4)])
    r=np.random.Generator(np.random.PCG64(7310301))
    samples=means[r.integers(4,size=(10000,4))].mean(axis=1)
    exact=np.array([means[list(index)].mean() for index in itertools.product(range(4),repeat=4)])
    return {'mean':float(means.mean()),'seed_means':means.tolist(),'ci95_frozen_mc':[float(v) for v in np.percentile(samples,[2.5,97.5])],
        'ci95_exact_empirical_four_seed_bootstrap':[float(v) for v in np.percentile(exact,[2.5,97.5])],
        'leave_one_seed_out_means':[float(np.delete(means,i).mean()) for i in range(4)],
        'states_negative_zero_positive':[int((values<0).sum()),int((values==0).sum()),int((values>0).sum())],
        'state_min':float(values.min()),'state_max':float(values.max()),'per_state':values.tolist()}

def contrast(row,mode,panel,control):
    b=row['compositions'][mode]['branches']
    v=b['highloss'][panel+'_loss'] if control=='highloss' else np.mean([b[f'uniform_{d}'][panel+'_loss'] for d in range(4)])
    return float(b['mir'][panel+'_loss']-v)

report={'scope':'posthoc saved-output analysis only; four historical stream clusters, not fresh confirmation or future retention','contrasts':{},'absolute_changes':{},'tie_audit':{}}
for mode in ['joint','sequential']:
    for panel in ['old','new']:
        for control in ['highloss','uniform']:
            key=f'{mode}_{panel}_mir_minus_{control}'
            report['contrasts'][key]=summary([contrast(row,mode,panel,control) for row in rows])
for mode in ['joint','sequential']:
    for arm in ['mir','highloss']:
        for panel in ['old','new']:
            for origin in ['original','fresh_only']:
                key=f'{mode}_{panel}_{arm}_change_from_{origin}'
                report['absolute_changes'][key]=summary([row['compositions'][mode]['branches'][arm][f'{panel}_change_from_{origin}'] for row in rows])
report['composition_interaction_old_highloss']=summary([contrast(row,'joint','old','highloss')-contrast(row,'sequential','old','highloss') for row in rows])
for key,savedkey in [('joint_old_mir_minus_highloss','joint_mir_minus_highloss'),('joint_old_mir_minus_uniform','joint_mir_minus_uniform'),('sequential_old_mir_minus_highloss','sequential_mir_minus_highloss')]:
    got=report['contrasts'][key]; want=saved['summaries'][savedkey]
    assert abs(got['mean']-want['mean'])<1e-12
    np.testing.assert_allclose(got['ci95_frozen_mc'],want['ci95'],rtol=0,atol=1e-12)
for arm in ['mir','highloss']:
    cutoffs=[item for row in rows for item in row['selections'][arm]['cutoffs']]
    report['tie_audit'][arm]={'state_selection_changes':sum(row[arm+'_tie_selection_changed'] for row in rows),'class_cutoff_records':len(cutoffs),
        'exact_straddling_cutoffs':sum(item['straddling'] for item in cutoffs),'cutoffs_with_multiple_exact_equals':sum(item['equal_count']>1 for item in cutoffs),
        'cutoffs_with_multiple_near_items':sum(item['near_count']>1 for item in cutoffs)}
report['input_output_bindings']={'freeze_sha256':json.loads((ROOT/'provenance/source-input-freeze.projection.json').read_text())['original_sha256'],
    'summary_sha256':hashlib.sha256((OUT/'summary.json').read_bytes()).hexdigest(),
    'complete_sha256':hashlib.sha256((OUT/'COMPLETE.json').read_bytes()).hexdigest(),
    'observed_external_exit':json.loads((ROOT/'provenance/execution-summary.projection.json').read_text())['observed_external_exit'],
    'source_rows_sha256':{f'seed-{row["seed"]}-block-{row["block"]}.json':hashlib.sha256((OUT/f'seed-{row["seed"]}-block-{row["block"]}.json').read_bytes()).hexdigest() for row in rows}}
parser=argparse.ArgumentParser(description='Recount SAVED JSON only; default is read-only stdout. No images, checkpoints or model code.')
parser.add_argument('--output',type=Path,help='Optional NEW write-once output JSON; existing files are never replaced.')
args=parser.parse_args()
if args.output is not None:
    with args.output.open('x') as f: json.dump(report,f,sort_keys=True,indent=2,allow_nan=False); f.write('\n')
print(json.dumps(report,sort_keys=True,indent=2,allow_nan=False))
