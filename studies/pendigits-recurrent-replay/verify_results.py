import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json, pathlib, hashlib,numpy as np
from experiment import metrics,SEEDS,ARMS,ROOT
out=ROOT/'results'; rows=json.loads((out/'metrics.json').read_text()); done=json.loads((out/'COMPLETE.json').read_text()); summary=json.loads((out/'summary.json').read_text()); freeze=json.loads((ROOT/'freeze.json').read_text())
assert not (out/'INCOMPLETE').exists(); assert len(rows)==30 and done['cases']==30
assert hashlib.sha256((ROOT/'experiment.py').read_bytes()).hexdigest()==freeze['source_sha256']
assert hashlib.sha256((ROOT/'PROTOCOL.md').read_bytes()).hexdigest()==freeze['protocol_sha256']
test=np.load(out/'test-inputs.npz'); checked=0
for s in SEEDS:
 counts=[]
 for a in ARMS:
  rr=[r for r in rows if r['seed']==s and r['arm']==a]; assert len(rr)==1
  row=rr[0]; assert len(row['boundaries'])==10; counts.append(row['counts_per_arm'])
  for i in range(10):
   z=np.load(out/f'{s}-{a}-{i}.npz'); w=[z[f'w{j}'] for j in range(4)]; m,p,probs=metrics(w,test['x'],test['y']); assert np.array_equal(p,z['predictions']); assert np.array_equal(probs,z['probabilities']); saved=row['boundaries'][i]
   assert m['macro']==saved['macro'] and m['micro']==saved['micro'] and m['recalls']==saved['recalls']
   checked+=1
  assert row['final_macro']==row['boundaries'][-1]['macro']
 assert counts[0]==counts[1]==counts[2]
 assert counts[0]['actual_steps']==counts[0]['virtual_steps']
rng=np.random.default_rng(20261025); d=np.array([next(a['final_macro'] for a in rows if a['seed']==s and a['arm']=='mir')-next(a['final_macro'] for a in rows if a['seed']==s and a['arm']=='uniform') for s in SEEDS]); ci=np.quantile(rng.choice(d,size=(10000,10),replace=True).mean(1),[.025,.975]); assert np.array_equal(ci,summary['paired_seed_bootstrap_95']); assert d.mean()==summary['mean_difference']
result={'verification':'pass','checkpoint_prediction_pairs_reproduced':checked,'all_30_cases_accounted':True,'within_seed_compute_counts_equal':True,'bootstrap_recomputed':True,'source_and_protocol_match_freeze':True}; (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result))
