import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import pathlib,json,hashlib,datetime,io,numpy as np
from sklearn.model_selection import train_test_split
P=pathlib.Path(__file__).resolve().parent; O=P/'results'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
freeze=json.loads((P/'freeze.json').read_text()); review=json.loads((P/'independent-review.json').read_text()); rows=json.loads((O/'metrics.json').read_text()); summary=json.loads((O/'summary.json').read_text())
for key,file in [('source_sha256','experiment.py'),('protocol_sha256','PROTOCOL.md')]:assert sha(P/file)==freeze[key]==review[key]
assert sha(P/'independent-review.json')==freeze['review_sha256']
assert sha(P.parent/'pendigits.tra')==freeze['training_sha256']
assert (P/'independent-review.json').stat().st_mtime<=(P/'freeze.json').stat().st_mtime<(O/'test-inputs.npz').stat().st_mtime<(O/'COMPLETE.json').stat().st_mtime
assert not (O/'INCOMPLETE').exists() and len(rows)==30
D=np.loadtxt(P.parent/'pendigits.tra',delimiter=','); X=D[:,:16]/100; Y=D[:,16].astype(int); T=np.load(O/'test-inputs.npz'); tx,ty=T['x'],T['y']; S=np.load(O/'split.npz')
tr,dv=train_test_split(np.arange(len(Y)),test_size=.2,random_state=20261024,stratify=Y)
assert np.array_equal(tr,S['train']) and np.array_equal(dv,S['development'])
assert len(set(tr)|set(dv))==7494 and not set(tr)&set(dv)
sets=[set(map(tuple,a)) for a in (X[tr],X[dv],tx)]
assert all(not sets[i]&sets[j] for i,j in [(0,1),(0,2),(1,2)])
assert tx.shape==(3498,16) and np.isfinite(tx).all() and set(ty)==set(range(10))
by={(r['seed'],r['arm']):r for r in rows}; assert len(by)==30
checked=0; recomputed={}; counts_all={}; traces={}
for seed in range(31001,31011):
 rng=np.random.default_rng(seed+100000); parts={}
 for c in range(10):
  ids=tr[Y[tr]==c].copy(); rng.shuffle(ids); parts[c]=np.array_split(ids,2)
 pairs=[(0,1),(2,3),(4,5),(6,7),(8,9)]; phases=[]
 for visit,ps in enumerate([pairs,pairs[::-1]]):
  for pair in ps:
   ids=np.concatenate([parts[c][visit] for c in pair]); rng.shuffle(ids); phases.append((pair,ids))
 full=np.concatenate([ids for _,ids in phases]); assert len(full)==len(tr) and set(full)==set(tr)
 br=np.random.default_rng(seed+200000); cr=np.random.default_rng(seed+300000); ur=np.random.default_rng(seed+400000); buf=[]; seen=0; trace=hashlib.sha256()
 count=dict(actual_steps=0,virtual_steps=0,candidate_pre_post_scored=0,fresh_presentations=0,replay_presentations=0)
 for pair,ids in phases:
  for start in range(0,len(ids),32):
   fresh=ids[start:start+32]
   for _ in range(4):
    ci=cr.choice(len(buf),size=min(128,len(buf)),replace=False) if buf else []
    candidate=np.array([buf[i] for i in ci],dtype=int); assert not set(candidate)&set(fresh)
    if len(candidate):ur.choice(len(candidate),size=32,replace=len(candidate)<32)
    trace.update(candidate.tobytes()); count['actual_steps']+=1; count['virtual_steps']+=1; count['candidate_pre_post_scored']+=2*len(candidate); count['fresh_presentations']+=len(fresh); count['replay_presentations']+=32 if len(candidate) else 0
   for idx in fresh:
    seen+=1
    if len(buf)<256:buf.append(int(idx))
    else:
     j=br.integers(seen)
     if j<256:buf[j]=int(idx)
 assert seen==len(tr) and len(buf)==256
 counts_all[str(seed)]=count; traces[str(seed)]=trace.hexdigest()
 for arm in ['uniform','mir','high_loss']:
  row=by[seed,arm]; assert row['counts_per_arm']==count and len(row['boundaries'])==10
  seen_classes=set(); boundaries=[]
  for phase,(pair,_) in enumerate(phases):
   z=np.load(O/f'{seed}-{arm}-{phase}.npz'); w=[z[f'w{i}'] for i in range(4)]; assert sum(a.size for a in w)==1738 and all(np.isfinite(a).all() for a in w)
   logits=np.maximum(tx@w[0]+w[1],0)@w[2]+w[3]; logits-=logits.max(axis=1,keepdims=True); prob=np.exp(logits); prob/=prob.sum(axis=1,keepdims=True); pred=prob.argmax(axis=1)
   assert np.array_equal(pred,z['predictions']) and np.array_equal(prob,z['probabilities'])
   recalls=np.array([(pred[ty==c]==c).mean() for c in range(10)]); saved=row['boundaries'][phase]; seen_classes.update(pair); old=sorted(seen_classes-set(pair)); acq=recalls[list(pair)].mean()
   assert np.array_equal(recalls,saved['recalls']) and recalls.mean()==saved['macro'] and (pred==ty).mean()==saved['micro']
   assert acq==saved['acquisition'] and (saved['old_accuracy'] is None if not old else saved['old_accuracy']==recalls[old].mean())
   ce=-np.log(np.maximum(prob[np.arange(len(ty)),ty],1e-300)).mean(); assert ce==saved['ce']
   boundaries.append((float(recalls.mean()),float(acq))); checked+=1
  assert row['final_macro']==boundaries[-1][0] and row['mean_acquisition']==np.mean([v[1] for v in boundaries])
  recomputed[seed,arm]=(boundaries[-1][0],np.mean([v[1] for v in boundaries]))
diff=np.array([recomputed[s,'mir'][0]-recomputed[s,'uniform'][0] for s in range(31001,31011)]); acq=np.array([recomputed[s,'mir'][1]-recomputed[s,'uniform'][1] for s in range(31001,31011)])
boot=np.random.default_rng(20261025).choice(diff,size=(10000,10),replace=True).mean(axis=1); ci=np.quantile(boot,[.025,.975]); support=bool(diff.mean()>=.01 and ci[0]>0 and acq.mean()>=-.02)
assert np.array_equal(diff,summary['mir_minus_uniform']) and np.array_equal(ci,summary['paired_seed_bootstrap_95']) and diff.mean()==summary['mean_difference'] and acq.mean()==summary['mean_acquisition_difference'] and support==summary['support']
means={a:float(np.mean([recomputed[s,a][0] for s in range(31001,31011)])) for a in ['uniform','mir','high_loss']};assert means==summary['mean_final_macro']
assert summary['wall_seconds']<600 and summary['peak_rss_bytes']<1024**3
receipt={'audit':'PASS','checkpoints_independently_recomputed':checked,'source_and_protocol_hash_match_preexecution_review':True,'file_timestamp_order':'review < freeze < saved test inputs < complete; source control flow also asserts freeze hashes before downloading/opening test','split_identity_and_all_pairwise_exact_feature_duplicate_checks':'PASS','stream_reservoir_candidate_reconstruction':'PASS; counts match all arms, 5995 distinct training examples per seed, no current-batch replay leakage','counts_per_seed_per_arm':counts_all,'reconstructed_candidate_trace_sha256':traces,'bootstrap':'independently reproduced 10000 paired-seed resamples, RNG20261025','mean_final_macro':means,'mir_minus_uniform_pp':float(diff.mean()*100),'ci95_pp':(ci*100).tolist(),'acquisition_difference_pp':float(acq.mean()*100),'support':support,'reported_resource_usage':{'wall_seconds':summary['wall_seconds'],'peak_rss_bytes':summary['peak_rss_bytes']},'limitations':['No model training or tuning performed by audit.','Filesystem ordering and frozen code support preregistration timing, but are not tamper-proof independent attestation of all historical test access.','Counts verified by independent stream reconstruction and frozen shared-loop inspection; no per-update runtime RNG trace was persisted.','Resource values are producer-process measurements, not independently sampled during execution.','Only one dataset/fixed split/artificial recurrence; algorithmic seeds do not provide independent writer/population replication.','High-loss is a descriptive secondary result, not confirmatory superiority.','MIR hypothesis unsupported; CI crosses zero, so this is not proof of inferiority or equivalence. Acquisition guardrail also failed.']}
(O/'independent-audit.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
