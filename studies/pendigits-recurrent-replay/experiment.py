import os
os.environ['OPENBLAS_NUM_THREADS']='1'; os.environ['OMP_NUM_THREADS']='1'
import numpy as np, json, time, hashlib, pathlib, urllib.request, zipfile, io, resource, sys
from sklearn.model_selection import train_test_split
ROOT=pathlib.Path(__file__).resolve().parent
ARMS=['uniform','high_loss','mir']; SEEDS=list(range(31001,31011)); LR=.05

def init(seed):
 r=np.random.default_rng(seed)
 return [r.normal(size=(16,64))*np.sqrt(2/16),np.zeros(64),r.normal(size=(64,10))*np.sqrt(2/64),np.zeros(10)]
def forward(w,x):
 h=np.maximum(0,x@w[0]+w[1]); z=h@w[2]+w[3]; z-=z.max(1,keepdims=True); p=np.exp(z); p/=p.sum(1,keepdims=True); return h,p

def losses(w,x,y):
 return -np.log(np.maximum(forward(w,x)[1][np.arange(len(y)),y],1e-300))
def step(w,x,y):
 h,p=forward(w,x); p[np.arange(len(y)),y]-=1; p/=len(y)
 dh=(p@w[2].T)*(h>0)
 grads=[x.T@dh,dh.sum(0),h.T@p,p.sum(0)]
 assert all(np.isfinite(g).all() for g in grads), 'Nonfinite gradient'
 return [v-LR*g for v,g in zip(w,grads)]
def metrics(w,x,y):
 p=forward(w,x)[1]; assert np.isfinite(p).all(); pred=p.argmax(1); recalls=[float(np.mean(pred[y==c]==c)) for c in range(10)]
 return {'macro':float(np.mean(recalls)),'micro':float(np.mean(pred==y)),'recalls':recalls,'ce':float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-300)).mean())},pred,p

def selftest():
 r=np.random.default_rng(0); x=r.normal(size=(32,16)); y=r.integers(0,10,32); w=init(0)
 before=losses(w,x,y).mean(); ww=step(w,x,y); assert losses(ww,x,y).mean()<before
 assert all(np.array_equal(a,b) for a,b in zip(w,init(0)))
 assert all(np.array_equal(a,b) for a,b in zip(ww,step(init(0),x,y)))
 def rng_trace(seed):
  br=np.random.default_rng(seed+200000); cr=np.random.default_rng(seed+300000); buf=[]; trace=[]
  for n in range(300):
   if len(buf)<256: buf.append(n)
   else:
    j=br.integers(n+1)
    if j<256: buf[j]=n
   trace.append([buf[i] for i in cr.choice(len(buf),size=min(128,len(buf)),replace=False)])
  return buf,trace
 assert rng_trace(31001)==rng_trace(31001)
 # Finite-difference agreement for a weight (not at a ReLU boundary).
 eps=1e-5; i,j=2,3; plus=[v.copy() for v in w]; minus=[v.copy() for v in w]
 plus[0][i,j]+=eps; minus[0][i,j]-=eps
 fd=(losses(plus,x,y).mean()-losses(minus,x,y).mean())/(2*eps)
 grad=(w[0][i,j]-ww[0][i,j])/LR; assert abs(fd-grad)<1e-6
 t=time.monotonic()
 for _ in range(100): w=step(w,x,y)
 print(json.dumps({'selftest':'pass','synthetic_100_updates_seconds':time.monotonic()-t}))

def run():
 started=time.monotonic(); out=ROOT/'results'; out.mkdir(exist_ok=False)
 (out/'INCOMPLETE').write_text('Execution started; only COMPLETE.json establishes completion.\n')
 freeze=json.loads((ROOT/'freeze.json').read_text())
 assert hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()==freeze['source_sha256']
 assert hashlib.sha256((ROOT/'PROTOCOL.md').read_bytes()).hexdigest()==freeze['protocol_sha256']
 raw=(ROOT.parent/'pendigits.tra').read_bytes(); assert hashlib.sha256(raw).hexdigest()==freeze['training_sha256']
 data=np.loadtxt(io.BytesIO(raw),delimiter=','); assert data.shape==(7494,17) and np.isfinite(data).all() and set(data[:,16])==set(range(10)); x=data[:,:16]/100.; y=data[:,16].astype(int)
 train,dev=train_test_split(np.arange(len(y)),test_size=.2,random_state=20261024,stratify=y)
 np.savez_compressed(out/'split.npz',train=train,development=dev)
 # Official held-out test is opened only now, after the source/protocol freeze.
 archive=urllib.request.urlopen(freeze['source_url'],timeout=30).read(); assert hashlib.sha256(archive).hexdigest()==freeze['archive_sha256']
 testbytes=zipfile.ZipFile(io.BytesIO(archive)).read('pendigits.tes'); test=np.loadtxt(io.BytesIO(testbytes),delimiter=','); assert test.shape==(3498,17) and np.isfinite(test).all() and set(test[:,16])==set(range(10)); tx=test[:,:16]/100.; ty=test[:,16].astype(int)
 np.savez_compressed(out/'test-inputs.npz',x=tx,y=ty)
 assert not set(map(tuple,x[train])).intersection(map(tuple,tx)), 'Train/test duplicate features'
 assert not set(map(tuple,x[train])).intersection(map(tuple,x[dev])), 'Train/dev duplicate features'
 assert not set(map(tuple,x[dev])).intersection(map(tuple,tx)), 'Dev/test duplicate features'
 rows=[]
 for seed in SEEDS:
  rng=np.random.default_rng(seed+100000); parts={}
  for c in range(10):
   ids=train[y[train]==c].copy(); rng.shuffle(ids); parts[c]=np.array_split(ids,2)
  pairs=[(0,1),(2,3),(4,5),(6,7),(8,9)]
  phase_ids=[]
  for visit,ps in enumerate([pairs,list(reversed(pairs))]):
   for pair in ps:
    ids=np.concatenate([parts[c][visit] for c in pair]); rng.shuffle(ids); phase_ids.append((pair,ids))
  models={a:init(seed) for a in ARMS}; buffers=[]; seen=0; updates=0; per={a:[] for a in ARMS}; seen_classes=set()
  buffer_rng=np.random.default_rng(seed+200000); candidate_rng=np.random.default_rng(seed+300000); uniform_rng=np.random.default_rng(seed+400000)
  count={'actual_steps':0,'virtual_steps':0,'candidate_pre_post_scored':0,'fresh_presentations':0,'replay_presentations':0}
  for phase,(pair,ids) in enumerate(phase_ids):
   for start in range(0,len(ids),32):
    fresh=ids[start:start+32]
    for repeat in range(4):
     if time.monotonic()-started>600: raise TimeoutError('600-second prespecified cap')
     if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>1024**3: raise MemoryError('1 GiB sampled RSS cap')
     ci=candidate_rng.choice(len(buffers),size=min(128,len(buffers)),replace=False) if buffers else []
     candidates=np.array([buffers[i] for i in ci],dtype=int)
     pick=uniform_rng.choice(len(candidates),size=32,replace=len(candidates)<32) if len(candidates) else []
     for arm,w in models.items():
      virtual=step(w,x[fresh],y[fresh])
      if len(candidates):
       pre=losses(w,x[candidates],y[candidates]); post=losses(virtual,x[candidates],y[candidates])
       if arm=='uniform': selected=candidates[pick]
       else:
        ranking=np.argsort(-(post-pre if arm=='mir' else pre),kind='stable')
        selected=candidates[np.resize(ranking[:32],32)]
       joint=np.concatenate([fresh,selected])
      else: joint=fresh
      models[arm]=step(w,x[joint],y[joint])
     updates+=1; count['actual_steps']+=1; count['virtual_steps']+=1; count['candidate_pre_post_scored']+=2*len(candidates); count['fresh_presentations']+=len(fresh); count['replay_presentations']+=32 if len(candidates) else 0
    for idx in fresh:
     seen+=1
     if len(buffers)<256: buffers.append(int(idx))
     else:
      pos=buffer_rng.integers(seen)
      if pos<256: buffers[pos]=int(idx)
   seen_classes.update(pair)
   for arm,w in models.items():
    m,pred,probs=metrics(w,tx,ty); old=sorted(seen_classes-set(pair)); m.update(phase=phase,current_pair=list(pair),acquisition=float(np.mean([m['recalls'][c] for c in pair])),old_accuracy=float(np.mean([m['recalls'][c] for c in old])) if old else None)
    per[arm].append(m); np.savez_compressed(out/f'{seed}-{arm}-{phase}.npz',**{f'w{i}':v for i,v in enumerate(w)},predictions=pred,probabilities=probs)
  for arm in ARMS:
   row={'seed':seed,'arm':arm,'boundaries':per[arm],'final_macro':per[arm][-1]['macro'],'mean_acquisition':float(np.mean([m['acquisition'] for m in per[arm]])),'counts_per_arm':count.copy()}; rows.append(row)
  (out/'metrics.json').write_text(json.dumps(rows,indent=2)+'\n'); print(json.dumps({'seed_completed':seed,'elapsed_seconds':time.monotonic()-started}),flush=True)
 r=np.random.default_rng(20261025); d=np.array([next(a['final_macro'] for a in rows if a['seed']==s and a['arm']=='mir')-next(a['final_macro'] for a in rows if a['seed']==s and a['arm']=='uniform') for s in SEEDS]); bs=r.choice(d,size=(10000,len(d)),replace=True).mean(1); ci=np.quantile(bs,[.025,.975]); acq=np.array([next(a['mean_acquisition'] for a in rows if a['seed']==s and a['arm']=='mir')-next(a['mean_acquisition'] for a in rows if a['seed']==s and a['arm']=='uniform') for s in SEEDS]); summary={'mean_final_macro':{a:float(np.mean([v['final_macro'] for v in rows if v['arm']==a])) for a in ARMS},'mir_minus_uniform':d.tolist(),'mean_difference':float(d.mean()),'paired_seed_bootstrap_95':ci.tolist(),'mean_acquisition_difference':float(acq.mean()),'support':bool(d.mean()>=.01 and ci[0]>0 and acq.mean()>=-.02),'wall_seconds':time.monotonic()-started,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'test_sha256':hashlib.sha256(testbytes).hexdigest(),'test_rows':len(ty)}
 (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 assert time.monotonic()-started<=600, 'Final walltime cap'
 assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=1024**3, 'Final RSS cap'
 (out/'COMPLETE.json').write_text(json.dumps({'cases':len(rows),'checkpoints':len(list(out.glob('31*.npz'))),'source_sha256':freeze['source_sha256']})); (out/'INCOMPLETE').unlink(); print(json.dumps(summary),flush=True)
if __name__=='__main__':
 if '--selftest' in sys.argv: selftest()
 elif '--run' in sys.argv: run()
