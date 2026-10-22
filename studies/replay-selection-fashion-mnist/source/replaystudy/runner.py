"""Streaming orchestration and separate read-only evaluator functions.

train_stream receives no validation/test arrays. Audit sink has no return channel.
The Python process is trusted; API separation is not an OS security boundary.
"""
import hashlib, json, os, resource, time, tracemalloc
from pathlib import Path
import numpy as np
from . import core as c

def digest_bytes(value): return hashlib.sha256(value).hexdigest()
def file_sha(path): return digest_bytes(Path(path).read_bytes())
def dump(path,value):
    with open(path,'x') as f: json.dump(value,f,sort_keys=True,indent=2,allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
def rss():
    try: return int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
    except OSError: return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024

def require_threads():
    if any(int(os.environ.get(k,'1'))>2 for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')): raise ValueError('maximum two CPU threads; use run.py launcher')
    if Path('/proc/self/task').exists() and len(list(Path('/proc/self/task').iterdir()))>2: raise ValueError('actual process thread count exceeds two')

def check_forecast(forecast):
    for key,limit in [('cpu_seconds',7200),('output_bytes',250*1024**2),('rss_bytes',1024**3)]:
        if not 0<=forecast[key]<=limit: raise ValueError(f'resource forecast exceeds {key}: {forecast[key]} > {limit}')

def recalls(pred,y):
    if set(np.unique(y))!=set(range(10)): raise ValueError('all ten labels required')
    return np.array([np.mean(pred[y==k]==k) for k in range(10)])

def evaluate(p,x,y,pairs):
    pred=np.concatenate([np.argmax(c.forward(p,x[i:i+256])[1],axis=1) for i in range(0,len(y),256)])
    rec=recalls(pred,y)
    return rec.reshape(-1),np.array([rec[pair].mean() for pair in pairs]),pred

def rankdata(x):
    order=np.argsort(x,kind='stable'); rank=np.empty(len(x),np.float64)
    for v in np.unique(x):
        indices=np.flatnonzero(x==v); rank[indices]=np.flatnonzero(x[order]==v).mean()
    return rank

def correlation(a,b):
    a=rankdata(a); b=rankdata(b)
    if a.std()==0 or b.std()==0: return None
    return float(np.corrcoef(a,b)[0,1])

def audit(p,fx,fy,bx,by,bids,cand,selected,eta,vx,vy,old_labels,seed,update):
    start=time.perf_counter(); cpu=time.process_time()
    # All branches are copied and discarded; no return to optimizer or selection.
    virtual,current,delta=c.score(p,fx,fy,bx,by,eta)
    q=c.quotas(by[cand],10,update-1)
    mirror=c.select('mir',bids[cand],by[cand],q,current[cand],delta[cand],c.rng(seed+update,'audit'))
    weak=c.select('highloss',bids[cand],by[cand],q,current[cand],delta[cand],c.rng(seed+update,'audit'))
    shuffled=c.select('shuffledscore',bids[cand],by[cand],q,current[cand],delta[cand],c.rng(seed+update,'audit'))
    # chosen deployment IDs and shuffled within-class candidate-score IDs.
    branch=c.step(virtual,bx[selected],by[selected],eta)
    control=c.step(virtual,bx[cand[shuffled]],by[cand[shuffled]],eta)
    old=np.isin(vy,old_labels)
    effect=float(c.losses(branch,vx[old],vy[old]).mean()-c.losses(control,vx[old],vy[old]).mean())
    return {'update':update,'candidate_rank_vs_exhaustive_harm':correlation(delta[cand],c.losses(virtual,bx[cand],by[cand])-c.losses(p,bx[cand],by[cand])),
      'selected_mean_harm':float(delta[selected].mean()),'eligible_mean_harm':float(delta.mean()),
      'mir_highloss_overlap':len(set(cand[mirror])&set(cand[weak]))/10,
      'selected_minus_shuffled_old_loss':effect,'old_labels':[int(k) for k in old_labels],
      'audit_operations':c.selection_ops(len(by))+2*c.loss_ops(len(cand))+2*c.update_ops(10)+2*c.loss_ops(int(old.sum())),
      'wall_seconds':time.perf_counter()-start,'cpu_seconds':time.process_time()-cpu}

def train_stream(x,y,ids,*,seed,arm,eta,output,block_size=2000,software_only=False,audit_sink=None,cpu_deadline=None):
    if arm not in c.ARMS: raise ValueError(arm)
    if not software_only and (len(y)!=10000 or block_size!=2000): raise ValueError('frozen stream dimensions')
    if len(y)%16 or block_size%16: raise ValueError('partial batches forbidden')
    output=Path(output)
    if not output.is_dir(): raise ValueError('caller must exclusively reserve case directory')
    if any(output.iterdir()): raise FileExistsError('case directory not empty')
    tracemalloc.start()
    # These are externally delivered observations; past input rows never queried.
    p=c.initialize(seed); buffer=c.Reservoir(seed)
    candidate_rng=c.rng(seed,'candidate'); uniform_rng=c.rng(seed,'uniform'); shuffle_rng=c.rng(seed,'shuffle'); extra_rng=c.rng(seed,'extra')
    budget=c.ComputeBudget(); candidate_hash=hashlib.sha256(); reservoir_hash=hashlib.sha256(); quota_hash=hashlib.sha256()
    ledger={'committed_base_updates':0,'extra_updates':0,'virtual_gradients':0,'candidate_forward_examples':0,'fresh_presentations':0,'current_repeat_presentations':0,'historical_presentations':0,'extra_historical_presentations':0,'extra_current_presentations':0,'operations_used':0,'operations_allowed':0,'native_operations':0,'negative_scores':0,'zero_scores':0,'score_count':0,'updates_with_ties':0,'candidate_draws':0,'base_selected_items':0,'extra_selected_items':0}
    checkpoints=[]; start=time.perf_counter(); start_cpu=time.process_time(); audit_seconds=0.; audit_cpu=0.; io_seconds=0.; peak_rss=rss(); snapshot_allocated_peak=0
    trace=open(output/'selection-trace.jsonl','x')
    try:
      for batch_start in range(0,len(y),16):
        fx=x[batch_start:batch_start+16]; fy=y[batch_start:batch_start+16]; fids=ids[batch_start:batch_start+16]
        for repetition in range(4):
          u=ledger['committed_base_updates']; bx,by,bids=buffer.arrays()
          empty=not len(by)
          if empty: bx,by,bids=fx,fy,fids
          cand=c.candidates(by,u,candidate_rng); cx,cy=bx[cand],by[cand]
          candidate_hash.update(np.asarray(bids[cand],dtype='<u8').tobytes()); ledger['candidate_draws']+=len(cand)
          q=c.quotas(cy,10,u)
          quota_hash.update(np.asarray(list(q.items()),dtype='<i8').tobytes())
          if arm!='uniform-extra-replay':
            virtual,current,delta=c.score(p,fx,fy,cx,cy,eta)
            ledger['virtual_gradients']+=1; ledger['candidate_forward_examples']+=2*len(cand)
            ledger['negative_scores']+=int(np.sum(delta<0)); ledger['zero_scores']+=int(np.sum(delta==0)); ledger['score_count']+=len(delta)
            ledger['updates_with_ties']+=int(any(len(np.unique(delta[cy==k]))<sum(cy==k) for k in q))
            sel=c.select(arm,bids[cand],cy,q,current,delta,shuffle_rng if arm=='shuffledscore' else uniform_rng)
            selected=cand[sel]
            # Virtual p is transient and never committed.
          else: selected=cand[c.select('uniform',bids[cand],cy,q,np.zeros(len(cand)),np.zeros(len(cand)),uniform_rng)]
          if empty: selected=c.first_repeat(len(fy))
          if audit_sink is not None and u>=500 and (u+1)%25==0:
            a=time.perf_counter(); ac=time.process_time()
            audit_sink(tuple(a.copy() for a in p),fx.copy(),fy.copy(),bx.copy(),by.copy(),bids.copy(),cand.copy(),selected.copy(),u+1)
            audit_seconds+=time.perf_counter()-a; audit_cpu+=time.process_time()-ac
          p=c.step(p,np.concatenate((fx,bx[selected])),np.concatenate((fy,by[selected])),eta)
          ledger['committed_base_updates']+=1; ledger['fresh_presentations']+=16; ledger['base_selected_items']+=10
          ledger['current_repeat_presentations' if empty else 'historical_presentations']+=10
          extra_trace=[]
          allowance=c.diagnostic_ops(len(cand)); ledger['operations_allowed']+=allowance
          if arm=='uniform-extra-replay':
            extra=budget.advance(len(cand))
            for _ in range(extra):
              if empty: extra_sel=c.first_repeat(len(fy))
              else: extra_sel=c.candidates(by,u,extra_rng,total=10)
              p=c.step(p,bx[extra_sel],by[extra_sel],eta)
              ledger['extra_updates']+=1; ledger['extra_selected_items']+=10
              ledger['extra_current_presentations' if empty else 'extra_historical_presentations']+=10
              extra_trace.extend(bids[extra_sel].tolist())
            ledger['operations_used']=budget.used; ledger['native_operations']=budget.used
          else:
            ledger['operations_used']+=allowance
            ledger['native_operations']+=c.update_ops(26)+(c.loss_ops(len(cand)) if arm=='highloss' else 0 if arm=='uniform' else c.selection_ops(len(cand)))
          if ledger['operations_used']>ledger['operations_allowed']: raise AssertionError('compute cap')
          io_start=time.perf_counter()
          trace.write(json.dumps({'u':u,'resident':buffer.n,'coverage':np.bincount(buffer.y[:buffer.n],minlength=10).tolist(),'historical':not empty,'selected_ids':bids[selected].tolist(),'extra_selected_ids':extra_trace},separators=(',',':'))+'\n')
          if (u+1)%25==0: trace.flush(); os.fsync(trace.fileno())
          io_seconds+=time.perf_counter()-io_start
          peak_rss=max(peak_rss,rss())
          if peak_rss>1024**3: raise MemoryError('sampled RSS gate exceeded')
          if time.process_time()-start_cpu>7200 or (cpu_deadline is not None and time.process_time()>cpu_deadline): raise TimeoutError('case exceeds entire-study CPU budget')
          snapshot_allocated_peak=max(snapshot_allocated_peak,tracemalloc.get_traced_memory()[1])
        buffer.admit(fx,fy,fids); reservoir_hash.update(np.asarray(buffer.ids[:buffer.n],dtype='<u8').tobytes())
        if (batch_start+16)%block_size==0:
          block=(batch_start+16)//block_size-1
          path=output/f'block-{block}.npz'
          io_start=time.perf_counter()
          with open(path,'xb') as f: np.savez(f,**{f'p{i}':v for i,v in enumerate(p)}); f.flush(); os.fsync(f.fileno())
          checkpoints.append({'path':path.name,'sha256':file_sha(path),'block':block})
          io_seconds+=time.perf_counter()-io_start
      trace.flush(); os.fsync(trace.fileno())
    finally:
      trace.close(); tracemalloc.stop()
    ledger.update({'residual_operations':ledger['operations_allowed']-ledger['operations_used'],'wall_seconds':time.perf_counter()-start,'cpu_seconds':time.process_time()-start_cpu,'learner_io_wall_seconds':io_seconds,'training_wall_excluding_audit_io':time.perf_counter()-start-audit_seconds-io_seconds,'audit_wall_seconds':audit_seconds,'audit_cpu_seconds':audit_cpu,'training_cpu_seconds_excluding_audit':time.process_time()-start_cpu-audit_cpu,'sampled_peak_rss_bytes':peak_rss,'traced_peak_allocated_bytes':snapshot_allocated_peak,'trainable_scalars':6634,'trainable_raw_bytes':26536,'reservoir_raw_bytes':158600,'virtual_parameter_bytes':0 if arm=='uniform-extra-replay' else 26536,'unique_observations':len(y),'model_rng_states':{k:v.bit_generator.state for k,v in [('candidate',candidate_rng),('uniform',uniform_rng),('shuffle',shuffle_rng),('extra',extra_rng),('reservoir',buffer.generator)]},'reservoir_counter':buffer.seen,'audit_log_access':'append-only; no learner reads','persistent_array_bytes':26536+158600,'rng_state_json_bytes':sum(len(json.dumps(v.bit_generator.state).encode()) for v in (candidate_rng,uniform_rng,shuffle_rng,extra_rng,buffer.generator)), 'python_reservoir_counter_bytes':__import__('sys').getsizeof(buffer.seen)})
    return {'checkpoints':checkpoints,'ledger':ledger,'candidate_trace_sha256':candidate_hash.hexdigest(),'reservoir_trace_sha256':reservoir_hash.hexdigest(),'quota_trace_sha256':quota_hash.hexdigest()}

def paired_summary(differences):
    d=np.asarray(differences,np.float64)
    if d.shape!=(20,) or not np.isfinite(d).all(): raise ValueError('exactly twenty paired differences')
    r=np.random.Generator(np.random.PCG64(73103)); samples=d[r.integers(20,size=(10000,20))].mean(axis=1)
    lo,hi=np.percentile(samples,[2.5,97.5]); mean=float(d.mean())
    return {'mean':mean,'ci95':[float(lo),float(hi)],'individual_differences':d.tolist(),'decision':'supports_prespecified_threshold' if mean>=.01 and lo>0 else 'practical_threshold_unsupported' if hi<.01 else 'inconclusive'}


def mean_forgetting(matrix):
    matrix=np.asarray(matrix)
    if matrix.shape!=(5,5) or not np.isfinite(matrix).all(): raise ValueError('finite 5x5 matrix required')
    return float(np.mean([np.max(matrix[k:4,k])-matrix[4,k] for k in range(4)]))
