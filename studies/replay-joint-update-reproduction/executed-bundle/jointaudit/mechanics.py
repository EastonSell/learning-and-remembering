"""Pure mechanics; no filesystem, data loading or historical checkpoint access."""
import hashlib
import numpy as np
from vendor import deployed_core as c


def state_digest(parameters):
    h=hashlib.sha256()
    for a in parameters:
        h.update(str((a.dtype.str,a.shape)).encode()); h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def branch_update(p,fx,fy,rx,ry,eta,mode):
    """Start BOTH update compositions from an independent byte-identical clone."""
    if mode not in ('joint','sequential'): raise ValueError('unknown composition')
    if not np.isfinite(eta) or eta<=0: raise ValueError('eta must be positive and finite')
    if not len(fy) or not len(ry): raise ValueError('empty update')
    q=tuple(a.copy() for a in p)
    if mode=='joint':
        return c.step(q,np.concatenate((fx,rx)),np.concatenate((fy,ry)),eta)
    return c.step(c.step(q,fx,fy,eta),rx,ry,eta)


def _validate(ids,labels,quota,scores=None):
    ids=np.asarray(ids); labels=np.asarray(labels)
    if ids.ndim!=1 or labels.shape!=ids.shape or len(np.unique(ids))!=len(ids): raise ValueError('unique candidate IDs and matching labels required')
    if not np.issubdtype(ids.dtype,np.integer) or not np.issubdtype(labels.dtype,np.integer): raise ValueError('integer IDs and labels required')
    if set(quota)!=set(int(k) for k in np.unique(labels)): raise ValueError('quota class mismatch')
    for k,n in quota.items():
        if not isinstance(n,(int,np.integer)) or not 0<=n<=np.sum(labels==k): raise ValueError('invalid quota')
    if scores is not None and (np.shape(scores)!=ids.shape or not np.isfinite(scores).all()): raise ValueError('finite matching scores required')


def _priorities(ids,seed):
    canonical=np.argsort(ids,kind='stable')
    result=np.empty(len(ids),np.int64)
    result[canonical]=np.random.default_rng(seed).permutation(len(ids))
    return result


def ranked_selection(ids,labels,quota,scores,tie_policy,seed):
    """Strict scores rank first. Randomness affects exact ties only."""
    _validate(ids,labels,quota,scores)
    if tie_policy not in ('source_id','random'): raise ValueError('unknown tie policy')
    priority=ids if tie_policy=='source_id' else _priorities(ids,seed)
    selected=[]; logs=[]
    for k in sorted(quota):
        n=int(quota[k]); pos=np.flatnonzero(labels==k)
        chosen=pos[np.lexsort((priority[pos],-scores[pos]))[:n]]
        selected.extend(chosen.tolist())
        if not n:
            logs.append({'label':int(k),'quota':0,'cutoff':None,'straddling':False}); continue
        cutoff=float(scores[chosen[-1]]); equal=pos[scores[pos]==cutoff]; above=pos[scores[pos]>cutoff]
        near=pos[np.abs(scores[pos].astype(np.float64)-cutoff)<=1e-7]
        chosen_set=set(chosen.tolist()); tied_selected=[int(ids[i]) for i in equal if i in chosen_set]
        tied_rejected=[int(ids[i]) for i in equal if i not in chosen_set]
        logs.append({'label':int(k),'quota':n,'cutoff':cutoff,'above_count':len(above),'equal_count':len(equal),
            'tied_selected_ids':sorted(tied_selected),'tied_rejected_ids':sorted(tied_rejected),
            'straddling':bool(tied_selected and tied_rejected),'near_count':len(near),'near_ids':sorted(int(ids[i]) for i in near),
            'selected_ids':[int(ids[i]) for i in chosen]})
    return np.asarray(selected,np.int64),logs


def uniform_selection(ids,labels,quota,seed):
    _validate(ids,labels,quota)
    priority=_priorities(ids,seed); selected=[]
    for k in sorted(quota):
        pos=np.flatnonzero(labels==k)
        selected.extend(pos[np.argsort(priority[pos],kind='stable')[:quota[k]]].tolist())
    return np.asarray(selected,np.int64)


def macro_loss(losses,labels):
    losses=np.asarray(losses,dtype=np.float64); labels=np.asarray(labels)
    if losses.ndim!=1 or labels.shape!=losses.shape or not len(labels) or not np.isfinite(losses).all(): raise ValueError('nonempty finite losses required')
    return float(np.mean([np.mean(losses[labels==k],dtype=np.float64) for k in np.unique(labels)]))


def cluster_summary(rows,key,resamples=10000):
    if not rows or resamples<1: raise ValueError('nonempty rows and positive resamples required')
    grouped={}
    for row in rows:
        value=float(row[key])
        if not np.isfinite(value): raise ValueError('nonfinite contrast')
        grouped.setdefault(int(row['seed']),[]).append(value)
    seeds=sorted(grouped); means=np.array([np.mean(grouped[k]) for k in seeds],np.float64)
    r=np.random.Generator(np.random.PCG64(7310301)); boot=means[r.integers(len(means),size=(resamples,len(means)))].mean(axis=1)
    return {'mean':float(means.mean()),'ci95':[float(v) for v in np.percentile(boot,[2.5,97.5])],
        'seed_means':{str(k):float(v) for k,v in zip(seeds,means)},'cluster_count':len(means),'state_count':len(rows),
        'status':'posthoc_descriptive_only','warning':'few historical seed clusters; not fresh confirmation, dataset uncertainty, or retention evidence'}
