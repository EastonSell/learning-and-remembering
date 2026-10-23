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
        'seed_means':{str(k):float(v) for k,v in zip(seeds,means)},
        'state_sign_counts':{name:sum((float(row[key])<0 if name=='negative' else float(row[key])>0 if name=='positive' else float(row[key])==0) for row in rows) for name in ('negative','zero','positive')},
        'seed_sign_counts':{name:int(np.sum(means<0 if name=='negative' else means>0 if name=='positive' else means==0)) for name in ('negative','zero','positive')},'cluster_count':len(means),'state_count':len(rows),
        'status':'posthoc_descriptive_only','warning':'few historical seed clusters; not fresh confirmation, dataset uncertainty, or retention evidence'}


FACTOR_ETA=.3
ALPHA=16/26
BETA=10/26
PARAM_ATOL=2e-6
PARAM_RTOL=2e-6
LOSS_ATOL=2e-5
LOSS_RTOL=2e-6
RECONCILE_ATOL=1e-10
CELLS=('00','01','10','11')

class NumericalEquivalenceError(ValueError):
    def __init__(self,reason,evidence):
        super().__init__(reason); self.evidence=evidence


def _clone(p): return tuple(a.copy() for a in p)


def _bitwise_equal(a,b):
    return len(a)==len(b) and all(x.dtype==y.dtype and x.shape==y.shape and x.tobytes()==y.tobytes() for x,y in zip(a,b))


def parameter_equivalence(actual,reference):
    if len(actual)!=len(reference): raise ValueError('parameter count mismatch')
    rows=[]
    for i,(a,b) in enumerate(zip(actual,reference)):
        if a.dtype!=b.dtype or a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all(): raise ValueError('finite shape/dtype matched arrays required')
        difference=np.abs(a.astype(np.float64)-b.astype(np.float64))
        tolerance=PARAM_ATOL+PARAM_RTOL*np.abs(b.astype(np.float64))
        rows.append({'parameter':i,'max_abs_difference':float(difference.max()),
            'max_tolerance_fraction':float((difference/tolerance).max()),'within_tolerance':bool(np.all(difference<=tolerance))})
    return {'atol':PARAM_ATOL,'rtol':PARAM_RTOL,'arrays':rows,'within_tolerance':all(row['within_tolerance'] for row in rows)}


def prepare_factor_context(p,fx,fy,eta):
    """One shared fresh gradient; source-exact reference is checked, not assumed."""
    if eta!=FACTOR_ETA or len(fy)!=16 or len(fx)!=16: raise ValueError('locked eta=.3 and exactly16 fresh required')
    original=state_digest(p); batch=state_digest((fx,fy))
    gradient=c.loss_grad(_clone(p),fx,fy)[1]
    unit=c.updated(_clone(p),gradient,eta)
    deployed=c.updated(_clone(p),gradient,eta*ALPHA)
    direct=c.step(_clone(p),fx,fy,eta)
    checks={'fresh_unit_bitwise_equal':_bitwise_equal(unit,direct),'direct_fresh_sha256':state_digest(direct)}
    if not checks['fresh_unit_bitwise_equal']: raise NumericalEquivalenceError('fresh reference exact recovery failed',checks)
    if state_digest(p)!=original or state_digest((fx,fy))!=batch: raise AssertionError('fresh source/batch mutated')
    return {'original_sha256':original,'fresh_batch_sha256':batch,'eta':eta,'fresh_gradient':gradient,
        'fresh_unit':unit,'fresh_deployed':deployed,'checks':checks,
        'fresh_gradient_sha256':state_digest(gradient),'fresh_unit_sha256':state_digest(unit),'fresh_deployed_sha256':state_digest(deployed)}


def factorial_update(p,fx,fy,rx,ry,eta,context):
    """Exactly four scientific cells, plus source/numerical equivalence checks."""
    if eta!=FACTOR_ETA or len(fy)!=16 or len(fx)!=16 or len(ry)!=10 or len(rx)!=10: raise ValueError('locked16+10 and eta=.3 required')
    original=state_digest(p); batches=state_digest((fx,fy,rx,ry))
    if context['eta']!=eta or context['original_sha256']!=original or context['fresh_batch_sha256']!=state_digest((fx,fy)): raise ValueError('context must match common source state/fresh batch/rate')
    for key in ('fresh_gradient','fresh_unit','fresh_deployed'):
        if state_digest(context[key])!=context[key+'_sha256']: raise ValueError('shared context mutated')
    gf=context['fresh_gradient']; full=context['fresh_unit']; deployed=context['fresh_deployed']
    gr0=c.loss_grad(_clone(p),rx,ry)[1]
    grd=c.loss_grad(_clone(deployed),rx,ry)[1]
    gru=c.loss_grad(_clone(full),rx,ry)[1]
    cells={
        '00':c.step(_clone(p),np.concatenate((fx,rx)),np.concatenate((fy,ry)),eta),
        '01':c.updated(_clone(deployed),grd,eta*BETA),
        '10':c.updated(_clone(p),tuple(a+b for a,b in zip(gf,gr0)),eta),
        '11':c.updated(_clone(full),gru,eta)}
    mathematical00=c.updated(_clone(p),tuple(np.asarray(ALPHA,dtype=a.dtype)*a+np.asarray(BETA,dtype=b.dtype)*b for a,b in zip(gf,gr0)),eta)
    sequential_reference=c.step(_clone(full),rx,ry,eta)
    checks={'sequential_bitwise_equal':_bitwise_equal(cells['11'],sequential_reference),
        'direct_sequential_sha256':state_digest(sequential_reference),
        'normalized_joint_math':parameter_equivalence(mathematical00,cells['00'])}
    if not checks['sequential_bitwise_equal'] or not checks['normalized_joint_math']['within_tolerance']:
        raise NumericalEquivalenceError('factorial endpoint equivalence failed; do not change tolerances',checks)
    if state_digest(p)!=original or state_digest((fx,fy,rx,ry))!=batches: raise AssertionError('common state/batches mutated')
    for key in ('fresh_gradient','fresh_unit','fresh_deployed'):
        if state_digest(context[key])!=context[key+'_sha256']: raise AssertionError('shared context mutated by factorial')
    return {'cells':cells,'mathematical00':mathematical00,'checks':checks,
        'replay_gradient_state_sha256':{'00':original,'10':original,'01':state_digest(deployed),'11':state_digest(full)},
        'replay_gradient_sha256':{'original':state_digest(gr0),'deployed_fresh':state_digest(grd),'unit_fresh':state_digest(gru)}}


def loss_equivalence(mathematical,actual):
    difference=float(mathematical)-float(actual)
    tolerance=LOSS_ATOL+LOSS_RTOL*abs(float(actual))
    return {'mathematical00_loss':float(mathematical),'actual00_loss':float(actual),'difference':difference,
        'atol':LOSS_ATOL,'rtol':LOSS_RTOL,'tolerance':tolerance,'within_tolerance':bool(np.isfinite(difference) and abs(difference)<=tolerance)}


def decompose(d):
    if set(d)!=set(CELLS) or not all(np.isfinite(v) for v in d.values()): raise ValueError('four finite cell contrasts required')
    d={key:float(value) for key,value in d.items()}
    w=d['10']-d['00']; l=d['01']-d['00']; interaction=d['11']-d['10']-d['01']+d['00']; total=d['11']-d['00']
    residual=total-(w+l+interaction)
    if abs(residual)>RECONCILE_ATOL: raise NumericalEquivalenceError('component reconciliation failed',{'contrasts':d,'residual':residual,'atol':RECONCILE_ATOL})
    return {'weight_at_original':w,'location_at_deployed':l,'interaction':interaction,'total_sequential_minus_joint':total,
        'weight_at_updated':d['11']-d['01'],'location_at_unit':d['11']-d['10'],'reconciliation_residual':residual}
