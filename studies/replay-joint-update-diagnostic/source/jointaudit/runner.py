"""Read-only retained-state preparation and bounded JOINT diagnostic.

Import and freeze/preflight do not load parameters, decode images, or score data.
Public reference derivative: scientific execution is not packaged here.
"""
import gzip
import hashlib
import json
import os
from pathlib import Path
import resource
import struct
import time
import numpy as np
from vendor import deployed_core as c
from . import mechanics as m

ROOT=Path(__file__).resolve().parents[1]
HISTORY=ROOT.parent/'external-inputs/historical/results/evaluation-v2'
DATA=ROOT.parent/'external-inputs/data'
SNAPSHOT=ROOT.parent/'external-inputs/historical/source/replaystudy'
SEEDS=(2001,2002,2003,2004)
BLOCKS=(0,1,2,3)
ETA=.3


def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024**2),b''): h.update(chunk)
    return h.hexdigest()


def write_once(path,value):
    with Path(path).open('x') as f:
        json.dump(value,f,sort_keys=True,indent=2,allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())


def verify_freeze(frozen):
    if not isinstance(frozen.get('files'),dict) or not frozen['files']: raise ValueError('missing source/input freeze')
    for path,want in frozen['files'].items():
        if sha256_file(path)!=want: raise ValueError(f'freeze mismatch: {path}')


class ResourceCaps:
    def __init__(self,output=None,started=None):
        self.cpu0,self.wall0=started if started is not None else (time.process_time(),time.monotonic())
        self.output=Path(output) if output else None
    def check_values(self,*,cpu,wall,rss,output):
        for key,value,cap in [('cpu',cpu,180),('wall',wall,300),('rss',rss,768*1024**2),('output',output,32*1024**2)]:
            if value>cap: raise RuntimeError(f'entire-command {key} cap exceeded: {value}>{cap}')
    def snapshot(self):
        return {'cpu':time.process_time()-self.cpu0,'wall':time.monotonic()-self.wall0,
            'rss':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            'output':sum(p.stat().st_size for p in self.output.rglob('*') if p.is_file()) if self.output else 0}
    def check(self): self.check_values(**self.snapshot())


def reconstruct_reservoir(x,y,ids,seed,stop):
    if stop<16 or stop%16 or stop>len(y): raise ValueError('invalid admission boundary')
    result=c.Reservoir(seed)
    for start in range(0,stop,16): result.admit(x[start:start+16],y[start:start+16],ids[start:start+16])
    return result


def _panel_loss(p,x,y,guard):
    chunks=[]
    for start in range(0,len(y),256):
        guard(); chunks.append(c.losses(p,x[start:start+256],y[start:start+256]).astype(np.float64)); guard()
    return m.macro_loss(np.concatenate(chunks),y)


def audit_state(*,p,fx,fy,fids,bx,by,bids,oldx,oldy,oldids,newx,newy,newids,seed,block,guard=lambda:None):
    """All selectors and update compositions share this ONE state and inputs."""
    guard(); original=m.state_digest(p)
    groups=[set(np.asarray(a).tolist()) for a in (fids,bids,oldids,newids)]
    if any(groups[i]&groups[j] for i in range(4) for j in range(i)): raise ValueError('fresh/candidate/evaluator ID overlap')
    if len(fy)!=16 or len(fids)!=len(fy): raise ValueError('exactly16 fresh examples required')
    for x,y,ids in [(fx,fy,fids),(bx,by,bids),(oldx,oldy,oldids),(newx,newy,newids)]:
        if len(x)!=len(y) or len(y)!=len(ids) or len(set(ids.tolist()))!=len(ids) or not np.isfinite(x).all(): raise ValueError('malformed batch/panel')
    # Named domains, none derived from loss, outcome, or array order.
    def rng(domain,draw=0): return np.random.Generator(np.random.PCG64(np.random.SeedSequence([7310401,seed,block,domain,draw])))
    def scalar_seed(domain,draw=0): return int(rng(domain,draw).integers(0,2**63,dtype=np.int64))
    u=500*(block+1); cand=c.candidates(by,u,rng(1)); cx,cy,ids=bx[cand],by[cand],bids[cand]
    quota=c.quotas(cy,10,u)
    if sum(quota.values())!=10: raise ValueError('exactly ten replay quota slots required')
    virtual,current,delta=c.score(tuple(a.copy() for a in p),fx,fy,cx,cy,ETA)
    guard(); choices={}; tie_seed=scalar_seed(2)
    for name,scores,policy in [('mir',delta,'source_id'),('highloss',current,'source_id'),('mir_random_ties',delta,'random'),('highloss_random_ties',current,'random')]:
        selected,cutoffs=m.ranked_selection(ids,cy,quota,scores,policy,tie_seed)
        choices[name]={'positions':selected,'ids':ids[selected].tolist(),'tie_policy':policy,'cutoffs':cutoffs}
    for draw in range(4):
        selected=m.uniform_selection(ids,cy,quota,scalar_seed(3,draw))
        choices[f'uniform_{draw}']={'positions':selected,'ids':ids[selected].tolist(),'tie_policy':'uniform','draw':draw}
    baseline={}; fresh={}
    for key,x,y in [('old',oldx,oldy),('new',newx,newy)]:
        baseline[key]=_panel_loss(p,x,y,guard); fresh[key]=_panel_loss(virtual,x,y,guard)
    compositions={}
    for mode in ('joint','sequential'):
        branches={}
        for name,chosen in choices.items():
            take=chosen['positions']; guard()
            # Reuse immutable theta_f: exactly the same fresh-only SGD for all sequential branches.
            branch=(m.branch_update(p,fx,fy,cx[take],cy[take],ETA,'joint') if mode=='joint'
                else c.step(tuple(a.copy() for a in virtual),cx[take],cy[take],ETA))
            record={'start_state_sha256':original,'end_state_sha256':m.state_digest(branch)}
            for key,x,y in [('old',oldx,oldy),('new',newx,newy)]:
                value=_panel_loss(branch,x,y,guard)
                record[key+'_loss']=value; record[key+'_change_from_original']=value-baseline[key]
                record[key+'_change_from_fresh_only']=value-fresh[key]
            branches[name]=record
        uniform=float(np.mean([branches[f'uniform_{draw}']['old_loss'] for draw in range(4)]))
        compositions[mode]={'branches':branches,'mir_minus_highloss':branches['mir']['old_loss']-branches['highloss']['old_loss'],
            'mir_minus_uniform':branches['mir']['old_loss']-uniform,
            'mir_random_ties_minus_highloss_random_ties':branches['mir_random_ties']['old_loss']-branches['highloss_random_ties']['old_loss']}
    if m.state_digest(p)!=original: raise AssertionError('source state mutated')
    return {'seed':seed,'block':block,'state_sha256':original,'eta':ETA,'zero_based_update_offset':u,
        'fresh_ids':fids.tolist(),'fresh_labels':fy.tolist(),'resident_ids':bids.tolist(),
        'candidate_ids':ids.tolist(),'candidate_labels':cy.tolist(),'current_loss':current.tolist(),'signed_virtual_delta':delta.tolist(),
        'quota':{str(k):v for k,v in quota.items()},'tie_seed':tie_seed,
        'old_panel_ids':oldids.tolist(),'old_labels':sorted(int(k) for k in np.unique(oldy)),
        'new_panel_ids':newids.tolist(),'new_labels':sorted(int(k) for k in np.unique(newy)),
        'selections':{name:{k:v for k,v in sel.items() if k!='positions'} for name,sel in choices.items()},
        'baseline_old_new':baseline,'fresh_only_old_new':fresh,'compositions':compositions,
        'joint_mir_minus_highloss':compositions['joint']['mir_minus_highloss'],
        'joint_mir_minus_uniform':compositions['joint']['mir_minus_uniform'],
        'sequential_mir_minus_highloss':compositions['sequential']['mir_minus_highloss'],
        'joint_minus_sequential_contrast':compositions['joint']['mir_minus_highloss']-compositions['sequential']['mir_minus_highloss'],
        'random_tie_contrast_shift':compositions['joint']['mir_random_ties_minus_highloss_random_ties']-compositions['joint']['mir_minus_highloss'],
        'mir_tie_selection_changed':set(choices['mir']['ids'])!=set(choices['mir_random_ties']['ids']),
        'highloss_tie_selection_changed':set(choices['highloss']['ids'])!=set(choices['highloss_random_ties']['ids'])}


def _case(seed): return HISTORY/f'seed-{seed}__uniform__eta-0.3'


def input_paths():
    paths=[DATA/'prepared-v1'/name for name in ['manifest.json','train_ids.npy','val_ids.npy','train_raw_sha256.npy','train_transformed_sha256.npy']]
    paths += [DATA/'raw'/name for name in ['train-images-idx3-ubyte.gz','train-labels-idx1-ubyte.gz','LICENSE']]
    paths += [HISTORY/'artifacts.json',HISTORY/'manifest.json',HISTORY/'cases.jsonl',HISTORY.parent/'evaluation-lock-v2.json',SNAPSHOT/'core.py',SNAPSHOT/'runner.py']
    for seed in SEEDS:
        paths += [_case(seed)/'stream-indices.npy',_case(seed)/'class-pairs.json']
        paths += [_case(seed)/f'training/block-{block}.npz' for block in BLOCKS]
    return paths


def local_freezable_files(root):
    root=Path(root)
    return sorted(p for p in root.rglob('*') if p.is_file() and p.suffix in ('.py','.md','.json')
        and not p.is_relative_to(root/'verification') and not p.is_relative_to(root/'results')
        and not p.name.startswith('review-freeze'))


def make_review_freeze():
    """Hash and metadata checks ONLY; no np.load, image decode, or inference."""
    inventory=json.loads((HISTORY/'artifacts.json').read_text())
    files={}
    for path in input_paths():
        actual=sha256_file(path)
        if path.is_relative_to(HISTORY) and path.name!='artifacts.json':
            if inventory.get(str(path.relative_to(HISTORY)))!=actual: raise ValueError(f'historical inventory mismatch: {path}')
        files[str(path)]=actual
    manifest=json.loads((DATA/'prepared-v1/manifest.json').read_text())
    for name in ['train-images-idx3-ubyte.gz','train-labels-idx1-ubyte.gz','LICENSE']:
        if files[str(DATA/'raw'/name)]!=manifest['assets'][name]['sha256']: raise ValueError('raw training source mismatch')
    for name in ['train_ids','val_ids','train_raw_sha256','train_transformed_sha256']:
        if files[str(DATA/'prepared-v1'/f'{name}.npy')]!=manifest['arrays'][name]['sha256']: raise ValueError('split/hash-array manifest mismatch')
    if files[str(SNAPSHOT/'core.py')]!=sha256_file(ROOT/'vendor/deployed_core.py'): raise ValueError('vendored deployed kernel mismatch')
    lock=json.loads((HISTORY.parent/'evaluation-lock-v2.json').read_text())
    if lock['rates']['diagnostic_eta']!=ETA: raise ValueError('historical locked rate mismatch')
    local=local_freezable_files(ROOT)
    files.update({str(path):sha256_file(path) for path in local})
    return {'kind':'posthoc-joint-audit-source-input-freeze-v2','scientific_execution_performed':False,
        'files':files,'seeds':list(SEEDS),'blocks':list(BLOCKS),'origin':'uniform','eta':ETA,
        'official_test_opened':False,'checkpoint_arrays_loaded':False,'image_arrays_decoded':False,
        'state_count':16,'cluster_count':4,'epsilon_nats':.005,'caps':{'cpu_soft_seconds':180,'cpu_hard_backstop_seconds':181,'wall_signal_seconds':300,'address_space_limit_bytes':768*1024**2,'observed_rss_gate_bytes':768*1024**2,'artifact_output_limit_bytes':32*1024**2,'threads':1}}


def _train_arrays():
    # Deliberately NO test path or network-capable data helper exists here.
    with gzip.open(DATA/'raw/train-images-idx3-ubyte.gz','rb') as f: images=f.read(16+60000*784+1)
    with gzip.open(DATA/'raw/train-labels-idx1-ubyte.gz','rb') as f: labels=f.read(8+60000+1)
    if len(images)!=16+60000*784 or struct.unpack('>IIII',images[:16])!=(2051,60000,28,28): raise ValueError('training images malformed')
    if len(labels)!=8+60000 or struct.unpack('>II',labels[:8])!=(2049,60000): raise ValueError('training labels malformed')
    raw=np.frombuffer(images,np.uint8,offset=16).reshape(60000,28,28)
    y=np.frombuffer(labels,np.uint8,offset=8)
    train=np.load(DATA/'prepared-v1/train_ids.npy',allow_pickle=False); val=np.load(DATA/'prepared-v1/val_ids.npy',allow_pickle=False)
    if train.shape!=(10000,) or val.shape!=(5000,) or len(set(train))!=10000 or len(set(val))!=5000 or set(train)&set(val): raise ValueError('split size/ID overlap')
    if not np.array_equal(np.bincount(y[train],minlength=10),[1000]*10) or not np.array_equal(np.bincount(y[val],minlength=10),[500]*10): raise ValueError('split class counts')
    for name in ['train_raw_sha256','train_transformed_sha256']:
        hashes=np.load(DATA/'prepared-v1'/f'{name}.npy',allow_pickle=False)
        if set(hashes[train])&set(hashes[val]): raise ValueError('raw/transformed cross-split duplicate')
    def transform(ids):
        return (raw[ids].reshape(-1,14,2,14,2).mean(axis=(2,4),dtype=np.float32).reshape(-1,196)/np.float32(255)).astype(np.float32)
    return transform(train),y[train],train,transform(val),y[val],val


def execute(frozen,output,*,admission=None):
    """Only the early-admitted launcher may enter this scientific path."""
    output=Path(output)
    if admission is None or not admission.limits_installed or not admission.owned or admission.output!=output:
        raise ValueError('early stdlib admission with owned output is required')
    caps=admission
    verify_freeze(frozen); caps.check()
    if any(os.environ.get(k)!='1' for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']): raise ValueError('single-thread environment required')
    if Path('/proc/self/task').exists() and len(list(Path('/proc/self/task').iterdir()))!=1: raise ValueError('actual thread count must be one')
    x,y,ids,vx,vy,vids=_train_arrays(); caps.check(); rows=[]
    caps.write_json('run-input-freeze.json',frozen)
    for seed in SEEDS:
        order,pairs=c.stream_order(y,seed); sx,sy,sids=x[order],y[order],ids[order]
        saved=np.load(_case(seed)/'stream-indices.npy',allow_pickle=False)
        if not np.array_equal(saved,sids): raise ValueError('saved historical stream mismatch')
        if json.loads((_case(seed)/'class-pairs.json').read_text())!=pairs.tolist(): raise ValueError('historical class-pair mismatch')
        for block in BLOCKS:
            caps.check(); boundary=2000*(block+1); path=_case(seed)/f'training/block-{block}.npz'
            with np.load(path,allow_pickle=False) as archive:
                if set(archive.files)!={f'p{i}' for i in range(4)}: raise ValueError('checkpoint fields')
                p=tuple(archive[f'p{i}'].copy() for i in range(4))
            if any(a.dtype!=np.float32 or a.shape!=shape or not np.isfinite(a).all() for a,shape in zip(p,c.SHAPES)): raise ValueError('checkpoint arrays invalid')
            reservoir=reconstruct_reservoir(sx,sy,sids,seed,boundary); bx,by,bids=reservoir.arrays()
            old=np.isin(vy,pairs[:block+1].reshape(-1)); new=np.isin(vy,pairs[block+1])
            row=audit_state(p=p,fx=sx[boundary:boundary+16],fy=sy[boundary:boundary+16],fids=sids[boundary:boundary+16],
                bx=bx,by=by,bids=bids,oldx=vx[old],oldy=vy[old],oldids=vids[old],
                newx=vx[new],newy=vy[new],newids=vids[new],seed=seed,block=block,guard=caps.check)
            row['checkpoint_sha256']=sha256_file(path); rows.append(row)
            caps.write_json(f'seed-{seed}-block-{block}.json',row); caps.check()
    if len(rows)!=16: raise AssertionError('incomplete state bundle')
    summaries={key:m.cluster_summary(rows,key) for key in ['joint_mir_minus_highloss','joint_mir_minus_uniform','sequential_mir_minus_highloss','joint_minus_sequential_contrast','random_tie_contrast_shift']}
    caps.write_json('summary.json',{'status':'posthoc_diagnostic_not_confirmation','summaries':summaries,
        'epsilon_nats':.005,'yardstick_sensitivities':[.001,.01],
        'mir_tie_selection_changed_states':sum(row['mir_tie_selection_changed'] for row in rows),
        'highloss_tie_selection_changed_states':sum(row['highloss_tie_selection_changed'] for row in rows),
        'resources_sampled_before_summary_write':caps.snapshot(),'future_retention_tested':False},terminal=True)
    caps.finalize_success(states=16,seed_clusters=4)
