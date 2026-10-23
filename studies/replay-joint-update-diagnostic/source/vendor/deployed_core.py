"""Learner primitives. This module cannot load data, read audit files or score splits.

Logical scalar-operation accounting is architecture-defined, not hardware FLOPs.
Copies, RNG, comparisons and indexing are reported separately from arithmetic ops.
"""
import numpy as np

ARMS=('mir','highloss','uniform','shuffledscore','uniform-extra-replay')
SHAPES=((196,32),(32,),(32,10),(10,))
PARAMETERS=6634

def rng(seed,name):
    names={'init':1,'stream':2,'reservoir':3,'candidate':4,'uniform':5,'shuffle':6,'extra':7,'audit':8}
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed,names[name]])))

def initialize(seed):
    r=rng(seed,'init')
    return (r.normal(0,np.sqrt(2/196),(196,32)).astype(np.float32),np.zeros(32,np.float32),
            r.normal(0,np.sqrt(2/32),(32,10)).astype(np.float32),np.zeros(10,np.float32))

def forward(p,x):
    h=np.maximum(x@p[0]+p[1],0)
    z=h@p[2]+p[3]
    shifted=z-z.max(axis=1,keepdims=True)
    e=np.exp(shifted); probability=e/e.sum(axis=1,keepdims=True)
    logp=shifted-np.log(e.sum(axis=1,keepdims=True))
    if not np.isfinite(logp).all(): raise ValueError('nonfinite forward')
    return h,probability,logp

def losses(p,x,y):
    return -forward(p,x)[2][np.arange(len(y)),y]

def loss_grad(p,x,y):
    h,d,lp=forward(p,x)
    loss=-lp[np.arange(len(y)),y]
    d[np.arange(len(y)),y]-=1
    d/=len(y)
    gw2=h.T@d; gb2=d.sum(axis=0)
    dh=d@p[2].T; dh*=h>0
    gw1=x.T@dh; gb1=dh.sum(axis=0)
    g=(gw1,gb1,gw2,gb2)
    if not all(np.isfinite(a).all() for a in g): raise ValueError('nonfinite gradient')
    return loss,g

def updated(p,g,eta):
    return tuple(a-np.asarray(eta,dtype=a.dtype)*b for a,b in zip(p,g))

def step(p,x,y,eta):
    return updated(p,loss_grad(p,x,y)[1],eta)

def score(p,fx,fy,cx,cy,eta):
    v=step(p,fx,fy,eta)
    current=losses(p,cx,cy); delta=losses(v,cx,cy)-current
    if not np.isfinite(delta).all(): raise ValueError('nonfinite delta')
    return v,current,delta

def quotas(labels,total,offset):
    classes=np.unique(labels); counts={int(k):int(np.sum(labels==k)) for k in classes}
    q={int(k):0 for k in classes}
    if not len(classes): return q
    cycle=np.roll(classes,-(offset%len(classes)))
    while sum(q.values())<min(total,len(labels)):
        for label in cycle:
            k=int(label)
            if q[k]<counts[k]: q[k]+=1
            if sum(q.values())==min(total,len(labels)): break
    return q

def candidates(labels,offset,generator,total=50):
    q=quotas(labels,total,offset)
    return np.concatenate([generator.choice(np.flatnonzero(labels==k),n,replace=False) for k,n in q.items()]).astype(np.int64) if q else np.empty(0,np.int64)

def select(arm,ids,labels,q,current,delta,generator):
    if not np.isfinite(current).all() or not np.isfinite(delta).all(): raise ValueError('nonfinite selector')
    selected=[]
    for k,n in q.items():
        pos=np.flatnonzero(labels==k)
        if arm in ('uniform','uniform-extra-replay'):
            chosen=generator.choice(pos,n,replace=False)
        else:
            s=current[pos] if arm=='highloss' else delta[pos]
            if arm=='shuffledscore': s=generator.permutation(s)
            if arm not in ('highloss','mir','shuffledscore'): raise ValueError(arm)
            chosen=pos[np.lexsort((ids[pos],-s))[:n]]
        selected.extend(chosen)
    return np.asarray(selected,np.int64)

def first_repeat(fresh_count,count=10):
    return np.arange(count)%fresh_count

class Reservoir:
    def __init__(self,seed,capacity=200):
        self.x=np.empty((capacity,196),np.float32); self.y=np.empty(capacity,np.uint8)
        self.ids=np.empty(capacity,np.uint64); self.n=0; self.seen=0; self.generator=rng(seed,'reservoir')
    @property
    def raw_bytes(self): return self.x.nbytes+self.y.nbytes+self.ids.nbytes
    def admit(self,x,y,ids):
        for a,b,i in zip(x,y,ids):
            self.seen+=1
            slot=self.n if self.n<len(self.y) else int(self.generator.integers(self.seen))
            if slot<len(self.y):
                self.x[slot]=a; self.y[slot]=b; self.ids[slot]=i
                self.n=min(self.n+1,len(self.y))
    def arrays(self): return self.x[:self.n],self.y[:self.n],self.ids[:self.n]

def stream_order(labels,seed):
    r=rng(seed,'stream'); pairs=r.permutation(10).reshape(5,2)
    parts=[r.permutation(np.flatnonzero(np.isin(labels,p))) for p in pairs]
    if any(len(a)!=2000 for a in parts): raise ValueError('need exactly 1000 training examples per class')
    return np.concatenate(parts),pairs

# Matrix dot uses k multiplies + k-1 additions per output, including k=1.
def dot_ops(m,k,n): return m*n*(2*k-1)
def forward_ops(n):
    # two dots + biases, ReLU comparisons, max comparisons, subtract, exp,
    # two sums (implementation does two), division, log, logp subtraction.
    return dot_ops(n,196,32)+n*32+n*32+dot_ops(n,32,10)+n*10+n*9+n*10+n*10+2*n*9+n*10+n+n*10

def loss_ops(n): return forward_ops(n)+n # loss negation

def gradient_ops(n):
    # d[label]-=1, divide all logits, three dot products, bias sums, ReLU
    # derivative comparison + mask multiplication; loss computed too.
    return loss_ops(n)+n+n*10+dot_ops(32,n,10)+10*(n-1)+dot_ops(n,10,32)+2*n*32+dot_ops(196,n,32)+32*(n-1)

def update_ops(n): return gradient_ops(n)+2*PARAMETERS

def selection_ops(n): return update_ops(16)+2*loss_ops(n)+n # virtual plus candidate delta

def diagnostic_ops(n): return update_ops(26)+selection_ops(n)

class ComputeBudget:
    def __init__(self): self.residual=0; self.used=0; self.allowed=0; self.extra_updates=0
    def advance(self,candidate_count):
        allowed=diagnostic_ops(candidate_count); base=update_ops(26)
        self.allowed+=allowed; self.used+=base; self.residual+=allowed-base
        count=self.residual//update_ops(10)
        self.used+=count*update_ops(10); self.residual-=count*update_ops(10); self.extra_updates+=count
        assert self.used<=self.allowed
        return count
