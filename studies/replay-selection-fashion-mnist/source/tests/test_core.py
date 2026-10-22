import unittest
import numpy as np
from replaystudy import core as c

class CoreTests(unittest.TestCase):
 def test_counts(self):
  p=c.initialize(3); self.assertEqual(sum(a.size for a in p),6634); self.assertEqual(sum(a.nbytes for a in p),26536)
 def test_gradient(self):
  p=tuple(a.astype(np.float64) for a in c.initialize(3)); x=np.random.default_rng(4).random((7,196)); y=np.arange(7)
  _,g=c.loss_grad(p,x,y)
  for k,idx in [(0,(5,3)),(1,(3,)),(2,(3,2)),(3,(2,))]:
   q=[a.copy() for a in p]; q[k][idx]+=1e-5
   r=[a.copy() for a in p]; r[k][idx]-=1e-5
   v=(c.losses(q,x,y).mean()-c.losses(r,x,y).mean())/2e-5
   self.assertAlmostEqual(v,g[k][idx],places=6)
 def test_virtual_nonmutation(self):
  p=c.initialize(1); originals=[v.copy() for v in p]; x=np.ones((16,196),np.float32); y=np.arange(16)%10
  q,cur,delta=c.score(p,x,y,x[:6],y[:6],.1)
  self.assertTrue(all(np.array_equal(a,b) for a,b in zip(p,originals)))
  np.testing.assert_allclose(delta,c.losses(q,x[:6],y[:6])-cur)
 def test_ranking_disagrees_and_signed_ties(self):
  ids=np.array([7,3,8,1]); labels=np.array([0,0,1,1]); quotas={0:1,1:1}; rng=np.random.default_rng(1)
  cur=np.array([8.,1,2,2]); delta=np.array([-2.,-1,0,0])
  np.testing.assert_array_equal(c.select('mir',ids,labels,quotas,cur,delta,rng),[1,3])
  np.testing.assert_array_equal(c.select('highloss',ids,labels,quotas,cur,delta,rng),[0,3])
 def test_quotas(self):
  labels=np.array([0]*2+[1]*8+[2]*17)
  for offset in range(10):
   q=c.quotas(labels,10,offset); self.assertEqual(sum(q.values()),10); self.assertLessEqual(q[0],2)
  q=c.quotas(labels,50,0); self.assertEqual(sum(q.values()),27)
 def test_reservoir_identity(self):
  a=c.Reservoir(7); b=c.Reservoir(7)
  x=np.ones((16,196),np.float32); y=np.arange(16,dtype=np.uint8)%10
  for t in range(30):
   a.admit(x,y,np.arange(t*16,t*16+16,dtype=np.uint64)); b.admit(x,y,np.arange(t*16,t*16+16,dtype=np.uint64))
  np.testing.assert_array_equal(a.ids,b.ids); self.assertEqual(a.seen,480); self.assertEqual(a.raw_bytes,158600)
 def test_candidates_deterministic(self):
  labels=np.arange(200)%10
  a=c.candidates(labels,4,np.random.default_rng(42)); b=c.candidates(labels,4,np.random.default_rng(42))
  np.testing.assert_array_equal(a,b); self.assertEqual(len(set(a)),50); self.assertTrue(all(np.sum(labels[a]==k)==5 for k in range(10)))
 def test_nonfinite_rejected(self):
  with self.assertRaises(ValueError): c.select('mir',np.array([0]),np.array([0]),{0:1},np.array([0.]),np.array([np.nan]),np.random.default_rng(1))
 def test_operation_budget(self):
  budget=c.ComputeBudget(); total=0
  for n in [16,16,32,48,50]*10:
   extras=budget.advance(n); total+=c.diagnostic_ops(n)
   self.assertLessEqual(budget.used,total); self.assertGreaterEqual(extras,0)
  self.assertEqual(total-budget.used,budget.residual)
 def test_first_batch(self):
  np.testing.assert_array_equal(c.first_repeat(16,10),np.arange(10))
 def test_stream_counts(self):
  y=np.repeat(np.arange(10),1000); order,pairs=c.stream_order(y,123)
  self.assertEqual(len(set(order)),10000); self.assertEqual(pairs.shape,(5,2))
  for i in range(5): self.assertEqual(set(y[order[2000*i:2000*(i+1)]]),set(pairs[i]))

if __name__=='__main__': unittest.main()

class FrozenOperationTests(unittest.TestCase):
 def test_full_schedule_budget_constants(self):
  budget=c.ComputeBudget()
  for n in [16]*8+[32]*4+[48]*4+[50]*2484: budget.advance(n)
  self.assertEqual(budget.allowed,6200562008)
  self.assertEqual(budget.used,6200428622)
  self.assertEqual(budget.extra_updates,15853)
  self.assertEqual(budget.residual,133386)
