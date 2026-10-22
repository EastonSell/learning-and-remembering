import unittest, tempfile, pathlib
import numpy as np
from replaystudy import runner as r
from replaystudy import core as c
class RunnerTests(unittest.TestCase):
 def test_recall_missing_labels_fails(self):
  with self.assertRaises(ValueError): r.recalls(np.array([0]),np.array([0]))
 def test_recall_and_matrix(self):
  y=np.tile(np.arange(10),3); pred=y.copy(); pred[0]=1
  recall=r.recalls(pred,y); self.assertAlmostEqual(recall[0],2/3)
 def test_software_loop_first_batch_and_identity(self):
  x=np.random.default_rng(9).random((48,196),dtype=np.float32); y=np.arange(48,dtype=np.uint8)%10; ids=np.arange(48,dtype=np.uint64)
  rows=[]
  for arm in c.ARMS:
   with tempfile.TemporaryDirectory() as d:
    result=r.train_stream(x,y,ids,seed=1,arm=arm,eta=.03,output=pathlib.Path(d),block_size=16,software_only=True)
    self.assertEqual(result['ledger']['committed_base_updates'],12)
    self.assertEqual(result['ledger']['fresh_presentations'],192)
    self.assertEqual(result['ledger']['current_repeat_presentations'],40)
    self.assertEqual(result['ledger']['historical_presentations'],80)
    self.assertLessEqual(result['ledger']['operations_used'],result['ledger']['operations_allowed'])
    rows.append(result)
  self.assertEqual(len({a['reservoir_trace_sha256'] for a in rows}),1)
  self.assertEqual(len({a['candidate_trace_sha256'] for a in rows}),1)
  self.assertEqual(len({a['quota_trace_sha256'] for a in rows}),1)
 def test_audit_nonmutation(self):
  p=c.initialize(7); p0=[a.copy() for a in p]; x=np.random.default_rng(5).random((100,196),dtype=np.float32); y=np.repeat(np.arange(10),10).astype(np.uint8)
  result=r.audit(p,x[:16],y[:16],x,y,np.arange(100),np.arange(50),np.arange(10),.1,x,y,np.array([0,1]),1,650)
  self.assertTrue(all(np.array_equal(a,b) for a,b in zip(p,p0))); self.assertTrue(np.isfinite(result['selected_minus_shuffled_old_loss']))
 def test_bootstrap(self):
  z=r.paired_summary(np.ones(20)*.02); self.assertEqual(z['decision'],'supports_prespecified_threshold'); self.assertAlmostEqual(z['mean'],.02)
 def test_gate(self):
  with self.assertRaises(ValueError): r.check_forecast({'cpu_seconds':7201,'output_bytes':100,'rss_bytes':100})
if __name__=='__main__': unittest.main()

class DeterminismTests(unittest.TestCase):
 def test_checkpoint_bytes_repeat(self):
  x=np.random.default_rng(9).random((32,196),dtype=np.float32); y=np.arange(32,dtype=np.uint8)%10
  with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
   outputs=[]
   for path in (a,b): outputs.append(r.train_stream(x,y,np.arange(500,532),seed=22,arm='mir',eta=.1,output=pathlib.Path(path),block_size=16,software_only=True))
   self.assertEqual([p['sha256'] for p in outputs[0]['checkpoints']],[p['sha256'] for p in outputs[1]['checkpoints']])
   self.assertEqual(pathlib.Path(a,'selection-trace.jsonl').read_bytes(),pathlib.Path(b,'selection-trace.jsonl').read_bytes())
 def test_runtime_thread_gate(self):
  from unittest.mock import patch
  with patch.dict('os.environ',{'OPENBLAS_NUM_THREADS':'64'}):
   with self.assertRaises(ValueError): r.require_threads()

class OutcomeTests(unittest.TestCase):
 def test_forgetting_can_be_negative_and_excludes_preacquisition(self):
  m=np.zeros((5,5)); m[0,1]=.99
  for k in range(4): m[k:4,k]=.4
  m[-1,:]=.7
  self.assertAlmostEqual(r.mean_forgetting(m),-.3)
