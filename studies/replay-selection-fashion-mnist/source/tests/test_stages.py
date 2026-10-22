import unittest,tempfile,json
from pathlib import Path
from replaystudy import stages as s
class StageTests(unittest.TestCase):
 def test_source_freeze_roundtrip(self):
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'freeze.json'; s.freeze(path); s.verify_freeze(path)
   v=json.loads(path.read_text()); v['files']['replaystudy/core.py']='0'*64; path.write_text(json.dumps(v))
   with self.assertRaises(ValueError): s.verify_freeze(path)
 def test_no_authorization(self):
  with tempfile.TemporaryDirectory() as d:
   f=Path(d)/'f.json'; s.freeze(f); a=Path(d)/'a.json'; a.write_text('{}')
   with self.assertRaises(ValueError): s.authorize(f,a,'development')
 def test_rates(self):
  rows=[]
  for seed in [1001,1002]:
   for eta in [.03,.1,.3]:
    for arm in s.c.ARMS: rows.append({'seed':seed,'eta':eta,'arm':arm,'metrics':{'validation_recall':[.5 if arm!='uniform-extra-replay' or eta==.1 else .4]*10}})
  self.assertEqual(s.select_rates(rows),{'diagnostic_eta':.03,'extra_eta':.1})
 def test_incomplete_rates(self):
  with self.assertRaises(ValueError): s.select_rates([])
if __name__=='__main__': unittest.main()

class IntegrationTests(unittest.TestCase):
 def test_development_orchestration_with_synthetic_short_stream(self):
  import argparse,numpy as np
  from unittest.mock import patch
  from replaystudy import runner as r,evidence as e
  with tempfile.TemporaryDirectory() as d:
   root=Path(d); report=root/'review.txt'; report.write_text('synthetic test review only')
   freeze=root/'freeze.json'; s.freeze(freeze)
   auth=root/'auth.json'; auth.write_text(json.dumps({'source_freeze_sha256':r.file_sha(freeze),'parent_go_ahead':True,'approved_stages':['development'],'independent_review':{'passed':True,'reviewer':'synthetic unit fixture','report_path':str(report),'report_sha256':r.file_sha(report)}}))
   prepared=root/'prepared'; prepared.mkdir(); (prepared/'manifest.json').write_text('{}'); np.save(prepared/'train_ids.npy',np.arange(80))
   forecast=root/'forecast.json'; forecast.write_text(json.dumps({'cpu_seconds':50,'output_bytes':1000000,'rss_bytes':10000000}))
   x=np.random.default_rng(3).random((80,196),dtype=np.float32); y=np.arange(80,dtype=np.uint8)%10
   data={'x_train':x,'y_train':y,'train_ids':np.arange(100,180),'x_val':x,'y_val':y,'val_ids':np.arange(80)}
   args=argparse.Namespace(freeze=freeze,authorization=auth,prepared=prepared,forecast=forecast,data_root=root,output=root/'bundle',budget_journal=root/'budget.jsonl')
   train=r.train_stream; bundle_type=e.Bundle
   def software_bundle(path,**kw):
    # Reduced synthetic stream exercises orchestration only, never research acceptance.
    kw['stage']='software'
    return bundle_type(path,**kw)
   def short(*a,**kw): return train(*a,**kw,software_only=True,block_size=16)
   with patch('replaystudy.evidence.Bundle',side_effect=software_bundle),patch('replaystudy.stages.verify_forecast',return_value=json.loads(forecast.read_text())),patch('replaystudy.data.load_development',return_value=data),patch('replaystudy.core.stream_order',return_value=(np.arange(80),np.arange(10).reshape(5,2))),patch('replaystudy.runner.train_stream',side_effect=short): s.run_stage(args,'development')
   self.assertEqual(e.verify_bundle(root/'bundle')['case_count'],30)
   self.assertEqual(len(s.load_rows(root/'bundle')),30)

class FeasibilityTests(unittest.TestCase):
 def test_forecast_formula(self):
  rows=[{'ledger':{'cpu_seconds':.1,'sampled_peak_rss_bytes':1000000,'committed_base_updates':200},'artifact_bytes':50000} for _ in range(5)]
  f=s.forecast_resources(rows,40000000,scoring_cpu=.1)
  self.assertGreater(f['cpu_seconds'],0); self.assertLess(f['cpu_seconds'],7200)
  self.assertGreater(f['output_bytes'],40000000)

class EvaluationIntegrationTests(unittest.TestCase):
 def test_synthetic_evaluation_scores_only_after_all_training(self):
  import argparse,numpy as np
  from unittest.mock import patch
  from replaystudy import runner as r,evidence as e
  with tempfile.TemporaryDirectory() as d:
   root=Path(d); prepared=root/'prepared'; prepared.mkdir(); (prepared/'manifest.json').write_text('{}'); np.save(prepared/'train_ids.npy',np.arange(80))
   f=root/'freeze.json'; s.freeze(f); auth=root/'auth.json'; auth.write_text('{}'); lock=root/'lock.json'; lock.write_text('{}')
   forecast=root/'forecast.json'; forecast.write_text('{}')
   x=np.random.default_rng(3).random((80,196),dtype=np.float32); y=np.arange(80,dtype=np.uint8)%10
   data={'x_train':x,'y_train':y,'train_ids':np.arange(100,180),'x_val':x,'y_val':y,'val_ids':np.arange(80)}
   args=argparse.Namespace(freeze=f,authorization=auth,prepared=prepared,forecast=forecast,data_root=root,output=root/'bundle',budget_journal=root/'budget.jsonl',lock=lock)
   train=r.train_stream; base=e.Bundle
   class SoftwareOnly(base):
    def __init__(self,path,**kw):
     kw['stage']='software'; super().__init__(path,**kw)
    def append_case(self,row):
     row['metrics']['validation_recall']=row['metrics']['test_recall']; super().append_case(row)
   def short(*a,**kw): return train(*a,**kw,software_only=True,block_size=16)
   def test_loader(*a,**kw):
    self.assertEqual(len(list((root/'bundle').glob('*/trained-case.json'))),100)
    return {'x_test':x,'y_test':y,'keep_ids':np.arange(80)}
   with patch('replaystudy.evidence.Bundle',SoftwareOnly),patch('replaystudy.stages.authorize'),patch('replaystudy.stages.verify_forecast',return_value={}),patch('replaystudy.stages.verify_lock',return_value={'rates':{'diagnostic_eta':.03,'extra_eta':.1},'source_freeze_sha256':r.file_sha(f),'data_manifest_sha256':r.file_sha(prepared/'manifest.json'),'forecast_sha256':r.file_sha(forecast)}),patch('replaystudy.data.load_development',return_value=data),patch('replaystudy.data.load_test_for_scoring',side_effect=test_loader),patch('replaystudy.core.stream_order',return_value=(np.arange(80),np.arange(10).reshape(5,2))),patch('replaystudy.runner.train_stream',side_effect=short): s.run_stage(args,'evaluation')
   self.assertEqual(e.verify_bundle(root/'bundle')['case_count'],100)
   self.assertTrue((root/'bundle'/'first-test-scoring.json').exists())
