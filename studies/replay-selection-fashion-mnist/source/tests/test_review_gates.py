"""Regressions for independent review findings; no research data are scored."""
import json,tempfile,unittest,argparse
from pathlib import Path
from unittest.mock import patch
from replaystudy import stages as s

class ReviewGateTests(unittest.TestCase):
 def test_development_binding_rejects_wrong_stage_and_hashes(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d); (p/'manifest.json').write_text(json.dumps({'provenance':{'freeze':[{'sha256':'f'}],'data':[{'sha256':'d'}]}}))
   with patch('replaystudy.evidence.verify_bundle',return_value={'stage':'software'}):
    with self.assertRaisesRegex(ValueError,'development'): s.verify_development_binding(p,'f','d')
   with patch('replaystudy.evidence.verify_bundle',return_value={'stage':'development'}):
    s.verify_development_binding(p,'f','d')
    for fh,dh in [('other','d'),('f','other')]:
     with self.assertRaises(ValueError): s.verify_development_binding(p,fh,dh)
 def test_evaluation_binds_arguments_before_data_loading(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d); f=p/'freeze'; f.write_text('f'); prepared=p/'prepared'; prepared.mkdir(); (prepared/'manifest.json').write_text('d')
   args=argparse.Namespace(freeze=f,prepared=prepared,authorization=p/'auth',forecast=p/'forecast',lock=p/'lock',data_root=p)
   for key in ['source_freeze_sha256','data_manifest_sha256']:
    lock={'rates':{'diagnostic_eta':.03,'extra_eta':.1},'source_freeze_sha256':s.r.file_sha(f),'data_manifest_sha256':s.r.file_sha(prepared/'manifest.json')}; lock[key]='wrong'
    with patch('replaystudy.stages.authorize'),patch('replaystudy.stages.verify_forecast',return_value={}),patch('replaystudy.stages.verify_lock',return_value=lock),patch('replaystudy.data.load_development',side_effect=AssertionError('data loaded before binding')):
     with self.assertRaises(ValueError): s.run_stage(args,'evaluation')
 def test_overlapping_resource_roots_not_double_counted(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d); sub=p/'prepared'; sub.mkdir(); (sub/'x').write_bytes(b'abc')
   self.assertEqual(s.disk_bytes([p,sub]),3)
 def test_all_resource_dimensions_checked(self):
  with patch('replaystudy.stages.disk_bytes',return_value=1),patch('replaystudy.runner.rss',return_value=1),patch('replaystudy.stages.time.process_time',return_value=20):
   with self.assertRaises(TimeoutError): s.check_resources([],prior_cpu=7190,start_cpu=0)
  with patch('replaystudy.stages.disk_bytes',return_value=250*1024**2+1):
   with self.assertRaises(ValueError): s.check_resources([],prior_cpu=0,start_cpu=s.time.process_time())
  with patch('replaystudy.stages.disk_bytes',return_value=1),patch('replaystudy.runner.rss',return_value=1024**3+1):
   with self.assertRaises(MemoryError): s.check_resources([],prior_cpu=0,start_cpu=s.time.process_time())
 def test_external_prepared_path_is_counted(self):
  args=argparse.Namespace(data_root='/data',prepared='/external-prepared',output='/out',forecast='/feas/forecast.json')
  self.assertIn(Path('/external-prepared'),s.resource_paths(args,[]))

class LockIntegrationGateTests(unittest.TestCase):
 def test_make_lock_checks_development_stage_and_inputs(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d); f=p/'freeze'; f.write_text('f'); prepared=p/'prepared'; prepared.mkdir(); (prepared/'manifest.json').write_text('d'); dev=p/'dev'; dev.mkdir()
   (dev/'manifest.json').write_text(json.dumps({'provenance':{'freeze':[{'sha256':'other'}],'data':[{'sha256':s.r.file_sha(prepared/'manifest.json')}]}}))
   args=argparse.Namespace(freeze=f,prepared=prepared,authorization=p/'auth',development=dev,forecast=p/'forecast',output=p/'output')
   for stage in ['software','development']:
    with patch('replaystudy.stages.authorize',return_value={'no_prior_test_selection':True,'test_provenance_basis':'fixture'}),patch('replaystudy.evidence.verify_bundle',return_value={'stage':stage}),patch('replaystudy.stages.verify_forecast',side_effect=AssertionError('forecast reached before development binding')):
     with self.assertRaises(ValueError): s.make_lock(args)
   self.assertFalse(args.output.exists())
 def test_verify_lock_rechecks_development_binding(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d); f=p/'freeze'; f.write_text('f'); data=p/'data'; data.write_text('d'); dev=p/'dev'; dev.mkdir()
   (dev/'manifest.json').write_text(json.dumps({'provenance':{'freeze':[{'sha256':s.r.file_sha(f)}],'data':[{'sha256':'other'}]}}))
   v={'kind':'evaluation-lock-v1','complete':True,'no_prior_test_selection':True,'freeze_path':str(f),'source_freeze_sha256':s.r.file_sha(f),'data_manifest_path':str(data),'data_manifest_sha256':s.r.file_sha(data),'development_path':str(dev)}
   lock=p/'lock'; lock.write_text(json.dumps(v))
   with patch('replaystudy.stages.verify_freeze'),patch('replaystudy.evidence.verify_bundle',return_value={'stage':'development'}):
    with self.assertRaisesRegex(ValueError,'data provenance'): s.verify_lock(lock)
 def test_summary_rejects_software_stage_even_if_case_count_matches(self):
  with patch('replaystudy.evidence.verify_bundle',return_value={'stage':'software'}),patch('replaystudy.stages.load_rows',side_effect=AssertionError('loaded before stage validation')):
   with self.assertRaises(ValueError): s.summary(argparse.Namespace(bundle='synthetic',output='unused'))

class ResourceLifecycleTests(unittest.TestCase):
 def run_fault(self,phase):
  import numpy as np
  from replaystudy import runner as r,evidence as e
  with tempfile.TemporaryDirectory() as d:
   p=Path(d); prepared=p/'outside-data'; prepared.mkdir(); (prepared/'manifest.json').write_text('{}'); np.save(prepared/'train_ids.npy',np.arange(80))
   data_root=p/'raw-data'; data_root.mkdir(); f=p/'freeze'; s.freeze(f); auth=p/'auth'; auth.write_text('{}'); forecast=p/'forecast'; forecast.write_text('{}')
   args=argparse.Namespace(freeze=f,prepared=prepared,authorization=auth,forecast=forecast,data_root=data_root,output=p/'bundle',budget_journal=p/'budget.jsonl')
   x=np.random.default_rng(1).random((80,196),dtype=np.float32); y=np.arange(80,dtype=np.uint8)%10
   data={'x_train':x,'y_train':y,'train_ids':np.arange(80),'x_val':x,'y_val':y,'val_ids':np.arange(80)}
   state={'cpu':0.,'rss':1000,'disk':1000}; base=e.Bundle; train=r.train_stream; evaluate=r.evaluate
   class SoftwareOnly(base):
    def __init__(self,path,**kw): kw['stage']='software'; super().__init__(path,**kw)
    def complete(self):
     answer=super().complete()
     if phase=='finalization': state['disk']=250*1024**2+1
     return answer
   def load(*a,**kw):
    if phase=='loader': state['cpu']=7300.
    return data
   def score(*a,**kw):
    value=evaluate(*a,**kw)
    if phase=='scoring': state['rss']=1024**3+1
    return value
   def short(*a,**kw): return train(*a,**kw,software_only=True,block_size=16)
   one=e.expected_cases('development')[:1]
   with patch('replaystudy.stages.authorize'),patch('replaystudy.stages.verify_forecast',return_value={}),patch('replaystudy.evidence.expected_cases',return_value=one),patch('replaystudy.evidence.Bundle',SoftwareOnly),patch('replaystudy.data.load_development',side_effect=load),patch('replaystudy.core.stream_order',return_value=(np.arange(80),np.arange(10).reshape(5,2))),patch('replaystudy.runner.train_stream',side_effect=short),patch('replaystudy.runner.evaluate',side_effect=score),patch('replaystudy.runner.rss',side_effect=lambda:state['rss']),patch('replaystudy.stages.disk_bytes',side_effect=lambda paths:state['disk']),patch('replaystudy.stages.time.process_time',side_effect=lambda:state['cpu']):
    with self.assertRaises((TimeoutError,MemoryError,ValueError)): s.run_stage(args,'development')
   self.assertTrue((p/'bundle'/'FAILED').exists())
   with self.assertRaises(e.VerificationError): e.verify_bundle(p/'bundle')
   journal=[json.loads(v) for v in (p/'budget.jsonl').read_text().splitlines()]
   self.assertEqual(journal[-1]['status'],'finished')
   if phase=='loader': self.assertGreaterEqual(journal[-1]['cpu_seconds'],7200)
   if phase=='finalization': self.assertTrue((p/'bundle'/'COMPLETE').exists()) # retained, but FAILED makes it unacceptable
 def test_scoring_only_rss_overrun_retains_failure(self): self.run_fault('scoring')
 def test_finalization_only_disk_overrun_cannot_verify_complete(self): self.run_fault('finalization')
 def test_loader_cpu_overrun_is_charged_and_retained(self): self.run_fault('loader')

class ResourcePeakTests(unittest.TestCase):
 def test_resource_peak_retained_even_when_gate_fails(self):
  peaks={}
  with patch('replaystudy.stages.disk_bytes',return_value=10),patch('replaystudy.runner.rss',return_value=1024**3+1),patch('replaystudy.stages.time.process_time',return_value=12):
   with self.assertRaises(MemoryError): s.check_resources([],prior_cpu=1,start_cpu=10,observations=peaks)
  self.assertEqual(peaks['sampled_peak_rss_bytes'],1024**3+1)
  self.assertEqual(peaks['sampled_peak_output_bytes'],10)
  self.assertEqual(peaks['sampled_total_cpu_seconds'],3)
