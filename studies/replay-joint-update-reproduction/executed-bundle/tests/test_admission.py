"""Bounded stdlib-only admission/finalization probes, never scientific execution."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from jointaudit import admission as a

ROOT=Path(__file__).resolve().parents[1]

class AdmissionTests(unittest.TestCase):
    def startup_probe(self,mode):
        with tempfile.TemporaryDirectory() as temp:
            temp=Path(temp); frozen=temp/'freeze.json'; frozen.write_text('{}'); output=temp/'out'
            script=r'''
import json,resource,runpy,signal,sys,types
from pathlib import Path
root=Path(sys.argv[1]); frozen=sys.argv[2]; out=sys.argv[3]; mode=sys.argv[4]
fake=types.ModuleType('jointaudit.runner'); fake.ROOT=root
# Fake only the read-only verification boundary. It must occur after admission,
# before any real module import, dataset or checkpoint operation.
def verify(_):
    assert 'numpy' not in sys.modules
    limits={name:list(resource.getrlimit(getattr(resource,name))) for name in ['RLIMIT_CPU','RLIMIT_AS','RLIMIT_FSIZE']}
    Path(out,'observed-limits.json').write_text(json.dumps(limits))
    if mode=='cpu': signal.raise_signal(signal.SIGXCPU)
    raise RuntimeError('synthetic early verification failure')
fake.verify_freeze=verify
sys.modules['jointaudit.runner']=fake
sys.argv=[str(root/'run.py'),'--execute-reproduction','--freeze',frozen,'--output',out,'--training-assets',str(root)]
runpy.run_path(str(root/'run.py'),run_name='__main__')
'''
            result=subprocess.run([sys.executable,'-B','-c',script,str(ROOT),str(frozen),str(output),mode],capture_output=True,text=True,timeout=5,cwd=ROOT)
            self.assertNotEqual(result.returncode,0)
            self.assertNotEqual(result.returncode,-24)
            self.assertTrue((output/'FAILURE.json').exists(),result.stderr)
            self.assertFalse((output/'COMPLETE.json').exists())
            observed=json.loads((output/'observed-limits.json').read_text())
            self.assertEqual(observed['RLIMIT_CPU'],[180,181])
            self.assertEqual(observed['RLIMIT_AS'],[768*1024**2]*2)
            self.assertEqual(observed['RLIMIT_FSIZE'],[32*1024**2]*2)
            receipt=json.loads((output/'FAILURE.json').read_text())
            self.assertEqual(receipt['status'],'entire_bundle_invalid')
            self.assertLess((output/'FAILURE.json').stat().st_size,8192)

    def test_hard_limits_precede_import_and_hash_verification(self): self.startup_probe('early')
    def test_soft_cpu_signal_records_failure_not_default_termination(self): self.startup_probe('cpu')

    def test_completion_accounts_for_own_bytes_and_requires_zero_external_exit(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'out'; run=a.Admission(out); run.reserve_output()
            run.write_json('summary.json',{'synthetic':True},terminal=True)
            run.finalize_success(states=16,seed_clusters=4)
            receipt=json.loads((out/'COMPLETE.json').read_text())
            actual=sum(p.stat().st_size for p in out.iterdir())
            self.assertEqual(receipt['artifact_bytes_including_complete'],actual)
            self.assertIn('sampled_before_completion_write',receipt['resources'])
            a.verify_completion(out,exit_code=0)
            with self.assertRaises(ValueError): a.verify_completion(out,exit_code=1)
            (out/'unexpected').write_text('not bound')
            with self.assertRaises(ValueError): a.verify_completion(out,exit_code=0)

    def test_near_cap_completion_fails_and_reserved_failure_receipt_fits(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'out'; run=a.Admission(out); run.reserve_output()
            with (out/'synthetic-sparse').open('wb') as f: f.truncate(a.OUTPUT_LIMIT-a.FAILURE_RESERVE-1)
            with self.assertRaises(RuntimeError): run.finalize_success(states=16,seed_clusters=4)
            self.assertFalse((out/'COMPLETE.json').exists())
            run.record_failure(RuntimeError('near-cap failure'))
            self.assertTrue((out/'FAILURE.json').exists())
            self.assertLessEqual(sum(p.stat().st_size for p in out.iterdir()),a.OUTPUT_LIMIT)

    def test_regular_writes_reserve_terminal_evidence_and_never_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'out'; run=a.Admission(out); run.reserve_output()
            run.write_json('one.json',{'first':1})
            with self.assertRaises(FileExistsError): run.write_json('one.json',{'second':2})
            with (out/'synthetic-sparse').open('wb') as f: f.truncate(a.OUTPUT_LIMIT-a.FINAL_RESERVE)
            with self.assertRaises(RuntimeError): run.write_json('two.json',{'x':1})
            self.assertFalse((out/'two.json').exists())

if __name__=='__main__': unittest.main()
