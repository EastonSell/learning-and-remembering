"""Synthetic runner/admission regression checks; no real source data."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))

class RunnerTests(unittest.TestCase):
    def setUp(self):
        import run
        self.r=run
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.config=json.loads((ROOT/'protocol.json').read_text())
        rng=np.random.default_rng(40)
        self.x=rng.random((100,16));self.y=np.repeat(np.arange(10),10)
        self.tx=rng.random((20,16));self.ty=np.repeat(np.arange(10),2)
        self.arrays=(self.x,self.y,self.tx,self.ty,np.arange(100),np.array([],dtype=int))

    def test_missing_admission_blocks_before_data_access(self):
        with self.assertRaisesRegex(ValueError,'admission'):
            self.r.verify_admission(self.root,self.root/'absent.json')

    def test_admission_binds_exact_manifest_and_all_required_files(self):
        required=self.r.REQUIRED_FREEZE_FILES
        for name in required:
            p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(name)
        files={n:hashlib.sha256((self.root/n).read_bytes()).hexdigest() for n in required}
        m=self.root/'PRE_DATA_MANIFEST.json';m.write_text(json.dumps({'status':'FROZEN_PENDING_INDEPENDENT_REVIEW','files':files}))
        review=self.root/'review.json';review.write_text(json.dumps({'status':'APPROVED','manifest_sha256':hashlib.sha256(m.read_bytes()).hexdigest(),'reviewer':'independent synthetic fixture'}))
        a=self.root/'admission.json';a.write_text(json.dumps({'status':'APPROVED_FOR_EXECUTION','manifest_sha256':hashlib.sha256(m.read_bytes()).hexdigest(),'review_path':str(review),'review_sha256':hashlib.sha256(review.read_bytes()).hexdigest()}))
        self.r.verify_admission(self.root,a)
        (self.root/required[0]).write_text('changed')
        with self.assertRaisesRegex(ValueError,'hash'):
            self.r.verify_admission(self.root,a)

    def test_manifest_missing_runner_is_not_accepted(self):
        m=self.root/'PRE_DATA_MANIFEST.json';m.write_text(json.dumps({'status':'FROZEN_PENDING_INDEPENDENT_REVIEW','files':{}}))
        a=self.root/'admission.json';a.write_text(json.dumps({'status':'APPROVED_FOR_EXECUTION','manifest_sha256':hashlib.sha256(m.read_bytes()).hexdigest(),'reviewer':'fixture','review_sha256':'0'*64}))
        with self.assertRaisesRegex(ValueError,'required'):
            self.r.verify_admission(self.root,a)

    def test_raw_hash_checked_before_parse(self):
        p=self.root/'raw';p.write_bytes(b'not data')
        with self.assertRaisesRegex(ValueError,'hash'):
            self.r.read_member(p,'0'*64,(100,17))

    def test_raw_integer_labels_and_range_are_enforced(self):
        for label,coord in [(0.5,0),(0,101),(11,0)]:
            a=np.zeros((10,17));a[:,16]=np.arange(10);a[0,16]=label;a[0,0]=coord
            p=self.root/'raw';np.savetxt(p,a,delimiter=',')
            with self.assertRaises(ValueError):
                self.r.read_member(p,hashlib.sha256(p.read_bytes()).hexdigest(),(10,17))

    def test_all_36_synthetic_cases_finish_and_store_all_360_boundaries(self):
        out=self.root/'complete'
        self.r.execute_arrays(self.config,self.arrays,out,lambda:None,provenance={'kind':'SYNTHETIC_TEST_ONLY'})
        ledger=json.loads((out/'cases.json').read_text())
        self.assertEqual(len(ledger),36)
        self.assertEqual({r['status'] for r in ledger},{'complete'})
        self.assertEqual(len(list(out.glob('*.npz'))),362) # 360 boundaries + split + inputs
        complete=json.loads((out/'COMPLETE.json').read_text())
        self.assertEqual(complete['completed_cases'],36)
        self.assertFalse((out/'INCOMPLETE').exists())
        self.assertEqual(json.loads((out/'provenance.json').read_text())['kind'],'SYNTHETIC_TEST_ONLY')

    def test_failure_keeps_full_denominator_and_never_marks_complete(self):
        calls=[]
        def guard():
            calls.append(1)
            if len(calls)>15:raise TimeoutError('synthetic budget')
        out=self.root/'failed'
        with self.assertRaisesRegex(TimeoutError,'synthetic budget'):
            self.r.execute_arrays(self.config,self.arrays,out,guard,provenance={'kind':'SYNTHETIC_TEST_ONLY'})
        rows=json.loads((out/'cases.json').read_text())
        self.assertEqual(len(rows),36)
        self.assertEqual(sum(r['status']=='failed' for r in rows),1)
        self.assertEqual(sum(r['status']=='not_started' for r in rows),35)
        self.assertTrue((out/'INCOMPLETE').exists());self.assertFalse((out/'COMPLETE.json').exists())
        self.assertIn('synthetic budget',(out/'FAILURE.json').read_text())

    def test_late_cap_failure_invalidates_completion_marker(self):
        out=self.root/'late_cap'
        def guard():
            if (out/'COMPLETE.json').exists():raise TimeoutError('synthetic late cap')
        with self.assertRaisesRegex(TimeoutError,'synthetic late cap'):
            self.r.execute_arrays(self.config,self.arrays,out,guard,provenance={'kind':'SYNTHETIC_TEST_ONLY'})
        self.assertFalse((out/'COMPLETE.json').exists())
        self.assertTrue((out/'REJECTED_COMPLETE.json').exists())
        self.assertTrue((out/'INCOMPLETE').exists())

    def test_resources_saved_before_complete(self):
        out=self.root/'resources'
        self.r.execute_arrays(self.config,self.arrays,out,lambda:None,provenance={'kind':'SYNTHETIC_TEST_ONLY'})
        self.assertTrue((out/'resources.json').exists())
        self.assertGreater(json.loads((out/'resources.json').read_text())['wall_seconds'],0)

    def test_existing_output_is_never_overwritten(self):
        out=self.root/'exists';out.mkdir();(out/'sentinel').write_text('retain')
        with self.assertRaises(FileExistsError):
            self.r.execute_arrays(self.config,self.arrays,out,lambda:None,provenance={})
        self.assertEqual((out/'sentinel').read_text(),'retain')

    def test_exact_feature_leakage_is_rejected(self):
        tx=self.tx.copy();tx[0]=self.x[0]
        with self.assertRaisesRegex(ValueError,'duplicate'):
            self.r.check_leakage(self.x,tx,np.arange(100),np.array([],dtype=int))

if __name__=='__main__':unittest.main()
