"""Public package tests use hashing/saved JSON only, never a scientific rerun."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]

class ReleaseTests(unittest.TestCase):
    def module(self):
        path=ROOT/'verify_release.py'
        self.assertTrue(path.is_file(),'saved-only public release verifier is missing')
        spec=importlib.util.spec_from_file_location('public_verifier_subject',path)
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        return mod

    def tiny_package(self,root,mod):
        (root/'evidence.json').write_text('{"synthetic":true}\n')
        info=mod.identity(root/'evidence.json')
        (root/'PUBLIC-FILES.json').write_text(json.dumps({'files':{'evidence.json':info}}))
        return info

    def test_exact_inventory_rejects_changed_extra_missing_and_symlink_files(self):
        mod=self.module()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);info=self.tiny_package(root,mod)
            self.assertEqual(mod.verify_inventory(root),1)
            (root/'extra').write_text('extra')
            with self.assertRaises(ValueError):mod.verify_inventory(root)
            (root/'extra').unlink();(root/'evidence.json').write_text('changed')
            with self.assertRaises(ValueError):mod.verify_inventory(root)
            (root/'evidence.json').unlink()
            with self.assertRaises(ValueError):mod.verify_inventory(root)
            self.tiny_package(root,mod)
            (root/'link').symlink_to(root/'evidence.json')
            with self.assertRaises(ValueError):mod.verify_inventory(root)

    def test_existing_verification_output_is_preserved(self):
        self.module()
        with tempfile.TemporaryDirectory() as td:
            output=Path(td)/'report.json';output.write_text('PRESERVE')
            result=subprocess.run([sys.executable,'-B',str(ROOT/'verify_release.py'),'--output',str(output)],capture_output=True,text=True,timeout=20)
            self.assertNotEqual(result.returncode,0)
            self.assertEqual(output.read_text(),'PRESERVE')

    def test_saved_evidence_verification_never_imports_numpy(self):
        self.module()
        code=r'''
import importlib.abc,runpy,sys
class DenyNumpy(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,*args):
        if fullname=='numpy' or fullname.startswith('numpy.'):raise AssertionError('No NumPy or scientific computation allowed')
sys.meta_path.insert(0,DenyNumpy())
sys.argv=[sys.argv[1]]
runpy.run_path(sys.argv[0],run_name='__main__')
assert 'numpy' not in sys.modules
'''
        result=subprocess.run([sys.executable,'-B','-c',code,str(ROOT/'verify_release.py')],cwd='/tmp',capture_output=True,text=True,timeout=20)
        self.assertEqual(result.returncode,0,result.stderr)
        report=json.loads(result.stdout)
        self.assertTrue(report['all_sixteen_state_files_byte_identical'])
        self.assertTrue(report['scientific_summary_exact_except_measured_resource_sample'])
        self.assertFalse(report['model_execution_performed_by_verifier'])
        self.assertFalse(report['execution_independently_attested'])

if __name__=='__main__':unittest.main()

