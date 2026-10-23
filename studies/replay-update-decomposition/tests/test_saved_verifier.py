"""Saved-evidence-only checks; never import reference model or runner code."""
import copy
import importlib.util
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('saved_verifier',ROOT/'verify_saved.py')
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)

class SavedVerification(unittest.TestCase):
    def test_saved_bundle_and_all_350_metrics(self):
        result=v.verify(ROOT)
        self.assertEqual(result['recomputed_summaries'],350)
        self.assertEqual(result['states'],16)
        self.assertEqual(result['clusters'],4)
        self.assertFalse(result['scientific_execution_performed'])

    def test_rejects_changed_scalar_even_without_manifest_gate(self):
        row=v.load_json(ROOT/'evidence/seed-2005-block-0.json')
        row['cells']['00']['branches']['mir']['old_loss']+=0.01
        with self.assertRaisesRegex(v.VerificationError,'contrast|control|metric'):
            v.recompute_row(row)

    def test_rejects_extra_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'study';shutil.copytree(ROOT,root)
            (root/'unexpected.txt').write_text('not allowlisted')
            with self.assertRaisesRegex(v.VerificationError,'allowlist'):
                v.verify(root)

    def test_rejects_missing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'study';shutil.copytree(ROOT,root)
            (root/'evidence/summary.json').unlink()
            with self.assertRaisesRegex(v.VerificationError,'allowlist'):
                v.verify(root)

    def test_rejects_changed_saved_interval(self):
        summary=v.load_json(ROOT/'evidence/summary.json')['summaries']['old.mir_minus_highloss.cell00']
        changed=copy.deepcopy(summary);changed['ci95'][1]-=.01
        with self.assertRaisesRegex(v.VerificationError,'summary'):
            v.compare_summary(summary,changed)

    def test_rejects_nonfinite_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'bad.json';p.write_text('{"x": NaN}')
            with self.assertRaises(v.VerificationError):v.load_json(p)

    def test_rejects_duplicate_json_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'bad.json';p.write_text('{"x": 1, "x": 2}')
            with self.assertRaises(v.VerificationError):v.load_json(p)

    def test_rejects_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'study';shutil.copytree(ROOT,root)
            p=root/'evidence/summary.json';p.unlink();p.symlink_to(ROOT/'evidence/summary.json')
            with self.assertRaisesRegex(v.VerificationError,'symlink'):
                v.verify(root)

    def test_reference_source_not_imported(self):
        import sys
        v.verify(ROOT)
        self.assertNotIn('vendor.deployed_core',sys.modules)
        self.assertNotIn('jointaudit.mechanics',sys.modules)

if __name__=='__main__':unittest.main()
