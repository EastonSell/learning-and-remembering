"""Public redaction integrity and admission-binding checks; no real fitting."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))

class PublicDerivativeTests(unittest.TestCase):
    def test_public_manifest_is_explicitly_postrun_derivative(self):
        p=json.loads((ROOT/'PRE_DATA_MANIFEST.json').read_text())
        self.assertEqual(p.get('artifact_role'),'POSTRUN_PUBLIC_PROSE_REDACTION_DERIVATIVE')
        self.assertFalse(p['is_original_predata_freeze'])
        self.assertEqual(p['source_manifest_sha256'],hashlib.sha256((ROOT/'SOURCE_PRE_DATA_MANIFEST.json').read_bytes()).hexdigest())
        self.assertNotEqual(hashlib.sha256((ROOT/'PRE_DATA_MANIFEST.json').read_bytes()).hexdigest(),p['source_manifest_sha256'])

    def test_original_runtime_code_numeric_protocol_and_tests_are_exact(self):
        m=json.loads((ROOT/'SOURCE_PRE_DATA_MANIFEST.json').read_text())
        for name in ('engine.py','run.py','protocol.json','test_engine.py','test_runner.py'):
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),m['files'][name])
        for name in ('PROTOCOL.md','README.md'):
            self.assertNotEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),m['files'][name])

    def test_original_admission_is_rejected_for_changed_public_manifest(self):
        import run
        source=hashlib.sha256((ROOT/'SOURCE_PRE_DATA_MANIFEST.json').read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'admission.json'
            p.write_text(json.dumps({'status':'APPROVED_FOR_EXECUTION','manifest_sha256':source}))
            with self.assertRaisesRegex(ValueError,'exact frozen manifest'):
                run.verify_admission(ROOT,p)

    def test_fresh_synthetic_admission_must_bind_current_derivative(self):
        import run
        current=hashlib.sha256((ROOT/'PRE_DATA_MANIFEST.json').read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);review=root/'review.json';review.write_text(json.dumps({'status':'APPROVED','manifest_sha256':current,'reviewer':'synthetic fixture only'}))
            admission=root/'admission.json';admission.write_text(json.dumps({'status':'APPROVED_FOR_EXECUTION','manifest_sha256':current,'review_path':str(review),'review_sha256':hashlib.sha256(review.read_bytes()).hexdigest()}))
            result=run.verify_admission(ROOT,admission)
            self.assertEqual(result['manifest_sha256'],current)
            # This test checks integrity only; it neither loads data nor executes training.

    def test_every_declared_prose_redaction_digest_matches(self):
        m=json.loads((ROOT/'PUBLIC_PROVENANCE_DERIVATIONS.json').read_text())
        for name in ('README.md','PROTOCOL.md','PRE_DATA_MANIFEST.json'):
            self.assertEqual(m['files'][name]['public_sha256'],hashlib.sha256((ROOT/name).read_bytes()).hexdigest())

if __name__=='__main__':unittest.main()
