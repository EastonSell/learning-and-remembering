"""Release-audit regressions: incomplete comparisons must fail closed."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import numpy as np

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/verify_run.py'


class VerifyRunCoverageTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.ref=self.root/'reference';self.other=self.root/'candidate'
        self.ref.mkdir();(self.ref/'checkpoints').mkdir()
        self.rows=[dict(model='a',seed=s,split='balanced',accuracy=a,training_seconds=1.)
                   for s,a in [(11,.8),(23,.9)]]
        self.write_rows(self.ref,self.rows)
        for seed in (11,23):np.savez(self.ref/f'checkpoints/a_{seed}.npz',w=np.array([seed]))
        self.manifest()
        shutil.copytree(self.ref,self.other)

    def write_rows(self,folder,rows):
        (folder/'metrics.json').write_text(json.dumps(rows))

    def manifest(self):
        paths=[self.ref/'metrics.json']+list((self.ref/'checkpoints').glob('*.npz'))
        (self.ref/'manifest.json').write_text(json.dumps(dict(source_sha256={},artifact_sha256={
            str(p.relative_to(self.ref)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})))

    def run_compare(self,*extra):
        return subprocess.run([sys.executable,str(SCRIPT),str(self.ref),'--compare',str(self.other),*extra],
                              text=True,capture_output=True)

    def test_complete_comparison_passes(self):
        r=self.run_compare();self.assertEqual(r.returncode,0,r.stderr)

    def test_empty_and_truncated_metric_sets_fail(self):
        for rows in ([],self.rows[:1]):
            with self.subTest(rows=rows):
                self.write_rows(self.other,rows)
                r=self.run_compare();self.assertNotEqual(r.returncode,0,r.stdout)

    def test_duplicate_metric_keys_fail(self):
        self.write_rows(self.other,[self.rows[0],self.rows[0]])
        r=self.run_compare();self.assertNotEqual(r.returncode,0,r.stdout)

    def test_missing_metric_field_fails(self):
        rows=[dict(row) for row in self.rows];rows[0].pop('accuracy')
        self.write_rows(self.other,rows)
        r=self.run_compare();self.assertNotEqual(r.returncode,0,r.stdout)

    def test_missing_checkpoint_fails_even_when_reference_omission_allowed(self):
        (self.other/'checkpoints/a_23.npz').unlink()
        r=self.run_compare('--allow-missing-checkpoints');self.assertNotEqual(r.returncode,0,r.stdout)

    def test_omitted_reference_still_requires_all_candidate_checkpoints(self):
        (self.ref/'checkpoints/a_23.npz').unlink()
        r=self.run_compare('--allow-missing-checkpoints');self.assertEqual(r.returncode,0,r.stderr)
        (self.other/'checkpoints/a_23.npz').unlink()
        r=self.run_compare('--allow-missing-checkpoints');self.assertNotEqual(r.returncode,0,r.stdout)

    def test_partial_seed_requires_explicit_scope(self):
        self.write_rows(self.other,self.rows[:1]);(self.other/'checkpoints/a_23.npz').unlink()
        r=self.run_compare();self.assertNotEqual(r.returncode,0,r.stdout)
        r=self.run_compare('--compare-seeds','11');self.assertEqual(r.returncode,0,r.stderr)
        self.assertIn('11',r.stdout)
        r=self.run_compare('--compare-seeds','99');self.assertNotEqual(r.returncode,0,r.stdout)

    def test_duplicate_reference_keys_fail(self):
        self.write_rows(self.ref,[self.rows[0],self.rows[0]]);self.manifest()
        self.write_rows(self.other,[self.rows[0],self.rows[0]])
        r=self.run_compare();self.assertNotEqual(r.returncode,0,r.stdout)

    def test_nonfinite_metrics_fail(self):
        rows=[dict(row) for row in self.rows];rows[0]['accuracy']=float('nan')
        self.write_rows(self.other,rows)
        r=self.run_compare();self.assertNotEqual(r.returncode,0,r.stdout)

    def test_historical_tool_exception_does_not_apply_to_training_source(self):
        p=self.ref/'manifest.json';m=json.loads(p.read_text())
        old='b9be1d1fb77d531eeeabd97ff58caf7f8801be50a72589a21b3bcfa30ee1c34b'
        m['source_sha256']={'scripts/verify_run.py':old};p.write_text(json.dumps(m))
        r=self.run_compare();self.assertEqual(r.returncode,0,r.stderr)
        self.assertIn('archived snapshot',r.stdout)
        m['source_sha256']={'src/functional_circuits/replay_core.py':old};p.write_text(json.dumps(m))
        r=self.run_compare();self.assertNotEqual(r.returncode,0,r.stdout)

    def test_extra_checkpoint_fails(self):
        np.savez(self.other/'checkpoints/a_99.npz',w=np.array([99]))
        r=self.run_compare();self.assertNotEqual(r.returncode,0,r.stdout)


if __name__=='__main__':unittest.main()
