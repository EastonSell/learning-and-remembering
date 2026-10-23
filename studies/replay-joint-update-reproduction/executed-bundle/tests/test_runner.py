"""Synthetic-only orchestration and failure gates. No research-file loading."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from jointaudit import runner as r
from jointaudit import mechanics as m
from vendor import deployed_core as c

class RunnerTests(unittest.TestCase):
    def fixture(self):
        gen=np.random.default_rng(8)
        p=(gen.normal(size=(2,3)).astype('f')*.2,np.ones(3,'f')*.1,gen.normal(size=(3,4)).astype('f')*.2,np.zeros(4,'f'))
        return dict(p=p,fx=gen.random((16,2),dtype=np.float32),fy=np.tile([2,3],8),fids=np.arange(100,116),
            bx=gen.random((12,2),dtype=np.float32),by=np.repeat([0,1],6),bids=np.arange(12),
            oldx=gen.random((4,2),dtype=np.float32),oldy=np.repeat([0,1],2),oldids=np.arange(200,204),
            newx=gen.random((4,2),dtype=np.float32),newy=np.repeat([2,3],2),newids=np.arange(204,208),seed=2001,block=0)

    def test_audit_uses_all_common_inputs_logs_cutoffs_and_signed_outcomes(self):
        kw=self.fixture(); before=m.state_digest(kw['p']); got=r.audit_state(**kw)
        self.assertEqual(m.state_digest(kw['p']),before)
        self.assertEqual(got['state_sha256'],before); self.assertEqual(len(got['selections']),8)
        self.assertEqual(set(got['compositions']),{'joint','sequential'})
        self.assertEqual(len(got['candidate_ids']),12); self.assertEqual(got['quota'],{'0':5,'1':5})
        self.assertEqual(got['old_panel_ids'],[200,201,202,203]); self.assertEqual(got['fresh_ids'],list(range(100,116)))
        for name,sel in got['selections'].items():
            self.assertEqual(len(sel['ids']),10)
            if not name.startswith('uniform'): self.assertEqual(len(sel['cutoffs']),2)
        for mode in ['joint','sequential']:
            v=got['compositions'][mode]; self.assertIn('mir_minus_highloss',v); self.assertIn('mir_minus_uniform',v)
            ids=got['selections']['mir']['ids']; indices=[int(np.flatnonzero(kw['bids']==i)[0]) for i in ids]
            branch=m.branch_update(kw['p'],kw['fx'],kw['fy'],kw['bx'][indices],kw['by'][indices],.3,mode)
            expected=m.macro_loss(c.losses(branch,kw['oldx'],kw['oldy']),kw['oldy'])
            self.assertAlmostEqual(v['branches']['mir']['old_loss'],expected,places=7)

    def test_candidate_fresh_and_evaluator_overlap_fail_before_update(self):
        kw=self.fixture(); kw['oldids'][0]=kw['bids'][0]
        with self.assertRaises(ValueError): r.audit_state(**kw)
        kw=self.fixture(); kw['fids'][0]=kw['bids'][0]
        with self.assertRaises(ValueError): r.audit_state(**kw)

    def test_resource_caps_are_failure_gates_not_truncation(self):
        caps=r.ResourceCaps()
        caps.check_values(cpu=1,wall=1,rss=1,output=1)
        for key,limit in [('cpu',180),('wall',300),('rss',768*1024**2),('output',32*1024**2)]:
            d=dict(cpu=1,wall=1,rss=1,output=1); d[key]=limit+1
            with self.assertRaises(RuntimeError): caps.check_values(**d)

    def test_freeze_detects_source_and_input_changes_before_scoring(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); s=root/'source.py'; i=root/'input.npz'; s.write_text('source'); i.write_bytes(b'not loaded')
            frozen={'files':{p.name:r.sha256_file(p) for p in [s,i]}}
            with patch.object(r,'ROOT',root), patch.object(r,'RAW',root):
                r.verify_freeze(frozen)
                i.write_bytes(b'changed')
                with self.assertRaises(ValueError): r.verify_freeze(frozen)

    def test_no_overwrite_json_preserves_prior_artifact(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'x.json'; r.write_once(p,{'old':1})
            with self.assertRaises(FileExistsError): r.write_once(p,{'new':2})
            self.assertEqual(json.loads(p.read_text()),{'old':1})

    def test_caps_include_time_before_execute_and_freeze_ignores_results(self):
        import time
        caps=r.ResourceCaps(started=(time.process_time()-3,time.monotonic()-2))
        self.assertGreaterEqual(caps.snapshot()['cpu'],3)
        self.assertGreaterEqual(caps.snapshot()['wall'],2)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); (root/'results').mkdir(); (root/'verification').mkdir()
            for name in ['source.py','review-freeze.json','review-freeze-v1-initial.json','results/out.json','verification/note.json']:
                (root/name).write_text('{}')
            self.assertEqual([p.name for p in r.local_freezable_files(root)],['source.py'])

    def test_reservoir_reconstruction_matches_incremental_admissions(self):
        gen=np.random.default_rng(8); x=gen.random((48,196),dtype=np.float32); y=np.arange(48)%3; ids=np.arange(48)
        got=r.reconstruct_reservoir(x,y,ids,100,32)
        expected=c.Reservoir(100)
        for start in [0,16]: expected.admit(x[start:start+16],y[start:start+16],ids[start:start+16])
        for a,b in zip(got.arrays(),expected.arrays()): np.testing.assert_array_equal(a,b)
        self.assertEqual(got.seen,32)

if __name__=='__main__': unittest.main()
