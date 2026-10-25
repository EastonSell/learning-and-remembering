"""Synthetic-only checks. No PenDigits data is loaded by this suite."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import time
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

class EngineTests(unittest.TestCase):
    def setUp(self):
        import engine
        self.e = engine
        rng = np.random.default_rng(4)
        self.x = rng.normal(size=(100,16))
        self.y = np.repeat(np.arange(10),10)
        self.phases = self.e.build_phases(self.y, np.arange(100), 51001)

    def run_arm(self, arm, phases=None, callback=None):
        return self.e.train_case(self.x, self.y, self.phases if phases is None else phases,
                                 51001, arm, callback=callback)

    def test_model_mac_coefficients_are_exact(self):
        self.assertEqual(self.e.FORWARD_UNITS*128,16*64+64*10)
        self.assertEqual(self.e.TRAIN_UNITS*128,16*64+64*10+10*64+16*64+64*10)

    def test_uniform_has_no_virtual_steps_or_candidate_scoring(self):
        for arm in ('uniform_base','uniform_credit'):
            r=self.run_arm(arm)
            self.assertEqual(r['counts']['virtual_steps'],0)
            self.assertEqual(r['counts']['candidate_scoring_rows'],0)

    def test_empty_memory_creates_neither_virtual_work_nor_credit(self):
        phases=[((0,1),np.arange(5))]
        for arm in self.e.ARMS:
            r=self.run_arm(arm,phases)
            self.assertEqual(r['counts']['virtual_steps'],0)
            self.assertEqual(r['counts']['extra_steps'],0)
            self.assertEqual(r['credit'],0)
            self.assertEqual(r['counts']['fresh_presentations'],20)

    def test_credit_is_spent_without_borrowing_and_reconciles_executed_work(self):
        rows={a:self.run_arm(a) for a in self.e.ARMS}
        b,m,u=[rows[a] for a in self.e.ARMS]
        self.assertEqual(m['counts']['work_units']-u['counts']['work_units'],u['credit'])
        self.assertGreaterEqual(u['credit'],0)
        self.assertLess(u['credit'],992)
        self.assertGreater(u['counts']['extra_steps'],0)
        self.assertEqual(b['counts']['fresh_presentations'],u['counts']['fresh_presentations'])
        self.assertEqual(b['counts']['base_replay_trace_sha256'],u['counts']['base_replay_trace_sha256'])
        for a in self.e.ARMS:
            self.assertEqual(rows[a]['counts']['reservoir_trace_sha256'],b['counts']['reservoir_trace_sha256'])
            self.assertEqual(rows[a]['counts']['base_steps'],b['counts']['base_steps'])
        self.assertEqual(u['counts']['extra_replay_presentations'],32*u['counts']['extra_steps'])

    def test_accounting_checker_rejects_fabricated_credit_and_uniform_scoring(self):
        rows=[]
        for arm in self.e.ARMS:
            r=self.run_arm(arm);r.update(seed=51001,arm=arm);rows.append(r)
        self.e.validate_accounting(rows,[51001])
        for mutate in (
            lambda r:r[2].update(credit=r[2]['credit']+1),
            lambda r:r[0]['counts'].update(candidate_scoring_rows=1),
            lambda r:r[2]['counts'].update(base_replay_trace_sha256='wrong'),
            lambda r:r[2]['counts'].update(credited_units=0)):
            bad=copy.deepcopy(rows);mutate(bad)
            with self.assertRaises(ValueError):self.e.validate_accounting(bad,[51001])

    def test_reproducible_weights_and_counters(self):
        for arm in self.e.ARMS:
            a,b=self.run_arm(arm),self.run_arm(arm)
            self.assertEqual(a['counts'],b['counts'])
            for v,w in zip(a['weights'],b['weights']):
                np.testing.assert_array_equal(v,w)

    def test_phase_partition_uses_every_train_row_once(self):
        ids=np.concatenate([v for _,v in self.phases])
        np.testing.assert_array_equal(np.sort(ids),np.arange(100))
        self.assertEqual([list(p) for p,_ in self.phases],[[0,1],[2,3],[4,5],[6,7],[8,9],[8,9],[6,7],[4,5],[2,3],[0,1]])

    def test_synthetic_gradient_and_no_weight_mutation(self):
        w=self.e.init(5); before=[v.copy() for v in w]
        x,y=self.x[:8],self.y[:8]
        ww=self.e.step(w,x,y)
        self.assertLess(self.e.losses(ww,x,y).mean(),self.e.losses(w,x,y).mean())
        for a,b in zip(w,before): np.testing.assert_array_equal(a,b)
        eps=1e-5; plus=copy.deepcopy(w);minus=copy.deepcopy(w)
        plus[0][2,3]+=eps;minus[0][2,3]-=eps
        fd=(self.e.losses(plus,x,y).mean()-self.e.losses(minus,x,y).mean())/(2*eps)
        self.assertAlmostEqual(fd,(w[0][2,3]-ww[0][2,3])/.05,places=6)

    def test_nonfinite_gradient_is_rejected_without_python_assert(self):
        x=self.x[:2].copy(); x[0,0]=np.nan
        with self.assertRaisesRegex(ValueError,'finite'):
            self.e.step(self.e.init(0),x,self.y[:2])

    def test_bad_arm_rejected_before_any_training(self):
        with self.assertRaisesRegex(ValueError,'arm'):
            self.run_arm('mri')

    def test_boundary_evaluation_time_is_excluded(self):
        calls=[]
        def callback(phase,pair,w,partial):
            calls.append(phase);time.sleep(.01)
        r=self.run_arm('uniform_base',self.phases[:2],callback)
        self.assertEqual(calls,[0,1])
        self.assertGreaterEqual(r['case_seconds']-r['train_seconds'],.02)

    def test_resource_guard_can_stop_inside_case(self):
        checks=[]
        def guard():
            checks.append(1)
            if len(checks)>8: raise TimeoutError('synthetic cap')
        with self.assertRaisesRegex(TimeoutError,'synthetic cap'):
            self.e.train_case(self.x,self.y,self.phases,51001,'mir',check=guard)

    def test_execution_order_balances_each_position(self):
        orders=[self.e.case_order(i) for i in range(12)]
        self.assertEqual(len(set(orders)),6)
        for arm in self.e.ARMS:
            self.assertEqual([sum(o[p]==arm for o in orders) for p in range(3)],[4,4,4])

class AnalysisTests(unittest.TestCase):
    def setUp(self):
        import engine
        self.e=engine
        self.seeds=list(range(51001,51013))
        self.rows=[]
        for seed in self.seeds:
            for arm in self.e.ARMS:
                self.rows.append({'seed':seed,'arm':arm,'status':'complete',
                    'final_macro':.52 if arm=='mir' else .5,
                    'mean_acquisition':.5,'train_seconds':1. if arm=='mir' else 2.,
                    'boundaries':[{'macro':.52 if arm=='mir' else .5,'acquisition':.5}]*10})

    def test_paired_statistics_and_strict_dominance_gate(self):
        r=self.e.analyze(self.rows,self.seeds)
        p=r['comparisons']['uniform_credit']
        self.assertAlmostEqual(p['mean_macro_difference'],.02)
        self.assertAlmostEqual(p['geomean_runtime_ratio'],.5)
        self.assertTrue(p['accuracy_support'])
        self.assertTrue(p['within_environment_dominance'])
        self.assertEqual(len(p['pairs']),12)

    def test_high_accuracy_does_not_make_slow_mir_efficient(self):
        for r in self.rows:
            if r['arm']=='mir':r['train_seconds']=4.
        p=self.e.analyze(self.rows,self.seeds)['comparisons']['uniform_credit']
        self.assertTrue(p['accuracy_support']);self.assertFalse(p['within_environment_dominance'])

    def test_missing_duplicate_nonfinite_and_failed_cases_rejected(self):
        cases=[self.rows[:-1],self.rows+[self.rows[0]],copy.deepcopy(self.rows),copy.deepcopy(self.rows),copy.deepcopy(self.rows)]
        cases[2][0]['final_macro']=float('nan')
        cases[3][0]['status']='failed'
        cases[4][0]['train_seconds']=0
        for rows in cases:
            with self.assertRaises(ValueError):self.e.analyze(rows,self.seeds)

    def test_acquisition_gate_and_no_seed_replacement(self):
        for r in self.rows:
            if r['arm']=='mir':
                r['mean_acquisition']=.4
                r['boundaries']=[dict(b,acquisition=.4) for b in r['boundaries']]
        self.assertFalse(self.e.analyze(self.rows,self.seeds)['comparisons']['uniform_credit']['accuracy_support'])
        self.rows[0]['seed']=999
        with self.assertRaises(ValueError):self.e.analyze(self.rows,self.seeds)

    def test_summary_must_match_boundaries(self):
        self.rows[0]['final_macro']=.6
        with self.assertRaisesRegex(ValueError,'boundary'):
            self.e.analyze(self.rows,self.seeds)

    def test_secondary_comparison_is_not_confirmatory(self):
        p=self.e.analyze(self.rows,self.seeds)['comparisons']['uniform_base']
        self.assertIsNone(p['accuracy_support'])
        self.assertIsNone(p['within_environment_dominance'])

if __name__=='__main__':unittest.main()
