import unittest
import numpy as np
from functional_circuits import competence_controls as cc
from functional_circuits.data import generate
from functional_circuits.role_controls import make_control


class CompetenceControlTests(unittest.TestCase):
    def candidate(self, epoch, accuracy, contexts=None):
        return {'epoch': epoch, 'accuracy': accuracy,
                'context_accuracy': contexts or [accuracy, accuracy]}

    def test_selection_maximizes_competence_without_using_iia(self):
        a = [self.candidate(1, .75), self.candidate(4, .90), self.candidate(8, .91)]
        b = [self.candidate(1, .75), self.candidate(4, .905)]
        result = cc.select_pair(a, b, .02, .04, .70)
        self.assertEqual(result, (2, 1))
        for row in a+b:
            row['test_accuracy'] = 0
            row['memory_iia'] = 1
        self.assertEqual(cc.select_pair(a, b, .02, .04, .70), result)

    def test_selection_rejects_floor_gap_context_and_breaks_ties(self):
        for a, b in [(.69, .69), (.80, .85)]:
            self.assertIsNone(cc.select_pair([self.candidate(1,a)], [self.candidate(1,b)], .02,.04,.70))
        a = [self.candidate(1,.80,[.70,.90])]
        b = [self.candidate(1,.80,[.90,.70])]
        self.assertIsNone(cc.select_pair(a,b,.02,.04,.70))
        a = [self.candidate(8,.80),self.candidate(4,.80)]
        self.assertEqual(cc.select_pair(a,a,.02,.04,.70),(1,1))

    def test_shared_subset_denominators_and_empty_null(self):
        y=np.array([0,1,1,0]); cf=np.array([1,1,0,0])
        factual=[y.copy(),y.copy()]
        explicit=[cf.copy(),cf.copy()]
        factual[0][3]=1; explicit[1][1]=0
        patched=[np.array([1,0,1,0]),np.array([0,1,0,0])]
        got=cc.shared_subset_metrics(y,cf,factual,explicit,patched)
        self.assertEqual(got['examples'],2)
        self.assertEqual(got['changed_examples'],2)
        self.assertEqual(got['iia'],[.5,.5])
        self.assertEqual(got['changed_iia'],[.5,.5])
        got=cc.shared_subset_metrics(y,cf,[1-y,y],explicit,patched)
        self.assertEqual(got['examples'],0)
        self.assertEqual(got['iia'],[None,None])

    def test_explicit_counterfactual_replaces_only_memory(self):
        a,b=generate(100,11),generate(100,12)
        x=cc.explicit_counterfactual(a,b)
        np.testing.assert_array_equal(x[:,:6],a.x[:,:6])
        np.testing.assert_array_equal(x[:,6:12],b.x[:,6:12])
        np.testing.assert_array_equal(x[:,-1],a.x[:,-1])
        self.assertFalse(np.shares_memory(x,a.x))

    def test_tiny_run_repeats_and_protects_output(self):
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory
        protocol=cc.load_protocol()
        protocol.update(seeds=[101], epochs=[1,2], train_size=32,
                        selection_size=64,mapping_size=64,test_size=64,
                        minimum_accuracy=0.0, overall_accuracy_tolerance=1.0,
                        context_accuracy_tolerance=1.0)
        with TemporaryDirectory() as folder:
            p=Path(folder)/'protocol.json';p.write_text(json.dumps(protocol))
            for name in ('a','b'):
                cc.run(p,Path(folder)/name)
            for artifact in ('candidates.json','selections.json','metrics.json','summary.json'):
                self.assertEqual((Path(folder)/'a'/artifact).read_bytes(),(Path(folder)/'b'/artifact).read_bytes())
            rows=json.loads((Path(folder)/'a'/'metrics.json').read_text())
            self.assertEqual(len(rows),3)
            self.assertTrue(all(r['matched'] and r['parameters']==[132,132] for r in rows))
            with self.assertRaises(ValueError):
                cc.run(p,Path(folder)/'a')

    def test_aligned_memory_patch_is_explicit_input_replacement(self):
        # Thus 100% IIA conditional on correct explicit CF is structural,
        # not independent evidence of discovered causal abstraction.
        a,b=generate(100,11),generate(100,12)
        for seed in (3,7):
            model=make_control(seed,0)
            model.params['gate_w'][:]=2.0
            model.params['gate_b'][:]=-.3
            np.testing.assert_allclose(model.patch(a.x,b.x,[1]),
                model.forward(cc.explicit_counterfactual(a,b))[0],atol=1e-14)
        mixed=make_control(3,2)
        self.assertFalse(np.allclose(mixed.patch(a.x,b.x,[1]),
            mixed.forward(cc.explicit_counterfactual(a,b))[0]))
