"""Pure tiny synthetic mechanics. Never loads historical data/checkpoints."""
import copy
import unittest
import numpy as np
from jointaudit import mechanics as m
from vendor import deployed_core as c

class MechanicsTests(unittest.TestCase):
    def fixture(self):
        p=(np.array([[.8,-.2],[.3,.6]],np.float32),np.array([.1,.2],np.float32),
           np.array([[.4,-.3],[-.2,.7]],np.float32),np.array([.1,-.1],np.float32))
        return p,np.array([[1.,.2],[.4,.9]],np.float32),np.array([0,1]),np.array([[.2,1.]],np.float32),np.array([0])

    def test_joint_is_one_concatenated_mean_update_not_sequential(self):
        p,fx,fy,rx,ry=self.fixture()
        got=m.branch_update(p,fx,fy,rx,ry,.3,'joint')
        expected=c.step(p,np.concatenate([fx,rx]),np.concatenate([fy,ry]),.3)
        for a,b in zip(got,expected): np.testing.assert_array_equal(a,b)
        seq=m.branch_update(p,fx,fy,rx,ry,.3,'sequential')
        self.assertTrue(any(not np.allclose(a,b) for a,b in zip(got,seq)))

    def test_clones_leave_original_and_all_batches_unchanged(self):
        p,fx,fy,rx,ry=self.fixture(); original=copy.deepcopy((p,fx,fy,rx,ry))
        for mode in ['joint','sequential']: m.branch_update(p,fx,fy,rx,ry,.3,mode)
        for got,want in zip((p,fx,fy,rx,ry),original):
            for a,b in zip(got,want) if isinstance(got,tuple) else [(got,want)]: np.testing.assert_array_equal(a,b)

    def test_source_ties_straddle_cutoff_and_preserve_negative_scores(self):
        ids=np.array([30,10,20,40]); y=np.array([0,0,0,0]); scores=np.array([-.2,-.1,-.1,-.3])
        selected,log=m.ranked_selection(ids,y,{0:1},scores,'source_id',123)
        self.assertEqual(ids[selected].tolist(),[10])
        self.assertEqual(log[0]['cutoff'], -.1)
        self.assertEqual(log[0]['equal_count'],2)
        self.assertTrue(log[0]['straddling'])
        self.assertEqual(log[0]['tied_selected_ids'],[10]); self.assertEqual(log[0]['tied_rejected_ids'],[20])

    def test_random_tie_priorities_do_not_reorder_unequal_scores(self):
        ids=np.array([40,10,20,30]); y=np.zeros(4,int); scores=np.array([9.,1.,1.,1.])
        picks=[]
        for seed in range(20):
            selected,_=m.ranked_selection(ids,y,{0:2},scores,'random',seed)
            self.assertEqual(ids[selected][0],40); picks.append(int(ids[selected][1]))
        self.assertGreater(len(set(picks)),1)
        a,_=m.ranked_selection(ids,y,{0:2},scores,'random',5)
        perm=np.array([2,0,3,1]); b,_=m.ranked_selection(ids[perm],y[perm],{0:2},scores[perm],'random',5)
        self.assertEqual(ids[a].tolist(),ids[perm][b].tolist())

    def test_near_ties_logged_without_overriding_true_rank(self):
        ids=np.array([1,2,3]); y=np.zeros(3,int); scores=np.array([.1,.10000001,.3])
        selected,log=m.ranked_selection(ids,y,{0:2},scores,'source_id',0)
        self.assertEqual(ids[selected].tolist(),[3,2]); self.assertEqual(log[0]['equal_count'],1)
        self.assertFalse(log[0]['straddling']); self.assertEqual(log[0]['near_count'],2)

    def test_uniform_quota_matching_and_canonical_reproducibility(self):
        ids=np.arange(12); y=np.repeat([0,1,2],4); q={0:1,1:2,2:1}
        a=m.uniform_selection(ids,y,q,44); b=m.uniform_selection(ids[::-1],y[::-1],q,44)
        self.assertEqual(sorted(ids[a]),sorted(ids[::-1][b])); self.assertEqual(len(set(a)),4)
        self.assertEqual(np.bincount(y[a]).tolist(),[1,2,1])

    def test_bad_candidates_scores_and_quotas_fail(self):
        for ids,y,q,scores in [([1,1],[0,0],{0:1},[0.,1.]),([1,2],[0,0],{0:3},[0.,1.]),([1,2],[0,0],{0:1},[0.,np.nan])]:
            with self.assertRaises(ValueError): m.ranked_selection(np.array(ids),np.array(y),q,np.array(scores),'source_id',0)

    def test_macro_loss_is_equal_class_not_example_weight(self):
        self.assertEqual(m.macro_loss(np.array([1.,3.,8.]),np.array([0,0,1])),5.)
        with self.assertRaises(ValueError): m.macro_loss(np.array([np.inf]),np.array([0]))

    def test_bootstrap_clusters_seeds_not_states(self):
        rows=[{'seed':1,'contrast':-2.},{'seed':1,'contrast':0.},{'seed':2,'contrast':3.}]
        got=m.cluster_summary(rows,'contrast',resamples=100)
        self.assertEqual(got['seed_means'],{'1':-1.,'2':3.}); self.assertEqual(got['mean'],1.)
        self.assertEqual(got['cluster_count'],2); self.assertEqual(got['state_count'],3)
        self.assertEqual(got['status'],'posthoc_descriptive_only')

    def test_state_digest_detects_one_parameter_change(self):
        p,*_=self.fixture(); q=tuple(a.copy() for a in p); q[0][0,0]+=.01
        self.assertNotEqual(m.state_digest(p),m.state_digest(q))

if __name__=='__main__': unittest.main()
