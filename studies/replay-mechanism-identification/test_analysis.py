import unittest
from fractions import Fraction as Q
from analyze import toy

class AnalyticControl(unittest.TestCase):
    def test_literal_independent_endpoints(self):
        # Computed from F'(0)=1 and R'(x)=4x +/- 1/5, not from toy helper.
        expected={'00':(Q(-21,130),Q(-27,130)), '01':(Q(-129,1690),Q(-207,1690)), '10':(Q(-6,25),Q(-9,25)), '11':(Q(3,25),Q(0))}
        for k,(m,h) in expected.items():
            self.assertEqual(toy()[k]['endpoints'],{'mir':m,'highloss':h})
    def test_sign_reversal(self):
        x=toy()
        self.assertTrue(all(x[k]['D']<0 for k in ('00','01','10')))
        self.assertGreater(x['11']['D'],0)
    def test_actual_selector_scores(self):
        rm=lambda x:2*(x-Q(1,20))**2
        rh=lambda x:2*(x+Q(1,20))**2+1
        self.assertGreater(rh(0),rm(0))
        self.assertGreater(rm(-Q(3,10))-rm(0),rh(-Q(3,10))-rh(0))
    def test_exact_cubic_formula(self):
        a,b,r,h=Q(8,13),Q(5,13),Q(1,5),Q(4)
        for eta in (Q(3,10),Q(3,20),Q(1,100)):
            self.assertEqual(toy(eta)['interaction'],2*eta**3*r*h*(1-a*b*b))
        self.assertEqual(toy(Q(3,20))['interaction'],toy()['interaction']/8)
    def test_nearby_smaller_step_no_reversal(self):
        self.assertTrue(all(toy(Q(3,20))[k]['D']<0 for k in ('00','01','10','11')))
    def test_linear_evaluator_or_frozen_gradients_remove_interaction(self):
        self.assertEqual(toy(linear_eval=True)['interaction'],0)
        self.assertEqual(toy(h=Q(0))['interaction'],0)
    def test_symmetric_transport_attribution(self):
        def attribution(q0,q1,V):
            m0,d0=(q0[0]+q0[1])/2,q0[0]-q0[1]
            m1,d1=(q1[0]+q1[1])/2,q1[0]-q1[1]
            C=lambda m,d:V(m+d/2)-V(m-d/2)
            c00,c10,c01,c11=C(m0,d0),C(m1,d0),C(m0,d1),C(m1,d1)
            A=((c10-c00)+(c11-c01))/2
            B=((c01-c00)+(c11-c10))/2
            return A,B,c11-c00
        x=toy()
        effects=[]
        for w in ('0','1'):
            q0=tuple(x[w+'0']['endpoints'][s] for s in ('mir','highloss'))
            q1=tuple(x[w+'1']['endpoints'][s] for s in ('mir','highloss'))
            A,B,L=attribution(q0,q1,lambda v:(v+Q(1,20))**2/2)
            self.assertEqual(B,0); self.assertEqual(A+B,L);effects.append(A)
        self.assertEqual(effects[1]-effects[0],x['interaction'])
        # Identity remains exact with changed separation and nonquadratic evaluator.
        A,B,L=attribution((Q(-1),Q(2)),(Q(3),Q(5)),lambda v:v**4+v)
        self.assertEqual(A+B,L)
        self.assertNotEqual(A,0);self.assertNotEqual(B,0)
    def test_open_interval_sign_conditions(self):
        for h in (Q(3),Q(4),Q(6)):
            x=toy(h=h)
            self.assertTrue(all(x[k]['D']<0 for k in ('00','01','10')))
            self.assertGreater(x['11']['D'],0)
        self.assertEqual(toy(h=Q(25,9))['11']['D'],0)
        self.assertEqual(toy(h=Q(455,72))['01']['D'],0)
    def test_evaluator_null_differential_drift(self):
        qM0,qH0=(Q(0),Q(0)),(Q(1),Q(0))
        qM1,qH1=(Q(0),Q(1)),(Q(1),Q(-1))
        midpoint=lambda q,r:tuple((x+y)/2 for x,y in zip(q,r))
        difference=lambda q,r:tuple(x-y for x,y in zip(q,r))
        m0,m1=midpoint(qM0,qH0),midpoint(qM1,qH1)
        d0,d1=difference(qM0,qH0),difference(qM1,qH1)
        V=lambda q:q[0]**2/2
        C=lambda m,d:V(tuple(x+y/2 for x,y in zip(m,d)))-V(tuple(x-y/2 for x,y in zip(m,d)))
        c00,c10,c01,c11=C(m0,d0),C(m1,d0),C(m0,d1),C(m1,d1)
        self.assertEqual((c00,c10,c01,c11),(Q(-1,2),)*4)
        A=((c10-c00)+(c11-c01))/2
        B=((c01-c00)+(c11-c10))/2
        self.assertEqual((A,B),(0,0))
        self.assertEqual(sum((x-y)**2 for x,y in zip(d1,d0)),4)
    def test_location_preserves_selector_separation(self):
        x=toy()
        for w in ('0','1'):
            e0=x[w+'0']['endpoints'];e1=x[w+'1']['endpoints']
            self.assertEqual(e0['mir']-e0['highloss'],e1['mir']-e1['highloss'])

if __name__=='__main__':unittest.main(verbosity=2)
