"""Deterministic guardrails for dense intermediate AVSL-07 reward."""
import unittest
import numpy as np
from avsl_dense_process20 import signed_process,rollout,train_policy
from avsl_dapo_explorer20 import World,DIM

DATA='/tmp/avsl05-20site-structural-data/sign20_training_data.npz'

class DenseProcessTests(unittest.TestCase):
    def test_four_bonds_no_full_orbit_reward(self):
        w=World(DATA,'post_krylov',21)
        x=int(w.o.states[516])
        proc,n=signed_process(w,x,np.random.default_rng(22),4)
        self.assertTrue(np.isfinite(proc));self.assertEqual(n,4)
        self.assertEqual(w.evaluations,0)
        self.assertEqual(w.orbit_calls,0)
        self.assertLessEqual(w.o.counts()['unique_amp'],5)
        self.assertLessEqual(w.o.counts()['unique_sign'],5)

    def test_global_sign_invariance_of_local_proxy(self):
        w1=World(DATA,'post_krylov',21)
        w2=World(DATA,'post_krylov',21)
        raw_s=w2.s
        w2.s=lambda x:-raw_s(x)
        x=int(w1.o.states[902])
        a,_=signed_process(w1,x,np.random.default_rng(69),4)
        b,_=signed_process(w2,x,np.random.default_rng(69),4)
        self.assertAlmostEqual(a,b,places=13)

    def test_no_ed_sign_labels_in_features(self):
        w=World(DATA,'post_krylov',21)
        x=int(w.o.states[1021])
        opts=w.options(x,0)
        self.assertGreater(len(opts),0)
        self.assertEqual(w.o.counts()['unique_amp'],0)
        self.assertEqual(w.o.counts()['unique_sign'],0)

    def test_rollout_real_terminal_reward_only(self):
        w=World(DATA,'post_krylov',21)
        x=int(w.o.states[1021])
        tr=rollout(w,x,np.zeros(DIM),np.random.default_rng(42),.15,'dense_signed',4)
        self.assertEqual(len(tr['grads']),2)
        self.assertEqual(tr['nproc'],4)
        self.assertEqual(tr['success'],tr['gain']>1e-12)
        self.assertEqual(w.evaluations,1)

    def test_all_negative_binary_gets_no_group_update(self):
        w=World(DATA,'post_krylov',21)
        roots=[int(w.o.states[2037])]
        th,c,_,_=train_policy(w,roots,29,'binary',budget=4)
        self.assertTrue(np.isfinite(th).all())
        self.assertEqual(c['attempted'],4)
        self.assertEqual(c.get('updates',0),c.get('mixed_binary',0))

if __name__=='__main__':
    unittest.main()
