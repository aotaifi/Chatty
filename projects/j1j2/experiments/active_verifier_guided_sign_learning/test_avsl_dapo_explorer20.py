"""Physics + policy unit controls; no RL convergence claim."""
import unittest
import numpy as np
from avsl_dapo_explorer20 import World, trajectory, DIM, marshall
from avsl_cooperative20_validate import full_offdiag, energy_off

DATA='/tmp/avsl05-20site-structural-data/sign20_training_data.npz'

class AVSLDapo20Tests(unittest.TestCase):
    def test_exploration_geometry_no_oracle_calls(self):
        w=World(DATA,'post_krylov',13)
        x=int(w.o.states[120])
        opts=w.options(x,0)
        self.assertGreater(len(opts),0)
        self.assertEqual(w.o.n_amp_calls,0)
        self.assertEqual(w.o.n_sign_calls,0)
        for y,f in opts:
            self.assertEqual(y.bit_count(),10)
            self.assertEqual(len(f),DIM)
            self.assertTrue(np.isfinite(f).all())

    def test_terminal_score_matches_direct_fullH(self):
        w=World(DATA,'post_krylov',13)
        h=full_offdiag(w.o)
        for chamber in ('marshall','post_krylov'):
            w.chamber=chamber
            # One independent readout per chamber, start from freshly reset cache.
            w.scores.clear()
            signs=np.array([w.s(int(x)) for x in w.o.states],dtype=float)
            e0=energy_off(w.o,h,signs)
            for ix in (12,329,940):
                g,gain,fresh=w.score(int(w.o.states[ix]))
                self.assertTrue(fresh)
                sn=signs.copy()
                for member in w.o.members[g]:
                    sn[w.o.look[member]] *= -1
                direct=e0-energy_off(w.o,h,sn)
                self.assertAlmostEqual(gain,direct,places=11)

    def test_rollout_is_real_policy_not_offline_ranker(self):
        x=int(World(DATA,'marshall',13).o.states[15])
        w=World(DATA,'marshall',13)
        tr=trajectory(w,x,np.zeros(DIM),'learned',np.random.default_rng(43))
        self.assertEqual(tr['root'],x)
        self.assertEqual(tr['grad'].shape,(DIM,))
        self.assertEqual(w.policy_moves,2)
        self.assertGreater(w.feature_queries,0)
        self.assertEqual(w.evaluations,1)
        self.assertEqual(tr['positive'],tr['gain']>1e-12)

    def test_global_phase_not_learned_as_score(self):
        w=World(DATA,'post_krylov',13)
        x=int(w.o.states[100])
        self.assertEqual(w.s(x)**2,1)
        self.assertEqual(marshall(x,w.mask)**2,1)

if __name__=='__main__':unittest.main()
