"""Physics guards for AVSL-08 mixed FN walker support experiment."""
import unittest, math
import numpy as np
from avsl08_mixed_fn20 import Tilted20, ExactSignFN, make_fn_walkers, root_support, one_hop_frontier
from avsl_cooperative20 import Oracle,BONDS
from callable_fn_loop import CachedAmplitude

DATA='/tmp/avsl05-20site-structural-data/sign20_training_data.npz'

class TrueFNMixedTests(unittest.TestCase):
    def test_tilted_neigh_and_zz_diagonal_agree_npz(self):
        o=Oracle(DATA);lat=Tilted20(o);rng=np.random.default_rng(712)
        dd=np.load(DATA)['diag']
        for i in rng.integers(len(o.states),size=31):
            x=int(o.states[i])
            self.assertAlmostEqual(lat.diag_energy(x),dd[i],places=13)
            got={(int(y),float(J*.5)) for y,J,_ in lat.neigh(x)}
            expected={(int(y),float(h)) for y,h in o.neigh(x)}
            self.assertEqual(got,expected)

    def test_projector_local_energy_matches_signed_H_row(self):
        o=Oracle(DATA);lat=Tilted20(o)
        eng=ExactSignFN(lat,CachedAmplitude(lambda xs:np.log(np.asarray([o.a(int(z)) for z in xs]))),o)
        rng=np.random.default_rng(812)
        ix=rng.integers(0,len(o.states),size=35)
        for i in ix:
            x=int(o.states[i]);eng.ensure_local([x])
            sx=o.s(x);ax=o.a(x)
            physical=lat.diag_energy(x)+sum(.5*J*sx*o.s(y)*o.a(y)/ax for y,J,_ in lat.neigh(x))
            self.assertAlmostEqual(eng.local_cache[x][1],physical,places=9)
            d,eloc,ys,rates=eng.local_cache[x]
            self.assertGreaterEqual(min([0.,*rates]),0.)
            self.assertEqual(len(ys),len(rates))

    def test_m128_real_projector_not_metropolis_fake_snapshots(self):
        roots,info=make_fn_walkers(DATA,17771,M=128,beta=.16,burn=.05,n_init=256,init_burn=400,init_thin=3)
        self.assertEqual(info['M'],128)
        self.assertEqual(info['n_walkers'],128*info['snapshots'])
        self.assertGreater(info['snapshots'],0)
        self.assertGreater(info['all_oracle_cost']['unique_amp'],info['init_oracle_cost']['unique_amp'])
        self.assertGreater(info['fn_lattice_neighbor_edges'],0)
        self.assertTrue(np.isfinite(info['fn_energy_mean']))
        self.assertTrue(np.isfinite(info['local_energy_spread']))
        self.assertFalse(info['guide_is_converged_FN_eigenvector'])

    def test_support_is_orbit_invariant(self):
        o=Oracle(DATA)
        x=int(o.states[112]);g=o.key(x)
        y=o.members[g][-1]
        a=root_support(o,[x],{g:.1},{g:0.01})
        b=root_support(o,[int(y)],{g:.1},{g:0.01})
        self.assertEqual(a,b)
        self.assertIn(g,one_hop_frontier(o,[x]))

if __name__=='__main__':unittest.main()
