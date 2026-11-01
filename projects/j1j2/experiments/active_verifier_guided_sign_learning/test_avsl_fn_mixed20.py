"""AVSL-08 FN mixed Feynman-Kac process and audit correctness tests."""
import unittest
import numpy as np
from avsl_fn_mixed20 import MixedFN
from avsl_mixed_support_pilot20 import full_orbit_oracle

DATA='/tmp/avsl05-20site-structural-data/sign20_training_data.npz'
REF='/tmp/avsl_fn20_ref_post.npz'

class FNBranchingTests(unittest.TestCase):
    def test_jump_rates_positive_and_reverse_detailed_balance(self):
        w=MixedFN(DATA)
        rng=np.random.default_rng(20394)
        tested=0
        for x in w.o.states[rng.integers(len(w.o.states),size=40)]:
            xx=int(x); dst,rr,_,_,_=w.transitions(xx)
            for y,k in zip(dst[:4],rr[:4]):
                yd,yr,_,_,_=w.transitions(int(y))
                ix=list(map(int,yd)).index(xx)
                ratio=w.o.a(int(y))/w.o.a(xx)
                self.assertAlmostEqual(float(k/yr[ix]),ratio**2,places=9)
                tested+=1
        self.assertGreater(tested,20)
    def test_local_energy_equals_fn_operator_row(self):
        w=MixedFN(DATA)
        rng=np.random.default_rng(13431)
        for x in w.o.states[rng.integers(len(w.o.states),size=20)]:
            xx=int(x);dst,rates,_,_,el=w.transitions(xx)
            # FN gauge diagonal = bare diag + all sign-forbidden positive
            # edges; negative allowed edges remain off-diagonal -h.
            diag=float(w.diag[w.o.look[xx]])
            forb=0.;allow=0.
            for bits,h in __import__('avsl_fn_mixed20').BONDS:
                if (xx&bits).bit_count()!=1:continue
                y=xx^bits
                sign=int(w.sign[w.o.look[xx]])*int(w.sign[w.o.look[y]])
                amp_ratio=w.o.a(y)/w.o.a(xx)
                if sign>0:forb+=h*amp_ratio
                else:allow-=h*amp_ratio
            self.assertAlmostEqual(el,diag+forb+allow,places=10)
            self.assertAlmostEqual(sum(rates),-allow,places=10)
    def test_global_phase_invariance(self):
        a=MixedFN(DATA);b=MixedFN(DATA);b.sign=-b.sign
        for state in map(int,a.o.states[[501,2501,7001,20220]]):
            at=a.transitions(state);bt=b.transitions(state)
            np.testing.assert_equal(at[0],bt[0])
            np.testing.assert_allclose(at[1],bt[1])
            self.assertAlmostEqual(at[4],bt[4])
    def test_mixed_measure_energy_identity(self):
        q=np.load(REF)
        w=MixedFN(DATA)
        # Physical mixed ground state average of trial local energy E_L is
        # E_FN, independent of mismatch between guide a and true phi.
        # Evaluate vectorized using hopping arrays instead of millions of
        # Python calls; matrix relation follows from hermiticity.
        amp=w.guide;sign=w.sign.astype(float)
        phi=q['phi']
        mixed=q['mixed']
        diag=w.diag.astype(float)
        el=diag.copy()
        from avsl_cooperative20 import BONDS
        st=w.o.states
        look=w.o.look
        for bits,h in BONDS:
            mask=((st & bits)!=0)&((st&bits)!=bits)
            ix=np.flatnonzero(mask)
            jx=look[st[ix]^np.uint32(bits)]
            el[ix]+=h*sign[ix]*sign[jx]*amp[jx]/amp[ix]
        self.assertAlmostEqual(float(mixed@el),float(q['fn_energy']),places=7)
        self.assertAlmostEqual(float(mixed.sum()),1.,places=13)
    def test_sampler_is_branching_not_metropolis(self):
        w=MixedFN(DATA)
        samples,stats=w.sample(11098,n_walkers=12,beta=.8,dt=.1,burn=.3,record_every=2,init_burn=100)
        self.assertGreater(stats['n_jumps'],0)
        self.assertEqual(len(samples),stats['n_recorded'])
        self.assertEqual(stats['n_local_energy_evals'],len(w.cache))
        self.assertGreater(stats['n_amp_requests'],0)
        self.assertTrue(np.all(w.o.look[samples]>=0))
    def test_positive_orbit_gain_audit_reference(self):
        a,m,c=full_orbit_oracle(DATA,REF)
        self.assertEqual(len(a),4654)
        pos=[v for v in a.values() if v>1e-12]
        self.assertEqual(len(pos),198)
        self.assertAlmostEqual(sum(pos),.0008001565242074627,places=10)
        self.assertAlmostEqual(sum(m.values()),1.,places=11)

if __name__=='__main__':unittest.main()
