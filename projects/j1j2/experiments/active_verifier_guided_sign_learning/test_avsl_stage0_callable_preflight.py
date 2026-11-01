#!/usr/bin/env python3
"""Fast invariant checks for the bounded 6x6 callable FN preflight."""
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
from avsl_stage0_callable_preflight import (
    CountingLattice, AnalyticJastrow, independent_a2_chains, random_half_filled
)
from avsl_onpolicy_core import CostLedger, paired_local_difference
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from callable_fn_loop import CachedAmplitude, K1FNEngine


def ledger(p):
    return CostLedger(p,run_id='unit',code_commit='test',chamber_id='test',
       arm='test',device='cpu',model_hash='analytic',walker_seed=1,proposal_seed=2,training_seed=3)


class Stage0Tests(unittest.TestCase):
    def test_neighbour_symmetry_and_sector(self):
        lat=CountingLattice(6,.5)
        rng=np.random.default_rng(411)
        for _ in range(16):
            x=random_half_filled(lat.N,rng)
            ls=lat.neigh(x)
            self.assertEqual(len(ls),len(set(y for y,_,_ in ls)))
            for y,J,_ in ls:
                self.assertEqual(y.bit_count(),18)
                self.assertTrue(any(z==x and abs(J2-J)<1e-12 for z,J2,_ in lat.neigh(y)))

    def test_jastrow_covariance(self):
        lat=CountingLattice(6,.5)
        fun=AnalyticJastrow(lat)
        rng=np.random.default_rng(412)
        for _ in range(12):
            x=random_half_filled(lat.N,rng)
            y=0
            for i in range(lat.N):
                xx,yy=i%lat.L,i//lat.L
                j=((xx+1)%lat.L)+lat.L*((yy+2)%lat.L)
                y|=((x>>i)&1)<<j
            np.testing.assert_allclose(fun.log_amplitude_bits([x,y]),
                                       [fun.log_amplitude_bits([x])[0]]*2,rtol=0,atol=1e-12)

    def test_k1_local_field_matches_explicit_Marshall(self):
        lat=CountingLattice(6,.5)
        fun=AnalyticJastrow(lat)
        amp=CachedAmplitude(fun.log_amplitude_bits)
        eng=K1FNEngine(lat,amp)
        rng=np.random.default_rng(413)
        xs=[random_half_filled(lat.N,rng) for _ in range(9)]
        eng.ensure_r(xs)
        for x in xs:
            la=fun.log_amplitude_bits([x])[0]
            explicit=lat.diag_energy(x)
            for y,J,mr in lat.neigh(x):
                explicit+=.5*J*mr*math.exp(fun.log_amplitude_bits([y])[0]-la)
            self.assertAlmostEqual(explicit,eng.r_cache[x],places=10)

    def test_a2_proposal_detailed_balance(self):
        lat=CountingLattice(6,.5)
        fun=AnalyticJastrow(lat)
        rng=np.random.default_rng(414)
        for _ in range(30):
            x=random_half_filled(lat.N,rng)
            j,k,_,_=lat.bonds[int(rng.integers(len(lat.bonds)))]
            y=x^(1<<j)^(1<<k)
            lx,ly=fun.log_amplitude_bits([x,y])
            accxy=min(1.,math.exp(2*(ly-lx)))
            accyx=min(1.,math.exp(2*(lx-ly)))
            # Symmetric uniform bond proposal; stationary law pi ∝ exp(2 loga).
            self.assertAlmostEqual(math.exp(2*lx)*accxy,math.exp(2*ly)*accyx,places=9)

    def test_reproducible_independent_a2_chains(self):
        lat=CountingLattice(6,.5)
        fun=AnalyticJastrow(lat)
        with tempfile.TemporaryDirectory() as td:
            z=ledger(Path(td)/'events.jsonl')
            a,ra=independent_a2_chains(lat,
                 lambda x: fun.log_amplitude_bits([x])[0],
                 z,seed_base=876,chains=4,burn=10,thin=2,draws=8,phase='unit')
            b,rb=independent_a2_chains(lat,
                 lambda x: fun.log_amplitude_bits([x])[0],
                 z,seed_base=876,chains=4,burn=10,thin=2,draws=8,phase='unit')
            self.assertEqual(a,b)
            self.assertEqual(ra,rb)
            self.assertEqual(len(a),4)
            self.assertTrue(all(len(v)==8 and all(x.bit_count()==18 for x in v) for v in a))
            self.assertEqual(len((Path(td)/'events.jsonl').read_text().splitlines()),2)

    def test_paired_invariants(self):
        lat=CountingLattice(6,.5)
        fun=AnalyticJastrow(lat)
        loga=lambda x:fun.log_amplitude_bits([x])[0]
        neigh=lambda x:[(y,.5*J) for y,J,_ in lat.neigh(x)]
        sx=lambda x:lat.marshall(x)
        x=random_half_filled(36,np.random.default_rng(91))
        self.assertEqual(paired_local_difference(x,neigh,loga,sx,lambda y:1),0)
        self.assertEqual(paired_local_difference(x,neigh,loga,sx,lambda y:-1),0)


if __name__=='__main__':
    unittest.main()
