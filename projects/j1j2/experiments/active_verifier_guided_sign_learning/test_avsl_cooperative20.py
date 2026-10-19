#!/usr/bin/env python3
"""Independent full-H energy regression for the post-Krylov oracle cut scorer.

Run with AVSL_SIGN20_DATA=/path/to/regenerated/sign20_training_data.npz
No training. Exact reference is evaluation-only and not a scalable oracle.
"""
import os
import unittest
import numpy as np
from avsl_cooperative20 import Oracle, BONDS, patch, orbit_boundary, cut_gain, best_subset


@unittest.skipUnless(os.getenv("AVSL_SIGN20_DATA"), "regenerate exact 20-site FN data first")
class CooperativeReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.o=Oracle(os.environ["AVSL_SIGN20_DATA"])
        cls.patches=[patch(cls.o,int(cls.o.states[i]),101+i,8) for i in (100,1234,54321)]
        cls.all_keys=sorted({g for p in cls.patches for g in p})
        cls.boundaries={g:orbit_boundary(cls.o,g) for g in cls.all_keys}

    def direct_full_H(self, keys):
        """Construct both physical state vectors, evaluate all off-diagonal H hops."""
        o=self.o
        v=o.a_data * o.s_data
        eta=np.ones(len(o.states),dtype=np.int8)
        for g in keys:
            for x in o.members[g]:
                eta[o.look[x]]=-1
        new=v*eta
        eold=enew=0.
        for bits,h in BONDS:
            neigh=o.look[o.states ^ np.uint32(bits)]
            valid=neigh>=0
            idx=neigh[valid]
            eold += h*np.dot(v[valid],v[idx])
            enew += h*np.dot(new[valid],new[idx])
        return (eold-enew)/o.norm

    def test_direct_full_h_single_and_joint_energy(self):
        p=self.patches[0]
        for chosen in ([p[0]],[p[1]],[p[0],p[1]],[p[1],p[3],p[4]],
                       p[:8]):
            direct=self.direct_full_H(chosen)
            boundary=cut_gain(chosen,self.boundaries,self.o.norm)
            self.assertAlmostEqual(direct,boundary,delta=2e-11)

    def test_orbit_amplitude_and_sign_symmetry(self):
        o=self.o
        for g in self.all_keys:
            a=o.a(g);s=o.s(g)
            for x in o.members[g]:
                self.assertEqual(o.s(x),s)
                self.assertAlmostEqual(o.a(x),a,delta=1e-9*max(a,1e-10)+1e-11)

    def test_symmetric_orbit_couplings(self):
        o=self.o
        for g in self.all_keys:
            for h,w in self.boundaries[g].items():
                if h not in self.boundaries:
                    continue
                rev=self.boundaries[h].get(g,0.)
                self.assertAlmostEqual(w,rev,delta=1e-10)

    def test_subsets_never_worse_than_single_or_keep(self):
        for p in self.patches:
            v,mask,vals=best_subset(p,self.boundaries,self.o.norm)
            self.assertGreaterEqual(v,0.)
            self.assertGreaterEqual(v+1e-12,max(vals))
            chosen=[g for i,g in enumerate(p) if mask>>i&1]
            self.assertAlmostEqual(v,cut_gain(chosen,self.boundaries,self.o.norm),delta=1e-11)

    def test_global_sign_flip_invariant(self):
        o=self.o
        # Explicit physical whole-sector sign reversal: vector -> -vector.
        psi=o.a_data*o.s_data
        self.assertEqual(float(np.dot(psi,psi)),float(np.dot(-psi,-psi)))
        self.assertAlmostEqual(self.direct_full_H([]),0.,delta=1e-12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
