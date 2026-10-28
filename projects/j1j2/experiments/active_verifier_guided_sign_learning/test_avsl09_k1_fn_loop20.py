"""AVSL09 verifier and callable orbit-sign validation."""
import unittest, numpy as np
from avsl09_k1_fn_loop20 import K1Checker,fn_exact_reference,exact_frozen_a_gain,stage,evaluate_signs
from avsl_cooperative20 import Oracle,BONDS
DATA='/tmp/avsl05-20site-structural-data/sign20_training_data.npz'

class TestAVSL09(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.ns=fn_exact_reference(DATA)
  cls.arr=np.load(DATA)
 def test_local_field_is_physical_Hpsi_div_psi(self):
  k=K1Checker(DATA,threshold=0.)
  d=self.arr;ss=d['guide'].astype(float);aa=d['amp']
  psi=ss*aa
  hpsi=self.ns['Htot']@psi
  rng=np.random.default_rng(4242)
  for x in d['states'][rng.integers(len(d['states']),size=30)]:
   g,r,fresh=k.score(int(x))
   j=k.o.look[g]
   self.assertAlmostEqual(r,float(hpsi[j]/psi[j]),places=7)
 def test_all_K1_verifier_outputs_invariant_under_translation_and_spinflip(self):
  o=Oracle(DATA);k=K1Checker(DATA)
  x=int(o.states[41000]);g,r,fresh=k.score(x)
  self.assertEqual(g,o.key(x))
  for y in o.members[g]:
   h,rv,fresh2=k.score(y)
   self.assertEqual(g,h);self.assertEqual(rv,r)
 def test_full_sign_reprojection_energy_differential(self):
  ns=self.ns; a=self.arr['amp'].copy();s=self.arr['guide'].astype(float).copy()
  from scipy.sparse.linalg import LinearOperator
  rng=np.random.default_rng(911)
  o=Oracle(DATA)
  flip=set(o.key(int(x)) for x in self.arr['states'][rng.choice(len(o.states),size=3,replace=False)])
  sr=np.array([-s[i] if o.key(int(x)) in flip else s[i] for i,x in enumerate(o.states)])
  d=exact_frozen_a_gain(ns,a,s,sr)
  eold=float((s*a)@(ns['Htot']@(s*a)))/(a@a)
  enew=float((sr*a)@(ns['Htot']@(sr*a)))/(a@a)
  self.assertAlmostEqual(d,eold-enew,places=10)
 def test_comparator_proposals_do_not_consume_oracle_energy(self):
  policy=np.zeros(9)
  roots=self.arr['states'][100:400]
  for method in ('walker','random1','geometry1','learned1','random2','learned2'):
   flips,row=stage(DATA,roots,method,policy,82031,32,0.)
   self.assertEqual(row['k1_cost']['n_attempted'],32)
   self.assertEqual(row['k1_cost']['n_distinct_k1_verifications'],
                    row['k1_cost']['n_diagonal_calls'])
   self.assertEqual(len(flips),row['n_selected_orbits'])
   self.assertTrue(all(int(x)>=0 for x in flips))
 def test_ground_FN_baseline_reprojection(self):
  en,phi,info=self.ns['fn_step'](self.arr['amp'],self.arr['guide'].astype(float),
                              'AVSL09 baseline crosscheck',tol=1e-8)
  self.assertAlmostEqual(en,-10.417697678503282,places=6)
  self.assertLess(info['resid'],1e-5)

if __name__=='__main__':unittest.main()
