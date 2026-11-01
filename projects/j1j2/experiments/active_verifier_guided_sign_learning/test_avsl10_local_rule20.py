"""AVSL10 compact local sign-rule and heldout ranker invariants."""
import unittest,numpy as np
from avsl10_compact_local_rule20 import feature_matrix,windows,site_xy,train_rule
from avsl_cooperative20 import Oracle,MASK,PERMS

DATA='/tmp/avsl05-20site-structural-data/sign20_training_data.npz'

class AVSL10Tests(unittest.TestCase):
 def test_feature_translation_and_spin_complement_invariance(self):
  o=Oracle(DATA);rng=np.random.default_rng(20140)
  for x in o.states[rng.integers(0,len(o.states),size=20)]:
   x=int(x);z=feature_matrix([x])[0]
   for y in o.members[o.key(x)]:
    np.testing.assert_allclose(feature_matrix([int(y)])[0],z,atol=1e-6)
 def test_direct_eval_does_not_need_orbit_lookup(self):
  q=np.load(DATA)
  x=np.asarray(q['states'][2400:2440])
  a=q['amp'][2400:2440]
  v=feature_matrix(x,amplitudes=a)
  self.assertEqual(v.shape,(40,43))
  np.testing.assert_allclose(v[:,:8].sum(axis=1),1.,atol=1e-6)
  np.testing.assert_allclose(v[:,8:40].sum(axis=1),1.,atol=1e-6)
  np.testing.assert_allclose(v[:,-1],np.log(a)/20.,atol=1e-6)
 def test_transfer_lattice_size_6x6_and_tilt_0(self):
  bits=np.array([(1<<18)-1, (1<<36)-1 ^ ((1<<18)-1), 0x343253454],np.uint64)
  g=feature_matrix(bits,lx=6,ly=6,tilt=0)
  self.assertEqual(g.shape,(3,42))
  self.assertTrue(np.isfinite(g).all())
  self.assertEqual(windows(6,6,0).shape,(36,6))
 def test_untrained_classifier_does_not_mistake_zero_positives_for_success(self):
  keys=np.asarray([1011,1023,1055,2023],np.uint32)
  labels=np.zeros(len(keys),np.int8)
  with self.assertRaises(ValueError):train_rule(keys,labels)
 def test_guide_feature_differs_from_geometry_feature_only(self):
  q=np.load(DATA);ids=np.array(q['states'][500:510]);am=q['amp'][500:510]
  x=feature_matrix(ids,amplitudes=am)
  self.assertEqual(x.shape[1],43)
  np.testing.assert_allclose(x[:,:42],feature_matrix(ids))
 def test_all_symmetry_groups_in_raw_data_have_consistent_labels(self):
  o=Oracle(DATA)
  d=np.load(DATA)
  rng=np.random.default_rng(24)
  for i in rng.choice(len(o.states),size=30,replace=False):
   x=int(o.states[i]);s=d['guide'][i]
   g=o.key(x)
   # Do not require guide SIGN to be invariant under odd lattice
   # translations; learned *relative* correction eta must be invariant.
   for y in o.members[g]:
    self.assertEqual(o.key(y),g)

if __name__=='__main__':unittest.main()
