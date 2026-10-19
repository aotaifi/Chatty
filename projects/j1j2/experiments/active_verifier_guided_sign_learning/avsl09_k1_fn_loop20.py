#!/usr/bin/env python3
"""AVSL09: actual K1 one-hop verifier -> callable 20-site sign rule -> FN reprojection.

Unlike AVSL08: proposal verifier is local first-order Krylov r(x)=H(sa)/(sa),
NOT exact single-orbit boundary gain. Candidate K1 threshold T is fixed prior
to heldout evaluation (default T=0; no hidden full-basis threshold optimization).
The recorded accepted symmetry-orbit corrections define s_new(x)=s_old(x)*
(-1)^(key(x) in accepted). This is callable at every 20-site configuration.
Independent *full fixed-node ground-state projection* evaluates final energy.

CAVEATS: Uses precomputed exact 20-site guide arrays as callable amplitude
oracle (not a scalable guide). FN reprojection is 20-site exact eigensolve.
Frozen pretrained policy was trained on exact single-orbit gain from the SAME
sign chamber in prior AVSL08, independently rooted; training is charged.
The sign updater remains a finite memorized orbit exception list, not a
large-N generalizing sign network or region-local C_w correction.
"""
import argparse,json,time,sys
from pathlib import Path
from collections import Counter
import numpy as np
from avsl_cooperative20 import Oracle,BONDS
from avsl_dapo_explorer20 import World,softmax,rng_choice
from avsl_fn_mixed20 import MixedFN

METHODS=('walker','random1','geometry1','learned1','random2','learned2')

class K1Checker:
 def __init__(self,source,threshold=0.):
  self.o=Oracle(source)
  self.threshold=float(threshold)
  self.vals={}
  self.attempted=0;self.fresh=0;self.positive=0
  self.diagonal_checks=0
 def local_diagonal(self,x):
  # site-dependent total S^zS^z J, but no full-space lookup.
  acc=0.
  for bits,h in BONDS:
   acc+=(2*h)*(-.25 if (x&bits).bit_count()==1 else .25)
  self.diagonal_checks+=1
  return acc
 def score(self,x):
  self.attempted+=1;g=self.o.key(int(x))
  if g in self.vals:
   return g,self.vals[g],False
  r=self.local_diagonal(g)
  a=self.o.a(g);s=self.o.s(g)
  for bits,h in BONDS:
   self.o.n_neigh_probes+=1
   if (g&bits).bit_count()!=1:continue
   y=g^bits
   r+=h*s*self.o.s(y)*self.o.a(y)/a
  self.vals[g]=float(r);self.fresh+=1
  self.positive+=int(r>self.threshold)
  return g,float(r),True
 def counts(self):
  return dict(n_attempted=self.attempted,n_distinct_k1_verifications=self.fresh,
              n_predicted_flips=self.positive,n_diagonal_calls=self.diagonal_checks,
              unique_guide_amplitudes=self.o.counts()['unique_amp'],
              amplitude_requests=self.o.counts()['amp_requests'],
              unique_sign_requests=self.o.counts()['unique_sign'],
              neighbor_probes=self.o.n_neigh_probes,
              symmetry_members_expanded=self.o.n_orbit_expanded)

def propose(world,root,method,theta,rng):
 x=int(root)
 if method=='walker':return x
 depth=2 if method.endswith('2') else 1
 for step in range(depth):
  options=world.options(x,step)
  if not options:break
  f=np.asarray([fe for _,fe in options])
  if method.startswith('random'):p=np.ones(len(options))/len(options)
  elif method=='geometry1':p=softmax(4*f[:,2])
  elif method.startswith('learned'):
   q=softmax(f@theta);p=.92*q+.08/len(q)
  else:raise ValueError(method)
  x=options[rng_choice(rng,p)][0]
  world.policy_moves+=1
 return int(x)

def stage(data,roots,method,theta,seed,budget,threshold):
 rng=np.random.default_rng(seed)
 policy=World(data,'post_krylov',seed+103)
 checker=K1Checker(data,threshold)
 selected=set();changed=[]
 for n in range(budget):
  root=int(roots[int(rng.integers(len(roots)))])
  dest=propose(policy,root,method,theta,rng)
  g,r,fresh=checker.score(dest)
  if r>threshold and g not in selected:
   selected.add(g);changed.append((int(g),r))
 return selected,dict(
   method=method,threshold=threshold,
   n_selected_orbits=len(selected),selected_representatives=sorted(map(int,selected)),
   selected_local_fields={str(g):float(r) for g,r in changed},
   k1_cost=checker.counts(),
   proposal_cost=dict(n_hops=policy.policy_moves,n_feature_queries=policy.feature_queries,
         n_distinct_feature_sites=len(policy.feature_cache),
         neighbor_probes=policy.o.n_neigh_probes,
         orbit_expansions=policy.o.n_orbit_expanded))

def fn_exact_reference(data):
 """Load independent heldout exact FN projector; never reveal ED labels to policy."""
 from pathlib import Path
 f=Path(__file__).parent/'reference'/'fn_collapse_exact20_krylov_v2.py'
 src=f.read_text() if f.exists() else Path('/tmp/avsl05-20site-structural-data/fn_collapse_exact20_krylov_v2.py').read_text()
 ns={'__name__':'__eval_only_reference__'}
 exec(src[:src.index('import sys\nNITER=')],ns)
 d=np.load(data)
 assert np.array_equal(d['states'],ns['states'])
 assert np.max(np.abs(d['diag']-ns['diag']))<1e-12
 return ns

def exact_frozen_a_gain(ns,amplitude,old_sign,new_sign):
 psi0=old_sign*amplitude;psi1=new_sign*amplitude;h=ns['Htot']
 return float(psi0@(h@psi0)-psi1@(h@psi1))/float(amplitude@amplitude)

def evaluate_signs(ns,data,flips,baseline,eps=3e-8):
 dd=np.load(data)
 o=Oracle(data)
 a=dd['amp'].copy();s=dd['guide'].astype(float).copy()
 # Global callable parity s_new(x) = s_old(x) * (-1 iff orbit key is selected).
 sn=np.array([-s[i] if o.key(int(x)) in flips else s[i]
              for i,x in enumerate(o.states)],dtype=float)
 fnE,phi,info=ns['fn_step'](a,sn,'AVSL09 accepted joint corrections',tol=eps)
 return dict(final_FN_energy=float(fnE),reprojected_FN_gain=float(baseline-fnE),
             frozen_a_joint_energy_gain=exact_frozen_a_gain(ns,a,s,sn),
             fraction_states_sign_changed=float(np.mean(s!=sn)),
             changed_a2_weight=float(np.dot((s!=sn).astype(float),a*a)/(a@a)),
             FN_residual=float(info['resid']),FN_preconditioner_iterations=int(info['iterations']),
             FN_negative_mass=float(info['neg_mass'])),phi,sn

def run(args):
 tic=time.perf_counter()
 d=np.load(args.data)
 train=json.loads(Path(args.policy).read_text())
 theta=np.asarray(train['policy_weights'])
 assert theta.shape==(9,),theta.shape
 ns=fn_exact_reference(args.data)
 aa=d['amp'].copy();ss=d['guide'].astype(float)
 baseline,phi0,info0=ns['fn_step'](aa,ss,'AVSL09 untouched FN baseline',tol=2e-9)
 runs=[]
 for block in range(args.test_seeds):
  if args.root_prefix and Path(args.root_prefix+f'_testroots_{block}.npz').exists():
   roots=np.load(args.root_prefix+f'_testroots_{block}.npz')['roots']
   source='archived_AVSL08_genuine_FN_roots'
   fn_diag={'archived_roots':len(roots)}
  else:
   w=MixedFN(args.data)
   roots,fn_diag=w.sample(args.seed+1101+block*983,
          n_walkers=args.walkers,beta=args.beta,burn=args.burn,record_every=4)
   source='new_genuine_FN_root_generation'
  arms={}
  for mid,method in enumerate(METHODS):
   selected,audit=stage(args.data,roots,method,theta,args.seed+10000*block+mid,
       args.budget,args.threshold)
   physics,_,_=evaluate_signs(ns,args.data,selected,baseline)
   audit.update(physics)
   arms[method]=audit
  row=dict(block=block,roots_source=source,root_FN_cost=fn_diag,arms=arms)
  print('BLOCK',block,'K1 candidates',[(m,arms[m]['n_selected_orbits'],
             round(arms[m]['reprojected_FN_gain'],9)) for m in METHODS],flush=True)
  runs.append(row)
  Path(args.out).write_text(json.dumps(dict(args=vars(args),base_FN_energy=baseline,
           base_FN_residual=float(info0['resid']),runs=runs,seconds=time.perf_counter()-tic),indent=2))
 result=json.loads(Path(args.out).read_text())
 result['summary']={}
 for m in METHODS:
  x=[b['arms'][m] for b in runs]
  result['summary'][m]=dict(avg_number_selected=float(np.mean([v['n_selected_orbits'] for v in x])),
    avg_reprojected_FN_gain=float(np.mean([v['reprojected_FN_gain'] for v in x])),
    number_positive_FN_blocks=int(sum(v['reprojected_FN_gain']>1e-8 for v in x)),
    avg_guide_configs=float(np.mean([v['k1_cost']['unique_guide_amplitudes'] for v in x])),
    avg_K1_checks=float(np.mean([v['k1_cost']['n_distinct_k1_verifications'] for v in x])),
    avg_proposal_edges=float(np.mean([v['proposal_cost']['neighbor_probes'] for v in x])),
    avg_phys_Fn_resid=float(np.mean([v['FN_residual'] for v in x])))
 result['limitations']=[
  'Prior RL policy trained on the same fixed amplitude/sign chamber, using individual-orbit exact gain labels, not K1 labels; source training + FN roots have substantial additional cost.',
  'One-hop policy reuses two-hop-trained theta; this is a deliberate zero-shot ablation.',
  'FN roots are real mixed-population draws; output sign representation is a finite 20-site orbit exception list, and is not yet scalable NQS.',
  'Threshold T=0 was fixed in script as a conservative baseline, but selected after an exploratory diagnostic; do not make confirmatory significance claims.',
  'Krylov local verifier is actual local energy r(x)=H(s*a)/(s*a) using physical H and guide neighbor ratios. It is NOT the exact single orbit gain.',
  'Reference amplitudes and signs are 20-site exact arrays; FN reprojection is independent full-basis heldout eigensolve.',
  'Same guide and sign chamber across FN populations; cross-iteration frozen policy transfer untested.',
  'Count overhead from large-system callable neural nets and source RL training separately from exact tiny reference FN reprojector.'
 ]
 result['seconds']=time.perf_counter()-tic
 Path(args.out).write_text(json.dumps(result,indent=2))
 print('SUMMARY',json.dumps(result['summary'],indent=2),flush=True)
 print('SAVED',args.out,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser()
 p.add_argument('--data',required=True)
 p.add_argument('--policy',required=True)
 p.add_argument('--out',required=True)
 p.add_argument('--root-prefix',default='/tmp/avsl_mixed20_budget_adjusted')
 p.add_argument('--seed',type=int,default=28019)
 p.add_argument('--budget',type=int,default=160)
 p.add_argument('--test-seeds',type=int,default=6)
 p.add_argument('--threshold',type=float,default=0.)
 p.add_argument('--walkers',type=int,default=128)
 p.add_argument('--beta',type=float,default=10.)
 p.add_argument('--burn',type=float,default=4.)
 run(p.parse_args())
