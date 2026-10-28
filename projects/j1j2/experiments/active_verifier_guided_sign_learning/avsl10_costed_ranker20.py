#!/usr/bin/env python3
"""AVSL10B: costs of compact sign-correction model as *K1-check ALLOCATOR*.
Train labels confined to FN blocks 0,1,2. Heldout genuine FN root populations
3,4,5 only. Compare fixed-size identical walker-adjacent candidate pool.

Physical amplitudes used by guide-score model counted: each scoring amp(x) must
be obtained by on-demand query, plus each K1 verifier's neighbor amplitudes.
Geometry-only score has zero amplitude overhead; random likewise.
Charged verifications pass K1 local field r>0 and apply *only verified* orbit
changes. Outputs coherent sign field + independently reprojected FN energy.
Training 1214 source K1 labels and earlier RL policy costs remain additional.
"""
import json,argparse,time
from pathlib import Path
import numpy as np
from avsl10_compact_local_rule20 import feature_matrix
from avsl_cooperative20 import Oracle,BONDS
from avsl09_k1_fn_loop20 import K1Checker,fn_exact_reference,evaluate_signs

METHODS=['random','geometry_rank','amplitude_rare','amplitude_common','local_motif_rank','local_motif_plus_guide_rank']

def score_candidates(gids,oracle,rule,amp_mode):
 xx=np.asarray(gids,dtype=np.uint32)
 amps=np.asarray([oracle.a(int(g)) for g in gids]) if amp_mode else None
 feat=feature_matrix(xx,amplitudes=amps)
 return ((feat-rule['mean'])/rule['std'])@rule['coefs']+rule['bias']

def pool(roots,oracle,seed,B=2000):
 rng=np.random.default_rng(seed)
 opts=[]
 # Enumerate valid Heisenberg moves, but do NOT access amplitudes / labels.
 for trial in range(B):
  x=int(roots[int(rng.integers(len(roots)))])
  bits,h=BONDS[int(rng.integers(len(BONDS)))]
  if (x&bits).bit_count()!=1:continue
  y=x^bits
  opts.append(oracle.key(y))
 # identical deduplicated pool across rankers
 return sorted(set(opts)),dict(n_raw_attempts=B,n_valid_hops=len(opts),n_distinct_orbits=len(set(opts)),
                               symmetry_member_expansion=oracle.n_orbit_expanded,
                               bond_probes=B)

def run(args):
 st=time.monotonic()
 d=np.load(args.data)
 rulefiles=dict(local_motif_rank=args.local,local_motif_plus_guide_rank=args.guide)
 rules={}
 for name,path in rulefiles.items():
  z=json.load(open(path))
  assert z['args']['train_blocks']==[0,1,2],z['args']['train_blocks']
  rules[name]={k:(np.asarray(v,dtype=float) if k in ('coefs','mean','std') else v) for k,v in z['learned_rule'].items()}
  assert len(rules[name]['coefs'])==(43 if 'guide' in name else 42)
 ns=fn_exact_reference(args.data)
 originalE,_,_=ns['fn_step'](d['amp'],d['guide'].astype(float),'AVSL10 costed FN original',tol=2e-9)
 blocks=[]
 for b in args.test_blocks:
  roots=np.load(args.root_prefix+f'_testroots_{b}.npz')['roots']
  src=Oracle(args.data)
  gids,pcost=pool(roots,src,args.seed+b*991,args.pool_attempts)
  print('POOL',b,pcost,flush=True)
  rs={}
  for mid,method in enumerate(METHODS):
   checker=K1Checker(args.data,0.)
   scoring_oracle=checker.o
   rng=np.random.default_rng(args.seed+1000*b+mid*771)
   if method=='random':
    ranking=np.arange(len(gids));rng.shuffle(ranking)
   elif method=='geometry_rank':
    features=feature_matrix(gids)
    ranking=np.argsort(-features[:,41],kind='stable')
   elif method in ('amplitude_rare','amplitude_common'):
    amps=np.asarray([scoring_oracle.a(int(g)) for g in gids])
    ranking=np.argsort(amps if method=='amplitude_rare' else -amps,kind='stable')
   else:
    m=rules[method]
    scores=score_candidates(gids,scoring_oracle,m,amp_mode=(method=='local_motif_plus_guide_rank'))
    ranking=np.argsort(-scores,kind='stable')
   accepted=set();inspected=[]
   for idx in ranking:
    if len(inspected)>=args.max_checks:break
    g=int(gids[int(idx)])
    kg,r,_=checker.score(g)
    inspected.append([kg,float(r)])
    if r>0:accepted.add(kg)
    # Stop once the method's TOTAL guide amplitude requests meet cap.
    if checker.o.counts()['unique_amp']>=args.amp_budget:break
   phys,_,_=evaluate_signs(ns,args.data,accepted,originalE)
   c=checker.counts()
   rs[method]=dict(n_pool_orbits=len(gids),n_verifier_attempts=len(inspected),
         n_K1_positive=len(accepted),verified_K1_positive_orbits=sorted(map(int,accepted)),
         verification_log=inspected,first_positive_position=next(
              (j for j,(g,r) in enumerate(inspected) if r>0),None),
         verifier_costs=c,post_FN_physics=phys)
   print('ARM',b,method,'K1',len(inspected),'flips',len(accepted),
         'FN gain',round(phys['reprojected_FN_gain'],8),
         'unique guide',c['unique_guide_amplitudes'],flush=True)
  blocks.append(dict(block=b,root_pool_cost=pcost,arms=rs))
  Path(args.out).write_text(json.dumps(dict(args=vars(args),blocks=blocks,
                                            initial_FN_energy=originalE),indent=2))
 result=json.load(open(args.out))
 result['summary']={m:dict(
    mean_FN_gain=float(np.mean([v['arms'][m]['post_FN_physics']['reprojected_FN_gain'] for v in blocks])),
    n_positive_energy_blocks=sum(v['arms'][m]['post_FN_physics']['reprojected_FN_gain']>1e-8 for v in blocks),
    mean_K1_checks=float(np.mean([v['arms'][m]['n_verifier_attempts'] for v in blocks])),
    mean_K1_positive=float(np.mean([v['arms'][m]['n_K1_positive'] for v in blocks])),
    mean_unique_guide=float(np.mean([v['arms'][m]['verifier_costs']['unique_guide_amplitudes'] for v in blocks]))) for m in METHODS}
 result['limitations']=['Training used 1214 costly K1 source labels for both learned rankers, excluded from heldout amplitude cap and not amortized.',
   'Local+guide ranker spends one guide amplitude query per scored candidate; included in its verifier cache cost.',
   'FN roots are archived real mixed projector samples and pool is identical per test block; heldout across FN root populations but same underlying sign/amplitude chamber.',
   'Candidate geometry rank is a single, cheap NNN antiparallel-bond count; amplitude_rare and amplitude_common are untrained one-amplitude-query/candidate controls.',
   'K1 threshold T=0 is exploratory; compact models serve as rankers ONLY, sign correction is accepted only after actual K1 verification.',
   'Finite orbit exception sign list after verification still not a scalable output representation.',
   'Symmetry dedup in 20site pool uses explicit translation/complement enumeration, itself not scalable in this form.',
   'Fully independent FN reproject performed after all K1-accepted joint changes, but source-training and FN root-generation costs are not covered by guide cap.']
 result['seconds']=time.monotonic()-st
 Path(args.out).write_text(json.dumps(result,indent=2))
 print('SUMMARY',json.dumps(result['summary'],indent=2),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser()
 p.add_argument('--data',required=True)
 p.add_argument('--local',required=True);p.add_argument('--guide',required=True)
 p.add_argument('--out',required=True)
 p.add_argument('--root-prefix',default='/tmp/avsl_mixed20_budget_adjusted')
 p.add_argument('--test-blocks',type=int,nargs='+',default=[3,4,5])
 p.add_argument('--pool-attempts',type=int,default=3000)
 p.add_argument('--amp-budget',type=int,default=4000)
 p.add_argument('--max-checks',type=int,default=160)
 p.add_argument('--seed',type=int,default=101923)
 run(p.parse_args())

