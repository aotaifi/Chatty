#!/usr/bin/env python3
"""AVSL10 cross-chamber replication of costed compact ranker:
  old post-Krylov FN state -> verified stage-1 signs -> independent FN -> new
  real mixed FN walkers -> current K1 checks -> joint sign -> FN reprojection.

Source ranker is FROZEN after training with K1 fields on old sign chamber
only. Crucial guide-only control calculated for each new sign chamber.
All pool candidates from mixed FN, candidate scoring and K1 checks charged.
Zero second-stage learning/updates. Finite 20-site state arrays remain oracles.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from avsl10_costed_ranker20 import pool,score_candidates
from avsl10_compact_local_rule20 import feature_matrix
from avsl09_k1_fn_loop20 import K1Checker,fn_exact_reference,evaluate_signs
from avsl_fn_mixed20 import MixedFN
from avsl_cooperative20 import Oracle

ARMS=['random','amplitude_rare','local_motif_rank','local_motif_plus_guide_rank']

def run(args):
 sources=json.load(open(args.two))
 assert all(b['first_block'] in args.blocks for b in sources['runs'])
 rules={}
 for name,path in [('local_motif_rank',args.local),('local_motif_plus_guide_rank',args.guide)]:
  p=json.load(open(path))
  assert p['args']['train_blocks']==[0,1,2]
  rules[name]={k:np.asarray(v,float) if k in ('coefs','mean','std') else v for k,v in p['learned_rule'].items()}
 ns=fn_exact_reference(args.source_data)
 report=[]
 for src in sources['runs']:
  b=src['first_block']
  if b not in args.blocks:continue
  z=src['arms'][args.source_arm]
  path=z['newdata_path']
  if not Path(path).is_file():raise FileNotFoundError(path)
  inp=np.load(path)
  base,phi,info=ns['fn_step'](inp['amp'],inp['guide'].astype(float),'AVSL10 changed no-sign FN control',tol=2e-9)
  sim=MixedFN(path,'post_krylov')
  roots,smpl=sim.sample(args.seed+991*b,n_walkers=args.walkers,beta=args.beta,burn=args.burn,record_every=4)
  pool_oracle=Oracle(path)
  ids,pcost=pool(roots,pool_oracle,args.seed+272*b,args.pool_attempts)
  zres={}
  for j,arm in enumerate(ARMS):
   k1=K1Checker(path,0.)
   rng=np.random.default_rng(args.seed+1219*b+91*j)
   if arm=='random':
    order=rng.permutation(len(ids))
   elif arm=='amplitude_rare':
    zz=np.asarray([k1.o.a(int(x)) for x in ids])
    order=np.argsort(zz,kind='stable')
   else:
    model=rules[arm]
    gg=np.asarray([k1.o.a(int(x)) for x in ids]) if len(model['coefs'])==43 else None
    scores=(feature_matrix(ids,amplitudes=gg)-model['mean'])/model['std']
    scores=scores@model['coefs']+model['bias']
    order=np.argsort(-scores,kind='stable')
   flips=set();inspected=[]
   for t in order:
    if len(inspected)>=args.max_checks:break
    g,field,_=k1.score(ids[int(t)])
    inspected.append((int(g),float(field)))
    if field>0:flips.add(g)
    if k1.o.counts()['unique_amp']>=args.amp_budget:break
   phys,_,_=evaluate_signs(ns,path,flips,base)
   zres[arm]=dict(first_arm=args.source_arm,post_first_FN_energy=float(base),
    n_K1_checked=len(inspected),n_K1_positive=len(flips),
    K1_orbits=sorted(map(int,flips)),verifier_cost=k1.counts(),physical=phys)
   print('CHAMBER',b,arm,'K1',len(inspected),'pos',len(flips),
      'delta FN',round(phys['reprojected_FN_gain'],9),flush=True)
  report.append(dict(block=b,first_stage=z['first_fn'],source_fn_reprojection_residual=float(info['resid']),
                     sampler=smpl,pool=pcost,second_arms=zres))
  Path(args.out).write_text(json.dumps(dict(args=vars(args),chambers=report),indent=2))
 final=json.load(open(args.out))
 final['summary']={name:dict(
    mean_second_sign_only_FN_gain=float(np.mean([row['second_arms'][name]['physical']['reprojected_FN_gain'] for row in report])),
    n_positive_second_sign_gain=int(sum(row['second_arms'][name]['physical']['reprojected_FN_gain']>1e-8 for row in report)),
    mean_amp_queries=float(np.mean([row['second_arms'][name]['verifier_cost']['unique_guide_amplitudes'] for row in report])),
    mean_K1_checks=float(np.mean([row['second_arms'][name]['n_K1_checked'] for row in report]))) for name in ARMS}
 final['limitations']=[
  'Only three changed 20-site sign chambers, generated from prior geometry1 stage1 FN runs; no 6x6 claims.',
  'Source learner trained on old chamber from FN root populations 0,1,2; test sign chambers were changed first by heldout 0,2,5. Block 0/2 have overlapping ROOT population seed indices but none of the changed sign chambers appears in training.',
  'No policy retraining; full training and initial FN sampling cost must be charged for amortization.',
  'Per-arm 4000 unique guide amplitude cap does not equalize symmetry expansions, FN sampling or training expenses.',
  'Guide is exact 20-site phi FN from previous projection and all physical reprojectors are full-space oracle; scalable NQS unavailable.',
  'Finite orbit sign list remains accepted composite correction; compact NN only RANKS Krylov queries.']
 Path(args.out).write_text(json.dumps(final,indent=2))
 print('SUMMARY',json.dumps(final['summary'],indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser()
 p.add_argument('--source-data',required=True)
 p.add_argument('--two',required=True)
 p.add_argument('--local',required=True);p.add_argument('--guide',required=True)
 p.add_argument('--out',required=True)
 p.add_argument('--source-arm',default='geometry1')
 p.add_argument('--blocks',nargs='+',type=int,default=[0,2,5])
 p.add_argument('--seed',type=int,default=28340)
 p.add_argument('--walkers',type=int,default=128)
 p.add_argument('--beta',type=float,default=6.);p.add_argument('--burn',type=float,default=2.)
 p.add_argument('--pool-attempts',type=int,default=3000)
 p.add_argument('--amp-budget',type=int,default=4000)
 p.add_argument('--max-checks',type=int,default=160)
 run(p.parse_args())
