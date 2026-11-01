#!/usr/bin/env python3
"""AVSL09B: frozen explorer cross-chamber, 2 genuine FN-K1-FN updates on 20 sites.

Stage 1 from archived heldout FN populations (AVSL09). Apply a callable symmetry-
invariant sign rule and exact 20-site FN projection. Stage 2 draws NEW actual
branching FN walkers from the changed sign chamber, keeps the *same frozen
policy*, runs costed Krylov checks, makes a NEW coherent composite sign field
and independently reprojects FN.

Exact 20-site phi is used as the next positive FN guide, and is not a scalable
implementation; the first stage frozen policy was pretrained on same original
post-Krylov chamber with 400 costly single-orbit labels. This is a mechanistic
cross-iteration pilot only, not amortized scalable policy evidence.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from avsl_cooperative20 import Oracle
from avsl_fn_mixed20 import MixedFN
from avsl09_k1_fn_loop20 import fn_exact_reference,stage,exact_frozen_a_gain

def apply_fn(ns,data,old_keys,add_keys,base):
 o=Oracle(data)
 q=np.load(data)
 old_s=q['guide'].astype(float)
 new_s=np.asarray([(-old_s[i] if o.key(int(x)) in add_keys else old_s[i])
                  for i,x in enumerate(o.states)],dtype=float)
 guide=q['amp'].copy()
 energy,phi,info=ns['fn_step'](guide,new_s,'AVSL09 two-step FN checker',tol=2e-9)
 return new_s,phi,dict(energy=energy,fn_residual=float(info['resid']),
                gain_from_initial_E=float(base-energy),
                gain_from_old_FN_E=None,
                fixed_guide_joint_gain=exact_frozen_a_gain(ns,guide,old_s,new_s),
                n_orbit_corrections=len(add_keys),
                fractional_states_changed=float(np.mean(old_s!=new_s)))

def run(args):
 p=json.loads(Path(args.first).read_text())
 assert p['args']['data']==args.data
 q=np.load(args.data)
 theta=np.asarray(json.load(open(args.policy))['policy_weights'])
 ns=fn_exact_reference(args.data)
 initialE=float(p['base_FN_energy'])
 run=[]
 for index,block in enumerate(args.blocks):
  row={'first_block':block,'arms':{}}
  for j,method in enumerate(args.methods):
   first=p['runs'][block]['arms'][method]
   first_keys=set(map(int,first['selected_representatives']))
   s1,a1,info1=apply_fn(ns,args.data,set(),first_keys,initialE)
   assert abs(info1['energy']-first['final_FN_energy'])<3e-7,(method,info1,first)
   newdata=Path(args.out).with_suffix('').as_posix()+f'_block{block}_{method}_intermediate.npz'
   np.savez_compressed(newdata,states=q['states'],amp=a1,
                       guide=s1.astype(np.int8),diag=q['diag'],maskA=q['maskA'])
   sim=MixedFN(newdata)
   roots,meta=sim.sample(args.seed+block*131+j*793,n_walkers=args.walkers,
         beta=args.beta,burn=args.burn,record_every=4)
   keys2,proposals=stage(newdata,roots,method,theta,args.seed+12000+block*981+j,
                        args.budget,args.threshold)
   s2,a2,info2=apply_fn(ns,newdata,first_keys,keys2,initialE)
   info2['gain_from_old_FN_E']=float(info1['energy']-info2['energy'])
   res=dict(method=method,initial_fn_energy=initialE,
            first_fn=info1,second_fn=info2,
            first_k1=first,second_k1=proposals,
            second_FN_walker_cost=meta,
            newdata_path=newdata)
   row['arms'][method]=res
   print('BLOCK',block,method,'first dE',round(initialE-info1['energy'],9),
      'second additional',round(info1['energy']-info2['energy'],9),
      'total',round(initialE-info2['energy'],9),
      'second K1 flips',len(keys2),flush=True)
  run.append(row)
  Path(args.out).write_text(json.dumps(dict(args=vars(args),runs=run,
                    limitations=__doc__),indent=2))
 print('DONE',args.out)
if __name__=='__main__':
 p=argparse.ArgumentParser()
 p.add_argument('--first',required=True);p.add_argument('--data',required=True)
 p.add_argument('--policy',required=True);p.add_argument('--out',required=True)
 p.add_argument('--blocks',type=int,nargs='+',default=[0,2])
 p.add_argument('--methods',nargs='+',default=['random1','learned2'])
 p.add_argument('--seed',type=int,default=26019)
 p.add_argument('--walkers',type=int,default=128)
 p.add_argument('--beta',type=float,default=6.)
 p.add_argument('--burn',type=float,default=2.)
 p.add_argument('--budget',type=int,default=160)
 p.add_argument('--threshold',type=float,default=0.)
 run(p.parse_args())
