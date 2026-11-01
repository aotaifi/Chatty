#!/usr/bin/env python3
"""AVSL10: learn *compact symmetry-invariant local motif correction rule* from
actual scarce K1 verifications, not from ED signs nor global oracle labels.

Feature vector: histogram of 2x2 (8 complement-quotiented motifs) and 2x3
(32 complement-quotiented motifs) spin windows + total NN/NNN antiparallel
bond fractions (42 features), pooled over translations.
Generic tilted rectangular torus `site` uses Lx,Ly,tilt.
Current s_k remains caller-supplied; output eta_theta(x)=sign(score(x)-cut).
No exact-amplitude or symmetry-orbit lookup at inference.

Training K1 examples from REAL FN walker roots, one/two legal Hamiltonian
hops and costs explicitly recorded. Objective logistic BCE with L2 only;
no physics oracle outside queried K1 labels. Cut selected from TRAIN labels.
Heldout FN physics checks every configuration using exact 20site E_FN (evaluation only).
"""
import argparse,json,hashlib,time
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from avsl_cooperative20 import Oracle,BONDS
from avsl09_k1_fn_loop20 import K1Checker,propose,fn_exact_reference,exact_frozen_a_gain
from avsl_dapo_explorer20 import World

def site_xy(x,y,lx=4,ly=5,tilt=1):
 q,r=divmod(int(y),ly)
 return ((int(x)-tilt*q)%lx)+lx*r

def windows(lx=4,ly=5,tilt=1,width=2,height=3):
 return np.asarray([[site_xy(x+dx,y+dy,lx,ly,tilt)
    for dy in range(height) for dx in range(width)]
    for y in range(ly) for x in range(lx)],dtype=np.int32)

def feature_matrix(states,lx=4,ly=5,tilt=1,amplitudes=None):
 """O(number of sites * windows) fast vectorized convolution + global pooling."""
 states=np.asarray(states,dtype=np.uint64)
 n=len(states); out=np.zeros((n,42),dtype=np.float32)
 for size,dim,start in [(2,2,0),(2,3,8)]:
  ww=windows(lx,ly,tilt,size,dim)
  for row in ww:
   c=np.zeros(n,np.uint32)
   for j,bit in enumerate(row):c|=(((states>>np.uint64(bit))&1).astype(np.uint32)<<j)
   mask=(1<<(size*dim))-1
   code=np.minimum(c,c^mask)
   np.add.at(out,(np.arange(n),start+code),1.0)
 out[:,:40]/=(lx*ly)
 # nn and nnn density, spin complement and translation invariant
 count1=np.zeros(n,np.float32);count2=np.zeros(n,np.float32)
 for y in range(ly):
  for x in range(lx):
   a=site_xy(x,y,lx,ly,tilt)
   for dx,dy,which in [(1,0,1),(0,1,1),(1,1,2),(-1,1,2)]:
    b=site_xy(x+dx,y+dy,lx,ly,tilt)
    v=(((states>>np.uint64(a))^(states>>np.uint64(b)))&1).astype(np.float32)
    (count1 if which==1 else count2)[:]+=v
 out[:,40]=count1/(2*lx*ly)
 out[:,41]=count2/(2*lx*ly)
 if amplitudes is not None:
  aa=np.asarray(amplitudes,dtype=float)
  assert aa.shape==(n,)
  out=np.column_stack((out,np.log(np.maximum(aa,1e-220))/(lx*ly)))
 return out

def collect(data,roots_prefix,seed,train_blocks,cap=160,methods=('random1','geometry1','random2','learned2'),theta=None):
 found={};costs=[]
 for b in train_blocks:
  fn=np.load(f'{roots_prefix}_testroots_{b}.npz')['roots']
  for mi,method in enumerate(methods):
   world=World(data,'post_krylov',seed+b*1049+mi*173)
   checker=K1Checker(data,0)
   rng=np.random.default_rng(seed+b*3799+mi*773)
   for t in range(cap):
    xx=int(fn[int(rng.integers(len(fn)))])
    dest=propose(world,xx,method,theta,rng)
    g,r,fresh=checker.score(dest)
    if g in found:
     if abs(found[g]-r)>1e-6:raise AssertionError('bad symmetry-consistent K1')
    else:found[g]=r
   costs.append(dict(block=b,method=method,**checker.counts(),
       proposal_hops=world.policy_moves,
       proposal_bond_probes=world.o.n_neigh_probes,
       proposal_feature_queries=world.feature_queries))
 keys=np.asarray(sorted(found),dtype=np.uint32)
 target=np.asarray([found[g]>0 for g in keys],dtype=np.int8)
 return keys,target,found,costs

def train_rule(keys,labels,reg=1.,amplitudes=None):
 x=feature_matrix(keys,amplitudes=amplitudes).astype(float)
 avg=x.mean(axis=0);std=x.std(axis=0)
 std=np.maximum(std,.04)
 z=(x-avg)/std
 yy=labels.astype(float)
 n1=int(yy.sum());n0=len(yy)-n1
 if n1<3:raise ValueError(f'Only {n1} positive source labels; cannot train a compact corrector')
 wts=np.where(yy==1,.5/n1,.5/n0)
 def obj(w):
  t=z@w[:-1]+w[-1]
  v=np.logaddexp(0,t)-yy*t
  loss=float(wts@v+reg*(w[:-1]@w[:-1])/2)
  gr=np.r_[z.T@(wts*(expit(t)-yy))+reg*w[:-1],float(wts@(expit(t)-yy))]
  return loss,gr
 w0=np.zeros(z.shape[1]+1)
 r=minimize(obj,w0,jac=True,method='L-BFGS-B',options=dict(maxiter=250,ftol=1e-12))
 sc=z@r.x[:-1]+r.x[-1]
 # Declare cut strictly from training labels. Restrict false positives:
 neg=sc[yy==0];pos=sc[yy==1]
 cut=max(float(np.quantile(neg,.995)),float(np.quantile(pos,.25)))
 predicted=sc>cut
 prec=float(np.sum(predicted & (yy==1))/max(1,np.sum(predicted)))
 return dict(coefs=r.x[:-1],bias=float(r.x[-1]),mean=avg,std=std,threshold=cut,
   n_train=len(keys),n_positive=n1,n_negative=n0,
   train_precision=prec,train_recall=float(np.sum(predicted & (yy==1))/n1),
   train_predicted=int(predicted.sum()),objective=float(r.fun),optimizer_ok=bool(r.success))

def apply_rule_to_all(data,rule,ns,baseline,threshold_override=None):
 inp=np.load(data);aa=inp['amp'];states=inp['states'];orig=inp['guide'].astype(float)
 xx=feature_matrix(states,amplitudes=aa if len(rule['coefs'])==43 else None).astype(float)
 z=(xx-rule['mean'])/rule['std']
 scores=z@rule['coefs']+rule['bias']
 cut=rule['threshold'] if threshold_override is None else threshold_override
 flip=scores>cut
 sn=np.where(flip,-orig,orig)
 fnE,phi,info=ns['fn_step'](aa,sn,'AVSL10 fully callable local motif sign update',tol=2e-9)
 o=Oracle(data)
 orbits={o.key(int(x)) for x in states[flip]}
 return dict(fn_energy=float(fnE),post_FN_energy_gain=float(baseline-fnE),
        fixed_a_joint_gain=exact_frozen_a_gain(ns,aa,orig,sn),
        n_flip_states=int(flip.sum()),n_flip_symmetry_orbits=len(orbits),
        flipped_a2_weight=float(np.dot(flip.astype(float),aa*aa)/(aa@aa)),
        fn_residual=float(info['resid'])),flip

def run(args):
 st=time.monotonic()
 policy=json.load(open(args.policy))
 theta=np.asarray(policy['policy_weights'])
 if args.train_archive:
  prev=json.load(open(args.train_archive))
  keys=np.asarray(prev['train_keys'],dtype=np.uint32)
  yy=np.asarray(prev['train_labels'],dtype=np.int8)
  fields={int(k):float(v) for k,v in prev['train_fields'].items()}
  costs=prev['training_costs']
 else:
  keys,yy,fields,costs=collect(args.data,args.root_prefix,args.seed,
                               args.train_blocks,cap=args.training_checks,
                               theta=theta)
 print('TRAIN',len(keys),'positive',int(yy.sum()),'fraction',float(yy.mean()),flush=True)
 amplitudes=None
 if args.use_guide:
  training_oracle=Oracle(args.data)
  amplitudes=np.asarray([training_oracle.a(int(k)) for k in keys])
 rule=train_rule(keys,yy,reg=args.reg,amplitudes=amplitudes)
 print('TRAIN RULE',rule['train_precision'],rule['train_recall'],
       'positive predicted',rule['train_predicted'],flush=True)
 ns=fn_exact_reference(args.data)
 d=np.load(args.data)
 base,_,info=ns['fn_step'](d['amp'],d['guide'].astype(float),'AVSL10 baseline FN',tol=2e-9)
 score,pred=apply_rule_to_all(args.data,rule,ns,base)
 print('MODEL PHYSICS',score,flush=True)
 checker=K1Checker(args.data,0.)
 # Assess oracle labels strictly AFTER training / threshold freeze; posthoc only.
 # Auditing every predicted flip can be extremely expensive; use exhaustive
 # 20-site local fields previously precomputed to avoid expensive repeat.
 ref=json.load(open(args.k1_reference))
 keyset=Oracle(args.data)
 all_orbits={}
 for i,x in enumerate(d['states']):
  g=keyset.key(int(x))
  if g not in all_orbits:all_orbits[g]=bool(pred[i])
  elif all_orbits[g]!=bool(pred[i]):raise AssertionError('motif features broke symmetry')
 labeled=set(map(int,keys))
 pred_keys={g for g,v in all_orbits.items() if v}
 seenpositive=set(map(int,keys[yy==1]))
 unseen_pred=pred_keys-labeled
 k1field={int(g):float(v) for g,v in ref['rkeys'].items()}
 tp=sum(k1field[g]>0 for g in unseen_pred)
 neg=sum(k1field[g]<=0 for g in unseen_pred)
 audit=dict(n_all_orbits=len(all_orbits),n_predicted_orbits=len(pred_keys),
   n_predicted_unseen_training_orbits=len(unseen_pred),
   K1_pos_predicted_unseen=int(tp),K1_neg_predicted_unseen=int(neg),
   K1_unseen_precision=float(tp/max(1,tp+neg)),
   K1_unseen_recall=float(tp/max(1,sum(g not in labeled and v>0 for g,v in k1field.items()))),
   n_positive_train_orbits=len(seenpositive))
 obj=dict(label='AVSL10 compact local motif sign-rule distillation from on-policy K1',
   args=vars(args),training_costs=costs,train_keys=keys.tolist(),
   train_labels=yy.tolist(),train_fields={str(g):float(v) for g,v in fields.items()},
   learned_rule={k:v.tolist() if hasattr(v,'tolist') else v for k,v in rule.items()},
   physics=score,posthoc_holdout=audit,
   initial_FN_energy=float(base),time_s=time.monotonic()-st,
   limitations=['Only local spin motif histogram features; no amplitude oracle or symmetry enumeration at inference.',
     'Labels from same current sign chamber; not a cross-iteration generalization demonstration.',
     'Full-basis K1 labels used only after freezing rule to audit unseen orbits.',
     'All configurations enumerated for the 20-site validation, but the rule is per-state callable O(N) for larger sizes.',
     'K1 label not equivalent to isolated orbit benefit or globally optimal sign update; composite flips can increase physical energy.',
     'Training source K1 checks arise from real FN root blocks; FN sampling cost and existing RL policy pretraining must also be charged.',
     'Basing architecture/hyperparams on exploratory results prevents confirmatory inference.'])
 Path(args.out).write_text(json.dumps(obj,indent=2))
 print('HELDOUT AUDIT',audit,'SAVED',args.out,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser()
 p.add_argument('--data',required=True);p.add_argument('--policy',required=True);p.add_argument('--k1-reference',default='/tmp/avsl09_k1_diag.json')
 p.add_argument('--out',required=True)
 p.add_argument('--root-prefix',default='/tmp/avsl_mixed20_budget_adjusted')
 p.add_argument('--train-blocks',type=int,nargs='+',default=[0,1,2])
 p.add_argument('--training-checks',type=int,default=220)
 p.add_argument('--reg',type=float,default=.1)
 p.add_argument('--use-guide',action='store_true')
 p.add_argument('--train-archive',default=None)
 p.add_argument('--seed',type=int,default=28190)
 run(p.parse_args())
