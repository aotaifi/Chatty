#!/usr/bin/env python3
import numpy as np
import scipy.sparse.linalg as sla
from finite_tau_matching_20site_exact import special_columns,D
from finite_tau_amplitude_regression_20site import endpoint,H
from finite_tau_learned_guide_gfmc_20site import load_model,guide_vec,propagate_fn_uniform_importance

def norm(v): v=np.asarray(v,float); return v/np.linalg.norm(v)
def err(s,v):
 p=v*v;p/=p.sum(); return float(np.sum(p[s!=np.where(v>=0,1,-1)]))
def estimate(w,y,tau,arr,net,mu,sd,prior=512):
 d,_,_,_,_,vals,rlab,rbs=arr
 mass=np.bincount(rlab[w],minlength=len(vals)).astype(float)
 gm=guide_vec(net,mu,sd,y,tau,arr)
 pm=np.bincount(rlab,weights=gm,minlength=len(vals));pm/=pm.sum()
 return norm(np.maximum(((mass+prior*pm)/rbs)[rlab],1e-14))

y=int(special_columns(np.random.default_rng(20260930),2)[1]); M=8192;dt=.05
net,mu,sd=load_model();arr=endpoint(y);d0=arr[0]
s=np.where(d0%2==0,1,-1).astype(np.int8)
ra=np.random.default_rng(250001);rb=np.random.default_rng(260001)
wa=np.full(M,y,np.int32);wb=wa.copy();ga=np.full(D,1/np.sqrt(D));gb=ga.copy()
exact=np.zeros(D);exact[y]=1.

for n in range(1,11):
 wa=propagate_fn_uniform_importance(wa,ga,s,dt,ra)
 wb=propagate_fn_uniform_importance(wb,gb,s,dt,rb)
 exact=norm(sla.expm_multiply(-dt*H,exact)); tau=n*dt
 ga=estimate(wa,y,tau,arr,net,mu,sd);gb=estimate(wb,y,tau,arr,net,mu,sd)
 fa=s*ga-dt*(H@(s*ga));fb=s*gb-dt*(H@(s*gb))
 sa=np.where(fa>=0,1,-1);sb=np.where(fb>=0,1,-1)
 flipA=sa!=s;flipB=sb!=s;accept=flipA&flipB&(sa==sb)
 sn=s.copy();sn[accept]=sa[accept]
 exn=norm(sla.expm_multiply(-dt*H,exact));true=np.where(exn>=0,1,-1)
 trueflip=true!=s;p=exn*exn;p/=p.sum()
 false=float(np.sum(p[accept & ~trueflip]))
 missed=float(np.sum(p[trueflip & ~accept]))
 union=flipA|flipB; su=s.copy();su[union]=-s[union]
 falseu=float(np.sum(p[union & ~trueflip])); missu=float(np.sum(p[trueflip & ~union]))
 print({'tau':tau,'carry':err(s,exn),'consensus':err(sn,exn),'union':err(su,exn),
        'true_flip_mass':float(np.sum(p[trueflip])),
        'false_accept':false,'missed_true':missed,'union_false':falseu,'union_missed':missu,
        'accept_mass_exact':float(np.sum(p[accept]))},flush=True)
 s=su
