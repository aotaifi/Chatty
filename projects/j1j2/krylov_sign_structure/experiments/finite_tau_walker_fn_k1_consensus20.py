#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import scipy.sparse.linalg as sla

from finite_tau_matching_20site_exact import special_columns,D
from finite_tau_amplitude_regression_20site import endpoint,H
from finite_tau_learned_guide_gfmc_20site import load_model,guide_vec,propagate_fn_uniform_importance

ROOT=Path("/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure")

def norm(v):
    v=np.asarray(v,float); return v/np.linalg.norm(v)

def mismatch(s,psi):
    p=psi*psi;p/=p.sum()
    return float(np.sum(p[s!=np.where(psi>=0,1,-1)]))

def estimate(w,y,tau,arr,net,mu,sd,prior):
    d,_,_,_,rich,vals,rlab,rbs=arr
    mass=np.bincount(rlab[w],minlength=len(vals)).astype(float)
    gm=guide_vec(net,mu,sd,y,tau,arr)
    pm=np.bincount(rlab,weights=gm,minlength=len(vals));pm/=pm.sum()
    post=mass+prior*pm
    return norm(np.maximum((post/rbs)[rlab],1e-14))

def run(y,M,seed,dt=.05,beta=.5,prior=512.):
    net,mu,sd=load_model();arr=endpoint(y);d=arr[0]
    s=np.where(d%2==0,1,-1).astype(np.int8)
    ra=np.random.default_rng(seed);rb=np.random.default_rng(seed+10000)
    wa=np.full(M,y,np.int32);wb=np.full(M,y,np.int32)
    ga=np.full(D,1/np.sqrt(D));gb=ga.copy()
    exact=np.zeros(D);exact[y]=1.;rows=[]
    for n in range(1,int(round(beta/dt))+1):
        wa=propagate_fn_uniform_importance(wa,ga,s,dt,ra)
        wb=propagate_fn_uniform_importance(wb,gb,s,dt,rb)
        exact=norm(sla.expm_multiply(-dt*H,exact))
        tau=n*dt
        ga=estimate(wa,y,tau,arr,net,mu,sd,prior)
        gb=estimate(wb,y,tau,arr,net,mu,sd,prior)

        fa=s*ga-dt*(H@(s*ga)); fb=s*gb-dt*(H@(s*gb))
        sa=np.where(fa>=0,1,-1).astype(np.int8)
        sb=np.where(fb>=0,1,-1).astype(np.int8)
        flipA=sa!=s; flipB=sb!=s
        accept=flipA & flipB & (sa==sb)
        sn=s.copy();sn[accept]=sa[accept]

        exn=norm(sla.expm_multiply(-dt*H,exact))
        row={"tau":tau,"ampfidA":float(np.dot(ga,np.abs(exact))**2),
             "ampfidB":float(np.dot(gb,np.abs(exact))**2),
             "sign_err_current":mismatch(s,exact),
             "next_err_consensus":mismatch(sn,exn),
             "next_err_A":mismatch(sa,exn),"next_err_B":mismatch(sb,exn),
             "accepted_flip_mass":float(np.sum((.5*(ga*ga+gb*gb))[accept])),
             "replica_flip_disagree_mass":float(np.sum((.5*(ga*ga+gb*gb))[flipA!=flipB]))}
        rows.append(row);print("ROW",M,row,flush=True)
        s=sn
    return rows

def main():
    y=int(special_columns(np.random.default_rng(20260930),2)[1])
    out={"y":y,"runs":{}}
    for M in (8192,32768):
        out["runs"][str(M)]=run(y,M,190000+M)
    path=ROOT/"results/finite_tau_walker_fn_k1_consensus20.json"
    path.write_text(json.dumps(out,indent=2));print("WROTE",path,flush=True)

if __name__=="__main__":main()
