#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import scipy.sparse.linalg as sla

from finite_tau_matching_20site_exact import special_columns,D
from finite_tau_amplitude_regression_20site import endpoint,H
from finite_tau_learned_guide_gfmc_20site import (
    load_model,guide_vec,propagate_fn_uniform_importance
)

ROOT=Path("/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure")

def norm(v):
    v=np.asarray(v,float); return v/np.linalg.norm(v)

def mismatch(s,psi):
    p=psi*psi; p/=p.sum()
    return float(np.sum(p[s!=np.where(psi>=0,1,-1)]))

def run(y,M,seed,dt=.05,beta=.5,prior=512.):
    net,mu,sd=load_model(); arr=endpoint(y)
    d,_,_,_,rich,vals,rlab,rbs=arr
    s=np.where(d%2==0,1,-1).astype(np.int8)
    rng=np.random.default_rng(seed)
    walkers=np.full(M,y,np.int32)
    g=np.full(D,1/np.sqrt(D))
    exact=np.zeros(D); exact[y]=1.
    rows=[]

    for n in range(1,int(round(beta/dt))+1):
        walkers=propagate_fn_uniform_importance(walkers,g,s,dt,rng)
        exact=norm(sla.expm_multiply(-dt*H,exact))
        mass=np.bincount(rlab[walkers],minlength=len(vals)).astype(float)
        gm=guide_vec(net,mu,sd,y,n*dt,arr)
        pm=np.bincount(rlab,weights=gm,minlength=len(vals)); pm/=pm.sum()
        post=mass+prior*pm
        gnew=(post/rbs)[rlab]; gnew=norm(np.maximum(gnew,1e-14))

        psi_est=s*gnew
        exact_next=norm(sla.expm_multiply(-dt*H,exact))
        pred=psi_est-dt*(H@psi_est)
        sn=np.where(pred>=0,1,-1).astype(np.int8)

        row={"tau":n*dt,
             "amp_fid":float(np.dot(gnew,np.abs(exact))**2),
             "full_fid":float(np.dot(psi_est,exact)**2),
             "sign_err_current":mismatch(s,exact),
             "k1_next_sign_err":mismatch(sn,exact_next),
             "flip_mass":float(np.sum((gnew*gnew)[sn!=s])),
             "unique":int(len(np.unique(walkers)))}
        rows.append(row); print("ROW",M,seed,row,flush=True)
        s=sn; g=gnew
    return rows

def main():
    y=int(special_columns(np.random.default_rng(20260930),2)[1])
    out={"y":y,"runs":{}}
    for M in (8192,32768):
        for rep in (0,1):
            seed=130000+M+rep
            key=f"M{M}_r{rep}"
            out["runs"][key]=run(y,M,seed)
    path=ROOT/"results/finite_tau_walker_fn_k1_loop20.json"
    path.write_text(json.dumps(out,indent=2))
    print("WROTE",path,flush=True)

if __name__=="__main__": main()
