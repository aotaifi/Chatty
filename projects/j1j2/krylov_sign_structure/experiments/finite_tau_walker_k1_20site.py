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

def sampled_amp(y,M,seed,dt=.05,beta=.5):
    net,mu,sd=load_model(); arr=endpoint(y)
    d,_,_,_,rich,vals,rlab,rbs=arr
    s=np.where(d%2==0,1,-1).astype(np.int8)
    rng=np.random.default_rng(seed); w=np.full(M,y,np.int32)
    for n in range(1,int(round(beta/dt))+1):
        tp=(n-1)*dt
        g=guide_vec(net,mu,sd,y,tp,arr) if n>1 else np.full(D,1/np.sqrt(D))
        w=propagate_fn_uniform_importance(w,g,s,dt,rng)
    mass=np.bincount(rlab[w],minlength=len(vals)).astype(float)
    a=(mass/rbs)[rlab]
    return norm(a),s,w

def test(M,seed,y,dt=.05,beta=.5):
    a,s,w=sampled_amp(y,M,seed,dt,beta)
    psi=s*a
    pred=psi-dt*(H@psi)
    sk=np.where(pred>=0,1,-1)

    e=np.zeros(D); e[y]=1.
    ex=norm(sla.expm_multiply(-beta*H,e))
    exn=norm(sla.expm_multiply(-dt*H,ex))
    fixed=mismatch(s,exn)
    k1=mismatch(sk,exn)
    amp_fid=float(np.dot(a,np.abs(ex))**2)
    return {"M":M,"seed":seed,"amp_fid_tau05":amp_fid,
            "fixed_sign_err_tau055":fixed,
            "walker_k1_sign_err_tau055":k1,
            "unique":int(len(np.unique(w)))}

def main():
    y=int(special_columns(np.random.default_rng(20260930),2)[1])
    rows=[]
    for M in (8192,32768):
        for rep in (0,1):
            row=test(M,91000+M+rep,y)
            rows.append(row); print("ROW",row,flush=True)
    path=ROOT/"results/finite_tau_walker_k1_20site.json"
    path.write_text(json.dumps({"y":y,"rows":rows},indent=2))
    print("WROTE",path,flush=True)

if __name__=="__main__": main()
