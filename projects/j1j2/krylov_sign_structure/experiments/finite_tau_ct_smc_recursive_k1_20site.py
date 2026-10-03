#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import scipy.sparse.linalg as sla

from finite_tau_matching_20site_exact import D,special_columns
from finite_tau_ct_local_ratio_20site import H,distance_sign,normalized,mismatch
from finite_tau_ct_iterative_fn_krylov20 import ct_fn_params,cv_update
from finite_tau_ct_smc_fn20 import run_smc,ct_block,systematic

ROOT=Path("/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure")
M=100000; DT=.05; T0=.5; T1=1.0; PRIOR=.01

def reconstruct(y,s):
    g=np.ones(D)/np.sqrt(D)
    for it in range(3):
        F,J,lam,pot=ct_fn_params(s,g)
        h1,_=run_smc(y,T0,M,DT,J,lam,pot,101000+100*it+1)
        h2,_=run_smc(y,T0,M,DT,J,lam,pot,101000+100*it+2)
        g,eta,_=cv_update(g,h1,h2,PRIOR)
        print("RECON",it,eta,flush=True)
    return g

def init_replicas(y,s,g):
    F,J,lam,pot=ct_fn_params(s,g)
    h1,r1=run_smc(y,T0,M,DT,J,lam,pot,104001)
    h2,r2=run_smc(y,T0,M,DT,J,lam,pot,104002)
    gn,eta,_=cv_update(g,h1,h2,PRIOR)
    rg1=np.random.default_rng(104101); rg2=np.random.default_rng(104102)
    st1=systematic(h1/h1.sum(),rg1,M).astype(np.int32)
    st2=systematic(h2/h2.sum(),rg2,M).astype(np.int32)
    print("INIT",{"eta":eta,"u1":int(np.unique(st1).size),"u2":int(np.unique(st2).size)},flush=True)
    return st1,st2,gn

def k1(s,a):
    pred=s*a-DT*(H@(s*a))
    return np.where(pred>=0,1.0,-1.0)

def exact_col(y,t):
    e=np.zeros(D);e[y]=1.
    return normalized(sla.expm_multiply(-t*H,e))

def run_branch(y,st10,st20,g0,s0,adaptive,label):
    st1=st10.copy();st2=st20.copy();g=g0.copy();s=s0.copy()
    rg1=np.random.default_rng(210001);rg2=np.random.default_rng(220001)
    rows=[];t=T0
    while t<T1-1e-12:
        F,J,lam,pot=ct_fn_params(s,g)
        st1,ess1,u1=ct_block(st1,DT,J,lam,pot,rg1)
        st2,ess2,u2=ct_block(st2,DT,J,lam,pot,rg2)
        t=round(t+DT,10)
        h1=np.bincount(st1,minlength=D).astype(float)
        h2=np.bincount(st2,minlength=D).astype(float)
        q1=normalized(h1);q2=normalized(h2);q=normalized(h1+h2)
        gn,eta,_=cv_update(g,h1,h2,PRIOR)

        ex=exact_col(y,t)
        full=float(np.dot(s*q,ex)**2)
        amp=float(np.dot(q,np.abs(ex))**2)
        row={"tau":t,"full_fidelity":full,"amp_fidelity":amp,
             "sign_mismatch":mismatch(s,ex),
             "replica_cross":float(np.dot(q1,q2)**2),
             "ess":[ess1,ess2],"unique":[u1,u2],"eta":eta}

        if t<T1-1e-12:
            exn=exact_col(y,t+DT)
            sn=k1(s,gn) if adaptive else s.copy()
            row["next_sign_mismatch"]=mismatch(sn,exn)
            row["flip_weight_current"]=float(np.sum((ex*ex)[sn!=s]))
            s=sn
        rows.append(row)
        print(label,json.dumps(row,sort_keys=True),flush=True)
        g=gn
    return rows

def main():
    y=int(special_columns(np.random.default_rng(20260930),2)[1])
    s0=distance_sign(y)
    g3=reconstruct(y,s0)
    st1,st2,g4=init_replicas(y,s0,g3)

    adaptive=run_branch(y,st1,st2,g4,s0,True,"ADAPT")
    fixed=run_branch(y,st1,st2,g4,s0,False,"FIXED")

    out={"y":y,"M":M,"dt":DT,"adaptive":adaptive,"fixed":fixed}
    path=ROOT/"results/finite_tau_ct_smc_recursive_k1_20site.json"
    path.write_text(json.dumps(out,indent=2))
    print("WROTE",path,flush=True)

if __name__=="__main__":main()
