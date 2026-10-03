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
M=100000; TAU=.5; DT=.05; BLOCK=.05

def reconstruct(y,s):
    g=np.ones(D)/np.sqrt(D)
    hist=[]
    for it in range(3):
        F,J,lam,pot=ct_fn_params(s,g)
        h1,r1=run_smc(y,TAU,M,BLOCK,J,lam,pot,101000+100*it+1)
        h2,r2=run_smc(y,TAU,M,BLOCK,J,lam,pot,101000+100*it+2)
        gn,eta,curve=cv_update(g,h1,h2,.01)
        hist.append({"iter":it,"eta":eta,
                     "min_ess":[min(z["ess_pre_resample"] for z in r1),
                                min(z["ess_pre_resample"] for z in r2)]})
        print("RECON",hist[-1],flush=True)
        g=gn
    return g,hist

def sample_tau05_population(y,s,g):
    F,J,lam,pot=ct_fn_params(s,g)
    h1,r1=run_smc(y,TAU,M,BLOCK,J,lam,pot,104001)
    h2,r2=run_smc(y,TAU,M,BLOCK,J,lam,pot,104002)
    h=h1+h2
    q=normalized(h)
    # Build one M-particle population from the two-replica amplitude estimate.
    rng=np.random.default_rng(104003)
    prob=h/h.sum()
    idx=systematic(prob,rng,M)
    states=idx.astype(np.int32)
    return states,q,h1,h2,r1,r2

def exacts(y):
    e=np.zeros(D);e[y]=1.
    p05=normalized(sla.expm_multiply(-TAU*H,e))
    p055=normalized(sla.expm_multiply(-(TAU+DT)*H,e))
    return p05,p055

def next_sign(s,a):
    pred=s*a-DT*(H@(s*a))
    return np.where(pred>=0,1.0,-1.0)

def one_block(states,s,g,seed):
    F,J,lam,pot=ct_fn_params(s,g)
    rng=np.random.default_rng(seed)
    out,ess,uniq=ct_block(states,DT,J,lam,pot,rng)
    h=np.bincount(out,minlength=D).astype(float)
    return normalized(h),ess,uniq

def main():
    y=int(special_columns(np.random.default_rng(20260930),2)[1])
    s0=distance_sign(y); psi05,psi055=exacts(y)

    g3,recon=reconstruct(y,s0)
    F3,J3,lam3,pot3=ct_fn_params(s0,g3)
    afn=normalized(np.maximum(sla.expm_multiply(-TAU*F3,np.eye(1,D,y)[0]),0.0))
    print("AFN05",{"full":float(np.dot(s0*afn,psi05)**2),
                   "amp":float(np.dot(afn,np.abs(psi05))**2)},flush=True)

    states,q,h1,h2,r1,r2=sample_tau05_population(y,s0,g3)
    g4,eta4,curve4=cv_update(g3,h1,h2,.01)
    s_oracle=next_sign(s0,afn)
    s_sample=next_sign(s0,g4)

    print("SIGNS",{
        "carry":mismatch(s0,psi055),
        "oracle_fn_k1":mismatch(s_oracle,psi055),
        "sample_guide_k1":mismatch(s_sample,psi055),
        "sample_vs_oracle_mass":float(np.sum((psi055*psi055)[s_sample!=s_oracle])),
        "eta4":eta4,
        "pop_amp_fid05":float(np.dot(q,np.abs(psi05))**2)
    },flush=True)

    branches={}
    for name,sb,seed in [
        ("carry",s0,105001),
        ("k1_sample",s_sample,105002),
        ("k1_oracleFN",s_oracle,105003),
    ]:
        qb,ess,uniq=one_block(states.copy(),sb,g4,seed)
        row={"full_fidelity_055":float(np.dot(sb*qb,psi055)**2),
             "amp_fidelity_055":float(np.dot(qb,np.abs(psi055))**2),
             "sign_mismatch_055":mismatch(sb,psi055),
             "ess":ess,"unique":uniq}
        branches[name]=row; print("BRANCH",name,row,flush=True)

    out={"y":y,"M":M,"tau0":TAU,"dt":DT,"reconstruct":recon,
         "g4_eta":eta4,
         "population_amp_fidelity_05":float(np.dot(q,np.abs(psi05))**2),
         "signs":{"carry":mismatch(s0,psi055),
                  "k1_sample":mismatch(s_sample,psi055),
                  "k1_oracleFN":mismatch(s_oracle,psi055)},
         "branches":branches}
    path=ROOT/"results/finite_tau_ct_smc_composed_k1_block20.json"
    path.write_text(json.dumps(out,indent=2))
    np.savez_compressed(ROOT/"results/finite_tau_ct_smc_composed_k1_state20.npz",
        g3=g3,g4=g4,s0=s0,s_sample=s_sample,s_oracle=s_oracle,q05=q)
    print("WROTE",path,flush=True)

if __name__=="__main__":main()
