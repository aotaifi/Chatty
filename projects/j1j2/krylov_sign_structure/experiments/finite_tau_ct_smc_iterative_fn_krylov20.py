#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import scipy.sparse.linalg as sla
from finite_tau_matching_20site_exact import D,special_columns
from finite_tau_ct_local_ratio_20site import H,distance_sign,normalized
from finite_tau_ct_iterative_fn_krylov20 import ct_fn_params,cv_update,k1_error
from finite_tau_ct_smc_fn20 import run_smc

ROOT=Path("/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--M",type=int,default=100000)
    ap.add_argument("--tau",type=float,default=.5)
    ap.add_argument("--iters",type=int,default=3)
    ap.add_argument("--block",type=float,default=.05)
    ap.add_argument("--prior-frac",type=float,default=.01)
    ap.add_argument("--dt-k1",type=float,default=.05)
    ap.add_argument("--out",default=str(ROOT/"results/finite_tau_ct_smc_iterative_fn_krylov20.json"))
    args=ap.parse_args()
    y=int(special_columns(np.random.default_rng(20260930),2)[1])
    s=distance_sign(y)
    e=np.zeros(D); e[y]=1.0
    psi=normalized(sla.expm_multiply(-args.tau*H,e))
    psi_next=normalized(sla.expm_multiply(-(args.tau+args.dt_k1)*H,e))

    g=np.ones(D)/np.sqrt(D); rows=[]
    for it in range(args.iters):
        F,J,lam,pot=ct_fn_params(s,g)
        afn=normalized(np.maximum(sla.expm_multiply(-args.tau*F,e),0.0))
        full=float(np.dot(s*afn,psi)**2)
        h1,r1=run_smc(y,args.tau,args.M,args.block,J,lam,pot,101000+100*it+1)
        h2,r2=run_smc(y,args.tau,args.M,args.block,J,lam,pot,101000+100*it+2)
        q1=normalized(h1); q2=normalized(h2)
        gn,eta,curve=cv_update(g,h1,h2,args.prior_frac)
        row={"iter":it,"fn_full_fidelity":full,
             "replica_fn_fidelity":[float(np.dot(q1,afn)**2),float(np.dot(q2,afn)**2)],
             "replica_cross_fidelity":float(np.dot(q1,q2)**2),
             "min_block_ess":[min(z["ess_pre_resample"] for z in r1),
                              min(z["ess_pre_resample"] for z in r2)],
             "eta_cv":eta,"cv_curve":curve,
             "guide_k1_next_sign_mismatch":k1_error(s,gn,psi_next,args.dt_k1)}
        rows.append(row); print("ITER",json.dumps(row,sort_keys=True),flush=True)
        g=gn

    F,J,lam,pot=ct_fn_params(s,g)
    afinal=normalized(np.maximum(sla.expm_multiply(-args.tau*F,e),0.0))
    final={"full_fidelity":float(np.dot(s*afinal,psi)**2),
           "amp_fidelity":float(np.dot(afinal,np.abs(psi))**2),
           "k1_next_sign_mismatch":k1_error(s,afinal,psi_next,args.dt_k1)}
    out={"y":y,"M":args.M,"tau":args.tau,"block":args.block,
         "rows":rows,"final":final}
    print("FINAL",json.dumps(final,sort_keys=True),flush=True)
    Path(args.out).write_text(json.dumps(out,indent=2))
    print("WROTE",args.out,flush=True)

if __name__=="__main__": main()
