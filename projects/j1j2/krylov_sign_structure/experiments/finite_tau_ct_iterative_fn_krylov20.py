#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from finite_tau_matching_20site_exact import D,special_columns
from finite_tau_ct_local_ratio_20site import (
    H,diag,R,C,HV,IDX,distance_sign,sample_paths,normalized,mismatch)

ROOT=Path("/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure")

def ct_fn_params(s,g):
    prod=s[R]*s[C]; good=prod<0; bad=~good
    floor=1e-10/np.sqrt(D)
    gg=np.maximum(np.asarray(g,float),floor)
    lam=np.bincount(R[good],weights=HV[good],minlength=D)
    shift=np.bincount(R[bad],weights=HV[bad]*gg[C[bad]]/gg[R[bad]],minlength=D)
    F=sp.coo_matrix((
        np.concatenate([-HV[good],diag+shift]),
        (np.concatenate([R[good],IDX]),np.concatenate([C[good],IDX]))
    ),shape=H.shape).tocsr()
    J=sp.coo_matrix((HV[good],(R[good],C[good])),shape=H.shape).tocsr()
    pot=lam-(diag+shift)
    return F,J,lam,pot

def cv_update(g,h1,h2,prior_frac):
    old=np.maximum(g/g.sum(),1e-300)
    p1=h1/h1.sum(); p2=h2/h2.sum()
    b1=np.maximum(p1+prior_frac/D,1e-300)
    b2=np.maximum(p2+prior_frac/D,1e-300)
    curve=[]
    for eta in np.linspace(0.05,1.0,20):
        q1=old**(1-eta)*b1**eta; q1/=q1.sum()
        q2=old**(1-eta)*b2**eta; q2/=q2.sum()
        ce=-0.5*(np.sum(p2*np.log(q1))+np.sum(p1*np.log(q2)))
        curve.append((float(eta),float(ce)))
    eta=min(curve,key=lambda z:z[1])[0]
    pm=.5*(p1+p2)
    bm=np.maximum(pm+prior_frac/D,1e-300)
    gn=normalized(old**(1-eta)*bm**eta)
    return gn,eta,curve

def k1_error(s,a,psi_next,dt):
    pred=s*a-dt*(H@(s*a))
    sn=np.where(pred>=0,1.0,-1.0)
    return mismatch(sn,psi_next)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--M",type=int,default=100000)
    ap.add_argument("--tau",type=float,default=.5)
    ap.add_argument("--iters",type=int,default=3)
    ap.add_argument("--prior-frac",type=float,default=.01)
    ap.add_argument("--dt-k1",type=float,default=.05)
    ap.add_argument("--out",default=str(ROOT/"results/finite_tau_ct_iterative_fn_krylov20.json"))
    args=ap.parse_args()
    y=int(special_columns(np.random.default_rng(20260930),2)[1])
    s=distance_sign(y)
    e=np.zeros(D); e[y]=1.0
    psi=normalized(sla.expm_multiply(-args.tau*H,e))
    psi_next=normalized(sla.expm_multiply(-(args.tau+args.dt_k1)*H,e))
    g=np.ones(D)/np.sqrt(D)
    rows=[]
    for it in range(args.iters):
        F,J,lam,pot=ct_fn_params(s,g)
        afn=normalized(np.maximum(sla.expm_multiply(-args.tau*F,e),0.0))
        full=float(np.dot(s*afn,psi)**2)
        amp=float(np.dot(afn,np.abs(psi))**2)
        print("ITER",it,"FN",full,flush=True)

        h1,ess1,u1=sample_paths(y,args.tau,args.M,J,lam,pot,81000+100*it+1)
        h2,ess2,u2=sample_paths(y,args.tau,args.M,J,lam,pot,81000+100*it+2)
        q1=normalized(h1); q2=normalized(h2)
        repfid=[float(np.dot(q1,afn)**2),float(np.dot(q2,afn)**2)]
        cross=float(np.dot(q1,q2)**2)
        gn,eta,curve=cv_update(g,h1,h2,args.prior_frac)
        k1g=k1_error(s,gn,psi_next,args.dt_k1)
        row={"iter":it,"fn_full_fidelity":full,"fn_amp_fidelity":amp,
             "ess":[ess1,ess2],"unique":[u1,u2],
             "replica_fn_fidelity":repfid,"replica_cross_fidelity":cross,
             "eta_cv":eta,"cv_curve":curve,"guide_k1_next_sign_mismatch":k1g}
        rows.append(row)
        print("UPDATE",json.dumps(row,sort_keys=True),flush=True)
        g=gn
    F,J,lam,pot=ct_fn_params(s,g)
    afinal=normalized(np.maximum(sla.expm_multiply(-args.tau*F,e),0.0))
    final={"full_fidelity":float(np.dot(s*afinal,psi)**2),
           "amp_fidelity":float(np.dot(afinal,np.abs(psi))**2),
           "k1_next_sign_mismatch":k1_error(s,afinal,psi_next,args.dt_k1)}
    out={"y":y,"M":args.M,"tau":args.tau,"rows":rows,"final":final}
    print("FINAL",json.dumps(final,sort_keys=True),flush=True)
    Path(args.out).write_text(json.dumps(out,indent=2))
    print("WROTE",args.out,flush=True)
if __name__=="__main__": main()
