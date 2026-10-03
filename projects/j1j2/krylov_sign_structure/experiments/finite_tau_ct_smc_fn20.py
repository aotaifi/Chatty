#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import scipy.sparse.linalg as sla
from finite_tau_matching_20site_exact import D,special_columns
from finite_tau_ct_local_ratio_20site import (
    H,distance_sign,uniform_fn,normalized)

ROOT=Path("/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure")

def systematic(w,rng,M):
    w=np.asarray(w,float); w/=w.sum()
    c=np.cumsum(w); c[-1]=1.0
    u=rng.random()/M+np.arange(M)/M
    return np.searchsorted(c,u,side="right")

def ct_block(states,dt,J,lam,pot,rng):
    M=len(states)
    out=np.empty(M,np.int32); logw=np.empty(M,float)
    for q,x0 in enumerate(states):
        x=int(x0); t=0.0; lw=0.0
        while True:
            rate=float(lam[x])
            if rate<=0:
                lw+=float(pot[x])*(dt-t); break

            wait=float(rng.exponential(1.0/rate))
            if t+wait>=dt:
                lw+=float(pot[x])*(dt-t); break
            lw+=float(pot[x])*wait; t+=wait
            a,b=J.indptr[x],J.indptr[x+1]
            rates=J.data[a:b]
            u=rng.random()*rate
            j=int(np.searchsorted(np.cumsum(rates),u,side="right"))
            x=int(J.indices[a+min(j,len(rates)-1)])
        out[q]=x; logw[q]=lw
    logw-=np.max(logw); w=np.exp(logw)
    ess=float(w.sum()**2/np.dot(w,w))
    sel=systematic(w,rng,M)
    return out[sel],ess,int(np.unique(out[sel]).size)

def run_smc(y,tau,M,block,J,lam,pot,seed):
    rng=np.random.default_rng(seed)
    states=np.full(M,int(y),np.int32)
    rec=[]; t=0.0
    while t<tau-1e-14:
        dt=min(block,tau-t)
        states,ess,uniq=ct_block(states,dt,J,lam,pot,rng)
        t+=dt; rec.append({"tau":t,"ess_pre_resample":ess,"unique":uniq})
    hist=np.bincount(states,minlength=D).astype(float)
    return hist,rec

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--M",type=int,default=20000)
    ap.add_argument("--tau",type=float,default=.5)
    ap.add_argument("--block",type=float,default=.05)
    ap.add_argument("--out",default=str(ROOT/"results/finite_tau_ct_smc_fn20.json"))
    args=ap.parse_args()
    y=int(special_columns(np.random.default_rng(20260930),2)[1])
    s=distance_sign(y); F,J,lam,pot,bad=uniform_fn(s)
    e=np.zeros(D); e[y]=1.0
    afn=normalized(np.maximum(sla.expm_multiply(-args.tau*F,e),0.0))
    rows=[]; qs=[]
    for rep,seed in enumerate((91001,92001),1):
        h,rec=run_smc(y,args.tau,args.M,args.block,J,lam,pot,seed)
        q=normalized(h); qs.append(q)
        row={"rep":rep,"amp_fidelity":float(np.dot(q,afn)**2),
             "final_unique":int(np.count_nonzero(h)),"blocks":rec}
        rows.append(row); print("REP",json.dumps(row,sort_keys=True),flush=True)
    cross=float(np.dot(qs[0],qs[1])**2)
    out={"y":y,"M":args.M,"tau":args.tau,"block":args.block,
         "rows":rows,"cross_fidelity":cross}
    print("FINAL",{"cross_fidelity":cross},flush=True)
    Path(args.out).write_text(json.dumps(out,indent=2))
    print("WROTE",args.out,flush=True)
if __name__=="__main__": main()
