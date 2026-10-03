#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import scipy.sparse.linalg as sla

from closed_fn_krylov_exact4x4 import (
    build_H, basis, idx, site, canonical,
    marshall_signs, ground, build_fixed_node
)

ROOT=Path("/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure")
def normalize(v):
    n=float(np.linalg.norm(v))
    return np.asarray(v,float)/n if n>0 else np.asarray(v,float)

def anchor_sign(s,y):
    s=np.asarray(s,dtype=np.int8).copy()
    if s[y] < 0:
        s=-s
    return s

def signs_of(v, fallback, y):
    s=np.asarray(fallback,dtype=np.int8).copy()
    nz=np.abs(v)>1e-14*np.max(np.abs(v))
    s[nz]=np.where(v[nz]>=0,1,-1)
    return anchor_sign(s,y)

def state_index(bits):
    return idx[int(bits)]

def named_columns():
    neel=sum(1<<site(x,y) for y in range(4) for x in range(4) if (x+y)%2==0)
    stripe=sum(1<<site(x,y) for y in range(4) for x in range(4) if x%2==0)
    return {"neel":state_index(neel),"stripe":state_index(stripe),"random":137}
def run_column(H,diag,g0,y,dt,beta,floor):
    D=H.shape[0]
    a=np.zeros(D); a[y]=1.0
    exact=a.copy()
    sM=canonical(marshall_signs())
    s=anchor_sign(sM*int(sM[y]),y)
    sbase=s.copy()
    guide=g0.copy()
    records=[]
    nsteps=int(round(beta/dt))

    for n in range(1,nsteps+1):
        psi=a*s
        trial=psi-dt*np.asarray(H@psi)
        snew=signs_of(trial,s,y)

        F=build_fixed_node(H,diag,guide,s)
        anew=np.asarray(sla.expm_multiply(-dt*F,a),float)
        anew=np.maximum(anew,0.0)
        anew=normalize(anew)

        exact=np.asarray(sla.expm_multiply(-dt*H,exact),float)
        exact=normalize(exact)
        strue=signs_of(exact,snew,y)
        atr=np.abs(exact)
        p=atr*atr; p/=p.sum()

        mismatch=float(np.sum(p[snew!=strue]))
        baseline_mismatch=float(np.sum(p[sbase!=strue]))
        psihat=anew*snew
        fidelity=float(np.dot(psihat,exact)**2)
        amp_fid=float(np.dot(anew,atr)**2)
        change=float(np.sum((anew*anew)[snew!=s]))

        records.append(dict(step=n,tau=n*dt,sign_mismatch_mass=mismatch,
                            baseline_marshall_mismatch_mass=baseline_mismatch,
                            sign_overlap=1-2*mismatch,fidelity=fidelity,
                            amplitude_fidelity=amp_fid,sign_change_mass=change))
        a,s=anew,snew
        guide=np.maximum(a,floor*g0)
        guide=normalize(guide)
    return records
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dt",type=float,default=0.05)
    ap.add_argument("--beta",type=float,default=0.5)
    ap.add_argument("--floor",type=float,default=1e-10)
    ap.add_argument("--guide",choices=["j20","uniform"],default="j20")
    ap.add_argument("--columns",default="neel,stripe,random")
    ap.add_argument("--out",default=str(ROOT/"results/finite_tau_fn_sign_exact4x4.json"))
    args=ap.parse_args()

    H,diag=build_H(0.5)
    if args.guide=="j20":
        H0,_=build_H(0.0)
        _,v0=ground(H0)
        g0=normalize(np.abs(v0))
    else:
        g0=np.full(H.shape[0],1/np.sqrt(H.shape[0]),dtype=float)
    cols=named_columns()
    out={"method":{"J2":0.5,"dt":args.dt,"beta":args.beta,"floor":args.floor,
                   "initial_sign":"Marshall relative to column y",
                   "first_guide":args.guide,
                   "later_guide":"current transient FN amplitude with tiny positive floor",
                   "sign_update":"s[n+1]=sign((I-dt H) a[n]s[n])",
                   "oracle":"exact exp(-dt H) used only for diagnostics"},
         "columns":{}}
    for name in args.columns.split(","):
        name=name.strip(); y=cols[name]
        rec=run_column(H,diag,g0,y,args.dt,args.beta,args.floor)
        out["columns"][name]={"basis_index":int(y),"records":rec}
        last=rec[-1]
        print(name,"FINAL",json.dumps(last,sort_keys=True),flush=True)
    Path(args.out).write_text(json.dumps(out,indent=2))
    print("WROTE",args.out,flush=True)

if __name__=="__main__":
    main()
