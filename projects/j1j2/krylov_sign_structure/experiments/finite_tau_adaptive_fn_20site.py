#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import scipy.sparse.csgraph as cs
from finite_tau_matching_20site_exact import build_H, basis, D, special_columns

ROOT=Path("/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure")

def config_distance_sign(H,diag,y):
    A=(H-sp.diags(diag)).copy()
    A.data=np.ones_like(A.data)
    d=np.asarray(cs.shortest_path(A,directed=False,unweighted=True,indices=[y]))[0]
    return np.where((d.astype(np.int64)%2)==0,1,-1).astype(np.int8)

def build_fn_vec(H,diag,guide,s):
    co=H.tocoo()
    off=co.row!=co.col
    r=co.row[off].astype(np.int32); c=co.col[off].astype(np.int32); h=co.data[off]
    prod=s[r]*s[c]
    keep=prod<0
    g=np.maximum(np.asarray(guide,float),1e-14)
    shift=np.bincount(r[~keep],weights=h[~keep]*g[c[~keep]]/g[r[~keep]],minlength=H.shape[0])
    rr=np.concatenate([r[keep],np.arange(H.shape[0],dtype=np.int32)])
    cc=np.concatenate([c[keep],np.arange(H.shape[0],dtype=np.int32)])
    vv=np.concatenate([h[keep]*prod[keep],diag+shift])
    return sp.coo_matrix((vv,(rr,cc)),shape=H.shape).tocsr()
def run_col(H,diag,y,dt,beta,floor):
    s=config_distance_sign(H,diag,y)
    a=np.zeros(D); a[y]=1.0
    exact=a.copy()
    guide=np.full(D,1/np.sqrt(D),dtype=float)
    rec=[]
    for n in range(1,int(round(beta/dt))+1):
        F=build_fn_vec(H,diag,guide,s)
        a=np.asarray(sla.expm_multiply(-dt*F,a),float)
        a=np.maximum(a,0.0); a/=np.linalg.norm(a)
        exact=np.asarray(sla.expm_multiply(-dt*H,exact),float)
        exact/=np.linalg.norm(exact)
        atr=np.abs(exact); p=atr*atr
        mismatch=float(np.sum(p[s!=np.where(exact>=0,1,-1)]))
        amp_fid=float(np.dot(a,atr)**2)
        full_fid=float(np.dot(a*s,exact)**2)
        rec.append({"tau":n*dt,"sign_mismatch_mass":mismatch,
                    "amplitude_fidelity":amp_fid,"full_fidelity":full_fid})
        print("y",y,"step",n,rec[-1],flush=True)
        guide=np.maximum(a,floor/np.sqrt(D)); guide/=np.linalg.norm(guide)
    return rec
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dt",type=float,default=0.05)
    ap.add_argument("--beta",type=float,default=0.5)
    ap.add_argument("--floor",type=float,default=1e-8)
    ap.add_argument("--ncols",type=int,default=2)
    ap.add_argument("--seed",type=int,default=20260930)
    ap.add_argument("--out",default=str(ROOT/"results/finite_tau_adaptive_fn_20site.json"))
    args=ap.parse_args()
    rng=np.random.default_rng(args.seed)
    H,diag=build_H(.5)
    ys=special_columns(rng,args.ncols)
    out={"D":D,"dt":args.dt,"beta":args.beta,"floor":args.floor,"ys":ys,"columns":{}}
    for y in ys:
        out["columns"][str(y)]=run_col(H,diag,int(y),args.dt,args.beta,args.floor)
    Path(args.out).write_text(json.dumps(out,indent=2))
    print("WROTE",args.out,flush=True)

if __name__=="__main__":
    main()
