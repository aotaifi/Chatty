#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import scipy.sparse.csgraph as cs
from finite_tau_matching_20site_exact import build_H,D,special_columns

ROOT=Path("/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure")
H,diag=build_H(.5)
A=(H-sp.diags(diag)).copy(); A.data=np.ones_like(A.data)
co=H.tocoo(); off=co.row!=co.col
r=co.row[off].astype(np.int32); c=co.col[off].astype(np.int32); h=co.data[off]
idx=np.arange(D,dtype=np.int32)

def norm(v):
    v=np.asarray(v,float); return v/np.linalg.norm(v)

def build_fn(guide,s):
    prod=s[r]*s[c]; keep=prod<0
    g=np.maximum(np.asarray(guide,float),1e-8/np.sqrt(D))
    shift=np.bincount(r[~keep],weights=h[~keep]*g[c[~keep]]/g[r[~keep]],minlength=D)
    rr=np.concatenate([r[keep],idx]); cc=np.concatenate([c[keep],idx])
    vv=np.concatenate([h[keep]*prod[keep],diag+shift])
    return sp.coo_matrix((vv,(rr,cc)),shape=H.shape).tocsr()

def mismatch(s,psi):
    p=psi*psi; p/=p.sum()
    return float(np.sum(p[s!=np.where(psi>=0,1,-1)]))

def run(y,dt=.05,beta=.5,k1=True):
    dist=np.asarray(cs.shortest_path(A,directed=False,unweighted=True,indices=[y]))[0]
    s=np.where((dist.astype(np.int64)%2)==0,1.0,-1.0)
    a=np.zeros(D); a[y]=1.0
    exact=a.copy()
    guide=np.full(D,1/np.sqrt(D))
    rows=[]
    for n in range(1,int(round(beta/dt))+1):
        F=build_fn(guide,s)
        a=norm(np.maximum(sla.expm_multiply(-dt*F,a),0.0))
        exact=norm(sla.expm_multiply(-dt*H,exact))
        psi=s*a
        row={"tau":n*dt,
             "amp_fid":float(np.dot(a,np.abs(exact))**2),
             "full_fid":float(np.dot(psi,exact)**2),
             "sign_err_before":mismatch(s,exact)}
        guide=np.maximum(a,1e-8/np.sqrt(D)); guide=norm(guide)
        if k1:
            pred=psi-dt*(H@psi)
            sn=np.where(pred>=0,1.0,-1.0)
            row["k1_flip_mass"]=float(np.sum((a*a)[sn!=s]))
            row["sign_err_after_k1"]=mismatch(sn,exact)
            s=sn
        rows.append(row)
        print(("K1" if k1 else "FIX"),row,flush=True)
    return rows

def main():
    ys=special_columns(np.random.default_rng(20260930),2)
    y=int(ys[1]); print("HARD",y,flush=True)
    adaptive=run(y,k1=True)
    out={"y":y,"adaptive":adaptive}
    path=ROOT/"results/finite_tau_adaptive_fn_k1_20site.json"
    path.write_text(json.dumps(out,indent=2))
    print("WROTE",path,flush=True)

if __name__=="__main__": main()
