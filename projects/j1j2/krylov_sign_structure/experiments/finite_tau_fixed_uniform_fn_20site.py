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
rng=np.random.default_rng(20260930); ys=special_columns(rng,2)
rows=[]
for y in ys:
    dist=np.asarray(cs.shortest_path(A,directed=False,unweighted=True,indices=[int(y)]))[0]
    s=np.where((dist.astype(np.int64)%2)==0,1,-1).astype(np.int8)
    co=H.tocoo(); off=co.row!=co.col
    r=co.row[off]; c=co.col[off]; h=co.data[off]
    prod=s[r]*s[c]; keep=prod<0
    shift=np.bincount(r[~keep],weights=h[~keep],minlength=D)
    rr=np.concatenate([r[keep],np.arange(D)])
    cc=np.concatenate([c[keep],np.arange(D)])
    vv=np.concatenate([h[keep]*prod[keep],diag+shift])
    F=sp.coo_matrix((vv,(rr,cc)),shape=H.shape).tocsr()
    e=np.zeros(D); e[int(y)]=1.
    a=np.asarray(sla.expm_multiply(-.5*F,e),float); a=np.maximum(a,0); a/=np.linalg.norm(a)
    ex=np.asarray(sla.expm_multiply(-.5*H,e),float); ex/=np.linalg.norm(ex)
    at=np.abs(ex); p=ex*ex; ts=np.where(ex>=0,1,-1)
    row={"y":int(y),"tau":.5,
         "sign_mismatch_mass":float(np.sum(p[s!=ts])),
         "amplitude_fidelity":float(np.dot(a,at)**2),
         "full_fidelity":float(np.dot(a*s,ex)**2)}
    rows.append(row); print(row,flush=True)
path=ROOT/"results/finite_tau_fixed_uniform_fn_20site.json"
path.write_text(json.dumps({"rows":rows},indent=2)); print("WROTE",path)
