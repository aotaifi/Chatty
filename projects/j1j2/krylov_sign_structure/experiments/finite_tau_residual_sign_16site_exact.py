#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import scipy.sparse.csgraph as cs
from closed_fn_krylov_exact4x4 import build_H, basis, D, canonical, marshall_signs

ROOT=Path("/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure")
H,diag=build_H(.5)
Habs=2*sp.diags(diag)-H
rng=np.random.default_rng(20260930)
ys=np.array(rng.choice(np.arange(D),size=32,replace=False),dtype=int)
E=np.zeros((D,len(ys))); E[ys,np.arange(len(ys))]=1.0

Agraph=(H-sp.diags(diag)).copy(); Agraph.data=np.ones_like(Agraph.data)
dist=np.asarray(cs.shortest_path(Agraph,directed=False,unweighted=True,indices=ys)).T
pred=np.where((dist.astype(np.int64)%2)==0,1,-1)
rows=[]
for tau in (0.0125,0.025,0.05,0.1):
    G=np.asarray(sla.expm_multiply(-tau*H,E),float)
    A=np.asarray(sla.expm_multiply(-tau*Habs,E),float)
    guided=[]; raw=[]
    for k in range(len(ys)):
        den=float(np.sum(A[:,k]))
        guided.append(float(np.sum(pred[:,k]*G[:,k])/den))
        raw.append(float(np.sum(G[:,k])/den))
    gm=float(np.mean(guided)); rm=float(np.mean(raw))
    c=float(-np.log(gm)/(tau*tau))
    row={"tau":tau,"guided_mean":gm,"raw_mean":rm,"c_minuslog_over_tau2":c}
    rows.append(row); print(row,flush=True)

out={"N":16,"D":D,"ncols":len(ys),"rows":rows}
path=ROOT/"results/finite_tau_residual_sign_16site_exact.json"
path.write_text(json.dumps(out,indent=2))
print("WROTE",path)
