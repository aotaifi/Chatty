#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as cs
import scipy.sparse.linalg as sla
from closed_fn_krylov_exact4x4 import build_H, canonical, marshall_signs

ROOT=Path("/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure")
TAUS=[0.1,0.25,0.5,1.0,2.0,4.0]; NCOLS=32; SEED=20260930
H,diag=build_H(0.5)
rng=np.random.default_rng(SEED)
ys=np.array(rng.choice(np.arange(H.shape[0]),size=NCOLS,replace=False),dtype=int)
A=(H-sp.diags(diag)).copy(); A.data=np.ones_like(A.data)
dist=np.asarray(cs.shortest_path(A,directed=False,unweighted=True,indices=ys)).T
pred=np.where((dist.astype(np.int64)%2)==0,1,-1)
sm=canonical(marshall_signs()); base=sm[:,None]*sm[ys][None,:]
rows=[]
for tau in TAUS:
    E=np.zeros((H.shape[0],NCOLS)); E[ys,np.arange(NCOLS)]=1.0
    G=np.asarray(sla.expm_multiply(-tau*H,E),float)
    G/=np.linalg.norm(G,axis=0,keepdims=True)
    p=G*G; true=np.where(G>=0,1,-1)
    e=np.sum(p*(pred!=true),axis=0)
    em=np.sum(p*(base!=true),axis=0)
    row={"tau":tau,"shortest_mean":float(np.mean(e)),"shortest_max":float(np.max(e)),
         "marshall_mean":float(np.mean(em))}
    rows.append(row); print(row,flush=True)
out={"ncols":NCOLS,"seed":SEED,"rows":rows}
path=ROOT/"results/finite_tau_shortest_path_sweep_exact4x4.json"
path.write_text(json.dumps(out,indent=2))
print("WROTE",path)
