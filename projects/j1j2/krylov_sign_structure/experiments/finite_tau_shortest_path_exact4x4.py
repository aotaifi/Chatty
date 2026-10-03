#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as cs
import scipy.sparse.linalg as sla
from closed_fn_krylov_exact4x4 import build_H, basis, canonical, marshall_signs

ROOT=Path("/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure")
TAU=0.5; NCOLS=64; SEED=1234

H,diag=build_H(0.5)
rng=np.random.default_rng(SEED)
ys=np.array(rng.choice(np.arange(H.shape[0]),size=NCOLS,replace=False),dtype=int)

E=np.zeros((H.shape[0],NCOLS))
E[ys,np.arange(NCOLS)]=1.0
G=np.asarray(sla.expm_multiply(-TAU*H,E),float)
G/=np.linalg.norm(G,axis=0,keepdims=True)
p=G*G
true=np.where(G>=0,1,-1).astype(np.int8)
# Configuration graph: every nonzero off-diagonal exchange counts as one operator step.
A=(H-sp.diags(diag)).copy()
A.data=np.ones_like(A.data)
dist=cs.shortest_path(A,directed=False,unweighted=True,indices=ys)
dist=np.asarray(dist).T

pred=np.ones_like(true)
finite=np.isfinite(dist)
pred[finite]=np.where((dist[finite].astype(np.int64)%2)==0,1,-1)
err=np.sum(p*(pred!=true),axis=0)

sm=canonical(marshall_signs())
base=sm[:,None]*sm[ys][None,:]
marshall_err=np.sum(p*(base!=true),axis=0)

corr_true=true*base
corr_short=pred*base
corr_err=np.sum(p*(corr_short!=corr_true),axis=0)
out={
 "tau":TAU,"ncols":NCOLS,"seed":SEED,
 "marshall_mean_error":float(np.mean(marshall_err)),
 "shortest_path_mean_error":float(np.mean(err)),
 "shortest_path_max_error":float(np.max(err)),
 "correction_mean_error":float(np.mean(corr_err)),
 "per_column_error":[float(x) for x in err]
}
path=ROOT/"results/finite_tau_shortest_path_exact4x4.json"
path.write_text(json.dumps(out,indent=2))
print(json.dumps({k:v for k,v in out.items() if k!="per_column_error"},indent=2))
print("WROTE",path)
