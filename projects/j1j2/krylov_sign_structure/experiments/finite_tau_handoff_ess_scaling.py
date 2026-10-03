#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import scipy.sparse.linalg as sla
from closed_fn_krylov_exact4x4 import build_H as build16,D as D16
from finite_tau_matching_20site_exact import build_H as build20,D as D20

ROOT=Path("/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure")
rng=np.random.default_rng(20260930)

def run(build,D,N,ncols=8):
    H,_=build(.5)
    ys=np.asarray(rng.choice(np.arange(D),size=ncols,replace=False),int)
    E=np.zeros((D,ncols)); E[ys,np.arange(ncols)]=1.
    rows=[]
    for tau in (.05,.10,.15,.20,.25,.50):
        G=np.asarray(sla.expm_multiply(-tau*H,E),float)
        vals=[]
        for k in range(ncols):
            a=np.abs(G[:,k])
            ess=(np.sum(a*a)**2)/(np.sum(a)*np.sum(a**3))
            vals.append(float(ess))
        row={"N":N,"tau":tau,"mean_ess_fraction":float(np.mean(vals)),
             "min":float(np.min(vals)),"max":float(np.max(vals)),"values":vals}
        rows.append(row); print(row,flush=True)
    return rows

out={"N16":run(build16,D16,16),"N20":run(build20,D20,20)}
path=ROOT/"results/finite_tau_handoff_ess_scaling_16_20.json"
path.write_text(json.dumps(out,indent=2)); print("WROTE",path,flush=True)
