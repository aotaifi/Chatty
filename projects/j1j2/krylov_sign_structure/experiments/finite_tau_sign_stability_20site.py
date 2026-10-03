#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as cs
import scipy.sparse.linalg as sla
from finite_tau_matching_20site_exact import build_H,D,special_columns

ROOT=Path("/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure")
H,diag=build_H(.5)
rng=np.random.default_rng(20260930)
ys=special_columns(rng,8)
A=(H-sp.diags(diag)).copy(); A.data=np.ones_like(A.data)
dist=np.asarray(cs.shortest_path(A,directed=False,unweighted=True,indices=ys))
pred=np.where((dist.astype(np.int64)%2)==0,1,-1).astype(np.int8)
E=np.zeros((D,len(ys))); E[ys,np.arange(len(ys))]=1.0

taus=np.array([.02,.05,.1,.15,.25,.5,.75,1.0,1.25,1.5,2.0])
prev=None; rows=[]
for tau in taus:
    G=np.asarray(sla.expm_multiply(-tau*H,E),float)
    for k,y in enumerate(ys):
        g=G[:,k]; nrm=np.linalg.norm(g); g=g/nrm
        p=g*g; s=np.where(g>=0,1,-1).astype(np.int8)
        mm=float(np.sum(p[s!=pred[k]]))
        ov=float(abs(np.sum(p*s*pred[k])))
        nz=np.abs(g)>1e-14
        frac=float(np.mean(s[nz]!=pred[k,nz])) if np.any(nz) else 0.0
        cross_mass=0.0; cross_count=0
        if prev is not None:
            gp=prev[:,k]
            sp=np.where(gp>=0,1,-1).astype(np.int8)
            flip=(sp!=s) & (np.abs(gp)>1e-14) & (np.abs(g)>1e-14)
            cross_count=int(np.sum(flip))
            cross_mass=float(np.sum(p[flip]))
        rows.append({"tau":float(tau),"y":int(y),"weighted_mismatch":mm,
                     "sign_overlap":ov,"unweighted_mismatch_fraction":frac,
                     "crossing_count_since_prev_tau":cross_count,
                     "crossing_mass_current":cross_mass})
    print("TAU",tau,
          "mean_mm",float(np.mean([r["weighted_mismatch"] for r in rows if r["tau"]==float(tau)])),
          "max_mm",float(np.max([r["weighted_mismatch"] for r in rows if r["tau"]==float(tau)])),
          "mean_cross_mass",float(np.mean([r["crossing_mass_current"] for r in rows if r["tau"]==float(tau)])),
          flush=True)
    prev=G.copy()

out={"D":D,"ys":[int(x) for x in ys],"taus":taus.tolist(),"rows":rows}
path=ROOT/"results/finite_tau_sign_stability_20site.json"
path.write_text(json.dumps(out,indent=2))
print("WROTE",path,flush=True)
