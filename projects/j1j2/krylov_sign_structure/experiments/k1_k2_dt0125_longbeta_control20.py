#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import scipy.sparse.linalg as sla
from finite_tau_matching_20site_exact import D
from finite_tau_ct_local_ratio_20site import H,normalized,mismatch

z=np.load("../results/reconstruct_M100000_y59279.npz")
y=int(z["y"]); s=z["s"].astype(float); logg=z["logg"]
g=normalized(np.exp(np.clip(logg-logg.max(),-700,0)))
p1=s*g; p2=p1.copy()
dt=.0125
check={.5}
t=.5
rows=[]
e=np.zeros(D);e[y]=1.
exact={}
for tt in [.5,1.,1.5,2.,2.5,3.]:
    exact[tt]=normalized(sla.expm_multiply(-tt*H,e))
def rec(tt,v1,v2):
    ex=exact[tt]
    return {
      "tau":tt,
      "k1_fid":float(np.dot(v1,ex)**2),
      "k2_fid":float(np.dot(v2,ex)**2),
      "k1_sign":mismatch(np.where(v1>=0,1.,-1.),ex),
      "k2_sign":mismatch(np.where(v2>=0,1.,-1.),ex)}
rows.append(rec(.5,p1,p2))
nsteps=int(round((3.-.5)/dt))
for k in range(1,nsteps+1):
    h1=H@p1
    p1=normalized(p1-dt*h1)
    h1=H@p2
    p2=normalized(p2-dt*h1+.5*dt*dt*(H@h1))
    t=round(.5+k*dt,10)
    if any(abs(t-q)<1e-9 for q in [1.,1.5,2.,2.5,3.]):
        row=rec(t,p1,p2);rows.append(row);print("ROW",json.dumps(row,sort_keys=True),flush=True)
out={"dt":dt,"M":100000,"rows":rows}
Path("../results/k1_k2_dt0125_longbeta_control20.json").write_text(json.dumps(out,indent=2))
print("FINAL",json.dumps(rows[-1],sort_keys=True),flush=True)
