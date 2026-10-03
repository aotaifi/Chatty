#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import scipy.sparse.linalg as sla
from finite_tau_matching_20site_exact import D
from finite_tau_ct_local_ratio_20site import H,normalized

z=np.load("../results/reconstruct_M100000_y59279.npz")
y=int(z["y"]);s=z["s"].astype(float);logg=z["logg"]
g=normalized(np.exp(np.clip(logg-logg.max(),-700,0)))
psi=s*g;dt=.0125
h1=H@psi;h2=H@h1
e1=h1/psi;e2=h2/psi
r1=1-dt*e1;r2=r1+.5*dt*dt*e2
eps=np.log(np.maximum(np.abs(r2),1e-300))-np.log(np.maximum(np.abs(r1),1e-300))
e=np.zeros(D);e[y]=1.
ex=normalized(sla.expm_multiply(-.5*H,e));p=ex*ex;p/=p.sum()
tot=float(np.sum(p*eps*eps))
rows=[]
for th in [0.001,0.002,0.005,0.01,0.02,0.05]:
    m=np.abs(eps)>th
    rows.append({"kind":"eps","threshold":th,"state_frac":float(m.mean()),
                 "physical_mass":float(p[m].sum()),
                 "sqerr_share":float(np.sum(p[m]*eps[m]*eps[m])/tot)})
for th in [0.2,0.4,0.6,0.8,1.0]:
    m=np.abs(r1)<th
    rows.append({"kind":"r1","threshold":th,"state_frac":float(m.mean()),
                 "physical_mass":float(p[m].sum()),
                 "sqerr_share":float(np.sum(p[m]*eps[m]*eps[m])/tot)})
out={"eps_rms_phys":float(np.sqrt(tot)),"rows":rows}
Path("../results/diag_e2_remainder_sparsity20.json").write_text(json.dumps(out,indent=2))
print("RESULT",json.dumps(out,sort_keys=True),flush=True)
