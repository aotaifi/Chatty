#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import scipy.sparse.linalg as sla
from finite_tau_matching_20site_exact import D,special_columns
from finite_tau_ct_local_ratio_20site import H,distance_sign,normalized
from finite_tau_ct_smc_edgeratio_longbeta20 import reconstruct

dt=.0125
y=int(special_columns(np.random.default_rng(20260930),2)[1])
s=distance_sign(y)
_,_,logg=reconstruct(y,s,100000)
g=normalized(np.exp(np.clip(logg-logg.max(),-700,0)))
psi=s*g
h1=H@psi; h2=H@h1
e1=h1/psi; e2=h2/psi
r1=1-dt*e1
r2=r1+.5*dt*dt*e2
f1=np.log(np.maximum(np.abs(r1),1e-300))
f2=np.log(np.maximum(np.abs(r2),1e-300))
flin=-dt*e1

e=np.zeros(D);e[y]=1
exact=normalized(sla.expm_multiply(-.5*H,e))
pp=exact*exact; pp/=pp.sum()
HO=H.copy();HO.setdiag(0);HO.eliminate_zeros()
rg=np.random.default_rng(12345)
xs=rg.choice(D,10000,p=pp)
src=[];dst=[];ww=[]
for x in xs:
    lo,hi=HO.indptr[x],HO.indptr[x+1]
    js=HO.indices[lo:hi];hs=np.abs(HO.data[lo:hi])
    src.extend([int(x)]*len(js));dst.extend(js.tolist());ww.extend(hs.tolist())
src=np.asarray(src);dst=np.asarray(dst);ww=np.asarray(ww,float)
target=f2[dst]-f2[src]
def wrms(z): return float(np.sqrt(np.sum(ww*z*z)/np.sum(ww)))
out={
 "nedge":int(len(src)),
 "target_rmse":wrms(target),
 "linear_E1_rmse":wrms((flin[dst]-flin[src])-target),
 "log_R1_rmse":wrms((f1[dst]-f1[src])-target),
 "exact_R2_selfcheck":wrms((f2[dst]-f2[src])-target),
 "E1_abs_median":float(np.median(np.abs(e1[xs]))),
 "E2_abs_median":float(np.median(np.abs(e2[xs]))),
 "R2_minus_R1_rms_state_phys":float(np.sqrt(np.sum(pp*(f2-f1)**2)))
}
Path("../results/diag_k2_energy_features20.json").write_text(json.dumps(out,indent=2))
print("RESULT",json.dumps(out,sort_keys=True),flush=True)
