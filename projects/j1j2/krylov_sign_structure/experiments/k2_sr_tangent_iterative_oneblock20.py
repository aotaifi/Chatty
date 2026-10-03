#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np

from finite_tau_matching_20site_exact import D,special_columns
from finite_tau_ct_local_ratio_20site import H,distance_sign,normalized
from finite_tau_ct_smc_edgeratio_longbeta20 import reconstruct,exact_sequence
from finite_tau_localratio_mlp_oracle20 import make_wd
from k2_sr_tangent_recursive20 import sr_project

DT=.05
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--M",type=int,required=True)
    ap.add_argument("--ns",type=int,default=2000)
    ap.add_argument("--h",type=int,default=8)
    ap.add_argument("--iters",type=int,default=4)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    y=int(special_columns(np.random.default_rng(20260930),2)[1])
    wd=make_wd(H,[y])[0]; s=distance_sign(y)
    exact=exact_sequence(y,.55)
    _,_,logg=reconstruct(y,s,a.M)
    g0=normalized(np.exp(np.clip(logg-logg.max(),-700,0)))
    psi=s*g0; h1=H@psi
    psi2=normalized(psi-DT*h1+.5*DT*DT*(H@h1))
    amp2=np.abs(psi2); sn=np.where(psi2>=0,1.,-1.)
    g=g0.copy(); rows=[]
    for it in range(1,a.iters+1):
        targ=np.log(np.maximum(amp2,1e-300))-np.log(np.maximum(g,1e-300))
        f,sd=sr_project(g,targ,y,wd,.55,a.ns,a.h,1700001+a.M+100*it)
        g=normalized(g*np.exp(np.clip(f,-20,20)))
        row={"iter":it,"target_fid":float(np.dot(g,amp2)**2),
             "physical_full":float(np.dot(sn*g,exact[.55])**2),**sd}
        rows.append(row); print("ITER",json.dumps(row,sort_keys=True),flush=True)
    out={"M":a.M,"ns":a.ns,"h":a.h,
         "base_full":float(np.dot(psi,exact[.5])**2),
         "exact_k2_full":float(np.dot(psi2,exact[.55])**2),"rows":rows}
    Path(a.out).write_text(json.dumps(out,indent=2))
    print("FINAL",json.dumps(out,sort_keys=True),flush=True)
if __name__=="__main__": main()
