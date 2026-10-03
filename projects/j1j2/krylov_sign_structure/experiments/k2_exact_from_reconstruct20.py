#!/usr/bin/env python3
import argparse,json,numpy as np
from pathlib import Path
from finite_tau_matching_20site_exact import D,special_columns
from finite_tau_ct_local_ratio_20site import H,distance_sign,normalized,mismatch
from finite_tau_ct_smc_edgeratio_longbeta20 import reconstruct,exact_sequence

DT=.05
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--M",type=int,required=True)
    ap.add_argument("--beta",type=float,default=3.0)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    y=int(special_columns(np.random.default_rng(20260930),2)[1])
    exact=exact_sequence(y,a.beta)
    s=distance_sign(y)
    _,_,logg=reconstruct(y,s,a.M)
    g=normalized(np.exp(np.clip(logg-logg.max(),-700,0)))
    psi=s*g
    rows=[]; t=.5
    while True:
        ex=exact[t]
        rows.append({"tau":t,"full":float(np.dot(psi,ex)**2),
                     "amp":float(np.dot(np.abs(psi),np.abs(ex))**2),
                     "sign":mismatch(np.where(psi>=0,1.,-1.),ex)})
        if t>=a.beta-1e-12: break
        h1=H@psi
        psi=normalized(psi-DT*h1+.5*DT*DT*(H@h1))
        t=round(t+DT,10)
    for r in rows:
        if r["tau"] in (.5,1.,1.5,2.,2.5,3.):
            print("K2EX",json.dumps(r,sort_keys=True),flush=True)
    Path(a.out).write_text(json.dumps({"M":a.M,"rows":rows},indent=2))
if __name__=="__main__": main()
