#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np
from finite_tau_matching_20site_exact import D,special_columns
from finite_tau_ct_local_ratio_20site import H,diag,R,C,HV,distance_sign,normalized,mismatch
from finite_tau_ct_smc_edgeratio_longbeta20 import reconstruct,exact_sequence

DT=.05

def k1_field(s,logg):
    rat=np.exp(np.clip(logg[C]-logg[R],-20,20))
    off=np.bincount(R,weights=HV*s[C]*rat,minlength=D)
    return s*(1-DT*diag)-DT*off

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--M",type=int,default=25000)
    ap.add_argument("--beta",type=float,default=3.0)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    y=int(special_columns(np.random.default_rng(20260930),2)[1])
    s=distance_sign(y)
    exact=exact_sequence(y,a.beta)
    _,_,logg=reconstruct(y,s,a.M)
    rows=[];t=.5
    while True:
        g=normalized(np.exp(np.clip(logg-logg.max(),-700,0)))
        ex=exact[t]
        rows.append({"tau":t,
                     "full":float(np.dot(s*g,ex)**2),
                     "amp":float(np.dot(g,np.abs(ex))**2),
                     "sign":mismatch(s,ex)})
        if t>=a.beta-1e-12: break
        field=k1_field(s,logg)
        s=np.where(field>=0,1.0,-1.0)
        logg=logg+np.log(np.maximum(np.abs(field),1e-300))
        logg-=logg.max()
        t=round(t+DT,10)
    print("INIT",json.dumps(rows[0],sort_keys=True),flush=True)
    for q in rows:
        if q["tau"] in (1.0,1.5,2.0,2.5,3.0):
            print("K1AMP",json.dumps(q,sort_keys=True),flush=True)
    print("FINAL",json.dumps(rows[-1],sort_keys=True),flush=True)
    Path(a.out).write_text(json.dumps({"M":a.M,"beta":a.beta,"rows":rows},indent=2))
if __name__=="__main__":main()
