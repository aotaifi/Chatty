#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np

from finite_tau_matching_20site_exact import D,special_columns
from finite_tau_ct_local_ratio_20site import H,distance_sign,normalized,mismatch
from finite_tau_ct_smc_edgeratio_longbeta20 import ratio_params,k1_ratio,reconstruct,exact_sequence
from finite_tau_ct_smc_pathratio_longbeta20 import ct_block_stats
from finite_tau_ct_smc_longbeta_scaling20 import cv0

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--M",type=int,default=25000)
    ap.add_argument("--beta",type=float,default=3.0)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    y=int(special_columns(np.random.default_rng(20260930),2)[1])
    s=distance_sign(y);exact=exact_sequence(y,a.beta)
    st1,st2,logg=reconstruct(y,s,a.M)
    rg1=np.random.default_rng(1310001);rg2=np.random.default_rng(1320001)
    rows=[];t=.5
    while t<a.beta-1e-12:
        J,lam,pot=ratio_params(s,logg)
        st1,e1,u1,end1,dw1=ct_block_stats(st1,.05,J,lam,pot,rg1)
        st2,e2,u2,end2,dw2=ct_block_stats(st2,.05,J,lam,pot,rg2)
        t=round(t+.05,10)
        g=np.exp(np.clip(logg-logg.max(),-60,0));g=normalized(g)
        gn,eta,_=cv0(g,end1,end2)
        loggn=np.log(np.maximum(gn,1e-300));loggn-=loggn.max()
        h1=np.bincount(st1,minlength=D).astype(float);h2=np.bincount(st2,minlength=D).astype(float)
        q1=normalized(h1);q2=normalized(h2);q=normalized(h1+h2);ex=exact[t]
        row={"tau":t,"full":float(np.dot(s*q,ex)**2),"amp":float(np.dot(q,np.abs(ex))**2),
             "sign":mismatch(s,ex),"cross":float(np.dot(q1,q2)**2),
             "ess":[e1,e2],"unique":[u1,u2],"eta":eta}
        if t<a.beta-1e-12:
            sn=k1_ratio(s,loggn);row["next_sign"]=mismatch(sn,exact[round(t+.05,10)]);s=sn
        logg=loggn;rows.append(row)
        if abs((t*20)%5)<1e-9 or t in (.55,a.beta):
            print("WEIGHTEDDENSE",json.dumps(row,sort_keys=True),flush=True)
    Path(a.out).write_text(json.dumps({"y":y,"M":a.M,"beta":a.beta,"rows":rows},indent=2))
    print("SUMMARY",json.dumps(rows[-1],sort_keys=True),flush=True)
if __name__=="__main__":main()
