#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np
import scipy.sparse.linalg as sla

from finite_tau_matching_20site_exact import D,special_columns
from finite_tau_ct_local_ratio_20site import H,distance_sign,normalized
from finite_tau_ct_smc_edgeratio_longbeta20 import reconstruct
from finite_tau_localratio_mlp_oracle20 import feat,make_wd
from k2_sr_tangent_recursive20 import sr_project

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--M",type=int,required=True)
    ap.add_argument("--dt",type=float,required=True)
    ap.add_argument("--ns",type=int,default=2000)
    ap.add_argument("--h",type=int,default=8)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    y=int(special_columns(np.random.default_rng(20260930),2)[1])
    wd=make_wd(H,[y])[0]; s=distance_sign(y)
    _,_,logg=reconstruct(y,s,a.M)
    g=normalized(np.exp(np.clip(logg-logg.max(),-700,0))); psi=s*g
    h1=H@psi
    psi2=normalized(psi-a.dt*h1+.5*a.dt*a.dt*(H@h1))
    targ=np.log(np.maximum(np.abs(psi2),1e-300))-np.log(np.maximum(g,1e-300))
    f,sd=sr_project(g,targ,y,wd,.5+a.dt,a.ns,a.h,
                    1800001+a.M+int(round(10000*a.dt)))
    gn=normalized(g*np.exp(np.clip(f,-20,20)))
    sn=np.where(psi2>=0,1.0,-1.0)
    e=np.zeros(D);e[y]=1.
    ex05=normalized(sla.expm_multiply(-.5*H,e))
    ex=normalized(sla.expm_multiply(-a.dt*H,ex05))
    row={"M":a.M,"dt":a.dt,
         "base_full":float(np.dot(psi,ex05)**2),
         "exact_k2_full":float(np.dot(psi2,ex)**2),
         "projected_full":float(np.dot(sn*gn,ex)**2),
         "target_fid":float(np.dot(gn,np.abs(psi2))**2),
         "target_infid":float(1-np.dot(gn,np.abs(psi2))**2),**sd}
    print(json.dumps(row,sort_keys=True),flush=True)
    Path(a.out).write_text(json.dumps(row,indent=2))
if __name__=="__main__":main()
