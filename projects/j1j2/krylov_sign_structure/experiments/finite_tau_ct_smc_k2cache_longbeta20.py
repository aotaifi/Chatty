#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np

from finite_tau_matching_20site_exact import D,special_columns
from finite_tau_ct_local_ratio_20site import H,distance_sign,normalized,mismatch
from finite_tau_ct_smc_fn20 import ct_block
from finite_tau_ct_smc_edgeratio_longbeta20 import ratio_params,reconstruct,exact_sequence

DT=.05
HO=H.copy(); HO.setdiag(0); HO.eliminate_zeros()

def expand(states,radius):
    mask=np.zeros(D,bool); mask[np.asarray(states,np.int64)]=True
    frontier=np.flatnonzero(mask)
    for _ in range(radius):
        add=[]
        for x in frontier:
            a,b=HO.indptr[int(x)],HO.indptr[int(x)+1]
            add.extend(HO.indices[a:b].tolist())
        if not add: break
        old=mask.copy()
        mask[np.asarray(add,np.int64)]=True
        frontier=np.flatnonzero(mask & ~old)
    return np.flatnonzero(mask).astype(np.int32)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--M",type=int,default=25000)
    ap.add_argument("--beta",type=float,default=3.0)
    ap.add_argument("--radius",type=int,default=0)
    ap.add_argument("--seed-offset",type=int,default=0)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    y=int(special_columns(np.random.default_rng(20260930),2)[1])
    s=distance_sign(y); exact=exact_sequence(y,a.beta)
    st1,st2,logg=reconstruct(y,s,a.M)
    rg1=np.random.default_rng(1410001+a.seed_offset)
    rg2=np.random.default_rng(1420001+a.seed_offset)
    rows=[];t=.5
    while t<a.beta-1e-12:
        g=np.exp(np.clip(logg-logg.max(),-700,0))
        psi=s*g; h1=H@psi
        psi2=psi-DT*h1+.5*DT*DT*(H@h1)
        k2=normalized(psi2)

        J,lam,pot=ratio_params(s,logg)
        st1,e1,u1=ct_block(st1,DT,J,lam,pot,rg1)
        st2,e2,u2=ct_block(st2,DT,J,lam,pot,rg2)
        t=round(t+DT,10)

        active=expand(np.concatenate([st1,st2]),a.radius)
        gn=g.copy(); sn=s.copy()
        gn[active]=np.maximum(np.abs(psi2[active]),1e-300)
        sn[active]=np.where(psi2[active]>=0,1.0,-1.0)
        logg=np.log(np.maximum(gn,1e-300)); logg-=logg.max(); s=sn

        gg=normalized(gn); cache=normalized(s*gn)
        q1=normalized(np.bincount(st1,minlength=D).astype(float))
        q2=normalized(np.bincount(st2,minlength=D).astype(float))
        q=normalized(q1+q2); ex=exact[t]
        row={"tau":t,"radius":a.radius,"active":int(len(active)),
             "physical_mass_active":float(np.sum(ex[active]**2)),
             "k2_mass_active":float(np.sum(k2[active]**2)),
             "k2_target_full":float(np.dot(k2,ex)**2),
             "cache_to_k2":float(np.dot(cache,k2)**2),
             "guide_full":float(np.dot(cache,ex)**2),
             "guide_amp":float(np.dot(gg,np.abs(ex))**2),
             "guide_sign":mismatch(s,ex),
             "walker_full":float(np.dot(s*q,ex)**2),
             "walker_amp":float(np.dot(q,np.abs(ex))**2),
             "cross":float(np.dot(q1,q2)**2),
             "ess":[e1,e2],"unique":[u1,u2]}
        rows.append(row)
        if t in (.55,1.0,1.5,2.0,2.5,3.0,a.beta):
            print("K2CACHE",json.dumps(row,sort_keys=True),flush=True)
    Path(a.out).write_text(json.dumps({"M":a.M,"beta":a.beta,"radius":a.radius,
                                       "rows":rows},indent=2))
    print("FINAL",json.dumps(rows[-1],sort_keys=True),flush=True)
if __name__=="__main__":main()
