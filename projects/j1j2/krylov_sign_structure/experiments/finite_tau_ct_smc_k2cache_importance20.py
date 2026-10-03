#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np, scipy.sparse as sp

from finite_tau_matching_20site_exact import D,special_columns
from finite_tau_ct_local_ratio_20site import H,diag,R,C,HV,distance_sign,normalized,mismatch
from finite_tau_ct_smc_fn20 import ct_block,systematic
from finite_tau_ct_smc_edgeratio_longbeta20 import reconstruct,exact_sequence

DT=.05
HO=H.copy();HO.setdiag(0);HO.eliminate_zeros()

def expand(states,radius):
    mask=np.zeros(D,bool);mask[np.asarray(states,np.int64)]=True
    frontier=np.flatnonzero(mask)
    for _ in range(radius):
        add=[]
        for x in frontier:
            a,b=HO.indptr[int(x)],HO.indptr[int(x)+1]
            add.extend(HO.indices[a:b].tolist())
        if not add:break
        old=mask.copy();mask[np.asarray(add,np.int64)]=True
        frontier=np.flatnonzero(mask&~old)
    return np.flatnonzero(mask).astype(np.int32)

def imp_params(s,logg):
    rat=np.exp(np.clip(logg[C]-logg[R],-20,20))
    good=s[R]*s[C]<0;bad=~good
    rates=HV[good]*rat[good]
    lam=np.bincount(R[good],weights=rates,minlength=D)
    shift=np.bincount(R[bad],weights=HV[bad]*rat[bad],minlength=D)
    J=sp.coo_matrix((rates,(R[good],C[good])),shape=H.shape).tocsr()
    pot=lam-(diag+shift)
    return J,lam,pot

def reweight_states(st,delta_logq,rng):
    lw=np.clip(delta_logq[st],-50,50);lw-=lw.max();w=np.exp(lw)
    ess=float(w.sum()**2/np.dot(w,w))
    sel=systematic(w,rng,len(st))
    return st[sel],ess

def amp_from_imp(st,logg):
    h=np.bincount(st,minlength=D).astype(float)
    q=np.exp(np.clip(logg-logg.max(),-700,0))
    return normalized(h/np.maximum(q,1e-300))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--M",type=int,default=25000)
    ap.add_argument("--beta",type=float,default=3.0)
    ap.add_argument("--radius",type=int,default=1)
    ap.add_argument("--seed-offset",type=int,default=0)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    y=int(special_columns(np.random.default_rng(20260930),2)[1])
    s=distance_sign(y);exact=exact_sequence(y,a.beta)
    st1,st2,logg=reconstruct(y,s,a.M)
    rg1=np.random.default_rng(1510001+a.seed_offset)
    rg2=np.random.default_rng(1520001+a.seed_offset)
    # Convert uniform-importance amplitude walkers to f=q*a walkers.
    st1,ei1=reweight_states(st1,logg,rg1)
    st2,ei2=reweight_states(st2,logg,rg2)
    rows=[];t=.5
    while t<a.beta-1e-12:
        oldlogg=logg.copy()
        g=np.exp(np.clip(logg-logg.max(),-700,0))
        psi=s*g;h1=H@psi
        psi2=psi-DT*h1+.5*DT*DT*(H@h1);k2=normalized(psi2)

        J,lam,pot=imp_params(s,logg)
        st1,e1,u1=ct_block(st1,DT,J,lam,pot,rg1)
        st2,e2,u2=ct_block(st2,DT,J,lam,pot,rg2)
        t=round(t+DT,10)

        active=expand(np.concatenate([st1,st2]),a.radius)
        gn=g.copy();sn=s.copy()
        gn[active]=np.maximum(np.abs(psi2[active]),1e-300)
        sn[active]=np.where(psi2[active]>=0,1.0,-1.0)
        loggn=np.log(np.maximum(gn,1e-300));loggn-=loggn.max()

        # Change importance function q_old -> q_new before measuring/next block.
        delta=loggn-oldlogg
        st1,er1=reweight_states(st1,delta,rg1)
        st2,er2=reweight_states(st2,delta,rg2)
        logg=loggn;s=sn

        a1=amp_from_imp(st1,logg);a2=amp_from_imp(st2,logg);ae=normalized(a1+a2)
        p1=np.bincount(st1,minlength=D).astype(float); p1/=p1.sum()
        p2=np.bincount(st2,minlength=D).astype(float); p2/=p2.sum()
        ph=.5*(p1+p2)
        gg=normalized(gn);cache=normalized(s*gn);ex=exact[t]; pex=ex*ex
        pguide=gg*gg
        def bc2(p,q): return float(np.sum(np.sqrt(np.maximum(p,0)*np.maximum(q,0)))**2)
        row={"tau":t,"radius":a.radius,"active":int(len(active)),
             "physical_mass_active":float(np.sum(ex[active]**2)),
             "k2_target_full":float(np.dot(k2,ex)**2),
             "cache_to_k2":float(np.dot(cache,k2)**2),
             "guide_full":float(np.dot(cache,ex)**2),
             "guide_sign":mismatch(s,ex),
             "guide_prob_bc2":bc2(pguide,pex),
             "walker_prob_bc2":bc2(ph,pex),
             "walker_prob_tv":float(.5*np.sum(np.abs(ph-pex))),
             "walker_prob_cross_bc2":bc2(p1,p2),
             "walker_full":float(np.dot(s*ae,ex)**2),
             "walker_amp":float(np.dot(ae,np.abs(ex))**2),
             "walker_cross":float(np.dot(a1,a2)**2),
             "path_ess":[e1,e2],"reweight_ess":[er1,er2],
             "unique":[int(np.unique(st1).size),int(np.unique(st2).size)]}
        rows.append(row)
        if t in (.55,1.0,1.5,2.0,2.5,3.0,a.beta):
            print("K2IMP",json.dumps(row,sort_keys=True),flush=True)
    Path(a.out).write_text(json.dumps({"M":a.M,"beta":a.beta,"radius":a.radius,
        "init_reweight_ess":[ei1,ei2],"rows":rows},indent=2))
    print("FINAL",json.dumps(rows[-1],sort_keys=True),flush=True)
if __name__=="__main__":main()
