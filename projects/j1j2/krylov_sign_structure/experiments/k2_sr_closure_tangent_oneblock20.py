#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np

from finite_tau_matching_20site_exact import D,special_columns
from finite_tau_ct_local_ratio_20site import H,distance_sign,normalized
from finite_tau_ct_smc_edgeratio_longbeta20 import reconstruct
from finite_tau_localratio_mlp_oracle20 import feat,make_wd
from k2_sr_tangent_oneblock20 import TinyMLP
from k2_sr_edge_tangent_oneblock20 import edge_set,jac_for,wrmse

def wmean(x,w):
    return float(np.sum(w*x)/np.sum(w))

def wrms(x,w):
    return float(np.sqrt(np.sum(w*x*x)/np.sum(w)))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--M",type=int,default=100000)
    ap.add_argument("--dt",type=float,default=.0125)
    ap.add_argument("--ns",type=int,default=2000)
    ap.add_argument("--h",type=int,default=8)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()

    y=int(special_columns(np.random.default_rng(20260930),2)[1]); wd=make_wd(H,[y])[0]
    s=distance_sign(y); _,_,logg=reconstruct(y,s,a.M)
    g=normalized(np.exp(np.clip(logg-logg.max(),-700,0))); psi=s*g
    h1=H@psi; psi2=normalized(psi-a.dt*h1+.5*a.dt*a.dt*(H@h1)); amp2=np.abs(psi2)
    targ=np.log(np.maximum(amp2,1e-300))-np.log(np.maximum(g,1e-300))
    tau=.5+a.dt; p=g*g; p/=p.sum(); rg=np.random.default_rng(2100001+a.M+int(10000*a.dt))

    trx=rg.choice(D,a.ns,p=p); vax=rg.choice(D,a.ns,p=p)
    trsrc,trdst,trw=edge_set(trx)
    vasrc,vadst,vaw=edge_set(vax)

    # Aggregate the one-hop shell with physically induced path weights.
    # Source occurrences carry unit sample weight; neighbor occurrences carry |H_xy|.
    wc=np.bincount(trx,minlength=D).astype(float)
    edge_scale0=max(float(np.mean(trw)),1e-12)
    wc += np.bincount(trdst,weights=trw/edge_scale0,minlength=D)
    centers=np.flatnonzero(wc>0).astype(np.int32)
    cweights=wc[centers]

    # Two-hop edges out of the expanded center set, weighted by how strongly the
    # center was reached from the original g^2 sample and by |H_yz|.
    csrc,cdst,cw0=edge_set(centers)
    cweight_map=wc[csrc]
    cew=cweight_map*cw0

    # Feature normalization uses exactly the states touched by the closure equations.
    ids=np.concatenate([centers,cdst])
    X0=feat(ids,y,wd,tau)
    mu=X0.mean(0); sd=X0.std(0); sd[sd<1e-6]=1
    net=TinyMLP(X0.shape[1],a.h,123)

    # Weighted state residual equations on x plus one-hop y closure.
    J=jac_for(net,centers,y,wd,tau,mu,sd)
    tc=targ[centers]
    jm=np.sum(cweights[:,None]*J,axis=0)/np.sum(cweights)
    tm=wmean(tc,cweights)
    Jc=J-jm; tc=tc-tm
    sws=np.sqrt(cweights/max(float(np.mean(cweights)),1e-12))
    Jcw=Jc*sws[:,None]; tcw=tc*sws
    state_var=max(wmean(tc*tc,cweights),1e-12)

    # Edge-ratio equations on y->z, i.e. the second hop required by K2.
    Jy=jac_for(net,csrc,y,wd,tau,mu,sd)
    Jz=jac_for(net,cdst,y,wd,tau,mu,sd)
    A=Jz-Jy
    et=targ[cdst]-targ[csrc]
    ewn=cew/max(float(np.mean(cew)),1e-12)
    swe=np.sqrt(ewn)
    Aw=A*swe[:,None]; etw=et*swe
    edge_var=max(float(np.sum(cew*et*et)/np.sum(cew)),1e-12)

    S=(Jcw.T@Jcw)/(len(Jcw)*state_var)
    b=(Jcw.T@tcw)/(len(Jcw)*state_var)
    S+=(Aw.T@Aw)/(len(Aw)*edge_var)
    b+=(Aw.T@etw)/(len(Aw)*edge_var)

    scale=max(float(np.trace(S)/len(S)),1e-14)
    ev,U=np.linalg.eigh(S); ub=U.T@b

    # Independent validation: original g^2 states and their one-hop Hamiltonian edges.
    JV=jac_for(net,vax,y,wd,tau,mu,sd)
    tv=targ[vax]
    vmean=tv.mean(); JVc=JV-JV.mean(0,keepdims=True); tvc=tv-vmean
    Vx=jac_for(net,vasrc,y,wd,tau,mu,sd)
    Vy=jac_for(net,vadst,y,wd,tau,mu,sd)
    VA=Vy-Vx; vet=targ[vadst]-targ[vasrc]

    state_base=max(float(np.mean(tvc*tvc)),1e-12)
    edge_base=max(float(np.sum(vaw*vet*vet)/np.sum(vaw)),1e-12)
    lams=scale*np.array([1e-5,1e-4,1e-3,1e-2,1e-1,1.,10.])
    scores=[]; sols=[]; parts=[]
    for lam in lams:
        sol=U@(ub/(ev+lam))
        sr=float(np.mean((JVc@sol-tvc)**2))
        er=float(np.sum(vaw*(VA@sol-vet)**2)/np.sum(vaw))
        scores.append(sr/state_base+er/edge_base)
        sols.append(sol); parts.append((sr,er))
    k=int(np.argmin(scores)); sol=sols[k]; sr,er=parts[k]

    allf=[]
    for q in range(0,D,4096):
        ix=np.arange(q,min(D,q+4096))
        allf.append(net.jac((feat(ix,y,wd,tau)-mu)/sd)@sol)
    f=np.concatenate(allf); f-=np.sum(p*f)
    gp=normalized(g*np.exp(np.clip(f,-20,20)))

    touched=np.zeros(D,dtype=bool)
    touched[np.unique(np.concatenate([centers,cdst]))]=True
    pred=VA@sol; err=pred-vet; seen=touched[vadst]

    out={"M":a.M,"dt":a.dt,"ns":a.ns,"h":a.h,
         "closure1_unique":int(len(centers)),
         "closure2_unique":int(touched.sum()),
         "closure2_frac_D":float(touched.mean()),
         "train_edges2":int(len(csrc)),
         "lambda":float(lams[k]),"validation_score":float(scores[k]),
         "state_rmse_base":float(np.sqrt(state_base)),
         "state_rmse_proj":float(np.sqrt(sr)),
         "edge_rmse_base":float(np.sqrt(edge_base)),
         "edge_rmse_proj":float(np.sqrt(er)),
         "edge_seenY_frac":float(seen.mean()),
         "edge_base_seenY":wrmse(vet[seen],vaw[seen]) if np.any(seen) else None,
         "edge_proj_seenY":wrmse(err[seen],vaw[seen]) if np.any(seen) else None,
         "edge_base_unseenY":wrmse(vet[~seen],vaw[~seen]) if np.any(~seen) else None,
         "edge_proj_unseenY":wrmse(err[~seen],vaw[~seen]) if np.any(~seen) else None,
         "target_fid":float((gp@amp2)**2),
         "f_rms_p":float(np.sqrt(np.sum(p*f*f))),
         "f_absmax":float(np.max(np.abs(f)))}
    print(json.dumps(out,sort_keys=True),flush=True)
    Path(a.out).write_text(json.dumps(out,indent=2))

if __name__=="__main__": main()
