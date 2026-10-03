#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np

from finite_tau_matching_20site_exact import special_columns,D
from finite_tau_amplitude_regression_20site import H
from finite_tau_localratio_mlp_oracle20 import feat,make_wd
from finite_tau_weighted_smc_20site import (
    HO,labels_for_y,propagate_weighted,update_theta_weighted,systematic,ess
)
from finite_tau_margin_ratio_walkers20 import load_shared,margin_weights,refine
from finite_tau_shared_ratio_walkers20 import oracle_diagnostics,sampled_replay

ROOT=Path("/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure")

def collect_smc(y,M,seed,K=4,block=.05,beta=.5,ess_frac=.35):
    rg=np.random.default_rng(seed);d,lab,bs=labels_for_y(y)
    s=np.where(d%2==0,1,-1).astype(np.int8)
    theta=np.zeros(len(bs)); states=np.full(M,y,np.int32);pw=np.ones(M)
    wd=make_wd(H,[y])[0];AA=[];BB=[];WW=[];TT=[];SS=[];ZZ=[]
    for n in range(1,int(round(beta/block))+1):
        tau=n*block;old=theta.copy()
        states,pw,nr=propagate_weighted(states,pw,old,lab,s,block,.005,rg,ess_frac)
        aw=pw*np.exp(np.clip(-old[lab[states]],-40,40));aw*=M/aw.sum()
        src=[];dst=[];ww=[]

        for x,w0 in zip(states,aw):
            lo,hi=HO.indptr[int(x)],HO.indptr[int(x)+1];nb=HO.indices[lo:hi]
            kk=min(K,len(nb));pick=rg.choice(len(nb),size=kk,replace=False);fac=len(nb)/kk
            src.extend([int(x)]*kk);dst.extend(nb[pick].tolist());ww.extend([float(w0*fac)]*kk)
        src=np.asarray(src,np.int32);dst=np.asarray(dst,np.int32)
        AA.append(feat(src,y,wd,tau));BB.append(feat(dst,y,wd,tau));WW.append(np.asarray(ww,float))
        TT.append(np.full(len(src),tau));SS.append(src);ZZ.append(dst)
        new,_=update_theta_weighted(old,lab,bs,states,pw,.5,32.)
        pw*=np.exp(np.clip(new[lab[states]]-old[lab[states]],-30,30));pw*=M/pw.sum();theta=new
        if ess(pw)<ess_frac*M:
            states=states[systematic(pw,rg,M)];pw=np.ones(M);nr+=1
        print("SMC",seed,tau,"ESS",round(ess(pw),1),"unique",len(np.unique(states)),"resamp",nr,flush=True)
    return (np.vstack(AA),np.vstack(BB),np.concatenate(WW),np.concatenate(TT),
            np.concatenate(SS),np.concatenate(ZZ),wd,s)

def main():
    y=int(special_columns(np.random.default_rng(20260930),2)[1]);M=8192
    net,mu,sd=load_shared()
    A,B,W,T,S,Z,wd,s0=collect_smc(y,M,801001)
    VA,VB,VW,VT,VS,VZ,_,_=collect_smc(y,M,802001)
    MW=margin_weights(net,mu,sd,y,wd,s0,T,S,Z,W,lam=100.,sigma=.5)
    MVW=margin_weights(net,mu,sd,y,wd,s0,VT,VS,VZ,VW,lam=100.,sigma=.5)
    bestv,hist=refine(net,mu,sd,A,B,MW,VA,VB,MVW,epochs=12,seed=61)
    diag=oracle_diagnostics(net,mu,sd,y,wd,s0)
    for r in diag:
        if r["tau"] in (.1,.25,.5):print("ORACLE",r,flush=True)
    replay=sampled_replay(net,mu,sd,y,M,803001)
    out={"y":y,"M":M,"best_val":bestv,"history":hist,
         "oracle_diagnostics":diag,"sampled_replay":replay}
    path=ROOT/"results/finite_tau_smc_margin_ratio20.json"
    path.write_text(json.dumps(out,indent=2));print("WROTE",path,flush=True)

if __name__=="__main__":main()
