#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import scipy.sparse.linalg as sla

from finite_tau_matching_20site_exact import special_columns,D
from finite_tau_amplitude_regression_20site import endpoint,H
from finite_tau_shared_ratio_walkers20 import (
    MLP,feat,make_wd,load_model,amp_est,propagate_fn_uniform_importance,
    norm,pstep,ploss,predict_full,oracle_diagnostics,sampled_replay
)

ROOT=Path("/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure")
HO=H.copy(); HO.setdiag(0); HO.eliminate_zeros()

def load_shared():
    z=np.load(ROOT/"results/finite_tau_shared_ratio_walkers20_weights.npz")
    net=MLP(len(z["mu"]),64,1)
    for n in ("W1","b1","W2","b2","W3","b3"): getattr(net,n)[:]=z[n]
    return net,z["mu"],z["sd"]

def collect_meta(y,M,seed,K=4,dt=.05,beta=.5):
    bnet,bmu,bsd=load_model(); arr=endpoint(y); d0=arr[0]
    s=np.where(d0%2==0,1,-1).astype(np.int8)
    rg=np.random.default_rng(seed); walkers=np.full(M,y,np.int32)
    g=np.full(D,1/np.sqrt(D)); wd=make_wd(H,[y])[0]
    AA=[];BB=[];WW=[];TT=[];SS=[];ZZ=[]

    for n in range(1,int(round(beta/dt))+1):
        walkers=propagate_fn_uniform_importance(walkers,g,s,dt,rg)
        tau=n*dt; g=amp_est(walkers,y,tau,arr,bnet,bmu,bsd)
        src=[];dst=[];ww=[]
        for x in walkers:
            lo,hi=HO.indptr[int(x)],HO.indptr[int(x)+1]; nb=HO.indices[lo:hi]
            kk=min(K,len(nb)); pick=rg.choice(len(nb),size=kk,replace=False)
            src.extend([int(x)]*kk); dst.extend(nb[pick].tolist()); ww.extend([len(nb)/kk]*kk)
        src=np.asarray(src,np.int32); dst=np.asarray(dst,np.int32)
        AA.append(feat(src,y,wd,tau)); BB.append(feat(dst,y,wd,tau))
        WW.append(np.asarray(ww,float)); TT.append(np.full(len(src),tau))
        SS.append(src); ZZ.append(dst)
        print("COLLECT",seed,tau,len(src),flush=True)
    return (np.vstack(AA),np.vstack(BB),np.concatenate(WW),np.concatenate(TT),
            np.concatenate(SS),np.concatenate(ZZ),wd,s)

def margin_weights(net,mu,sd,y,wd,s,T,src,dst,base,dt=.05,lam=24.,sigma=.5):
    out=np.empty(len(base),float)
    for tau in np.unique(T):
        a=predict_full(net,mu,sd,y,wd,float(tau))
        q=1-dt*s*(H@(s*a))/np.maximum(a,1e-300)
        m=T==tau
        edge=np.minimum(np.abs(q[src[m]]),np.abs(q[dst[m]]))
        focus=1+lam*np.exp(-(edge/sigma)**2)
        out[m]=base[m]*focus
        print("FOCUS",float(tau),"mean",float(focus.mean()),
              "p95",float(np.quantile(focus,.95)),"max",float(focus.max()),flush=True)
    return out

def refine(net,mu,sd,A,B,W,VA,VB,VW,epochs=12,seed=31):
    A=(A-mu)/sd;B=(B-mu)/sd;VA=(VA-mu)/sd;VB=(VB-mu)/sd
    rg=np.random.default_rng(seed);bestv=1e99;best=None;hist=[]
    for ep in range(epochs):
        order=rg.permutation(len(A));ls=[]
        for q in range(0,len(order),4096):
            z=order[q:q+4096];ls.append(pstep(net,A[z],B[z],W[z],lr=5e-4))
        vv=ploss(net,VA,VB,VW);hist.append([ep+1,float(np.mean(ls)),vv])
        if vv<bestv: bestv=vv;best=[x.copy() for x in net.par()]
        if ep in (0,1,3,7,11): print("REFINE",hist[-1],flush=True)
    for dst,src in zip(net.par(),best):dst[:]=src
    return bestv,hist

def main():
    y=int(special_columns(np.random.default_rng(20260930),2)[1]);M=8192
    net,mu,sd=load_shared()
    A,B,W,T,S,Z,wd,s0=collect_meta(y,M,501001)
    VA,VB,VW,VT,VS,VZ,_,_=collect_meta(y,M,502001)
    MW=margin_weights(net,mu,sd,y,wd,s0,T,S,Z,W)
    MVW=margin_weights(net,mu,sd,y,wd,s0,VT,VS,VZ,VW)
    bestv,hist=refine(net,mu,sd,A,B,MW,VA,VB,MVW)

    diag=oracle_diagnostics(net,mu,sd,y,wd,s0)
    for r in diag:
        if r["tau"] in (.1,.25,.5):print("ORACLE",r,flush=True)
    replay=sampled_replay(net,mu,sd,y,M,503001)
    out={"y":y,"M":M,"best_margin_val_ploss":bestv,"history":hist,
         "oracle_diagnostics":diag,"sampled_replay":replay}
    path=ROOT/"results/finite_tau_margin_ratio_walkers20.json"
    path.write_text(json.dumps(out,indent=2))
    np.savez_compressed(ROOT/"results/finite_tau_margin_ratio_walkers20_weights.npz",
        mu=mu,sd=sd,W1=net.W1,b1=net.b1,W2=net.W2,b2=net.b2,W3=net.W3,b3=net.b3)
    print("WROTE",path,flush=True)

if __name__=="__main__":main()
