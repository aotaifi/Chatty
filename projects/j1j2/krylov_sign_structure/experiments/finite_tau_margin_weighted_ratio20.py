#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import scipy.sparse.linalg as sla

from finite_tau_matching_20site_exact import special_columns,D
from finite_tau_amplitude_regression_20site import endpoint,H
from finite_tau_learned_guide_gfmc_20site import load_model,guide_vec,propagate_fn_uniform_importance
from finite_tau_localratio_mlp_oracle20 import MLP,feat,make_wd,mm
from finite_tau_shared_ratio_walkers20 import norm,pstep,ploss,predict_full

ROOT=Path("/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure")
HO=H.copy(); HO.setdiag(0); HO.eliminate_zeros()

def load_shared():
    z=np.load(ROOT/"results/finite_tau_shared_ratio_walkers20_weights.npz")
    net=MLP(len(z["mu"]),64,17)
    net.W1[:]=z["W1"];net.b1[:]=z["b1"];net.W2[:]=z["W2"];net.b2[:]=z["b2"];net.W3[:]=z["W3"];net.b3[:]=z["b3"]
    return net,z["mu"],z["sd"]

def amp_est(w,y,tau,arr,net,mu,sd,prior=512.):
    _,_,_,_,_,vals,rlab,rbs=arr
    mass=np.bincount(rlab[w],minlength=len(vals)).astype(float)
    gm=guide_vec(net,mu,sd,y,tau,arr)
    pm=np.bincount(rlab,weights=gm,minlength=len(vals));pm/=pm.sum()
    return norm(np.maximum(((mass+prior*pm)/rbs)[rlab],1e-14))

def collect(y,M,seed,base_net,bmu,bsd,K=4,dt=.05,beta=.5):
    prior_net,pmu,psd=load_model(); arr=endpoint(y); d0=arr[0]
    s=np.where(d0%2==0,1,-1).astype(np.int8)
    rg=np.random.default_rng(seed); walkers=np.full(M,y,np.int32)
    g=np.full(D,1/np.sqrt(D)); wd=make_wd(H,[y])[0]
    AA=[];BB=[];WW=[];TT=[];MM=[]
    for n in range(1,int(round(beta/dt))+1):
        walkers=propagate_fn_uniform_importance(walkers,g,s,dt,rg)
        tau=n*dt; g=amp_est(walkers,y,tau,arr,prior_net,pmu,psd)

        # sampled proxy margin from current ratio model; no oracle state used
        ah=predict_full(base_net,bmu,bsd,y,wd,tau)
        phi=s*ah; marg=phi-dt*(H@phi)
        scale=np.abs(phi)+dt*np.asarray(np.abs(H)@ah)
        rel=np.abs(marg)/np.maximum(scale,1e-30)

        src=[];dst=[];ww=[];mm=[]
        for x in walkers:
            lo,hi=HO.indptr[int(x)],HO.indptr[int(x)+1]; nb=HO.indices[lo:hi]
            kk=min(K,len(nb)); pick=rg.choice(len(nb),size=kk,replace=False)
            fac=len(nb)/kk
            src.extend([int(x)]*kk); dst.extend(nb[pick].tolist())
            ww.extend([fac]*kk); mm.extend([rel[int(x)]]*kk)
        src=np.asarray(src,np.int32);dst=np.asarray(dst,np.int32)
        AA.append(feat(src,y,wd,tau));BB.append(feat(dst,y,wd,tau))
        WW.append(np.asarray(ww,float));MM.append(np.asarray(mm,float));TT.append(np.full(len(src),tau))
        print("COLLECT",seed,tau,"proxy_q10",float(np.quantile(rel[walkers],.1)),flush=True)
    return np.vstack(AA),np.vstack(BB),np.concatenate(WW),np.concatenate(MM),np.concatenate(TT),wd,s

def weighted_train(A,B,W,MARG,VA,VB,VW,VMARG,epochs=20,seed=31,alpha=8.0,c=.12):
    X=np.vstack([A,B]);mu=X.mean(0);sd=X.std(0);sd[sd<1e-8]=1
    A=(A-mu)/sd;B=(B-mu)/sd;VA=(VA-mu)/sd;VB=(VB-mu)/sd
    # bounded emphasis: easy states weight~1, low-margin states up to 1+alpha
    boost=1+alpha*np.exp(-(MARG/c)**2)
    vboost=1+alpha*np.exp(-(VMARG/c)**2)
    TW=W*boost;VW2=VW*vboost
    net=MLP(A.shape[1],64,seed);rg=np.random.default_rng(seed)
    bestv=1e99;best=None;hist=[]
    for ep in range(epochs):
        order=rg.permutation(len(A));ls=[]
        for q in range(0,len(order),4096):
            z=order[q:q+4096];ls.append(pstep(net,A[z],B[z],TW[z]))
        vv=ploss(net,VA,VB,VW2);hist.append([ep+1,float(np.mean(ls)),vv])
        if vv<bestv: bestv=vv;best=[x.copy() for x in net.par()]
        if ep in (0,1,4,9,14,19): print("TRAIN",hist[-1],flush=True)
    for dst,src in zip(net.par(),best):dst[:]=src
    return net,mu,sd,bestv,hist

def exact_state_sign(y,tau,dt=.05):
    wd=make_wd(H,[y])[0]; d=(wd//1000).astype(np.int64)
    s=np.where(d%2==0,1,-1); psi=np.zeros(D);psi[y]=1.
    for n in range(1,int(round(tau/dt))+1):
        psi=norm(sla.expm_multiply(-dt*H,psi))
        if n<int(round(tau/dt)):
            a=np.abs(psi); s=np.where((s*a-dt*(H@(s*a)))>=0,1,-1)
    return psi,s,wd

def critical_rmse(net,mu,sd,y,tau,frac=.01,dt=.05):
    psi,s,wd=exact_state_sign(y,tau,dt);a=np.abs(psi)
    lp=[] 
    for q in range(0,D,4096):
        ix=np.arange(q,min(q+4096,D));lp.append(net.pred((feat(ix,y,wd,tau)-mu)/sd))
    lp=np.concatenate(lp)
    phi=s*a; margin=phi-dt*(H@phi)
    scale=np.abs(phi)+dt*np.asarray(np.abs(H)@a)
    rel=np.abs(margin)/np.maximum(scale,1e-30)
    active=np.flatnonzero((a*a)>1e-12)
    cut=np.quantile(rel[active],frac); crit=rel<=cut

    co=HO.tocoo();src=co.row.astype(np.int32);dst=co.col.astype(np.int32)
    mask=crit[src]&(a[src]>1e-15)&(a[dst]>1e-15)
    exact_lr=np.log(a[dst[mask]])-np.log(a[src[mask]])
    pred_lr=lp[dst[mask]]-lp[src[mask]]
    ww=a[src[mask]]**2
    return float(np.sqrt(np.average((pred_lr-exact_lr)**2,weights=ww))),float(cut)

def sampled_replay(net,mu,sd,y,M,seed,dt=.05,beta=.5):
    prior_net,pmu,psd=load_model();arr=endpoint(y);d0=arr[0]
    s=np.where(d0%2==0,1,-1).astype(np.int8)
    rg=np.random.default_rng(seed);walkers=np.full(M,y,np.int32)
    g=np.full(D,1/np.sqrt(D));wd=make_wd(H,[y])[0]
    exact=np.zeros(D);exact[y]=1.;rows=[]
    for n in range(1,int(round(beta/dt))+1):
        walkers=propagate_fn_uniform_importance(walkers,g,s,dt,rg)
        exact=norm(sla.expm_multiply(-dt*H,exact));tau=n*dt
        g=amp_est(walkers,y,tau,arr,prior_net,pmu,psd)
        ah=predict_full(net,mu,sd,y,wd,tau)
        pred=s*ah-dt*(H@(s*ah));sn=np.where(pred>=0,1,-1).astype(np.int8)
        exn=norm(sla.expm_multiply(-dt*H,exact))
        row={"tau":tau,"sign_err_current":mm(s,exact),
             "carry_next":mm(s,exn),"k1_next":mm(sn,exn),
             "ratio_ampfid":float(np.dot(ah,np.abs(exact))**2)}
        rows.append(row)
        if tau in (.25,.4,.5):print("REPLAY",row,flush=True)
        s=sn
    return rows

def main():
    y=int(special_columns(np.random.default_rng(20260930),2)[1]);M=8192
    base,bmu,bsd=load_shared()
    A,B,W,MARG,T,wd,s0=collect(y,M,501001,base,bmu,bsd)
    VA,VB,VW,VMARG,VT,_,_=collect(y,M,502001,base,bmu,bsd)
    print("DATA",len(W),"proxy<.12",float(np.mean(MARG<.12)),float(np.mean(VMARG<.12)),flush=True)

    net,mu,sd,bestv,hist=weighted_train(A,B,W,MARG,VA,VB,VW,VMARG)
    crit=[]
    for tau in (.25,.4,.5):
        rb,cut=critical_rmse(base,bmu,bsd,y,tau)
        rw,_=critical_rmse(net,mu,sd,y,tau)
        row={"tau":tau,"proxy_cut":cut,"baseline_critical_rmse":rb,"weighted_critical_rmse":rw}
        crit.append(row);print("CRIT",row,flush=True)

    replay=sampled_replay(net,mu,sd,y,M,503001)
    out={"y":y,"M":M,"best_val_weighted_ploss":bestv,"history":hist,
         "critical":crit,"replay":replay}
    path=ROOT/"results/finite_tau_margin_weighted_ratio20.json"
    path.write_text(json.dumps(out,indent=2))
    np.savez_compressed(ROOT/"results/finite_tau_margin_weighted_ratio20_weights.npz",
        mu=mu,sd=sd,W1=net.W1,b1=net.b1,W2=net.W2,b2=net.b2,W3=net.W3,b3=net.b3)
    print("WROTE",path,flush=True)

if __name__=="__main__":main()
