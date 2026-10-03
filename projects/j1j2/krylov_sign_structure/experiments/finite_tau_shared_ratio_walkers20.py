#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import scipy.sparse.linalg as sla

from finite_tau_matching_20site_exact import special_columns,D
from finite_tau_amplitude_regression_20site import endpoint,H
from finite_tau_learned_guide_gfmc_20site import load_model,guide_vec,propagate_fn_uniform_importance
from finite_tau_localratio_mlp_oracle20 import MLP,feat,make_wd,mm

ROOT=Path("/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure")
HO=H.copy(); HO.setdiag(0); HO.eliminate_zeros()

def norm(v):
    v=np.asarray(v,float); return v/np.linalg.norm(v)

def amp_est(w,y,tau,arr,net,mu,sd,prior=512.):
    _,_,_,_,_,vals,rlab,rbs=arr
    mass=np.bincount(rlab[w],minlength=len(vals)).astype(float)
    gm=guide_vec(net,mu,sd,y,tau,arr)
    pm=np.bincount(rlab,weights=gm,minlength=len(vals)); pm/=pm.sum()
    return norm(np.maximum(((mass+prior*pm)/rbs)[rlab],1e-14))

def collect(y,M,seed,K=4,dt=.05,beta=.5):
    net,mu,sd=load_model(); arr=endpoint(y); d0=arr[0]
    s=np.where(d0%2==0,1,-1).astype(np.int8)
    rg=np.random.default_rng(seed); walkers=np.full(M,y,np.int32)
    g=np.full(D,1/np.sqrt(D)); wd=make_wd(H,[y])[0]
    AA=[];BB=[];WW=[];TT=[]
    for n in range(1,int(round(beta/dt))+1):
        walkers=propagate_fn_uniform_importance(walkers,g,s,dt,rg)
        tau=n*dt; g=amp_est(walkers,y,tau,arr,net,mu,sd)
        src=[];dst=[];ww=[]
        for x in walkers:
            lo,hi=HO.indptr[int(x)],HO.indptr[int(x)+1]; nb=HO.indices[lo:hi]
            kk=min(K,len(nb)); pick=rg.choice(len(nb),size=kk,replace=False)
            src.extend([int(x)]*kk); dst.extend(nb[pick].tolist()); ww.extend([len(nb)/kk]*kk)
        AA.append(feat(np.asarray(src,np.int32),y,wd,tau))
        BB.append(feat(np.asarray(dst,np.int32),y,wd,tau))
        WW.append(np.asarray(ww,float)); TT.append(np.full(len(src),tau))
        print("COLLECT",seed,tau,len(src),len(np.unique(walkers)),flush=True)
    return np.vstack(AA),np.vstack(BB),np.concatenate(WW),np.concatenate(TT),wd,s

def pstep(net,A,B,w,lr=1e-3):
    za,ca=net.fwd(A); zb,cb=net.fwd(B)
    q=1/(1+np.exp(-np.clip(zb-za,-30,30)))
    ww=w/max(np.mean(w),1e-12); dz=ww*q/len(q)
    ga=net.grads(ca,-dz); gb=net.grads(cb,dz); gs=[u+v for u,v in zip(ga,gb)]
    net.t+=1
    for p,g,m,v in zip(net.par(),gs,net.m,net.v):
        m*=.9; m+=.1*g; v*=.999; v+=.001*g*g
        p-=lr*(m/(1-.9**net.t))/(np.sqrt(v/(1-.999**net.t))+1e-8)
    return float(np.average(np.logaddexp(0,zb-za),weights=w))

def ploss(net,A,B,w):
    return float(np.average(np.logaddexp(0,net.pred(B)-net.pred(A)),weights=w))

def predict_full(net,mu,sd,y,wd,tau):
    out=[]
    for q in range(0,D,4096):
        ix=np.arange(q,min(q+4096,D))
        out.append(net.pred((feat(ix,y,wd,tau)-mu)/sd))
    f=np.concatenate(out)
    a=np.exp(np.clip(f-f.max(),-60,0))
    return norm(a)

def train_shared(A,B,W,VA,VB,VW,epochs=20,seed=17):
    X=np.vstack([A,B]); mu=X.mean(0); sd=X.std(0); sd[sd<1e-8]=1
    A=(A-mu)/sd; B=(B-mu)/sd; VA=(VA-mu)/sd; VB=(VB-mu)/sd
    net=MLP(A.shape[1],64,seed); rg=np.random.default_rng(seed)
    bestv=1e99; best=None; hist=[]
    for ep in range(epochs):
        order=rg.permutation(len(A)); ls=[]
        for q in range(0,len(order),4096):
            z=order[q:q+4096]; ls.append(pstep(net,A[z],B[z],W[z]))
        vv=ploss(net,VA,VB,VW); hist.append([ep+1,float(np.mean(ls)),vv])
        if vv<bestv:
            bestv=vv; best=[x.copy() for x in net.par()]
        if ep in (0,1,4,9,14,19): print("TRAIN",hist[-1],flush=True)
    for dst,src in zip(net.par(),best): dst[:]=src
    return net,mu,sd,bestv,hist

def oracle_diagnostics(net,mu,sd,y,wd,s0,dt=.05,beta=.5):
    e=np.zeros(D); e[y]=1.; psi=e.copy(); s=s0.copy(); rows=[]
    for n in range(1,int(round(beta/dt))+1):
        psi=norm(sla.expm_multiply(-dt*H,psi)); tau=n*dt
        ah=predict_full(net,mu,sd,y,wd,tau)
        exnext=norm(sla.expm_multiply(-dt*H,psi))
        pred=s*ah-dt*(H@(s*ah)); sn=np.where(pred>=0,1,-1)
        rows.append({"tau":tau,"ampfid":float(np.dot(ah,np.abs(psi))**2),
                     "current_sign_err":mm(s,psi),
                     "carry_next":mm(s,exnext),"k1_next":mm(sn,exnext)})
        s=sn
    return rows

def sampled_replay(net,mu,sd,y,M,seed,dt=.05,beta=.5):
    bnet,bmu,bsd=load_model(); arr=endpoint(y); d0=arr[0]
    s=np.where(d0%2==0,1,-1).astype(np.int8)
    rg=np.random.default_rng(seed); walkers=np.full(M,y,np.int32)
    g=np.full(D,1/np.sqrt(D)); wd=make_wd(H,[y])[0]
    exact=np.zeros(D); exact[y]=1.; rows=[]
    for n in range(1,int(round(beta/dt))+1):
        walkers=propagate_fn_uniform_importance(walkers,g,s,dt,rg)
        exact=norm(sla.expm_multiply(-dt*H,exact)); tau=n*dt
        g=amp_est(walkers,y,tau,arr,bnet,bmu,bsd)
        ah=predict_full(net,mu,sd,y,wd,tau)
        pred=s*ah-dt*(H@(s*ah)); sn=np.where(pred>=0,1,-1).astype(np.int8)
        exnext=norm(sla.expm_multiply(-dt*H,exact))
        rows.append({"tau":tau,
                     "walker_ampfid":float(np.dot(g,np.abs(exact))**2),
                     "ratio_model_ampfid":float(np.dot(ah,np.abs(exact))**2),
                     "sign_err_current":mm(s,exact),
                     "carry_next":mm(s,exnext),"k1_next":mm(sn,exnext),
                     "unique":int(len(np.unique(walkers)))})
        print("REPLAY",rows[-1],flush=True); s=sn
    return rows

def main():
    y=int(special_columns(np.random.default_rng(20260930),2)[1]); M=8192
    A,B,W,T,wd,s0=collect(y,M,401001)
    VA,VB,VW,VT,_,_=collect(y,M,402001)
    print("DATA",len(W),len(VW),flush=True)
    net,mu,sd,bestv,hist=train_shared(A,B,W,VA,VB,VW)
    diag=oracle_diagnostics(net,mu,sd,y,wd,s0)
    for r in diag:
        if r["tau"] in (.1,.25,.5): print("ORACLE",r,flush=True)
    replay=sampled_replay(net,mu,sd,y,M,403001)
    out={"y":y,"M":M,"best_val_ploss":bestv,"history":hist,
         "oracle_diagnostics":diag,"sampled_replay":replay}
    path=ROOT/"results/finite_tau_shared_ratio_walkers20.json"
    path.write_text(json.dumps(out,indent=2))
    np.savez_compressed(ROOT/"results/finite_tau_shared_ratio_walkers20_weights.npz",
        mu=mu,sd=sd,W1=net.W1,b1=net.b1,W2=net.W2,b2=net.b2,W3=net.W3,b3=net.b3)
    print("WROTE",path,flush=True)

if __name__=="__main__": main()
