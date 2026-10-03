#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import scipy.sparse.linalg as sla

from finite_tau_matching_20site_exact import special_columns,D
from finite_tau_amplitude_regression_20site import endpoint,H
from finite_tau_learned_guide_gfmc_20site import load_model,guide_vec
from finite_tau_localratio_mlp_oracle20 import MLP,feat,make_wd,mm
from finite_tau_margin_ratio_walkers20 import collect_meta

ROOT=Path("/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure")

def norm(v):
    v=np.asarray(v,float); return v/np.linalg.norm(v)

def base_for_tau(bnet,bmu,bsd,y,arr,tau):
    return guide_vec(bnet,bmu,bsd,y,float(tau),arr)

def offsets_and_focus(bnet,bmu,bsd,y,arr,T,src,dst,s,basew,dt=.05,lam=24.,sigma=.5):
    off=np.empty(len(T),float); out=np.empty(len(T),float)
    for tau in np.unique(T):
        g=base_for_tau(bnet,bmu,bsd,y,arr,tau)
        lg=np.log(np.maximum(g,1e-300))
        q=1-dt*s*(H@(s*g))/np.maximum(g,1e-300)
        m=T==tau
        off[m]=lg[dst[m]]-lg[src[m]]
        edge=np.minimum(np.abs(q[src[m]]),np.abs(q[dst[m]]))
        out[m]=basew[m]*(1+lam*np.exp(-(edge/sigma)**2))
    return off,out

def rstep(net,A,B,off,w,lr=5e-4):
    za,ca=net.fwd(A); zb,cb=net.fwd(B)
    diff=off+zb-za
    q=1/(1+np.exp(-np.clip(diff,-30,30)))
    ww=w/max(np.mean(w),1e-12); dz=ww*q/len(q)
    ga=net.grads(ca,-dz); gb=net.grads(cb,dz); gs=[u+v for u,v in zip(ga,gb)]
    net.t+=1
    for p,g,m,v in zip(net.par(),gs,net.m,net.v):
        m*=.9;m+=.1*g;v*=.999;v+=.001*g*g
        p-=lr*(m/(1-.9**net.t))/(np.sqrt(v/(1-.999**net.t))+1e-8)
    return float(np.average(np.logaddexp(0,diff),weights=w))

def rloss(net,A,B,off,w):
    diff=off+net.pred(B)-net.pred(A)
    return float(np.average(np.logaddexp(0,diff),weights=w))

def train(A,B,O,W,VA,VB,VO,VW,epochs=15,seed=41):
    X=np.vstack([A,B]);mu=X.mean(0);sd=X.std(0);sd[sd<1e-8]=1
    A=(A-mu)/sd;B=(B-mu)/sd;VA=(VA-mu)/sd;VB=(VB-mu)/sd
    net=MLP(A.shape[1],64,seed);net.W3*=1e-3;net.b3[:]=0
    rg=np.random.default_rng(seed);bestv=1e99;best=None;hist=[]

    for ep in range(epochs):
        order=rg.permutation(len(A));ls=[]
        for q in range(0,len(order),4096):
            z=order[q:q+4096];ls.append(rstep(net,A[z],B[z],O[z],W[z]))
        vv=rloss(net,VA,VB,VO,VW);hist.append([ep+1,float(np.mean(ls)),vv])
        if vv<bestv:bestv=vv;best=[x.copy() for x in net.par()]
        if ep in (0,1,4,9,14):print("TRAIN",hist[-1],flush=True)
    for dst,src in zip(net.par(),best):dst[:]=src
    return net,mu,sd,bestv,hist

def total_amp(net,mu,sd,bnet,bmu,bsd,y,arr,wd,tau):
    ds=[]
    for q in range(0,D,4096):
        ix=np.arange(q,min(q+4096,D))
        ds.append(net.pred((feat(ix,y,wd,tau)-mu)/sd))
    delta=np.concatenate(ds); g=base_for_tau(bnet,bmu,bsd,y,arr,tau)
    la=np.log(np.maximum(g,1e-300))+delta
    a=np.exp(np.clip(la-la.max(),-60,0))
    return norm(a)

def evaluate(net,mu,sd,bnet,bmu,bsd,y,arr,wd,s0,dt=.05):
    e=np.zeros(D);e[y]=1.;psi=e.copy();s=s0.copy();rows=[]
    for n in range(1,11):
        psi=norm(sla.expm_multiply(-dt*H,psi));tau=n*dt
        ah=total_amp(net,mu,sd,bnet,bmu,bsd,y,arr,wd,tau)
        exn=norm(sla.expm_multiply(-dt*H,psi))
        sn=np.where((s*ah-dt*(H@(s*ah)))>=0,1,-1)
        rows.append({"tau":tau,"ampfid":float(np.dot(ah,np.abs(psi))**2),
                     "current_sign_err":mm(s,psi),
                     "carry_next":mm(s,exn),"k1_next":mm(sn,exn)})
        s=sn
    return rows

def main():
    y=int(special_columns(np.random.default_rng(20260930),2)[1]);M=8192
    arr=endpoint(y);bnet,bmu,bsd=load_model()
    A,B,W,T,S,Z,wd,s0=collect_meta(y,M,601001)
    VA,VB,VW,VT,VS,VZ,_,_=collect_meta(y,M,602001)
    O,MW=offsets_and_focus(bnet,bmu,bsd,y,arr,T,S,Z,s0,W)
    VO,MVW=offsets_and_focus(bnet,bmu,bsd,y,arr,VT,VS,VZ,s0,VW)
    net,mu,sd,bestv,hist=train(A,B,O,MW,VA,VB,VO,MVW)
    rows=evaluate(net,mu,sd,bnet,bmu,bsd,y,arr,wd,s0)
    for r in rows:
        if r["tau"] in (.1,.25,.5):print("RESULT",r,flush=True)

    out={"y":y,"M":M,"best_val_loss":bestv,"history":hist,"rows":rows}
    path=ROOT/"results/finite_tau_residual_ratio_walkers20.json"
    path.write_text(json.dumps(out,indent=2))
    np.savez_compressed(ROOT/"results/finite_tau_residual_ratio_walkers20_weights.npz",
        mu=mu,sd=sd,W1=net.W1,b1=net.b1,W2=net.W2,b2=net.b2,W3=net.W3,b3=net.b3)
    print("WROTE",path,flush=True)

if __name__=="__main__":main()
