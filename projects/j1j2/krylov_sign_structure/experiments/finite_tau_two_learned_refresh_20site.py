#!/usr/bin/env python3
import json
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from finite_tau_uniform_one_refresh_20site import (
    ROOT,H,HO,D,diag,f1,f2,arrays,feat,train,build_fn_from_guide,ce
)
from finite_tau_matching_20site_exact import special_columns
from finite_tau_learned_amplitude_20site import weighted_auc

def systematic(w,rng,M):
    w=np.asarray(w,float); w/=w.sum(); c=np.cumsum(w); c[-1]=1
    return np.searchsorted(c,rng.random()/M+np.arange(M)/M,side="right")

def propagate_vec(y,logg,s,M,seed,tau=.5,dtmax=.005):
    rg=np.random.default_rng(seed); w=np.full(M,int(y),np.int32); cache={}
    def local(x):
        x=int(x)
        if x in cache:return cache[x]
        a,b=HO.indptr[x],HO.indptr[x+1]; ys=HO.indices[a:b]; hs=HO.data[a:b]
        rat=np.exp(np.clip(logg[ys]-logg[x],-30,30)); opp=s[ys]!=s[x]
        rates=hs[opp]*rat[opp]; dfn=float(diag[x]+np.sum(hs[~opp]*rat[~opp]))
        z=(dfn,ys[opp],rates); cache[x]=z; return z
    t=0.
    while t<tau-1e-14:
        dat=[local(int(x)) for x in w]
        d=np.array([z[0] for z in dat]); el=np.array([z[0]-z[2].sum() for z in dat])
        eref=float(el.mean()); mx=max(0.,float(np.max(d-eref)))
        dt=min(dtmax,tau-t,0.8/mx if mx>0 else dtmax)
        nxt=np.empty(M,np.int32); bw=np.empty(M)
        for k,(x,(dv,ys,rates)) in enumerate(zip(w,dat)):
            stay=1-dt*(dv-eref); mov=dt*rates; tot=stay+mov.sum()
            u=rg.random()*tot
            if u<stay:nxt[k]=x
            else:
                j=np.searchsorted(np.cumsum(mov),u-stay,side="right"); nxt[k]=int(ys[min(j,len(ys)-1)])
            bw[k]=tot
        w=nxt[systematic(bw,rg,M)]; t+=dt
    return w

def fit_classifier(XA,YA,XB,YB,seed):
    net,mu,sd=train(XA,YA,seed=seed,epochs=30)
    sa=net.pred((XA-mu)/sd); sb=net.pred((XB-mu)/sd)
    grid=np.linspace(.05,2,196); loss=np.array([ce(sb,YB,e) for e in grid]); eta=float(grid[np.argmin(loss)])
    return net,mu,sd,eta,weighted_auc(sa,YA,np.ones(len(YA))),weighted_auc(sb,YB,np.ones(len(YB)))
def main():
    ys=special_columns(np.random.default_rng(20260930),2); M=8192
    # iteration 1: uniform guide -> g1
    XA=[];YA=[];XB=[];YB=[]; meta={}
    from finite_tau_uniform_one_refresh_20site import walkers_uniform
    for k,y in enumerate(ys):
        x,yv,m=walkers_uniform(int(y),M,71001+k);XA.append(x);YA.append(yv);meta[int(y)]=m
        x,yv,_=walkers_uniform(int(y),M,72001+k);XB.append(x);YB.append(yv)
    XA=np.concatenate(XA);YA=np.concatenate(YA);XB=np.concatenate(XB);YB=np.concatenate(YB)
    n1,mu1,sd1,e1,a1,b1=fit_classifier(XA,YA,XB,YB,7)
    print("ITER1_TRANSFER",a1,b1,e1,flush=True)
    g1={}
    for y in ys:
        d,n2,s=meta[int(y)]; F=feat(d,n2,np.arange(D))
        lg=e1*n1.pred((F-mu1)/sd1); lg-=lg.max(); g1[int(y)]=(lg,s,d,n2)
    # iteration 2 data: walkers from g1, references from g1^2
    XA=[];YA=[];XB=[];YB=[]
    for k,y in enumerate(ys):
        lg,s,d,n2=g1[int(y)]
        for seed,XX,YY in [(73001+k,XA,YA),(74001+k,XB,YB)]:
            w=propagate_vec(int(y),lg,s,M,seed)
            rg=np.random.default_rng(seed+9000); q=np.exp(np.clip(2*lg,-60,0)); q/=q.sum()
            ref=rg.choice(D,size=M,replace=True,p=q)
            XX.append(np.concatenate([feat(d,n2,w),feat(d,n2,ref)]))
            YY.append(np.concatenate([np.ones(M),np.zeros(M)]))
    XA=np.concatenate(XA);YA=np.concatenate(YA);XB=np.concatenate(XB);YB=np.concatenate(YB)
    n2m,mu2,sd2,e2,a2,b2=fit_classifier(XA,YA,XB,YB,17)
    print("ITER2_TRANSFER",a2,b2,e2,flush=True)
    rows=[]
    for y in ys:
        lg1,s,d,n2=g1[int(y)]; F=feat(d,n2,np.arange(D))
        resid=e2*n2m.pred((F-mu2)/sd2); lg2=lg1+resid; lg2-=lg2.max(); gg=np.exp(np.clip(lg2,-60,0))
        e=np.zeros(D);e[int(y)]=1.
        FF=build_fn_from_guide(s,gg)
        a=np.asarray(sla.expm_multiply(-.5*FF,e),float);a=np.maximum(a,0);a/=np.linalg.norm(a)
        ex=np.asarray(sla.expm_multiply(-.5*H,e),float);ex/=np.linalg.norm(ex)
        at=np.abs(ex)
        row={"y":int(y),"two_refresh_full_fidelity":float(np.dot(a*s,ex)**2),
             "two_refresh_amplitude_fidelity":float(np.dot(a,at)**2)}
        rows.append(row);print(row,flush=True)
    path=ROOT/"results/finite_tau_two_learned_refresh_20site.json"
    path.write_text(json.dumps({"iter1_auc":[a1,b1],"iter1_eta":e1,"iter2_auc":[a2,b2],"iter2_eta":e2,"rows":rows},indent=2))
    print("WROTE",path)
if __name__=="__main__":main()
