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
HO=H.copy();HO.setdiag(0);HO.eliminate_zeros()

def norm(v): v=np.asarray(v,float);return v/np.linalg.norm(v)

def est(w,y,tau,arr,net,mu,sd,prior=512):
    _,_,_,_,_,vals,rlab,rbs=arr
    mass=np.bincount(rlab[w],minlength=len(vals)).astype(float)
    gm=guide_vec(net,mu,sd,y,tau,arr)
    pm=np.bincount(rlab,weights=gm,minlength=len(vals));pm/=pm.sum()
    return norm(np.maximum(((mass+prior*pm)/rbs)[rlab],1e-14))

def paired_loop(y,M,seed):
    net,mu,sd=load_model();arr=endpoint(y);s=np.where(arr[0]%2==0,1,-1).astype(np.int8)
    ra=np.random.default_rng(seed);rb=np.random.default_rng(seed+1)
    wa=np.full(M,y,np.int32);wb=wa.copy();ga=np.full(D,1/np.sqrt(D));gb=ga.copy()
    for n in range(1,11):
        wa=propagate_fn_uniform_importance(wa,ga,s,.05,ra);wb=propagate_fn_uniform_importance(wb,gb,s,.05,rb)
        ga=est(wa,y,n*.05,arr,net,mu,sd);gb=est(wb,y,n*.05,arr,net,mu,sd)
        if n<10:
            fa=s*ga-.05*(H@(s*ga));fb=s*gb-.05*(H@(s*gb))
            flip=(np.sign(fa)!=s)|(np.sign(fb)!=s);s=s.copy();s[flip]*=-1
    return wa,wb,s

def pairs(w,y,wd,K=4,seed=1):
    rg=np.random.default_rng(seed);src=[];dst=[];ww=[]
    for x in w:
        lo,hi=HO.indptr[int(x)],HO.indptr[int(x)+1]; nb=HO.indices[lo:hi]
        kk=min(K,len(nb));pick=rg.choice(len(nb),size=kk,replace=False)
        src.extend([int(x)]*kk);dst.extend(nb[pick].tolist());ww.extend([len(nb)/kk]*kk)
    src=np.asarray(src,np.int32);dst=np.asarray(dst,np.int32)
    return feat(src,y,wd),feat(dst,y,wd),np.asarray(ww,float)

def pstep(net,A,B,w,lr=1e-3):
    za,ca=net.fwd(A);zb,cb=net.fwd(B);q=1/(1+np.exp(-np.clip(zb-za,-30,30)))
    ww=w/max(np.mean(w),1e-12);dz=ww*q/len(q)
    ga=net.grads(ca,-dz);gb=net.grads(cb,dz);gs=[u+v for u,v in zip(ga,gb)]
    net.t+=1
    for p,g,m,v in zip(net.par(),gs,net.m,net.v):
        m*=.9;m+=.1*g;v*=.999;v+=.001*g*g
        p-=lr*(m/(1-.9**net.t))/(np.sqrt(v/(1-.999**net.t))+1e-8)
    return float(np.average(np.logaddexp(0,zb-za),weights=w))

def ploss(net,A,B,w):
    return float(np.average(np.logaddexp(0,net.pred(B)-net.pred(A)),weights=w))

def main():
    y=int(special_columns(np.random.default_rng(20260930),2)[1]);M=32768
    print("WALK",flush=True);wa,wb,s=paired_loop(y,M,300001);wd=make_wd(H,[y])[0]
    A,B,W=pairs(wa,y,wd,4,1);VA,VB,VW=pairs(wb,y,wd,4,2)
    X=np.vstack([A,B]);mu=X.mean(0);sd=X.std(0);sd[sd<1e-8]=1
    A=(A-mu)/sd;B=(B-mu)/sd;VA=(VA-mu)/sd;VB=(VB-mu)/sd
    net=MLP(A.shape[1],64,9);rg=np.random.default_rng(9);bestv=1e99;best=None
    for ep in range(30):
        order=rg.permutation(len(A));ls=[]
        for q in range(0,len(order),2048):
            z=order[q:q+2048];ls.append(pstep(net,A[z],B[z],W[z]))
        vv=ploss(net,VA,VB,VW)
        if vv<bestv:
            bestv=vv;best=[x.copy() for x in net.par()]
        if ep in (0,1,4,9,19,29):print("TRAIN",ep+1,np.mean(ls),vv,flush=True)
    for dst,src in zip(net.par(),best): dst[:]=src
    fs=[]
    for q in range(0,D,4096):
        ix=np.arange(q,min(q+4096,D));fs.append(net.pred((feat(ix,y,wd)-mu)/sd))
    f=np.concatenate(fs);ah=np.exp(np.clip(f-f.max(),-60,0));ah=norm(ah)

    e=np.zeros(D);e[y]=1.;ex=norm(sla.expm_multiply(-.5*H,e));exn=norm(sla.expm_multiply(-.05*H,ex))
    sn=np.where((s*ah-.05*(H@(s*ah)))>=0,1,-1)
    row={"M":M,"npairs":len(A),"val_ploss":ploss(net,VA,VB,VW),
         "current_sign_error":mm(s,ex),"model_amplitude_fidelity":float(np.dot(ah,np.abs(ex))**2),
         "k1_next_sign_error":mm(sn,exn),"carry_next_sign_error":mm(s,exn)}
    print("RESULT",row,flush=True)
    (ROOT/"results/finite_tau_raw_pseudolikelihood_walkers20.json").write_text(json.dumps(row,indent=2))

if __name__=="__main__":main()
