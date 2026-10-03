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

def walker_loop(y,M,seed,dt=.05,beta=.5,prior=512.):
    net,mu,sd=load_model(); arr=endpoint(y)
    d,_,_,_,rich,vals,rlab,rbs=arr
    s=np.where(d%2==0,1,-1).astype(np.int8)
    rg=np.random.default_rng(seed); w=np.full(M,y,np.int32)
    g=np.full(D,1/np.sqrt(D))
    for n in range(1,int(round(beta/dt))+1):
        w=propagate_fn_uniform_importance(w,g,s,dt,rg)
        mass=np.bincount(rlab[w],minlength=len(vals)).astype(float)
        gm=guide_vec(net,mu,sd,y,n*dt,arr)
        pm=np.bincount(rlab,weights=gm,minlength=len(vals)); pm/=pm.sum()
        post=mass+prior*pm
        g=(post/rbs)[rlab]; g=norm(np.maximum(g,1e-14))
        if n<int(round(beta/dt)):
            s=np.where((s*g-dt*(H@(s*g)))>=0,1,-1).astype(np.int8)
    return w,s,g

def edge_dataset(counts,y,wd,minc=2,maxpairs=50000,seed=1):
    rg=np.random.default_rng(seed); src=[];dst=[]
    obs=np.flatnonzero(counts>=minc)
    for i in obs:
        lo,hi=HO.indptr[i],HO.indptr[i+1]
        ns=HO.indices[lo:hi]; ns=ns[counts[ns]>=minc]
        if len(ns):
            src.extend([int(i)]*len(ns)); dst.extend(map(int,ns))
    src=np.asarray(src,np.int32); dst=np.asarray(dst,np.int32)
    if len(src)>maxpairs:
        take=rg.choice(len(src),size=maxpairs,replace=False);src=src[take];dst=dst[take]
    t=np.log(counts[dst].astype(float))-np.log(counts[src].astype(float))
    return feat(src,y,wd),feat(dst,y,wd),t.astype(float),src,dst

def train_ratio(A,B,epochs=30,seed=7):
    net=MLP(A[0].shape[1],64,seed); rg=np.random.default_rng(seed)
    for ep in range(epochs):
        order=rg.permutation(len(A[2]));ls=[]
        for q in range(0,len(order),1024):
            z=order[q:q+1024];ls.append(net.step(A[0][z],A[1][z],A[2][z]))
        vr=float(np.sqrt(np.mean((net.pred(B[1])-net.pred(B[0])-B[2])**2)))
        if ep in (0,1,4,9,19,29): print("TRAIN",ep+1,float(np.mean(ls)),vr,flush=True)
    return net

def main():
    y=int(special_columns(np.random.default_rng(20260930),2)[1]); M=32768
    print("WALKERS",flush=True)
    wa,sa,ga=walker_loop(y,M,171001); wb,sb,gb=walker_loop(y,M,171002)
    ca=np.bincount(wa,minlength=D); cb=np.bincount(wb,minlength=D)
    wd=make_wd(H,[y])[0]
    A=edge_dataset(ca,y,wd,2,50000,1); B=edge_dataset(cb,y,wd,2,50000,2)
    print("PAIRS",len(A[2]),len(B[2]),"unique",len(np.unique(wa)),len(np.unique(wb)),flush=True)
    net=train_ratio(A,B)
    fs=[]
    for q in range(0,D,4096):
        ix=np.arange(q,min(q+4096,D));fs.append(net.pred(feat(ix,y,wd)))
    f=np.concatenate(fs); ah=np.exp(np.clip(f-f.max(),-60,0)); ah=norm(ah)

    e=np.zeros(D);e[y]=1.; ex=norm(sla.expm_multiply(-.5*H,e))
    exn=norm(sla.expm_multiply(-.05*H,ex))
    sn=np.where((sa*ah-.05*(H@(sa*ah)))>=0,1,-1)
    p=ex*ex
    sab=float(np.sum(p[sa!=sb]))
    row={"M":M,"pair_train":len(A[2]),"pair_val":len(B[2]),
         "replica_sign_disagreement_tau05":sab,
         "model_amplitude_fidelity":float(np.dot(ah,np.abs(ex))**2),
         "current_sign_error":mm(sa,ex),
         "ratio_model_k1_next_sign_error":mm(sn,exn)}
    print("RESULT",row,flush=True)
    (ROOT/"results/finite_tau_localratio_mlp_walkers20.json").write_text(json.dumps(row,indent=2))

if __name__=="__main__": main()
