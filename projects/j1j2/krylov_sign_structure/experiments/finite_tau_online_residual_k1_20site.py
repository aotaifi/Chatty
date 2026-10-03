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

def norm(v):
    v=np.asarray(v,float);return v/np.linalg.norm(v)

def pairs(w,y,wd,tau,K,rg):
    src=[];dst=[];ww=[]
    for x in w:
        lo,hi=HO.indptr[int(x)],HO.indptr[int(x)+1];nb=HO.indices[lo:hi]
        kk=min(K,len(nb));pick=rg.choice(len(nb),size=kk,replace=False)
        src.extend([int(x)]*kk);dst.extend(nb[pick].tolist());ww.extend([len(nb)/kk]*kk)
    src=np.asarray(src,np.int32);dst=np.asarray(dst,np.int32)
    return feat(src,y,wd,tau),feat(dst,y,wd,tau),np.asarray(ww,float),src,dst

def step(net,A,B,off,w,lr=5e-4):
    za,ca=net.fwd(A);zb,cb=net.fwd(B);diff=off+zb-za
    q=1/(1+np.exp(-np.clip(diff,-30,30)));ww=w/max(w.mean(),1e-12);dz=ww*q/len(q)
    ga=net.grads(ca,-dz);gb=net.grads(cb,dz);gs=[u+v for u,v in zip(ga,gb)]
    net.t+=1
    for p,g,m,v in zip(net.par(),gs,net.m,net.v):
        m*=.9;m+=.1*g;v*=.999;v+=.001*g*g
        p-=lr*(m/(1-.9**net.t))/(np.sqrt(v/(1-.999**net.t))+1e-8)
    return float(np.average(np.logaddexp(0,diff),weights=w))

def loss(net,A,B,off,w):
    return float(np.average(np.logaddexp(0,off+net.pred(B)-net.pred(A)),weights=w))

def fit_block(net,A,B,O,W,VA,VB,VO,VW,epochs,rg):
    bestv=1e99;best=None
    for ep in range(epochs):
        order=rg.permutation(len(A))
        for q in range(0,len(order),2048):
            z=order[q:q+2048];step(net,A[z],B[z],O[z],W[z])
        vv=loss(net,VA,VB,VO,VW)
        if vv<bestv:bestv=vv;best=[x.copy() for x in net.par()]
    for dst,src in zip(net.par(),best):dst[:]=src
    return bestv

def total_guide(net,bnet,bmu,bsd,y,arr,wd,tau,clip=3.):
    g0=guide_vec(bnet,bmu,bsd,y,tau,arr)
    out=[]
    for q in range(0,D,4096):
        ix=np.arange(q,min(q+4096,D));out.append(net.pred(feat(ix,y,wd,tau)))
    delta=np.clip(np.concatenate(out),-clip,clip)
    g=g0*np.exp(delta);return norm(np.maximum(g,1e-14))

def run(y,M=8192,dt=.05,beta=.5,seed=911001):
    arr=endpoint(y);d0=arr[0];s=np.where(d0%2==0,1,-1).astype(np.int8);wd=make_wd(H,[y])[0]
    bnet,bmu,bsd=load_model();ra=np.random.default_rng(seed);rb=np.random.default_rng(seed+1);rt=np.random.default_rng(seed+2)
    wa=np.full(M,y,np.int32);wb=wa.copy();g=np.full(D,1/np.sqrt(D))
    net=MLP(3*20+3,64,83);net.W3*=0;net.b3[:]=0
    exact=np.zeros(D);exact[y]=1.;rows=[]
    for n in range(1,int(round(beta/dt))+1):
        wa=propagate_fn_uniform_importance(wa,g,s,dt,ra);wb=propagate_fn_uniform_importance(wb,g,s,dt,rb)
        tau=n*dt;A,B,W,src,dst=pairs(wa,y,wd,tau,4,ra);VA,VB,VW,vsrc,vdst=pairs(wb,y,wd,tau,4,rb)
        g0=guide_vec(bnet,bmu,bsd,y,tau,arr);lg=np.log(np.maximum(g0,1e-300))
        O=lg[dst]-lg[src];VO=lg[vdst]-lg[vsrc]
        vl=fit_block(net,A,B,O,W,VA,VB,VO,VW,8 if n==1 else 4,rt)
        g=total_guide(net,bnet,bmu,bsd,y,arr,wd,tau)

        exact=norm(sla.expm_multiply(-dt*H,exact));exnext=norm(sla.expm_multiply(-dt*H,exact))
        sn=np.where((s*g-dt*(H@(s*g)))>=0,1,-1).astype(np.int8)
        row={"tau":tau,"val_loss":vl,"ampfid":float(np.dot(g,np.abs(exact))**2),
             "sign_err_current":mm(s,exact),"carry_next":mm(s,exnext),"k1_next":mm(sn,exnext),
             "uniqueA":int(len(np.unique(wa))),"uniqueB":int(len(np.unique(wb)))}
        rows.append(row);print("ONLINE_RESID",row,flush=True);s=sn
    return rows

def main():
    y=int(special_columns(np.random.default_rng(20260930),2)[1]);rows=run(y)
    path=ROOT/"results/finite_tau_online_residual_k1_20site.json"
    path.write_text(json.dumps({"y":y,"rows":rows},indent=2));print("WROTE",path,flush=True)

if __name__=="__main__":main()
