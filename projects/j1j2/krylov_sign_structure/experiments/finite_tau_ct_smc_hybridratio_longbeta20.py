#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np

from finite_tau_matching_20site_exact import D,special_columns
from finite_tau_ct_local_ratio_20site import H,distance_sign,normalized,mismatch
from finite_tau_ct_smc_fn20 import ct_block
from finite_tau_localratio_mlp_oracle20 import MLP,feat,make_wd
from finite_tau_learned_amplitude_20site import weighted_auc
from finite_tau_ct_smc_edgeratio_longbeta20 import (
    ratio_params,k1_ratio,edge_data,wstep,reconstruct,exact_sequence,full_score
)

DT=.05

def adam_apply(net,grads,lr):
    net.t+=1
    for p,g,m,v in zip(net.par(),grads,net.m,net.v):
        m*=.9;m+=.1*g;v*=.999;v+=.001*g*g
        p-=lr*(m/(1-.9**net.t))/(np.sqrt(v/(1-.999**net.t))+1e-8)

def bce_step(net,X,y,lr=8e-4):
    z,c=net.fwd(X); p=1/(1+np.exp(-np.clip(z,-30,30)))
    dz=(p-y)/len(y); adam_apply(net,net.grads(c,dz),lr)
    return float(np.mean(np.logaddexp(0,z)-y*z))

def ce_score(z,y,eta=1.0):
    q=np.clip(eta*z,-40,40)
    return float(np.mean(np.logaddexp(0,q)-y*q))

def clone_params(net):
    return [x.copy() for x in net.par()]

def load_params(net,ps):
    for a,b in zip(net.par(),ps): a[:]=b

def hybrid_fit(st1,st2,logg,y,wd,tau,seed,nfit=20000,epochs=6,
               edge_epochs=3,minc=2,maxedges=50000):
    rg=np.random.default_rng(seed); n=min(nfit,len(st1))
    q=np.exp(np.clip(logg-logg.max(),-60,0)); q/=q.sum()
    ia=rg.choice(len(st1),n,replace=False); ib=rg.choice(len(st2),n,replace=False)
    ra=rg.choice(D,n,p=q); rb=rg.choice(D,n,p=q)
    xa=np.concatenate([st1[ia],ra]); ya=np.concatenate([np.ones(n),np.zeros(n)])
    xv=np.concatenate([st2[ib],rb]); yv=np.concatenate([np.ones(n),np.zeros(n)])
    XA=feat(xa,y,wd,tau); XV=feat(xv,y,wd,tau)
    net=MLP(XA.shape[1],64,seed)
    for ep in range(epochs):
        order=rg.permutation(len(XA))
        for k in range(0,len(order),2048):
            z=order[k:k+2048]; bce_step(net,XA[z],ya[z])
    base_ce=ce_score(net.pred(XV),yv)
    A=edge_data(np.bincount(st1,minlength=D).astype(float),logg,y,wd,tau,minc,maxedges,seed+11)
    V=edge_data(np.bincount(st2,minlength=D).astype(float),logg,y,wd,tau,minc,maxedges,seed+12)

    best=clone_params(net); best_rmse=np.inf; rmse0=np.nan
    if V is not None:
        VA,VB,vt,vw,_,_=V
        rmse0=float(np.sqrt(np.average(vt*vt,weights=vw)))
        pd=net.pred(VB)-net.pred(VA)
        best_rmse=float(np.sqrt(np.average((pd-vt)**2,weights=vw)))
    if A is not None and V is not None:
        AA,AB,at,aw,_,_=A
        for ep in range(edge_epochs):
            # retain density-ratio identification while sharpening edge differences
            order=rg.permutation(len(XA))
            for k in range(0,len(order),4096):
                z=order[k:k+4096]; bce_step(net,XA[z],ya[z],4e-4)
            order=rg.permutation(len(at))
            for k in range(0,len(order),2048):
                z=order[k:k+2048]; wstep(net,AA[z],AB[z],at[z],aw[z],3e-4)
            pd=net.pred(VB)-net.pred(VA)
            rm=float(np.sqrt(np.average((pd-vt)**2,weights=vw)))
            if rm<best_rmse:
                best_rmse=rm;best=clone_params(net)
    load_params(net,best)

    sv=net.pred(XV)
    val_auc=float(weighted_auc(sv,yv,np.ones(len(yv))))
    grid=np.linspace(0,1,21)
    ce_losses=np.array([ce_score(sv,yv,e) for e in grid])
    eta_ce=float(grid[np.argmin(ce_losses)])

    eta_edge=0.0; edge_gain=0.0; edge_rmse=best_rmse
    if V is not None:
        VA,VB,vt,vw,_,_=V
        pd=net.pred(VB)-net.pred(VA)
        edge_losses=np.array([np.average((e*pd-vt)**2,weights=vw) for e in grid])
        j=int(np.argmin(edge_losses)); eta_edge=float(grid[j])
        edge_rmse=float(np.sqrt(edge_losses[j]))
        if np.isfinite(rmse0) and rmse0>0:
            edge_gain=float((rmse0-edge_rmse)/rmse0)
    # Require a reproducible edge improvement; otherwise do not perturb F.
    eta=eta_edge if edge_gain>=0.005 else 0.0
    diag={"val_auc":val_auc,"val_ce":base_ce,"eta_ce":eta_ce,
          "eta_edge":eta_edge,"eta":eta,"edge_gain":edge_gain,
          "edge_rmse0":float(rmse0) if np.isfinite(rmse0) else None,
          "edge_rmse":float(edge_rmse) if np.isfinite(edge_rmse) else None,
          "train_edges":0 if A is None else int(len(A[2])),
          "val_edges":0 if V is None else int(len(V[2]))}
    return net,eta,diag

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--M",type=int,default=25000)
    ap.add_argument("--beta",type=float,default=2.5)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    y=int(special_columns(np.random.default_rng(20260930),2)[1])
    s=distance_sign(y);wd=make_wd(H,[y])[0]
    exact=exact_sequence(y,a.beta)
    st1,st2,logg=reconstruct(y,s,a.M)
    rg1=np.random.default_rng(910001);rg2=np.random.default_rng(920001)
    rows=[];t=.5

    while t<a.beta-1e-12:
        J,lam,pot=ratio_params(s,logg)
        st1,e1,u1=ct_block(st1,.05,J,lam,pot,rg1)
        st2,e2,u2=ct_block(st2,.05,J,lam,pot,rg2)
        t=round(t+.05,10)
        h1=np.bincount(st1,minlength=D).astype(float)
        h2=np.bincount(st2,minlength=D).astype(float)
        q1=normalized(h1);q2=normalized(h2);q=normalized(h1+h2)

        net,eta,hd=hybrid_fit(st1,st2,logg,y,wd,t,930000+int(100*t))
        if eta>0:
            f=full_score(net,y,wd,t)
            loggn=logg+eta*f;loggn-=loggn.max()
        else:
            loggn=logg.copy()

        ex=exact[t]
        row={"tau":t,"full":float(np.dot(s*q,ex)**2),
             "amp":float(np.dot(q,np.abs(ex))**2),"sign":mismatch(s,ex),
             "cross":float(np.dot(q1,q2)**2),"ess":[e1,e2],"unique":[u1,u2],
             **hd}
        if t<a.beta-1e-12:
            sn=k1_ratio(s,loggn)
            row["next_sign"]=mismatch(sn,exact[round(t+.05,10)])
            s=sn
        logg=loggn;rows.append(row)
        if abs((t*20)%5)<1e-9 or t in (.55,a.beta):
            print("HYBRID",json.dumps(row,sort_keys=True),flush=True)

    Path(a.out).write_text(json.dumps({"y":y,"M":a.M,"beta":a.beta,"rows":rows},indent=2))
    print("SUMMARY",json.dumps(rows[-1],sort_keys=True),flush=True)
if __name__=="__main__":main()
