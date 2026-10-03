#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np

from finite_tau_matching_20site_exact import D,special_columns
from finite_tau_ct_local_ratio_20site import H,distance_sign,normalized,mismatch
from finite_tau_ct_smc_fn20 import ct_block
from finite_tau_localratio_mlp_oracle20 import MLP,feat,make_wd
from finite_tau_ct_smc_pairratio_longbeta20 import make_pairs,apply
from finite_tau_ct_smc_edgeratio_longbeta20 import ratio_params,reconstruct,exact_sequence

DT=.05

def edge_step(net,A,B,targ,w,lr=8e-4):
    fi,ci=net.fwd(A); fj,cj=net.fwd(B)
    e=(fj-fi)-targ
    ww=w/max(w.mean(),1e-12)
    dz=2*ww*e/max(ww.sum(),1e-12)
    gi=net.grads(ci,-dz); gj=net.grads(cj,dz)
    apply(net,[a+b for a,b in zip(gi,gj)],lr)
    return float(np.average(e*e,weights=w))

def fit_edge(st1,st2,target,logg,y,wd,tau,seed,nfit=20000,K=4,epochs=4):
    A,B,src,dst,w=make_pairs(st1,y,wd,tau,seed+1,nfit,K)
    VA,VB,vsrc,vdst,vw=make_pairs(st2,y,wd,tau,seed+2,nfit,K)
    X=np.vstack([A,B]); mu=X.mean(0); sd=X.std(0); sd[sd<1e-6]=1
    A=(A-mu)/sd;B=(B-mu)/sd;VA=(VA-mu)/sd;VB=(VB-mu)/sd
    tt=target[dst]-target[src]; vt=target[vdst]-target[vsrc]
    net=MLP(A.shape[1],64,seed); net.W3[:]=0; net.b3[:]=0
    rg=np.random.default_rng(seed)
    base=float(np.sqrt(np.average(vt*vt,weights=vw)))
    best=base; bestp=[x.copy() for x in net.par()]
    for ep in range(epochs):
        order=rg.permutation(len(src))
        for q in range(0,len(order),2048):
            z=order[q:q+2048]; edge_step(net,A[z],B[z],tt[z],w[z])
        pred=net.pred(VB)-net.pred(VA)
        rm=float(np.sqrt(np.average((pred-vt)**2,weights=vw)))
        if rm<best: best=rm; bestp=[x.copy() for x in net.par()]
    for a,b in zip(net.par(),bestp): a[:]=b
    pred=net.pred(VB)-net.pred(VA)
    grid=np.linspace(0,1.5,31)
    losses=np.array([np.average((eta*pred-vt)**2,weights=vw) for eta in grid])
    j=int(np.argmin(losses)); eta=float(grid[j])
    val=float(np.sqrt(losses[j])); gain=(base-val)/max(base,1e-12)
    return net,mu,sd,eta,{"edge_rmse0":base,"edge_rmse":val,"edge_gain":float(gain),
                          "eta":eta,"train_edges":int(len(src)),"val_edges":int(len(vsrc))}

def full_score(net,mu,sd,y,wd,tau):
    out=[]
    for q in range(0,D,4096):
        ix=np.arange(q,min(q+4096,D))
        out.append(net.pred((feat(ix,y,wd,tau)-mu)/sd))
    return np.concatenate(out)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--M",type=int,default=25000)
    ap.add_argument("--beta",type=float,default=3.0)
    ap.add_argument("--nfit",type=int,default=20000)
    ap.add_argument("--K",type=int,default=4)
    ap.add_argument("--epochs",type=int,default=4)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    y=int(special_columns(np.random.default_rng(20260930),2)[1])
    wd=make_wd(H,[y])[0]; s=distance_sign(y)
    exact=exact_sequence(y,a.beta)
    st1,st2,logg=reconstruct(y,s,a.M)
    rg1=np.random.default_rng(1310001); rg2=np.random.default_rng(1320001)
    rows=[]; t=.5
    while t<a.beta-1e-12:
        g=np.exp(np.clip(logg-logg.max(),-700,0)); psi=s*g
        h1=H@psi; psi2=psi-DT*h1+.5*DT*DT*(H@h1)
        target=np.log(np.maximum(np.abs(psi2),1e-300))-np.log(np.maximum(g,1e-300))
        sn=np.where(psi2>=0,1.0,-1.0)

        J,lam,pot=ratio_params(s,logg)
        st1,e1,u1=ct_block(st1,DT,J,lam,pot,rg1)
        st2,e2,u2=ct_block(st2,DT,J,lam,pot,rg2)
        t=round(t+DT,10)

        net,mu,sd,eta,ed=fit_edge(st1,st2,target,logg,y,wd,t,
                                   1330001+int(round(100*t)),a.nfit,a.K,a.epochs)
        f=full_score(net,mu,sd,y,wd,t)
        # Trust region: K2 log-ratio correction per block should remain local.
        f=np.clip(f,-1.5,1.5)
        logg=logg+eta*f; logg-=logg.max(); s=sn

        q1=normalized(np.bincount(st1,minlength=D).astype(float))
        q2=normalized(np.bincount(st2,minlength=D).astype(float))
        q=normalized(q1+q2)
        gg=normalized(np.exp(np.clip(logg-logg.max(),-700,0)))
        ex=exact[t]; k2=normalized(psi2)
        row={"tau":t,"guide_full":float(np.dot(s*gg,ex)**2),
             "guide_amp":float(np.dot(gg,np.abs(ex))**2),"guide_sign":mismatch(s,ex),
             "walker_full":float(np.dot(s*q,ex)**2),"walker_amp":float(np.dot(q,np.abs(ex))**2),
             "cross":float(np.dot(q1,q2)**2),"k2_target_full":float(np.dot(k2,ex)**2),
             "ess":[e1,e2],"unique":[u1,u2],**ed}
        rows.append(row)
        if t in (.55,1.0,1.5,2.0,2.5,3.0,a.beta):
            print("K2EDGE",json.dumps(row,sort_keys=True),flush=True)
    Path(a.out).write_text(json.dumps({"M":a.M,"beta":a.beta,"rows":rows},indent=2))
    print("FINAL",json.dumps(rows[-1],sort_keys=True),flush=True)

if __name__=="__main__": main()
