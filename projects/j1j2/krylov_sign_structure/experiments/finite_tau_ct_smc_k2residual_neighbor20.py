#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np

from finite_tau_matching_20site_exact import D,special_columns
from finite_tau_ct_local_ratio_20site import H,distance_sign,normalized,mismatch
from finite_tau_ct_smc_fn20 import ct_block
from finite_tau_localratio_mlp_oracle20 import MLP,feat,make_wd
from finite_tau_ct_smc_edgeratio_longbeta20 import ratio_params,reconstruct,exact_sequence

DT=.05
HO=H.copy(); HO.setdiag(0); HO.eliminate_zeros()

def apply(net,grads,lr):
    net.t+=1
    for p,g,m,v in zip(net.par(),grads,net.m,net.v):
        m*=.9;m+=.1*g;v*=.999;v+=.001*g*g
        p-=lr*(m/(1-.9**net.t))/(np.sqrt(v/(1-.999**net.t))+1e-8)

def mse_step(net,X,targ,lr=8e-4):
    pred,cache=net.fwd(X)
    e=pred-targ
    apply(net,net.grads(cache,2*e/len(e)),lr)
    return float(np.mean(e*e))

def fit_delta(st1,st2,target,y,wd,tau,seed,nfit=20000,epochs=4):
    rg=np.random.default_rng(seed)
    def sample_expand(st):
        if len(st)>nfit:
            base=st[rg.choice(len(st),nfit,replace=False)].astype(np.int32)
        else:
            base=st.astype(np.int32)
        uu=np.unique(base)
        neigh=[]
        for x in uu:
            aa,bb=HO.indptr[int(x)],HO.indptr[int(x)+1]
            neigh.extend(HO.indices[aa:bb].tolist())
        neigh=np.unique(np.asarray(neigh,np.int32)) if neigh else np.empty(0,np.int32)
        if len(neigh)>nfit:
            neigh=neigh[rg.choice(len(neigh),nfit,replace=False)]
        return np.concatenate([base,neigh]).astype(np.int32), int(len(uu)), int(len(neigh))

    i1,u1,n1=sample_expand(st1); i2,u2,n2=sample_expand(st2)
    X1=feat(i1,y,wd,tau); X2=feat(i2,y,wd,tau)
    mu=X1.mean(0); sd=X1.std(0); sd[sd<1e-6]=1
    X1=(X1-mu)/sd; X2=(X2-mu)/sd
    t1=np.clip(target[i1],-8,8); t2=np.clip(target[i2],-8,8)
    lo,hi=np.quantile(t1,[.002,.998])
    net=MLP(X1.shape[1],64,seed); net.W3[:]=0; net.b3[:]=0
    best=np.sqrt(np.mean(t2*t2)); bestp=[x.copy() for x in net.par()]
    for ep in range(epochs):
        order=rg.permutation(len(i1))
        for q in range(0,len(order),2048):
            z=order[q:q+2048]; mse_step(net,X1[z],t1[z])
        pv=np.clip(net.pred(X2),lo,hi)
        rm=np.sqrt(np.mean((pv-t2)**2))
        if rm<best:
            best=rm; bestp=[x.copy() for x in net.par()]
    for aa,bb in zip(net.par(),bestp): aa[:]=bb
    pred=np.clip(net.pred(X2),lo,hi)
    grid=np.linspace(0,1.5,31)
    losses=np.array([np.mean((eta*pred-t2)**2) for eta in grid])
    j=int(np.argmin(losses)); eta=float(grid[j])
    base=float(np.sqrt(np.mean(t2*t2))); val=float(np.sqrt(losses[j]))
    gain=(base-val)/max(base,1e-12)
    return net,mu,sd,eta,float(lo),float(hi),{
        "delta_rmse0":base,"delta_rmse":val,"delta_gain":float(gain),"eta":eta,
        "train_n":int(len(i1)),"val_n":int(len(i2)),
        "train_unique_walkers":u1,"train_neighbor_unique":n1,
        "val_unique_walkers":u2,"val_neighbor_unique":n2,
        "clip_lo":float(lo),"clip_hi":float(hi)}

def full_score(net,mu,sd,y,wd,tau,lo,hi):
    out=[]
    for q in range(0,D,4096):
        ix=np.arange(q,min(q+4096,D))
        out.append(np.clip(net.pred((feat(ix,y,wd,tau)-mu)/sd),lo,hi))
    return np.concatenate(out)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--M",type=int,default=25000)
    ap.add_argument("--beta",type=float,default=3.0)
    ap.add_argument("--nfit",type=int,default=20000)
    ap.add_argument("--epochs",type=int,default=4)
    ap.add_argument("--seed-offset",type=int,default=0)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    y=int(special_columns(np.random.default_rng(20260930),2)[1])
    wd=make_wd(H,[y])[0]; s=distance_sign(y)
    exact=exact_sequence(y,a.beta)
    st1,st2,logg=reconstruct(y,s,a.M)
    rg1=np.random.default_rng(1210001+a.seed_offset)
    rg2=np.random.default_rng(1220001+a.seed_offset)
    rows=[]; t=.5
    while t<a.beta-1e-12:
        g=np.exp(np.clip(logg-logg.max(),-700,0))
        psi=s*g
        h1=H@psi
        psi2=psi-DT*h1+.5*DT*DT*(H@h1)
        target=np.log(np.maximum(np.abs(psi2),1e-300))-np.log(np.maximum(g,1e-300))
        sn=np.where(psi2>=0,1.0,-1.0)

        J,lam,pot=ratio_params(s,logg)
        st1,e1,u1=ct_block(st1,DT,J,lam,pot,rg1)
        st2,e2,u2=ct_block(st2,DT,J,lam,pot,rg2)
        t=round(t+DT,10)

        net,mu,sd,eta,lo,hi,dd=fit_delta(st1,st2,target,y,wd,t,
                                         1230001+a.seed_offset+int(round(100*t)),
                                         a.nfit,a.epochs)
        f=full_score(net,mu,sd,y,wd,t,lo,hi)
        logg=logg+eta*f; logg-=logg.max(); s=sn

        q1=normalized(np.bincount(st1,minlength=D).astype(float))
        q2=normalized(np.bincount(st2,minlength=D).astype(float))
        q=normalized(q1+q2)
        gg=normalized(np.exp(np.clip(logg-logg.max(),-700,0)))
        ex=exact[t]; k2=normalized(psi2)
        row={"tau":t,
             "guide_full":float(np.dot(s*gg,ex)**2),
             "guide_amp":float(np.dot(gg,np.abs(ex))**2),
             "guide_sign":mismatch(s,ex),
             "walker_full":float(np.dot(s*q,ex)**2),
             "walker_amp":float(np.dot(q,np.abs(ex))**2),
             "cross":float(np.dot(q1,q2)**2),
             "k2_target_full":float(np.dot(k2,ex)**2),
             "ess":[e1,e2],"unique":[u1,u2],**dd}
        rows.append(row)
        if t in (.55,1.0,1.5,2.0,2.5,3.0,a.beta):
            print("K2RES",json.dumps(row,sort_keys=True),flush=True)
    Path(a.out).write_text(json.dumps({"M":a.M,"beta":a.beta,"nfit":a.nfit,
                                       "epochs":a.epochs,"rows":rows},indent=2))
    print("FINAL",json.dumps(rows[-1],sort_keys=True),flush=True)
if __name__=="__main__": main()
