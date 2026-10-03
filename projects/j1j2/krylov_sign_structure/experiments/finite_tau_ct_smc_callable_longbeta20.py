#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np
import scipy.sparse.linalg as sla

from finite_tau_matching_20site_exact import D,N,basis,special_columns
from finite_tau_ct_local_ratio_20site import H,distance_sign,normalized,mismatch
from finite_tau_ct_iterative_fn_krylov20 import ct_fn_params,cv_update
from finite_tau_ct_smc_fn20 import run_smc,ct_block,systematic
from finite_tau_localratio_mlp_oracle20 import make_wd
from finite_tau_learned_amplitude_20site import MLP,weighted_auc

ROOT=Path("/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure")
DT=.05; T0=.5; PRIOR=.01
SEEDOFF=0

def seed(x): return int(x+SEEDOFF)

def rawfeat(idx,y,wd):
    ii=np.asarray(idx,np.int64); st=basis[ii]
    xb=(2*((st[:,None]>>np.arange(N,dtype=np.uint32))&1)-1).astype(np.float32)
    yst=basis[int(y)]
    yb=(2*((yst>>np.arange(N,dtype=np.uint32))&1)-1).astype(np.float32)
    d=(wd[ii]//1000).astype(np.float32)/10
    n2=np.rint(wd[ii]-1000*(wd[ii]//1000)).astype(np.float32)/10
    return np.column_stack([xb,xb*yb,d,n2])

def ce(scores,y,eta):
    z=np.clip(eta*scores,-40,40)
    return float(np.mean(np.logaddexp(0,z)-y*z))

def fit_residual(st1,st2,g,y,wd,seed,nfit=20000,epochs=8):
    rg=np.random.default_rng(seed); n=min(nfit,len(st1))
    ia=rg.choice(len(st1),size=n,replace=False); ib=rg.choice(len(st2),size=n,replace=False)
    q=np.maximum(g,1e-300); q=q/q.sum()
    ra=rg.choice(D,size=n,replace=True,p=q); rb=rg.choice(D,size=n,replace=True,p=q)
    xa=np.concatenate([st1[ia],ra]); ya=np.concatenate([np.ones(n),np.zeros(n)])
    xb=np.concatenate([st2[ib],rb]); yb=np.concatenate([np.ones(n),np.zeros(n)])
    XA=rawfeat(xa,y,wd); XB=rawfeat(xb,y,wd)
    mu=XA.mean(0); sd=XA.std(0); sd[sd<1e-6]=1
    ZA=(XA-mu)/sd; ZB=(XB-mu)/sd
    net=MLP(ZA.shape[1],64,seed); ww=np.ones(len(ya))
    for ep in range(epochs):
        order=rg.permutation(len(ZA))
        for q0 in range(0,len(order),4096):
            j=order[q0:q0+4096]; net.step(ZA[j],ya[j],ww[j],lr=1e-3)
    sa=net.pred(ZA); sb=net.pred(ZB)
    grid=np.linspace(0,1,21); losses=np.array([ce(sb,yb,e) for e in grid])
    eta=float(grid[np.argmin(losses)])
    return net,mu,sd,eta,float(weighted_auc(sa,ya,ww)),float(weighted_auc(sb,yb,np.ones(len(yb))))

def eval_full(net,mu,sd,y,wd):
    out=[]
    for q in range(0,D,4096):
        ix=np.arange(q,min(q+4096,D))
        out.append(net.pred((rawfeat(ix,y,wd)-mu)/sd))
    return np.concatenate(out)

def reconstruct(y,s,M):
    g=np.ones(D)/np.sqrt(D)
    for it in range(3):
        F,J,lam,pot=ct_fn_params(s,g)
        h1,_=run_smc(y,T0,M,DT,J,lam,pot,seed(101000+100*it+1))
        h2,_=run_smc(y,T0,M,DT,J,lam,pot,seed(101000+100*it+2))
        g,eta,_=cv_update(g,h1,h2,PRIOR)
        print("RECON",it,eta,flush=True)
    F,J,lam,pot=ct_fn_params(s,g)
    h1,_=run_smc(y,T0,M,DT,J,lam,pot,seed(104001))
    h2,_=run_smc(y,T0,M,DT,J,lam,pot,seed(104002))
    g,eta,_=cv_update(g,h1,h2,PRIOR)
    rg1=np.random.default_rng(seed(104101));rg2=np.random.default_rng(seed(104102))
    st1=systematic(h1/h1.sum(),rg1,M).astype(np.int32)
    st2=systematic(h2/h2.sum(),rg2,M).astype(np.int32)
    return st1,st2,g

def exact_sequence(y,beta):
    e=np.zeros(D);e[y]=1.;psi=normalized(sla.expm_multiply(-T0*H,e))
    out={T0:psi.copy()};t=T0
    while t<beta-1e-12:
        psi=normalized(sla.expm_multiply(-DT*H,psi));t=round(t+DT,10);out[t]=psi.copy()
    return out

def k1(s,a):
    z=s*a-DT*(H@(s*a))
    return np.where(z>=0,1.0,-1.0)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--M",type=int,default=25000)
    ap.add_argument("--beta",type=float,default=2.0)
    ap.add_argument("--out",required=True)
    ap.add_argument("--seed-offset",type=int,default=0)
    ap.add_argument("--fixed-sign",action="store_true")
    a=ap.parse_args()
    global SEEDOFF
    SEEDOFF=a.seed_offset
    y=int(special_columns(np.random.default_rng(20260930),2)[1])
    s=distance_sign(y); wd=make_wd(H,[y])[0]
    exact=exact_sequence(y,a.beta)
    st1,st2,g=reconstruct(y,s,a.M)
    rg1=np.random.default_rng(seed(410001));rg2=np.random.default_rng(seed(420001))
    rows=[];t=T0

    while t<a.beta-1e-12:
        F,J,lam,pot=ct_fn_params(s,g)
        st1,ess1,u1=ct_block(st1,DT,J,lam,pot,rg1)
        st2,ess2,u2=ct_block(st2,DT,J,lam,pot,rg2)
        t=round(t+DT,10)
        h1=np.bincount(st1,minlength=D).astype(float)
        h2=np.bincount(st2,minlength=D).astype(float)
        q1=normalized(h1);q2=normalized(h2);q=normalized(h1+h2)

        net,mu,sd,eta,aucA,aucB=fit_residual(st1,st2,g,y,wd,seed(500000+int(round(100*t))))
        score=eval_full(net,mu,sd,y,wd)
        lg=np.log(np.maximum(g,1e-300))+eta*score;lg-=lg.max()
        gn=normalized(np.exp(np.clip(lg,-60,0)))

        ex=exact[t]
        row={"tau":t,"full":float(np.dot(s*q,ex)**2),
             "amp":float(np.dot(q,np.abs(ex))**2),"sign":mismatch(s,ex),
             "cross":float(np.dot(q1,q2)**2),"ess":[ess1,ess2],"unique":[u1,u2],
             "eta":eta,"auc_train":aucA,"auc_val":aucB}

        if t<a.beta-1e-12:
            sn=s.copy() if a.fixed_sign else k1(s,gn)
            row["next_sign"]=mismatch(sn,exact[round(t+DT,10)])
            row["flip_weight"]=float(np.sum((ex*ex)[sn!=s]))
            s=sn
        rows.append(row);g=gn
        if abs((t*20)%5)<1e-9 or t in (.55,a.beta):
            print("CALLABLE",json.dumps(row,sort_keys=True),flush=True)

    out={"y":y,"M":a.M,"beta":a.beta,"dt":DT,"seed_offset":a.seed_offset,"fixed_sign":a.fixed_sign,"rows":rows}
    Path(a.out).write_text(json.dumps(out,indent=2))
    print("SUMMARY",json.dumps(rows[-1],sort_keys=True),flush=True)
    print("WROTE",a.out,flush=True)

if __name__=="__main__":main()
