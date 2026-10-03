#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from finite_tau_matching_20site_exact import D,special_columns
from finite_tau_ct_local_ratio_20site import H,diag,R,C,HV,IDX,distance_sign,normalized,mismatch
from finite_tau_ct_smc_fn20 import run_smc,ct_block,systematic
from finite_tau_ct_iterative_fn_krylov20 import ct_fn_params,cv_update
from finite_tau_localratio_mlp_oracle20 import MLP,feat,make_wd

ROOT=Path("/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure")
DT=.05; T0=.5; PRIOR=.01
HO=H.copy(); HO.setdiag(0); HO.eliminate_zeros()

def ratio_params(s,logg):
    prod=s[R]*s[C]; good=prod<0; bad=~good
    rat=np.exp(np.clip(logg[C]-logg[R],-20,20))
    lam=np.bincount(R[good],weights=HV[good],minlength=D)
    shift=np.bincount(R[bad],weights=HV[bad]*rat[bad],minlength=D)
    J=sp.coo_matrix((HV[good],(R[good],C[good])),shape=H.shape).tocsr()
    pot=lam-(diag+shift)
    return J,lam,pot

def k1_ratio(s,logg):
    rat=np.exp(np.clip(logg[C]-logg[R],-20,20))
    off=np.bincount(R,weights=HV*s[C]*rat,minlength=D)
    field=s*(1-DT*diag)-DT*off
    return np.where(field>=0,1.0,-1.0)

def edge_data(counts,logg,y,wd,tau,minc=2,maxedges=50000,seed=1):
    rg=np.random.default_rng(seed)
    src=[];dst=[];w=[]
    obs=np.flatnonzero(counts>=minc)
    for x in obs:
        a,b=HO.indptr[x],HO.indptr[x+1]
        nb=HO.indices[a:b]
        nb=nb[(nb>x)&(counts[nb]>=minc)]
        if len(nb):
            src.extend([int(x)]*len(nb)); dst.extend(nb.tolist())
    if not src:
        return None
    src=np.asarray(src,np.int32);dst=np.asarray(dst,np.int32)
    rel=2*counts[src].astype(float)*counts[dst]/np.maximum(counts[src]+counts[dst],1)
    hval=np.empty(len(src),float)
    for k,(x,z) in enumerate(zip(src,dst)):
        a,b=HO.indptr[x],HO.indptr[x+1]
        js=HO.indices[a:b]; hs=HO.data[a:b]
        hval[k]=abs(hs[np.searchsorted(js,z)]) if np.all(js[:-1]<=js[1:]) else 1.0
    ww=np.maximum(rel*hval,1e-8)
    if len(src)>maxedges:
        pr=ww/ww.sum(); take=rg.choice(len(src),size=maxedges,replace=False,p=pr)
        src,dst,ww=src[take],dst[take],ww[take]
    targ=np.log(counts[dst])-np.log(counts[src])-(logg[dst]-logg[src])
    return feat(src,y,wd,tau),feat(dst,y,wd,tau),targ.astype(float),ww,src,dst

def wstep(net,A,B,t,w,lr=8e-4):
    fi,ci=net.fwd(A);fj,cj=net.fwd(B);e=fj-fi-t
    ww=w/max(w.mean(),1e-12); dz=2*ww*e/max(ww.sum(),1e-12)
    gi=net.grads(ci,-dz);gj=net.grads(cj,dz);gs=[a+b for a,b in zip(gi,gj)]
    net.t+=1
    for p,g,m,v in zip(net.par(),gs,net.m,net.v):
        m*=.9;m+=.1*g;v*=.999;v+=.001*g*g
        p-=lr*(m/(1-.9**net.t))/(np.sqrt(v/(1-.999**net.t))+1e-8)
    return float(np.average(e*e,weights=w))

def fit_residual(A,V,seed,epochs=8):
    if A is None or V is None:
        return None,0.0,{},None,None
    XA,XB,t,w,_,_=A; VA,VB,vt,vw,_,_=V
    net=MLP(XA.shape[1],64,seed); rg=np.random.default_rng(seed)
    for ep in range(epochs):
        order=rg.permutation(len(t))
        for q in range(0,len(order),2048):
            z=order[q:q+2048]; wstep(net,XA[z],XB[z],t[z],w[z])
    pd=net.pred(VB)-net.pred(VA)
    grid=np.linspace(0,1,21)
    losses=np.array([np.average((e*pd-vt)**2,weights=vw) for e in grid])
    eta=float(grid[np.argmin(losses)])
    diag={"train_edges":int(len(t)),"val_edges":int(len(vt)),
          "val_rmse0":float(np.sqrt(np.average(vt*vt,weights=vw))),
          "val_rmse":float(np.sqrt(losses.min()))}
    return net,eta,diag,pd,vt

def full_score(net,y,wd,tau):
    out=[]
    for q in range(0,D,4096):
        ix=np.arange(q,min(q+4096,D))
        out.append(net.pred(feat(ix,y,wd,tau)))
    return np.concatenate(out)

def reconstruct(y,s,M):
    g=np.ones(D)/np.sqrt(D)
    for it in range(3):
        F,J,lam,pot=ct_fn_params(s,g)
        h1,_=run_smc(y,T0,M,DT,J,lam,pot,101000+100*it+1)
        h2,_=run_smc(y,T0,M,DT,J,lam,pot,101000+100*it+2)
        g,eta,_=cv_update(g,h1,h2,PRIOR)
        print("RECON",it,eta,flush=True)
    F,J,lam,pot=ct_fn_params(s,g)
    h1,_=run_smc(y,T0,M,DT,J,lam,pot,104001)
    h2,_=run_smc(y,T0,M,DT,J,lam,pot,104002)
    g,eta,_=cv_update(g,h1,h2,PRIOR)
    rg1=np.random.default_rng(104101);rg2=np.random.default_rng(104102)
    st1=systematic(h1/h1.sum(),rg1,M).astype(np.int32)
    st2=systematic(h2/h2.sum(),rg2,M).astype(np.int32)
    logg=np.log(np.maximum(g,1e-300));logg-=logg.max()
    return st1,st2,logg

def exact_sequence(y,beta):
    e=np.zeros(D);e[y]=1.;psi=normalized(sla.expm_multiply(-T0*H,e))
    out={T0:psi.copy()};t=T0
    while t<beta-1e-12:
        psi=normalized(sla.expm_multiply(-DT*H,psi));t=round(t+DT,10)
        out[t]=psi.copy()
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--M",type=int,default=25000)
    ap.add_argument("--beta",type=float,default=2.5)
    ap.add_argument("--epochs",type=int,default=8)
    ap.add_argument("--maxedges",type=int,default=50000)
    ap.add_argument("--minc",type=int,default=2)
    ap.add_argument("--seed-offset",type=int,default=0)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()

    y=int(special_columns(np.random.default_rng(20260930),2)[1])
    s=distance_sign(y);wd=make_wd(H,[y])[0]
    exact=exact_sequence(y,a.beta)
    st1,st2,logg=reconstruct(y,s,a.M)
    rg1=np.random.default_rng(610001+a.seed_offset)
    rg2=np.random.default_rng(620001+a.seed_offset)
    rows=[];t=T0

    while t<a.beta-1e-12:
        J,lam,pot=ratio_params(s,logg)
        st1,ess1,u1=ct_block(st1,DT,J,lam,pot,rg1)
        st2,ess2,u2=ct_block(st2,DT,J,lam,pot,rg2)
        t=round(t+DT,10)
        h1=np.bincount(st1,minlength=D).astype(float)
        h2=np.bincount(st2,minlength=D).astype(float)
        q1=normalized(h1);q2=normalized(h2);q=normalized(h1+h2)

        A=edge_data(h1,logg,y,wd,t,a.minc,a.maxedges,710000+a.seed_offset+int(100*t))
        V=edge_data(h2,logg,y,wd,t,a.minc,a.maxedges,720000+a.seed_offset+int(100*t))
        net,eta,ed,pd,vt=fit_residual(A,V,730000+a.seed_offset+int(100*t),a.epochs)
        if net is not None and eta>0:
            f=full_score(net,y,wd,t)
            loggn=logg+eta*f
            loggn-=loggn.max()
        else:
            loggn=logg.copy()

        ex=exact[t]
        row={"tau":t,"full":float(np.dot(s*q,ex)**2),
             "amp":float(np.dot(q,np.abs(ex))**2),
             "sign":mismatch(s,ex),
             "cross":float(np.dot(q1,q2)**2),
             "ess":[ess1,ess2],"unique":[u1,u2],
             "eta":eta,**ed}

        if t<a.beta-1e-12:
            sn=k1_ratio(s,loggn)
            row["next_sign"]=mismatch(sn,exact[round(t+DT,10)])
            row["flip_weight"]=float(np.sum((ex*ex)[sn!=s]))
            s=sn
        rows.append(row);logg=loggn
        if abs((t*20)%5)<1e-9 or t in (.55,a.beta):
            print("EDGERATIO",json.dumps(row,sort_keys=True),flush=True)

    out={"y":y,"M":a.M,"beta":a.beta,"dt":DT,
         "epochs":a.epochs,"maxedges":a.maxedges,"minc":a.minc,
         "seed_offset":a.seed_offset,"rows":rows}
    Path(a.out).write_text(json.dumps(out,indent=2))
    print("SUMMARY",json.dumps(rows[-1],sort_keys=True),flush=True)
    print("WROTE",a.out,flush=True)

if __name__=="__main__":main()
