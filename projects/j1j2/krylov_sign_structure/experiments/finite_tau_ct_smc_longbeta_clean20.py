#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np
import scipy.sparse.linalg as sla

from finite_tau_matching_20site_exact import D,special_columns
from finite_tau_ct_local_ratio_20site import H,distance_sign,normalized,mismatch
from finite_tau_ct_iterative_fn_krylov20 import ct_fn_params
from finite_tau_ct_smc_fn20 import run_smc,ct_block,systematic

ROOT=Path("/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure")
DT=.05; T0=.5; PRIOR=.01

def cv0(g,h1,h2):
    old=np.maximum(g/g.sum(),1e-300)
    p1=h1/h1.sum();p2=h2/h2.sum()
    b1=np.maximum(p1+PRIOR/D,1e-300);b2=np.maximum(p2+PRIOR/D,1e-300)
    curve=[]
    for eta in np.linspace(0,1,21):
        q1=old**(1-eta)*b1**eta;q1/=q1.sum()
        q2=old**(1-eta)*b2**eta;q2/=q2.sum()
        ce=-.5*(np.sum(p2*np.log(q1))+np.sum(p1*np.log(q2)))
        curve.append((float(eta),float(ce)))
    eta=min(curve,key=lambda z:z[1])[0]
    pm=.5*(p1+p2);bm=np.maximum(pm+PRIOR/D,1e-300)
    gn=normalized(old**(1-eta)*bm**eta) if eta>0 else normalized(old)
    return gn,eta,curve

def reconstruct(y,s,M):
    g=np.ones(D)/np.sqrt(D)
    etas=[]
    for it in range(3):
        F,J,lam,pot=ct_fn_params(s,g)
        h1,_=run_smc(y,T0,M,DT,J,lam,pot,101000+100*it+1)
        h2,_=run_smc(y,T0,M,DT,J,lam,pot,101000+100*it+2)
        g,eta,_=cv0(g,h1,h2);etas.append(eta)
        print("RECON",M,it,eta,flush=True)
    F,J,lam,pot=ct_fn_params(s,g)
    h1,_=run_smc(y,T0,M,DT,J,lam,pot,104001)
    h2,_=run_smc(y,T0,M,DT,J,lam,pot,104002)
    g,eta,_=cv0(g,h1,h2);etas.append(eta)
    rg1=np.random.default_rng(104101);rg2=np.random.default_rng(104102)
    st1=systematic(h1/h1.sum(),rg1,M).astype(np.int32)
    st2=systematic(h2/h2.sum(),rg2,M).astype(np.int32)
    print("INIT",M,eta,int(np.unique(st1).size),int(np.unique(st2).size),flush=True)
    return st1,st2,g,etas

def exact_sequence(y,beta):
    e=np.zeros(D);e[y]=1.;psi=normalized(sla.expm_multiply(-T0*H,e))
    out={T0:psi.copy()};t=T0
    while t<beta-1e-12:
        psi=normalized(sla.expm_multiply(-DT*H,psi));t=round(t+DT,10)
        out[t]=psi.copy()
    return out

def k1(s,a):
    z=s*a-DT*(H@(s*a))
    return np.where(z>=0,1.0,-1.0)

def branch(y,st10,st20,g0,s0,M,beta,exact,adaptive,label):
    st1=st10.copy();st2=st20.copy();g=g0.copy();s=s0.copy()
    rg1=np.random.default_rng(310001);rg2=np.random.default_rng(320001)
    rows=[];t=T0
    while t<beta-1e-12:
        F,J,lam,pot=ct_fn_params(s,g)
        st1,ess1,u1=ct_block(st1,DT,J,lam,pot,rg1)
        st2,ess2,u2=ct_block(st2,DT,J,lam,pot,rg2)
        t=round(t+DT,10)
        h1=np.bincount(st1,minlength=D).astype(float)
        h2=np.bincount(st2,minlength=D).astype(float)
        q1=normalized(h1);q2=normalized(h2);q=normalized(h1+h2)
        gn,eta,_=cv0(g,h1,h2)
        ex=exact[t]
        row={"tau":t,"full":float(np.dot(s*q,ex)**2),
             "amp":float(np.dot(q,np.abs(ex))**2),
             "sign":mismatch(s,ex),"cross":float(np.dot(q1,q2)**2),
             "ess":[ess1,ess2],"eta":eta,"unique":[u1,u2]}
        if t<beta-1e-12:
            sn=k1(s,gn) if adaptive else s.copy()
            row["next_sign"]=mismatch(sn,exact[round(t+DT,10)])
            row["flipped_mass"]=float(np.sum((ex*ex)[sn!=s]))
            s=sn
        rows.append(row)
        if abs((t*20)%5)<1e-9 or t in (.55,beta):
            print(label,M,json.dumps(row,sort_keys=True),flush=True)
        g=gn
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--M",type=int,required=True)
    ap.add_argument("--beta",type=float,default=2.0)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    y=int(special_columns(np.random.default_rng(20260930),2)[1]);s0=distance_sign(y)
    exact=exact_sequence(y,a.beta)
    st1,st2,g,etas=reconstruct(y,s0,a.M)
    ad=branch(y,st1,st2,g,s0,a.M,a.beta,exact,True,"ADAPT")
    fx=branch(y,st1,st2,g,s0,a.M,a.beta,exact,False,"FIXED")
    out={"y":y,"M":a.M,"beta":a.beta,"dt":DT,"init_etas":etas,
         "adaptive":ad,"fixed":fx}
    Path(a.out).write_text(json.dumps(out,indent=2))
    print("SUMMARY",json.dumps({
        "M":a.M,"beta":a.beta,
        "adaptive_final":ad[-1],
        "fixed_final":fx[-1]
    },sort_keys=True),flush=True)
    print("WROTE",a.out,flush=True)

if __name__=="__main__":main()
