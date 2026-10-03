#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np

from finite_tau_matching_20site_exact import D,special_columns
from finite_tau_ct_local_ratio_20site import H,distance_sign,normalized
from finite_tau_ct_smc_edgeratio_longbeta20 import reconstruct,exact_sequence
from finite_tau_localratio_mlp_oracle20 import feat,make_wd

DT=.05

class TinyMLP:
    def __init__(self,nin,h,seed):
        rg=np.random.default_rng(seed)
        self.W1=rg.normal(0,.12,(nin,h)); self.b1=np.zeros(h)
        self.W2=rg.normal(0,.12,(h,h)); self.b2=np.zeros(h)
        self.W3=rg.normal(0,.05,(h,1)); self.b3=np.zeros(1)
    def fwd(self,X):
        h1=np.tanh(X@self.W1+self.b1)
        h2=np.tanh(h1@self.W2+self.b2)
        y=(h2@self.W3+self.b3).ravel()
        return y,(X,h1,h2)
    def jac(self,X):
        _,(X,h1,h2)=self.fwd(X)
        n=len(X); h=h1.shape[1]
        d2=(1-h2*h2)*self.W3.ravel()[None,:]
        d1=(d2@self.W2.T)*(1-h1*h1)
        jW1=(X[:,:,None]*d1[:,None,:]).reshape(n,-1)
        jb1=d1
        jW2=(h1[:,:,None]*d2[:,None,:]).reshape(n,-1)
        jb2=d2
        jW3=h2
        jb3=np.ones((n,1))
        return np.concatenate([jW1,jb1,jW2,jb2,jW3,jb3],axis=1)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--M",type=int,default=25000)
    ap.add_argument("--ns",type=int,default=4000)
    ap.add_argument("--h",type=int,default=8)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    y=int(special_columns(np.random.default_rng(20260930),2)[1])
    wd=make_wd(H,[y])[0]; s=distance_sign(y)
    exact=exact_sequence(y,.55)
    _,_,logg=reconstruct(y,s,a.M)
    g=normalized(np.exp(np.clip(logg-logg.max(),-700,0)))
    psi=s*g
    h1=H@psi
    psi2=normalized(psi-DT*h1+.5*DT*DT*(H@h1))
    targ=np.log(np.maximum(np.abs(psi2),1e-300))-np.log(np.maximum(g,1e-300))

    p=g*g; p/=p.sum(); rg=np.random.default_rng(20261001+a.M+a.ns+a.h)
    tr=rg.choice(D,a.ns,p=p); va=rg.choice(D,a.ns,p=p)
    X=feat(tr,y,wd,.55); V=feat(va,y,wd,.55)
    mu=X.mean(0); sd=X.std(0); sd[sd<1e-6]=1
    X=(X-mu)/sd; V=(V-mu)/sd
    net=TinyMLP(X.shape[1],a.h,7)
    J=net.jac(X); JV=net.jac(V)
    # centered tangent removes arbitrary normalization direction
    Jc=J-J.mean(0,keepdims=True); tc=targ[tr]-targ[tr].mean()
    JVc=JV-J.mean(0,keepdims=True); tv=targ[va]-targ[va].mean()
    S=(Jc.T@Jc)/len(Jc); b=(Jc.T@tc)/len(Jc)
    scale=float(np.trace(S)/len(S))
    lambdas=scale*np.array([1e-5,1e-4,1e-3,1e-2,1e-1,1.,10.])
    vals=[]; sols=[]
    I=np.eye(S.shape[0])
    for lam in lambdas:
        sol=np.linalg.solve(S+lam*I,b)
        rm=float(np.sqrt(np.mean((JVc@sol-tv)**2)))
        vals.append(rm); sols.append(sol)
    k=int(np.argmin(vals)); sol=sols[k]; lam=float(lambdas[k])
    # evaluate tangent correction globally; clip only as a numerical trust region diagnostic
    allf=[]
    for q in range(0,D,4096):
        ix=np.arange(q,min(D,q+4096)); Z=(feat(ix,y,wd,.55)-mu)/sd
        allf.append(net.jac(Z)@sol)
    f=np.concatenate(allf); f-=np.sum(p*f)
    rows=[]
    ex=exact[.55]; sn=np.where(psi2>=0,1.0,-1.0)
    for cap in (None,.5,1.0,2.0):
        ff=f if cap is None else np.clip(f,-cap,cap)
        gn=normalized(g*np.exp(np.clip(ff,-20,20)))
        rows.append({"cap":cap,"full":float(np.dot(sn*gn,ex)**2),
                     "amp":float(np.dot(gn,np.abs(ex))**2),
                     "target_fid":float(np.dot(gn,np.abs(psi2))**2)})
    out={"M":a.M,"ns":a.ns,"h":a.h,"pdim":int(J.shape[1]),
         "lambda":lam,"val_rmse":vals[k],
         "base_full":float(np.dot(psi,exact[.5])**2),
         "exact_k2_full":float(np.dot(psi2,ex)**2),"rows":rows}
    print(json.dumps(out,sort_keys=True),flush=True)
    Path(a.out).write_text(json.dumps(out,indent=2))
if __name__=="__main__":main()
