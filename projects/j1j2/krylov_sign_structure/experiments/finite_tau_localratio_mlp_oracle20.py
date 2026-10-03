#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import scipy.sparse.csgraph as cs

from finite_tau_matching_20site_exact import build_H,basis,D,N,special_columns

ROOT=Path("/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure")
H,_=build_H(.5); HO=H.copy(); HO.setdiag(0); HO.eliminate_zeros()
rng=np.random.default_rng(20261001)

def make_wd(H,ys):
    co=(H-sp.diags(H.diagonal())).tocoo()
    cost=np.where(np.abs(co.data)>0.375,1000.,1001.)
    W=sp.coo_matrix((cost,(co.row,co.col)),shape=H.shape).tocsr()
    return np.asarray(cs.shortest_path(W,directed=False,indices=ys))

def bits_idx(idx):
    s=basis[np.asarray(idx,np.int64)]
    return (2*((s[:,None]>>np.arange(N,dtype=np.uint32))&1)-1).astype(np.float32)

def feat(idx,y,wd,tau=.5):
    xb=bits_idx(idx); yb=bits_idx([y])[0]
    d=(wd[idx]//1000).astype(np.float32)/10
    n2=np.rint(wd[idx]-1000*(wd[idx]//1000)).astype(np.float32)/10
    return np.column_stack([xb,np.repeat(yb[None,:],len(xb),0),xb*yb,
                            d,n2,np.full(len(xb),tau/.5,np.float32)])

class MLP:
    def __init__(self,nin,h=64,seed=1):
        rg=np.random.default_rng(seed)
        self.W1=rg.normal(0,.12,(nin,h)); self.b1=np.zeros(h)
        self.W2=rg.normal(0,.12,(h,h)); self.b2=np.zeros(h)
        self.W3=rg.normal(0,.12,(h,1)); self.b3=np.zeros(1)
        self.m=[np.zeros_like(x) for x in self.par()]; self.v=[np.zeros_like(x) for x in self.par()]; self.t=0
    def par(self): return [self.W1,self.b1,self.W2,self.b2,self.W3,self.b3]
    def fwd(self,X):
        h1=np.tanh(X@self.W1+self.b1); h2=np.tanh(h1@self.W2+self.b2)
        return (h2@self.W3+self.b3).ravel(),(X,h1,h2)
    def grads(self,cache,dz):
        X,h1,h2=cache
        g3=h2.T@dz[:,None]; gb3=np.array([dz.sum()])
        q=(dz[:,None]@self.W3.T)*(1-h2*h2)
        g2=h1.T@q; gb2=q.sum(0)
        q=(q@self.W2.T)*(1-h1*h1)
        return [X.T@q,q.sum(0),g2,gb2,g3,gb3]

    def step(self,Xi,Xj,t,lr=1e-3):
        fi,ci=self.fwd(Xi); fj,cj=self.fwd(Xj)
        e=fj-fi-t; dz=2*e/len(e)
        gi=self.grads(ci,-dz); gj=self.grads(cj,dz)
        gs=[a+b for a,b in zip(gi,gj)]; self.t+=1
        for p,g,m,v in zip(self.par(),gs,self.m,self.v):
            m*=.9;m+=.1*g;v*=.999;v+=.001*g*g
            p-=lr*(m/(1-.9**self.t))/(np.sqrt(v/(1-.999**self.t))+1e-8)
        return float(np.mean(e*e))
    def pred(self,X): return self.fwd(X)[0]

def current_state(y,dt=.05,tau=.5):
    wd=make_wd(H,[y])[0]; d=(wd//1000).astype(np.int64)
    s=np.where(d%2==0,1,-1); psi=np.zeros(D);psi[y]=1.
    for n in range(1,int(round(tau/dt))+1):
        psi=sla.expm_multiply(-dt*H,psi);psi/=np.linalg.norm(psi)
        if n<int(round(tau/dt)):
            a=np.abs(psi); pred=s*a-dt*(H@(s*a)); s=np.where(pred>=0,1,-1)
    return psi,s,wd

def pairs(y,psi,wd,n):
    a=np.abs(psi); p=a/a.sum(); ii=rng.choice(D,size=n,p=p)
    jj=np.empty(n,np.int32)
    for q,i in enumerate(ii):
        lo,hi=HO.indptr[i],HO.indptr[i+1]; ns=HO.indices[lo:hi]
        jj[q]=int(ns[rng.integers(len(ns))])
    t=np.log(np.maximum(a[jj],1e-30))-np.log(np.maximum(a[ii],1e-30))
    return feat(ii,y,wd),feat(jj,y,wd),np.clip(t,-8,8)

def mm(s,psi):
    p=psi*psi;p/=p.sum();return float(np.sum(p[s!=np.where(psi>=0,1,-1)]))

def main():
    y=int(special_columns(np.random.default_rng(20260930),2)[1])
    psi,s,wd=current_state(y)
    Xi,Xj,t=pairs(y,psi,wd,30000); Vi,Vj,vt=pairs(y,psi,wd,10000)
    net=MLP(Xi.shape[1],64,7)
    for ep in range(30):
        order=rng.permutation(len(t));ls=[]
        for q in range(0,len(order),1024):
            z=order[q:q+1024];ls.append(net.step(Xi[z],Xj[z],t[z]))
        vr=float(np.sqrt(np.mean((net.pred(Vj)-net.pred(Vi)-vt)**2)))
        if ep in (0,1,4,9,19,29): print("EPOCH",ep+1,np.mean(ls),vr,flush=True)

    fs=[]
    for q in range(0,D,4096): fs.append(net.pred(feat(np.arange(q,min(q+4096,D)),y,wd)))
    f=np.concatenate(fs); ah=np.exp(np.clip(f-f.max(),-60,0)); ah/=np.linalg.norm(ah)
    a=np.abs(psi);a/=np.linalg.norm(a)
    exn=sla.expm_multiply(-.05*H,psi);exn/=np.linalg.norm(exn)
    sn=np.where((s*ah-.05*(H@(s*ah)))>=0,1,-1)
    row={"y":y,"current_sign_err":mm(s,psi),
         "amplitude_fidelity":float(np.dot(ah,a)**2),
         "k1_next_sign_error":mm(sn,exn)}
    print("RESULT",row,flush=True)
    (ROOT/"results/finite_tau_localratio_mlp_oracle20.json").write_text(json.dumps(row,indent=2))

if __name__=="__main__": main()
