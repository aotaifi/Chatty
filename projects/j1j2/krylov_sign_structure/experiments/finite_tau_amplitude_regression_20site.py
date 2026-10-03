#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as cs
import scipy.sparse.linalg as sla
from finite_tau_matching_20site_exact import build_H,basis,D,NN,NNN,special_columns
from finite_tau_table_gfmc_20site import propagate,update_theta,systematic

ROOT=Path("/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure")
H,diag=build_H(.5); HO=(H-sp.diags(diag)).tocsr()
f1=np.zeros(D); f2=np.zeros(D)
for u,v in NN: f1 += (((basis>>u)^(basis>>v))&1)
for u,v in NNN: f2 += (((basis>>u)^(basis>>v))&1)

def endpoint(y):
    co=HO.tocoo(); cost=np.where(np.abs(co.data)>0.375,1000.,1001.)
    W=sp.coo_matrix((cost,(co.row,co.col)),shape=H.shape).tocsr()
    wd=np.asarray(cs.shortest_path(W,directed=False,indices=[y]))[0]
    d=(wd//1000).astype(int); n2=np.rint(wd-1000*d).astype(int)
    ec=np.rint(8*diag).astype(int)+100
    raw=d.astype(np.int64)*100000+n2.astype(np.int64)*1000+ec
    _,lab=np.unique(raw,return_inverse=True); bs=np.bincount(lab).astype(float)
    rich=np.column_stack([d,n2,diag,f1,f2])
    vals,rlab=np.unique(rich,axis=0,return_inverse=True)
    rbs=np.bincount(rlab,minlength=len(vals)).astype(float)
    return d,n2,lab.astype(np.int32),bs,rich,vals,rlab.astype(np.int32),rbs

def xrows(vals,y,tau):
    yc=[diag[y],f1[y],f2[y]]
    return np.column_stack([vals,np.tile(yc,(len(vals),1)),np.full(len(vals),tau)])

def collect(ys,M,seed,block=.05,beta=.5,tau_max=.005,eta=.5,prior=32.):
    rg=np.random.default_rng(seed); Xs=[]; Ts=[]; Ws=[]; G=[]; oracle={}
    for y in ys:
        arr=endpoint(int(y)); d,n2,lab,bs,rich,vals,rlab,rbs=arr
        sgn=np.where(d%2==0,1,-1).astype(np.int8)
        theta=np.zeros(len(bs)); walkers=np.full(M,int(y),dtype=np.int32)
        exact=np.zeros(D); exact[int(y)]=1.
        for n in range(1,int(round(beta/block))+1):
            tau=n*block; old=theta.copy()
            walkers,_=propagate(walkers,old,lab,sgn,block,tau_max,rg)
            lw=-old[lab[walkers]]; lw-=lw.max(); iw=np.exp(lw); iw*=M/iw.sum()
            cnt=np.bincount(rlab[walkers],minlength=len(vals)).astype(float)
            mass=np.bincount(rlab[walkers],weights=iw,minlength=len(vals)).astype(float)
            keep=cnt>=2
            target=np.log(np.maximum(mass[keep],1e-30)/rbs[keep])
            ww=cnt[keep]
            target-=np.average(target,weights=ww)
            Xs.append(xrows(vals[keep],int(y),tau)); Ts.append(target); Ws.append(ww)
            G.extend([(int(y),round(float(tau),8))]*int(np.sum(keep)))
            theta,rw,_,_,_=update_theta(old,lab,bs,walkers,eta,prior,-1.)
            walkers=walkers[systematic(rw,rg,M)]
            exact=np.asarray(sla.expm_multiply(-block*H,exact),float); exact/=np.linalg.norm(exact)
            oracle[(int(y),round(float(tau),8))]=(arr,exact.copy())
    return np.concatenate(Xs),np.concatenate(Ts),np.concatenate(Ws),np.asarray(G,dtype=object),oracle
class RegMLP:
    def __init__(self,nin,h=48,seed=1):
        rg=np.random.default_rng(seed)
        self.W1=rg.normal(0,.18,(nin,h)); self.b1=np.zeros(h)
        self.W2=rg.normal(0,.18,(h,h)); self.b2=np.zeros(h)
        self.W3=rg.normal(0,.18,(h,1)); self.b3=np.zeros(1)
        self.m=[np.zeros_like(x) for x in self.params()]; self.v=[np.zeros_like(x) for x in self.params()]; self.t=0
    def params(self): return [self.W1,self.b1,self.W2,self.b2,self.W3,self.b3]
    def forward(self,X):
        h1=np.tanh(X@self.W1+self.b1); h2=np.tanh(h1@self.W2+self.b2); z=(h2@self.W3+self.b3).ravel()
        return z,(X,h1,h2)
    def step(self,X,y,w,lr=1e-3):
        z,(X,h1,h2)=self.forward(X); ww=w/w.mean()
        dz=2*(z-y)*ww/len(y)
        gW3=h2.T@dz[:,None]; gb3=np.array([dz.sum()])
        dz2=(dz[:,None]@self.W3.T)*(1-h2*h2)
        gW2=h1.T@dz2; gb2=dz2.sum(0)
        dz1=(dz2@self.W2.T)*(1-h1*h1)
        gW1=X.T@dz1; gb1=dz1.sum(0)
        self.t+=1
        for p,g,m,v in zip(self.params(),[gW1,gb1,gW2,gb2,gW3,gb3],self.m,self.v):
            m*=.9; m+=.1*g; v*=.999; v+=.001*g*g
            p-=lr*(m/(1-.9**self.t))/(np.sqrt(v/(1-.999**self.t))+1e-8)
        return float(np.average((z-y)**2,weights=ww))
    def pred(self,X): return self.forward(X)[0]

def fit(X,t,w,epochs=300,seed=7):
    mu=X.mean(0); sd=X.std(0); sd[sd<1e-8]=1; Z=(X-mu)/sd
    net=RegMLP(Z.shape[1],48,seed); rg=np.random.default_rng(seed)
    for ep in range(epochs):
        o=rg.permutation(len(Z)); ls=[]
        for i in range(0,len(Z),1024):
            j=o[i:i+1024]; ls.append(net.step(Z[j],t[j],w[j]))
        if ep in (0,9,49,99,199,299): print("TRAIN",ep+1,np.mean(ls),flush=True)
    return net,mu,sd

def grouped_rmse(pred,t,w,g):
    se=0.; sw=0.
    keys=sorted(set(map(tuple,g.tolist())))
    for key in keys:
        m=np.array([tuple(x)==key for x in g])
        off=np.average(t[m]-pred[m],weights=w[m])
        se+=np.sum(w[m]*(pred[m]+off-t[m])**2); sw+=w[m].sum()
    return float(np.sqrt(se/sw))
def oracle_score(net,mu,sd,ys,oracle):
    rows=[]
    for y in ys:
        for tau in np.arange(.05,.5001,.05):
            arr,exact=oracle[(int(y),round(float(tau),8))]
            rich=arr[4]
            X=np.column_stack([rich,np.tile([diag[y],f1[y],f2[y]],(D,1)),np.full(D,tau)])
            z=net.pred((X-mu)/sd); z-=z.max()
            a=np.exp(np.clip(z,-60,0)); a/=np.linalg.norm(a)
            at=np.abs(exact); at/=np.linalg.norm(at)
            rows.append({"y":int(y),"tau":float(tau),"fidelity":float(np.dot(a,at)**2)})
    return rows

def main():
    rg=np.random.default_rng(20260930); ys=special_columns(rg,2)
    Xa,ta,wa,ga,oa=collect(ys,8192,41001)
    Xb,tb,wb,gb,ob=collect(ys,8192,42001)
    print("DATA",len(Xa),len(Xb),flush=True)
    net,mu,sd=fit(Xa,ta,wa)
    pa=net.pred((Xa-mu)/sd); pb=net.pred((Xb-mu)/sd)
    tr=grouped_rmse(pa,ta,wa,ga); va=grouped_rmse(pb,tb,wb,gb)
    print("RMSE","train",tr,"val",va,flush=True)
    sc=oracle_score(net,mu,sd,ys,ob)
    for r in sc:
        if abs(r["tau"]-.25)<1e-9 or abs(r["tau"]-.5)<1e-9: print("ORACLE",r,flush=True)
    out={"ys":[int(x) for x in ys],"train_rmse":tr,"val_rmse":va,"oracle":sc}
    path=ROOT/"results/finite_tau_amplitude_regression_20site.json"; path.write_text(json.dumps(out,indent=2))
    np.savez_compressed(ROOT/"results/finite_tau_amplitude_regression_20site_weights.npz",mu=mu,sd=sd,
        W1=net.W1,b1=net.b1,W2=net.W2,b2=net.b2,W3=net.W3,b3=net.b3)
    print("WROTE",path,flush=True)
if __name__=="__main__": main()
