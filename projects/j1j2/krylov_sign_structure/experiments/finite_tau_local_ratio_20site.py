#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as cs
import scipy.sparse.linalg as sla
from finite_tau_matching_20site_exact import build_H,basis,D,N,NN,NNN,special_columns
from finite_tau_table_gfmc_20site import propagate,update_theta,systematic

ROOT=Path("/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure")
H,diag=build_H(.5); HO=(H-sp.diags(diag)).tocsr()
f1=np.zeros(D); f2=np.zeros(D)
for u,v in NN: f1 += (((basis>>u)^(basis>>v))&1)
for u,v in NNN: f2 += (((basis>>u)^(basis>>v))&1)

def endpoint_arrays(y):
    co=HO.tocoo(); cost=np.where(np.abs(co.data)>0.375,1000.0,1001.0)
    W=sp.coo_matrix((cost,(co.row,co.col)),shape=H.shape).tocsr()
    wd=np.asarray(cs.shortest_path(W,directed=False,indices=[y]))[0]
    d=(wd//1000).astype(float); n2=(wd-1000*d).round().astype(float)
    ec=np.rint(8*diag).astype(np.int32)+100
    raw=d.astype(np.int64)*100000+n2.astype(np.int64)*1000+ec.astype(np.int64)
    _,lab=np.unique(raw,return_inverse=True)
    return d,n2,lab.astype(np.int32),np.bincount(lab).astype(float)

def features(arrs,idx,tau,y):
    d,n2,_,_=arrs; ii=np.asarray(idx,dtype=np.int64)
    return np.column_stack([d[ii],n2[ii],diag[ii],f1[ii],f2[ii],
                            np.full(len(ii),diag[y]),np.full(len(ii),f1[y]),
                            np.full(len(ii),f2[y]),np.full(len(ii),tau)])
def collect_replica(ys,M,seed,block_dt=.05,beta=.5,tau_max=.005,eta=.5,prior=32.):
    rng=np.random.default_rng(seed); counts={}; oracle={}; arrmap={}
    for ky,y0 in enumerate(ys):
        y=int(y0); arrs=endpoint_arrays(y); arrmap[y]=arrs
        d,n2,lab,bs=arrs; sgn=np.where((d.astype(np.int64)%2)==0,1,-1).astype(np.int8)
        theta=np.zeros(len(bs)); walkers=np.full(M,y,dtype=np.int32)
        exact=np.zeros(D); exact[y]=1.
        for n in range(1,int(round(beta/block_dt))+1):
            tau=round(n*block_dt,8); old=theta.copy()
            walkers,_=propagate(walkers,old,lab,sgn,block_dt,tau_max,rng)
            lw=-old[lab[walkers]]; lw-=lw.max(); ww=np.exp(lw); ww*=M/ww.sum()
            c=np.bincount(walkers,weights=ww,minlength=D)
            counts[(y,tau)]=c
            theta,rw,_,_,_=update_theta(old,lab,bs,walkers,eta,prior,-1.)
            walkers=walkers[systematic(rw,rng,M)]
            exact=np.asarray(sla.expm_multiply(-block_dt*H,exact),float); exact/=np.linalg.norm(exact)
            oracle[(y,tau)]=exact.copy()
    return counts,oracle,arrmap

def edge_dataset(counts,arrmap,minc=.15,maxabs=8.):
    XL=[]; XR=[]; T=[]; W=[]; meta=[]
    for (y,tau),c in counts.items():
        arrs=arrmap[y]
        vis=np.flatnonzero(c>minc)
        for x in vis:
            a,b=HO.indptr[x],HO.indptr[x+1]
            for z in HO.indices[a:b]:
                z=int(z)
                if z<=x or c[z]<=minc: continue
                t=float(np.clip(np.log(c[z]/c[x]),-maxabs,maxabs))
                w=float(np.sqrt(c[x]*c[z]))
                XL.append(features(arrs,[x],tau,y)[0]); XR.append(features(arrs,[z],tau,y)[0])
                T.append(t); W.append(w); meta.append((y,tau,int(x),z))
    return np.asarray(XL),np.asarray(XR),np.asarray(T),np.asarray(W),meta
class MLP:
    def __init__(self,nin,h=48,seed=7):
        rg=np.random.default_rng(seed)
        self.W1=rg.normal(0,.2,(nin,h)); self.b1=np.zeros(h)
        self.W2=rg.normal(0,.2,(h,h)); self.b2=np.zeros(h)
        self.W3=rg.normal(0,.2,(h,1)); self.b3=np.zeros(1)
        self.m=[np.zeros_like(x) for x in self.params()]; self.v=[np.zeros_like(x) for x in self.params()]; self.t=0
    def params(self): return [self.W1,self.b1,self.W2,self.b2,self.W3,self.b3]
    def forward(self,X):
        h1=np.tanh(X@self.W1+self.b1); h2=np.tanh(h1@self.W2+self.b2)
        z=(h2@self.W3+self.b3).ravel()
        return z,(X,h1,h2)
    def backward_from_dz(self,cache,dz):
        X,h1,h2=cache
        gW3=h2.T@dz[:,None]; gb3=np.array([dz.sum()])
        dh2=dz[:,None]@self.W3.T; dz2=dh2*(1-h2*h2)
        gW2=h1.T@dz2; gb2=dz2.sum(0)
        dh1=dz2@self.W2.T; dz1=dh1*(1-h1*h1)
        gW1=X.T@dz1; gb1=dz1.sum(0)
        return [gW1,gb1,gW2,gb2,gW3,gb3]
    def step_pairs(self,A,B,t,w,lr=1e-3):
        za,ca=self.forward(A); zb,cb=self.forward(B); e=za-zb-t
        ww=w/max(w.mean(),1e-12); dz=2*ww*e/len(e)
        ga=self.backward_from_dz(ca,dz); gb=self.backward_from_dz(cb,-dz)
        grads=[a+b for a,b in zip(ga,gb)]; self.t+=1
        for p,g,m,v in zip(self.params(),grads,self.m,self.v):
            m*=.9; m+=.1*g; v*=.999; v+=.001*g*g
            mh=m/(1-.9**self.t); vh=v/(1-.999**self.t)
            p-=lr*mh/(np.sqrt(vh)+1e-8)
        return float(np.average(e*e,weights=w))
    def pred(self,X): return self.forward(X)[0]

def wrmse(pred,t,w): return float(np.sqrt(np.average((pred-t)**2,weights=w)))
def train_ratio(A,B,t,w,seed=7,epochs=40,batch=2048):
    X=np.vstack([A,B]); mu=X.mean(0); sd=X.std(0); sd[sd<1e-8]=1
    A=(A-mu)/sd; B=(B-mu)/sd
    net=MLP(A.shape[1],48,seed); rg=np.random.default_rng(seed)
    for ep in range(epochs):
        o=rg.permutation(len(t)); ls=[]
        for i in range(0,len(t),batch):
            j=o[i:i+batch]; ls.append(net.step_pairs(A[j],B[j],t[j],w[j]))
        if ep in (0,4,9,19,29,39):
            pr=net.pred(A)-net.pred(B)
            print("TRAIN",ep+1,"rmse",wrmse(pr,t,w),"loss",np.mean(ls),flush=True)
    return net,mu,sd

def eval_pairs(net,mu,sd,A,B,t,w):
    p=net.pred((A-mu)/sd)-net.pred((B-mu)/sd)
    return wrmse(p,t,w),float(np.corrcoef(p,t)[0,1])

def exact_pair_rmse(net,mu,sd,A,B,w,meta,oracle,arrmap):
    pred=net.pred((A-mu)/sd)-net.pred((B-mu)/sd)
    tar=[]; ww=[]; pp=[]
    for k,(y,tau,x,z) in enumerate(meta):
        ex=np.abs(oracle[(y,tau)])
        if ex[x]<=1e-15 or ex[z]<=1e-15: continue
        tar.append(np.log(ex[x]/ex[z])); pp.append(pred[k]); ww.append(w[k])
    return wrmse(np.asarray(pp),np.asarray(tar),np.asarray(ww))

def full_fidelity(net,mu,sd,ys,oracle,arrmap):
    out=[]
    for y0 in ys:
        y=int(y0); arrs=arrmap[y]
        for tau in (0.25,0.5):
            X=features(arrs,np.arange(D),tau,y)
            z=net.pred((X-mu)/sd); z-=z.max()
            a=np.exp(np.clip(z,-60,0)); a/=np.linalg.norm(a)
            ex=np.abs(oracle[(y,round(tau,8))]); ex/=np.linalg.norm(ex)
            out.append({"y":y,"tau":tau,"fidelity":float(np.dot(a,ex)**2)})
    return out

def main():
    rg=np.random.default_rng(20260930); ys=special_columns(rg,2)
    ca,oa,aa=collect_replica(ys,8192,41001)
    cb,ob,ab=collect_replica(ys,8192,42001)
    A1,B1,t1,w1,m1=edge_dataset(ca,aa); A2,B2,t2,w2,m2=edge_dataset(cb,ab)
    print("EDGES","train",len(t1),"val",len(t2),flush=True)
    net,mu,sd=train_ratio(A1,B1,t1,w1)
    tr=eval_pairs(net,mu,sd,A1,B1,t1,w1); va=eval_pairs(net,mu,sd,A2,B2,t2,w2)
    exrm=exact_pair_rmse(net,mu,sd,A2,B2,w2,m2,ob,ab)
    ff=full_fidelity(net,mu,sd,ys,ob,ab)
    print("PAIR","train",tr,"val",va,"exact_val_rmse",exrm,flush=True)
    for r in ff: print("ORACLE",r,flush=True)
    out={"ys":[int(x) for x in ys],"ntrain":len(t1),"nval":len(t2),
         "train_rmse":tr[0],"train_corr":tr[1],"val_rmse":va[0],"val_corr":va[1],
         "exact_val_ratio_rmse":exrm,"oracle":ff}
    path=ROOT/"results/finite_tau_local_ratio_20site.json"; path.write_text(json.dumps(out,indent=2))
    np.savez_compressed(ROOT/"results/finite_tau_local_ratio_20site_weights.npz",
        mu=mu,sd=sd,W1=net.W1,b1=net.b1,W2=net.W2,b2=net.b2,W3=net.W3,b3=net.b3)
    print("WROTE",path,flush=True)
if __name__=="__main__": main()
