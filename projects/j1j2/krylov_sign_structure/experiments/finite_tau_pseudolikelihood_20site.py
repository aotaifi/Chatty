#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as cs
import scipy.sparse.linalg as sla
from finite_tau_matching_20site_exact import build_H,basis,D,N,NN,NNN,special_columns
from finite_tau_weighted_smc_20site import propagate_weighted,update_theta_weighted,systematic,ess

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

def feat(arrs,idx,tau,y):
    d,n2,_,_=arrs; ii=np.asarray(idx,dtype=np.int64)
    return np.column_stack([d[ii],n2[ii],diag[ii],f1[ii],f2[ii],
        np.full(len(ii),diag[y]),np.full(len(ii),f1[y]),np.full(len(ii),f2[y]),np.full(len(ii),tau)])
def collect_pairs(ys,M,seed,K=4,block_dt=.05,beta=.5,tau_max=.005,eta=.5,prior=32.,oracle=False):
    rng=np.random.default_rng(seed); AA=[]; BB=[]; WW=[]; META=[]; orc={}; amap={}
    for y0 in ys:
        y=int(y0); arrs=endpoint_arrays(y); amap[y]=arrs
        d,n2,lab,bs=arrs; sgn=np.where((d.astype(np.int64)%2)==0,1,-1).astype(np.int8)
        theta=np.zeros(len(bs)); states=np.full(M,y,dtype=np.int32); pw=np.ones(M)
        exact=np.zeros(D); exact[y]=1.
        for n in range(1,int(round(beta/block_dt))+1):
            tau=round(n*block_dt,8); old=theta.copy()
            states,pw,_=propagate_weighted(states,pw,old,lab,sgn,block_dt,tau_max,rng,.35)
            # importance weights convert f=g*a samples to amplitude-measure samples.
            aw=pw*np.exp(np.clip(-old[lab[states]],-40,40)); aw*=M/aw.sum()
            # K random neighbors per particle, unbiased for sum over physical exchange neighbors.
            xs=[]; zs=[]; ws=[]; meta=[]
            for x,w0 in zip(states,aw):
                a,b=HO.indptr[int(x)],HO.indptr[int(x)+1]; nb=HO.indices[a:b]
                if len(nb)==0: continue
                kk=min(K,len(nb)); pick=rng.choice(len(nb),size=kk,replace=False)
                fac=len(nb)/kk
                for j in pick:
                    xs.append(int(x)); zs.append(int(nb[j])); ws.append(float(w0*fac))
                    meta.append((y,tau,int(x),int(nb[j])))
            AA.append(feat(arrs,xs,tau,y)); BB.append(feat(arrs,zs,tau,y)); WW.append(np.asarray(ws)); META.extend(meta)
            new,cnt=update_theta_weighted(old,lab,bs,states,pw,eta,prior)
            pw*=np.exp(np.clip(new[lab[states]]-old[lab[states]],-30,30)); pw*=M/pw.sum(); theta=new
            if ess(pw)<.35*M:
                states=states[systematic(pw,rng,M)]; pw=np.ones(M)
            if oracle:
                exact=np.asarray(sla.expm_multiply(-block_dt*H,exact),float); exact/=np.linalg.norm(exact)
                orc[(y,tau)]=exact.copy()
    return np.vstack(AA),np.vstack(BB),np.concatenate(WW),META,orc,amap
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
        return (h2@self.W3+self.b3).ravel(),(X,h1,h2)
    def backward(self,c,dz):
        X,h1,h2=c
        gW3=h2.T@dz[:,None]; gb3=np.array([dz.sum()])
        dz2=(dz[:,None]@self.W3.T)*(1-h2*h2)
        gW2=h1.T@dz2; gb2=dz2.sum(0)
        dz1=(dz2@self.W2.T)*(1-h1*h1)
        return [X.T@dz1,dz1.sum(0),gW2,gb2,gW3,gb3]
    def step(self,A,B,w,lr=1e-3):
        za,ca=self.forward(A); zb,cb=self.forward(B)
        q=1/(1+np.exp(-np.clip(zb-za,-30,30)))
        ww=w/max(w.mean(),1e-12); dz=-ww*q/len(w)
        ga=self.backward(ca,dz); gb=self.backward(cb,-dz); grads=[a+b for a,b in zip(ga,gb)]
        self.t+=1
        for p,g,m,v in zip(self.params(),grads,self.m,self.v):
            m*=.9; m+=.1*g; v*=.999; v+=.001*g*g
            p-=lr*(m/(1-.9**self.t))/(np.sqrt(v/(1-.999**self.t))+1e-8)
        return float(np.average(np.logaddexp(0,zb-za),weights=w))
    def pred(self,X): return self.forward(X)[0]

def ploss(net,A,B,w):
    za=net.pred(A); zb=net.pred(B); return float(np.average(np.logaddexp(0,zb-za),weights=w))
def train(A,B,w,epochs=30,batch=4096,seed=7):
    X=np.vstack([A,B]); mu=X.mean(0); sd=X.std(0); sd[sd<1e-8]=1
    A=(A-mu)/sd; B=(B-mu)/sd; net=MLP(A.shape[1],48,seed); rg=np.random.default_rng(seed)
    for ep in range(epochs):
        o=rg.permutation(len(w))
        for i in range(0,len(w),batch):
            j=o[i:i+batch]; net.step(A[j],B[j],w[j])
        if ep in (0,4,9,19,29): print("TRAIN",ep+1,"ploss",ploss(net,A,B,w),flush=True)
    return net,mu,sd

def exact_ratio_rmse(net,mu,sd,A,B,w,meta,orc):
    pr=net.pred((A-mu)/sd)-net.pred((B-mu)/sd); t=[]; p=[]; ww=[]
    for k,(y,tau,x,z) in enumerate(meta):
        ex=np.abs(orc[(y,tau)])
        if ex[x]<=1e-15 or ex[z]<=1e-15: continue
        t.append(np.log(ex[x]/ex[z])); p.append(pr[k]); ww.append(w[k])
    t=np.asarray(t); p=np.asarray(p); ww=np.asarray(ww)
    return float(np.sqrt(np.average((p-t)**2,weights=ww))),float(np.corrcoef(p,t)[0,1])

def fidelity(net,mu,sd,ys,orc,amap):
    out=[]
    for y0 in ys:
        y=int(y0); arrs=amap[y]
        for tau in (.25,.5):
            X=feat(arrs,np.arange(D),tau,y); z=net.pred((X-mu)/sd); z-=z.max()
            a=np.exp(np.clip(z,-60,0)); a/=np.linalg.norm(a)
            ex=np.abs(orc[(y,round(tau,8))]); ex/=np.linalg.norm(ex)
            out.append({"y":y,"tau":tau,"fidelity":float(np.dot(a,ex)**2)})
    return out
def main():
    rg=np.random.default_rng(20260930); ys=special_columns(rg,2)
    A1,B1,w1,m1,_,_=collect_pairs(ys,4096,61001,K=4,oracle=False)
    A2,B2,w2,m2,orc,amap=collect_pairs(ys,4096,62001,K=4,oracle=True)
    print("PAIRS",len(w1),len(w2),flush=True)
    net,mu,sd=train(A1,B1,w1)
    tr=ploss(net,(A1-mu)/sd,(B1-mu)/sd,w1); va=ploss(net,(A2-mu)/sd,(B2-mu)/sd,w2)
    er=exact_ratio_rmse(net,mu,sd,A2,B2,w2,m2,orc)
    ff=fidelity(net,mu,sd,ys,orc,amap)
    print("VALID","train_pl",tr,"val_pl",va,"exact_ratio",er,flush=True)
    for r in ff: print("ORACLE",r,flush=True)
    out={"ys":[int(x) for x in ys],"npairs_train":len(w1),"npairs_val":len(w2),
         "train_ploss":tr,"val_ploss":va,"exact_ratio_rmse":er[0],"exact_ratio_corr":er[1],"oracle":ff}
    path=ROOT/"results/finite_tau_pseudolikelihood_20site.json"; path.write_text(json.dumps(out,indent=2))
    np.savez_compressed(ROOT/"results/finite_tau_pseudolikelihood_20site_weights.npz",
        mu=mu,sd=sd,W1=net.W1,b1=net.b1,W2=net.W2,b2=net.b2,W3=net.W3,b3=net.b3)
    print("WROTE",path,flush=True)
if __name__=="__main__": main()
