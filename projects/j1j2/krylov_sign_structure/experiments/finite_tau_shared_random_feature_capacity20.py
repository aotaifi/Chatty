#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as cs
import scipy.sparse.linalg as sla
from finite_tau_matching_20site_exact import build_H,basis,D,N,NN,NNN,special_columns

ROOT=Path("/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure")
rng=np.random.default_rng(20260930)
H,diag=build_H(.5)
HO=(H-sp.diags(diag)).tocoo()
cost=np.where(np.abs(HO.data)>0.375,1000.0,1001.0)
W=sp.coo_matrix((cost,(HO.row,HO.col)),shape=H.shape).tocsr()

# state-only polynomial features
f1=np.zeros(D,np.float64); f2=np.zeros(D,np.float64)
for u,v in NN: f1 += (((basis>>u)^(basis>>v))&1)
for u,v in NNN: f2 += (((basis>>u)^(basis>>v))&1)
ediag=np.asarray(diag,float)

def pair_raw(y,wd,tau):
    d=(wd//1000).astype(float); n2=np.rint(wd-1000*d)
    by=basis[y]
    xor=np.bitwise_xor(basis,by)
    # uint64 popcount via bytes lookup
    bytes_=xor.view(np.uint8).reshape(-1,xor.dtype.itemsize)
    pc=np.unpackbits(bytes_,axis=1).sum(1).astype(float)/2.0
    fx1=np.zeros(D,float); fx2=np.zeros(D,float)
    for u,v in NN: fx1 += (((xor>>u)^(xor>>v))&1)
    for u,v in NNN: fx2 += (((xor>>u)^(xor>>v))&1)
    return np.column_stack([d,n2,pc,ediag,f1,f2,
                            np.full(D,ediag[y]),np.full(D,f1[y]),np.full(D,f2[y]),
                            fx1,fx2,np.full(D,tau)])

def norm_raw(X):
    scales=np.array([10,10,10,20,40,40,20,40,40,40,40,0.5],float)
    return X/scales

ys=special_columns(rng,6)
train_y=ys[:4]; test_y=ys[4:]
wd=np.asarray(cs.shortest_path(W,directed=False,indices=ys))
taus=(0.1,0.25,0.5)
def random_map(X,R,b):
    Z=np.tanh(X@R+b)
    return np.column_stack([np.ones(len(X)),X,Z])

def fit_and_eval(hidden,lam=1e-5):
    rr=np.random.default_rng(991+hidden)
    R=rr.normal(scale=.8,size=(12,hidden))
    b=rr.uniform(-np.pi,np.pi,size=hidden)
    P=1+12+hidden
    A=np.eye(P)*lam; A[0,0]=1e-10
    B=np.zeros(P)
    train_rows=[]
    for tau in taus:
        E=np.zeros((D,len(train_y))); E[train_y,np.arange(len(train_y))]=1.0
        G=np.asarray(sla.expm_multiply(-tau*H,E),float)
        for k,y in enumerate(train_y):
            a=np.abs(G[:,k]); a/=np.linalg.norm(a); w=a*a
            X=norm_raw(pair_raw(y,wd[k],tau))
            Phi=random_map(X,R,b)
            target=np.log(np.maximum(a,1e-300))
            # weighted normal equations, weight p=a^2
            WP=Phi*w[:,None]
            A += Phi.T@WP
            B += Phi.T@(w*target)
    theta=np.linalg.solve(A,B)

    rows=[]
    for split,subys,offset in [("train",train_y,0),("test",test_y,len(train_y))]:
      for tau in taus:
        E=np.zeros((D,len(subys))); E[subys,np.arange(len(subys))]=1.0
        G=np.asarray(sla.expm_multiply(-tau*H,E),float)
        for k,y in enumerate(subys):
            a=np.abs(G[:,k]); a/=np.linalg.norm(a)
            X=norm_raw(pair_raw(y,wd[offset+k],tau))
            pred=random_map(X,R,b)@theta
            pred=np.exp(np.clip(pred-np.max(pred),-80,0)); pred/=np.linalg.norm(pred)
            fid=float(np.dot(pred,a)**2)
            rows.append({"split":split,"tau":tau,"y":int(y),"fidelity":fid})
            print("H",hidden,rows[-1],flush=True)
    return rows

out={"ys":[int(x) for x in ys],"train_y":[int(x) for x in train_y],"test_y":[int(x) for x in test_y],"models":{}}
for h in (32,64,128,256):
    out["models"][str(h)]=fit_and_eval(h)
path=ROOT/"results/finite_tau_shared_random_feature_capacity20.json"
path.write_text(json.dumps(out,indent=2))
print("WROTE",path)
