#!/usr/bin/env python3
import json, math
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as cs
import scipy.sparse.linalg as sla

import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
from flax.training.train_state import TrainState

from finite_tau_matching_20site_exact import build_H,basis,D,N,NN,NNN,special_columns
ROOT=Path("/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure")
jax.config.update("jax_platform_name","cpu")

# normalized graph adjacencies
def adj(bonds):
    A=np.zeros((N,N),np.float32)
    for u,v in bonds: A[u,v]=A[v,u]=1.0
    A/=np.maximum(A.sum(1,keepdims=True),1.0)
    return jnp.asarray(A)
A1=adj(NN); A2=adj(NNN)

class PairGNN(nn.Module):
    hidden:int=48
    layers:int=4
    @nn.compact
    def __call__(self,xs,ys,g):
        h=jnp.stack([xs,ys,xs*ys],axis=-1)
        h=nn.Dense(self.hidden)(h)
        for _ in range(self.layers):
            m1=jnp.einsum('ij,bjh->bih',A1,h)
            m2=jnp.einsum('ij,bjh->bih',A2,h)
            z=jnp.concatenate([h,m1,m2],axis=-1)
            dh=nn.Dense(self.hidden)(nn.gelu(nn.Dense(self.hidden*2)(z)))
            h=nn.gelu(h+dh)
        pooled=jnp.concatenate([jnp.mean(h,axis=1),jnp.mean(h*h,axis=1)],axis=-1)
        z=jnp.concatenate([pooled,g],axis=-1)
        z=nn.gelu(nn.Dense(64)(z))
        z=nn.gelu(nn.Dense(32)(z))
        return nn.Dense(1)(z)[:,0]

def bits(states):
    states=np.asarray(states,dtype=np.uint32)
    b=((states[:,None]>>np.arange(N,dtype=np.uint32))&1).astype(np.float32)
    return 2*b-1

def make_wd(H,ys):
    diag=H.diagonal()
    co=(H-sp.diags(diag)).tocoo()
    cost=np.where(np.abs(co.data)>0.375,1000.0,1001.0)
    W=sp.coo_matrix((cost,(co.row,co.col)),shape=H.shape).tocsr()
    return np.asarray(cs.shortest_path(W,directed=False,indices=ys))
def context_arrays(Gcol,y,wd,tau):
    a=np.abs(Gcol); a/=np.linalg.norm(a); p=a*a
    d=(wd//1000).astype(np.float32)
    n2=np.rint(wd-1000*d).astype(np.float32)
    target=np.log(np.maximum(a,1e-30))
    target-=np.sum(p*target)
    glob=np.column_stack([d/10.0,n2/10.0,np.full(D,tau/0.5,np.float32)])
    return a,p,target.astype(np.float32),glob.astype(np.float32)

@jax.jit
def train_step(st,x,y,g,t):
    def lossfn(params):
        pred=st.apply_fn({'params':params},x,y,g)
        return jnp.mean((pred-t)**2)
    loss,gr=jax.value_and_grad(lossfn)(st.params)
    return st.apply_gradients(grads=gr),loss

@jax.jit
def predict(params,apply_fn,x,y,g):
    return apply_fn({'params':params},x,y,g)

def main():
    rng=np.random.default_rng(20260930)
    H,_=build_H(.5)
    ys=special_columns(rng,6)
    train_y=ys[:4]; test_y=ys[4:]
    wd=make_wd(H,ys)
    taus=(0.1,0.25,0.5)
    contexts={}
    for tau in taus:
        E=np.zeros((D,len(ys))); E[ys,np.arange(len(ys))]=1.0
        G=np.asarray(sla.expm_multiply(-tau*H,E),float)
        G/=np.linalg.norm(G,axis=0,keepdims=True)
        for k,y in enumerate(ys):
            contexts[(k,tau)]=(G[:,k].copy(),)+context_arrays(G[:,k],y,wd[k],tau)[1:]

    ns=3000
    X=[];Y=[];GG=[];TT=[]
    for k,y in enumerate(train_y):
      for tau in taus:
        Gcol,p,targ,glob=contexts[(k,tau)]
        idx=rng.choice(D,size=ns,replace=True,p=p)
        X.append(bits(basis[idx])); Y.append(np.repeat(bits(np.array([basis[y]])),ns,axis=0))
        GG.append(glob[idx]); TT.append(targ[idx])
    X=np.concatenate(X);Y=np.concatenate(Y);GG=np.concatenate(GG);TT=np.concatenate(TT)
    perm=rng.permutation(len(X)); X=X[perm];Y=Y[perm];GG=GG[perm];TT=TT[perm]
    nval=len(X)//10
    val=(X[:nval],Y[:nval],GG[:nval],TT[:nval])
    X=X[nval:];Y=Y[nval:];GG=GG[nval:];TT=TT[nval:]
    print("TRAIN",len(X),"VAL",nval,"ys",ys,flush=True)

    model=PairGNN()
    params=model.init(jax.random.PRNGKey(7),jnp.asarray(X[:4]),jnp.asarray(Y[:4]),jnp.asarray(GG[:4]))['params']
    st=TrainState.create(apply_fn=model.apply,params=params,tx=optax.adamw(1e-3,weight_decay=1e-6))
    bs=2048
    history=[]
    for ep in range(8):
        order=rng.permutation(len(X))
        losses=[]
        for i in range(0,len(X),bs):
            ix=order[i:i+bs]
            st,loss=train_step(st,jnp.asarray(X[ix]),jnp.asarray(Y[ix]),jnp.asarray(GG[ix]),jnp.asarray(TT[ix]))
            losses.append(float(loss))
        vp=[]
        for i in range(0,nval,bs):
            pr=np.asarray(model.apply({'params':st.params},jnp.asarray(val[0][i:i+bs]),jnp.asarray(val[1][i:i+bs]),jnp.asarray(val[2][i:i+bs])))
            vp.append(np.mean((pr-val[3][i:i+bs])**2))
        vm=float(np.mean(vp)); history.append([ep,float(np.mean(losses)),vm])
        print("EPOCH",ep,"train",history[-1][1],"val",vm,flush=True)
    rows=[]
    for split,subys,offset in [('train',train_y,0),('test',test_y,len(train_y))]:
      for tau in taus:
        for kk,y in enumerate(subys):
            k=offset+kk
            Gcol,p,targ,glob=contexts[(k,tau)]
            pred=[]
            xb=bits(basis); yb=bits(np.array([basis[y]]))[0]
            for i in range(0,D,4096):
                yy=np.repeat(yb[None,:],min(4096,D-i),axis=0)
                pr=np.asarray(model.apply({'params':st.params},jnp.asarray(xb[i:i+4096]),jnp.asarray(yy),jnp.asarray(glob[i:i+4096])))
                pred.append(pr)
            pred=np.concatenate(pred)
            aa=np.exp(np.clip(pred-np.max(pred),-80,0)); aa/=np.linalg.norm(aa)
            target=np.abs(Gcol); target/=np.linalg.norm(target)
            fid=float(np.dot(aa,target)**2)
            row={'split':split,'tau':tau,'y':int(y),'fidelity':fid}
            rows.append(row);print("RESULT",row,flush=True)
    out={'ys':[int(z) for z in ys],'train_y':[int(z) for z in train_y],'test_y':[int(z) for z in test_y],
         'history':history,'rows':rows}
    path=ROOT/'results/finite_tau_shared_gnn_capacity20.json'
    path.write_text(json.dumps(out,indent=2))
    # save params as msgpack
    from flax import serialization
    (ROOT/'results/finite_tau_shared_gnn_capacity20.mpack').write_bytes(serialization.to_bytes(st.params))
    print("WROTE",path,flush=True)
if __name__=='__main__': main()
