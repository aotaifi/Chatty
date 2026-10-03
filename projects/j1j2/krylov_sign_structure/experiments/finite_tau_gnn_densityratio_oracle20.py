#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as cs
import scipy.sparse.linalg as sla
import jax, jax.numpy as jnp
from flax import linen as nn
import optax
from flax.training.train_state import TrainState
from finite_tau_matching_20site_exact import build_H,basis,D,N,NN,NNN,special_columns

ROOT=Path("/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure")
jax.config.update("jax_platform_name","cpu")

def adj(bonds):
    A=np.zeros((N,N),np.float32)
    for u,v in bonds:A[u,v]=A[v,u]=1
    A/=np.maximum(A.sum(1,keepdims=True),1)
    return jnp.asarray(A)
A1,A2=adj(NN),adj(NNN)

class PairGNN(nn.Module):
    hidden:int=32; layers:int=3
    @nn.compact
    def __call__(self,xs,ys,g):
        h=nn.Dense(self.hidden)(jnp.stack([xs,ys,xs*ys],-1))
        for _ in range(self.layers):
            m1=jnp.einsum('ij,bjh->bih',A1,h); m2=jnp.einsum('ij,bjh->bih',A2,h)
            z=jnp.concatenate([h,m1,m2],-1)
            dh=nn.Dense(self.hidden)(nn.gelu(nn.Dense(self.hidden*2)(z)))
            h=nn.gelu(h+dh)
        z=jnp.concatenate([jnp.mean(h,1),jnp.mean(h*h,1),g],-1)
        z=nn.gelu(nn.Dense(48)(z))
        return nn.Dense(1)(z)[:,0]

def bits(states):
    s=np.asarray(states,np.uint32)
    return (2*((s[:,None]>>np.arange(N,dtype=np.uint32))&1).astype(np.float32)-1)

def make_wd(H,ys):
    co=(H-sp.diags(H.diagonal())).tocoo()
    cost=np.where(np.abs(co.data)>0.375,1000.,1001.)
    W=sp.coo_matrix((cost,(co.row,co.col)),shape=H.shape).tocsr()
    return np.asarray(cs.shortest_path(W,directed=False,indices=ys))

def globals_from_wd(wd,tau,idx):
    d=(wd[idx]//1000).astype(np.float32)
    n2=np.rint(wd[idx]-1000*d).astype(np.float32)
    return np.column_stack([d/10,n2/10,np.full(len(idx),tau/.5,np.float32)]).astype(np.float32)

class DRState(TrainState):
    bias: jnp.ndarray

@jax.jit
def step(st,x,y,g,label,ctx):
    def lf(params,bias):
        F=st.apply_fn({'params':params},x,y,g)
        z=F+bias[ctx]
        loss=optax.sigmoid_binary_cross_entropy(z,label).mean()
        return loss
    (loss),(gp,gb)=jax.value_and_grad(lf,argnums=(0,1))(st.params,st.bias)
    st=st.apply_gradients(grads=gp)
    st=st.replace(bias=st.bias-1e-3*gb)
    return st,loss
from functools import partial
@partial(jax.jit, static_argnums=(1,))
def pred(params,apply_fn,x,y,g):
    return apply_fn({'params':params},x,y,g)

def eval_full(model,params,y,wd,tau,target):
    bs=4096; out=[]; yb=bits(np.array([basis[y]]))[0]
    for i in range(0,D,bs):
        j=min(i+bs,D); idx=np.arange(i,j)
        yy=np.repeat(yb[None,:],j-i,0)
        gg=globals_from_wd(wd,tau,idx)
        out.append(np.asarray(pred(params,model.apply,jnp.asarray(bits(basis[idx])),
                                   jnp.asarray(yy),jnp.asarray(gg))))
    F=np.concatenate(out); a=np.exp(np.clip(F-F.max(),-80,0)); a/=np.linalg.norm(a)
    t=np.abs(target);t/=np.linalg.norm(t)
    return float(np.dot(a,t)**2)

def main():
    rng=np.random.default_rng(20260930)
    H,_=build_H(.5); ys=special_columns(rng,6); train_y=ys[:4]; test_y=ys[4:]
    wd=make_wd(H,ys); tau=.1
    E=np.zeros((D,len(ys)));E[ys,np.arange(len(ys))]=1
    G=np.asarray(sla.expm_multiply(-tau*H,E),float)
    G/=np.linalg.norm(G,axis=0,keepdims=True)

    ntr=6000; nva=3000
    def dataset(n):
        XX=[];YY=[];GG=[];LL=[];CC=[]
        for k,y in enumerate(train_y):
            a=np.abs(G[:,k]); pf=a/a.sum()
            ip=rng.choice(D,size=n,replace=True,p=pf)
            iq=rng.integers(0,D,size=n)
            idx=np.concatenate([ip,iq])
            XX.append(bits(basis[idx]))
            yb=bits(np.array([basis[y]]))[0]; YY.append(np.repeat(yb[None,:],2*n,0))
            GG.append(globals_from_wd(wd[k],tau,idx))
            LL.append(np.concatenate([np.ones(n,np.float32),np.zeros(n,np.float32)]))
            CC.append(np.full(2*n,k,np.int32))
        return tuple(np.concatenate(z) for z in (XX,YY,GG,LL,CC))
    tr=dataset(ntr); va=dataset(nva)
    print("TRAIN",len(tr[0]),"VAL",len(va[0]),"ys",ys,flush=True)
    model=PairGNN(); params=model.init(jax.random.PRNGKey(11),jnp.asarray(tr[0][:4]),jnp.asarray(tr[1][:4]),jnp.asarray(tr[2][:4]))['params']
    st=DRState.create(apply_fn=model.apply,params=params,tx=optax.adamw(1e-3,weight_decay=1e-6),bias=jnp.zeros(len(train_y)))
    bs=4096; hist=[]; best=None;bestv=1e9
    for ep in range(8):
        order=rng.permutation(len(tr[0])); ls=[]
        for i in range(0,len(order),bs):
            ix=order[i:i+bs]
            st,l=step(st,*[jnp.asarray(z[ix]) for z in tr])
            ls.append(float(l))
        vl=[]
        for i in range(0,len(va[0]),bs):
            F=np.asarray(model.apply({'params':st.params},jnp.asarray(va[0][i:i+bs]),jnp.asarray(va[1][i:i+bs]),jnp.asarray(va[2][i:i+bs])))
            z=F+np.asarray(st.bias)[va[4][i:i+bs]]
            y=va[3][i:i+bs]; vl.append(np.mean(np.logaddexp(0,z)-y*z))
        v=float(np.mean(vl)); hist.append([ep,float(np.mean(ls)),v]);print("EPOCH",hist[-1],flush=True)
        if v<bestv: bestv=v; best=(jax.tree_util.tree_map(lambda x:np.asarray(x),st.params),np.asarray(st.bias))
    params=best[0]
    from flax import serialization
    (ROOT/'results/finite_tau_gnn_densityratio_oracle20.mpack').write_bytes(serialization.to_bytes(params))
    rows=[]
    for split,sub,off in [('train',train_y,0),('test',test_y,4)]:
        for kk,y in enumerate(sub):
            k=off+kk; fid=eval_full(model,params,y,wd[k],tau,G[:,k])
            row={'split':split,'y':int(y),'tau':tau,'fidelity':fid};rows.append(row);print("RESULT",row,flush=True)
    out={'ys':[int(x) for x in ys],'history':hist,'rows':rows}
    path=ROOT/'results/finite_tau_gnn_densityratio_oracle20.json';path.write_text(json.dumps(out,indent=2))
    from flax import serialization
    (ROOT/'results/finite_tau_gnn_densityratio_oracle20.mpack').write_bytes(serialization.to_bytes(params))
    print("WROTE",path,flush=True)
if __name__=='__main__':main()
