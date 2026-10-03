#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as cs
import jax, jax.numpy as jnp
import optax

from finite_tau_matching_20site_exact import build_H,basis,D,N,NN,NNN,special_columns
from finite_tau_ct_local_ratio_20site import distance_sign,normalized
from finite_tau_ct_smc_edgeratio_longbeta20 import reconstruct
from k2_sr_edge_tangent_oneblock20 import edge_set,wrmse

H,_=build_H(.5)

def adj(bonds):
    A=np.zeros((N,N),np.float32)
    for u,v in bonds:
        A[u,v]=A[v,u]=1.
    A/=np.maximum(A.sum(1,keepdims=True),1.)
    return jnp.asarray(A)
A1=adj(NN);A2=adj(NNN)

def bits_idx(idx):
    st=basis[np.asarray(idx,np.int64)]
    b=((st[:,None]>>np.arange(N,dtype=np.uint32))&1).astype(np.float32)
    return 2*b-1

def make_wd(y):
    co=(H-sp.diags(H.diagonal())).tocoo()
    cost=np.where(np.abs(co.data)>0.375,1000.,1001.)
    W=sp.coo_matrix((cost,(co.row,co.col)),shape=H.shape).tocsr()
    return np.asarray(cs.shortest_path(W,directed=False,indices=[y]))[0]

def glob(wd,tau,idx):
    idx=np.asarray(idx,np.int64)
    d=(wd[idx]//1000).astype(np.float32)
    n2=np.rint(wd[idx]-1000*d).astype(np.float32)
    return np.column_stack([d/10,n2/10,np.full(len(idx),tau/.5,np.float32)])

def init_params(key,h=32,layers=3):
    ks=jax.random.split(key,4+layers)
    def rn(k,sh,scale):
        return scale*jax.random.normal(k,sh)
    ps={"W0":rn(ks[0],(3,h),.2),"b0":jnp.zeros(h),
        "layers":tuple({"W":rn(ks[1+i],(3*h,h),.12),"b":jnp.zeros(h)} for i in range(layers)),
        "Wp":rn(ks[-3],(2*h+3,48),.12),"bp":jnp.zeros(48),
        "Wo":rn(ks[-2],(48,1),.12),"bo":jnp.zeros(1)}
    return ps

def apply(params,x,y,g):
    h=jnp.stack([x,y,x*y],axis=-1)
    h=jnp.tanh(jnp.einsum("bif,fh->bih",h,params["W0"])+params["b0"])
    for q in params["layers"]:
        m1=jnp.einsum("ij,bjh->bih",A1,h)
        m2=jnp.einsum("ij,bjh->bih",A2,h)
        z=jnp.concatenate([h,m1,m2],axis=-1)
        dh=jax.nn.gelu(jnp.einsum("bif,fh->bih",z,q["W"])+q["b"])
        h=jnp.tanh(h+dh)
    pool=jnp.concatenate([jnp.mean(h,axis=1),jnp.mean(h*h,axis=1),g],axis=-1)
    z=jax.nn.gelu(pool@params["Wp"]+params["bp"])
    return (z@params["Wo"]+params["bo"])[:,0]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--M",type=int,default=100000)
    ap.add_argument("--dt",type=float,default=.0125)
    ap.add_argument("--ns",type=int,default=2000)
    ap.add_argument("--epochs",type=int,default=20)
    ap.add_argument("--hidden",type=int,default=32)
    ap.add_argument("--layers",type=int,default=3)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()

    y=int(special_columns(np.random.default_rng(20260930),2)[1])
    wd=make_wd(y); s=distance_sign(y)
    _,_,logg=reconstruct(y,s,a.M)
    g=normalized(np.exp(np.clip(logg-logg.max(),-700,0))); p=g*g;p/=p.sum()
    psi=s*g; h1=H@psi
    psi2=normalized(psi-a.dt*h1+.5*a.dt*a.dt*(H@h1)); amp2=np.abs(psi2)
    targ=np.log(np.maximum(amp2,1e-300))-np.log(np.maximum(g,1e-300))
    tau=.5+a.dt; rg=np.random.default_rng(2500001+a.M)
    tr=rg.choice(D,a.ns,p=p); va=rg.choice(D,a.ns,p=p)
    tt=(targ[tr]-np.mean(targ[tr])).astype(np.float32)
    vt=(targ[va]-np.mean(targ[va])).astype(np.float32)
    xb=bits_idx(tr); vb=bits_idx(va); yb=bits_idx([y])[0]
    Y=np.repeat(yb[None,:],len(tr),axis=0); VY=np.repeat(yb[None,:],len(va),axis=0)
    G=glob(wd,tau,tr); VG=glob(wd,tau,va)

    params=init_params(jax.random.PRNGKey(7),a.hidden,a.layers)
    opt=optax.adamw(5e-4,weight_decay=1e-6); state=opt.init(params)

    @jax.jit
    def train_step(params,state,x,yv,gg,t):
        def lf(pp):
            pr=apply(pp,x,yv,gg)
            return jnp.mean((pr-t)**2)
        loss,gr=jax.value_and_grad(lf)(params)
        up,state=opt.update(gr,state,params)
        return optax.apply_updates(params,up),state,loss

    best=1e30;bestp=None;hist=[];bs=512
    for ep in range(a.epochs):
        order=rg.permutation(len(tr)); ls=[]
        for q in range(0,len(order),bs):
            z=order[q:q+bs]
            params,state,l=train_step(params,state,jnp.asarray(xb[z]),jnp.asarray(Y[z]),jnp.asarray(G[z]),jnp.asarray(tt[z]))
            ls.append(float(l))
        pv=np.asarray(apply(params,jnp.asarray(vb),jnp.asarray(VY),jnp.asarray(VG)))
        rm=float(np.sqrt(np.mean((pv-vt)**2)))
        hist.append([ep,float(np.sqrt(np.mean(ls))),rm])
        if rm<best:
            best=rm; bestp=jax.tree_util.tree_map(lambda x:np.asarray(x).copy(),params)
        if ep in (0,1,4,9,14,19,a.epochs-1):
            print("EPOCH",ep+1,"val_rmse",rm,flush=True)
    params=jax.tree_util.tree_map(jnp.asarray,bestp)

    allf=[]
    for q in range(0,D,4096):
        ix=np.arange(q,min(q+4096,D))
        yy=np.repeat(yb[None,:],len(ix),axis=0)
        allf.append(np.asarray(apply(params,jnp.asarray(bits_idx(ix)),jnp.asarray(yy),jnp.asarray(glob(wd,tau,ix)))))
    f=np.concatenate(allf); f-=np.sum(p*f)
    gp=normalized(g*np.exp(np.clip(f,-20,20)))

    src,dst,w=edge_set(va); pe=f[dst]-f[src]; te=targ[dst]-targ[src]
    seenmask=np.zeros(D,dtype=bool);seenmask[np.unique(tr)]=True
    seen=seenmask[dst]
    out={"M":a.M,"dt":a.dt,"ns":a.ns,"hidden":a.hidden,"layers":a.layers,
         "best_val_rmse":best,"target_fid":float((gp@amp2)**2),
         "edge_seenY_frac":float(seen.mean()),
         "edge_base_rmse":wrmse(te,w),"edge_proj_rmse":wrmse(pe-te,w),
         "edge_base_seenY":wrmse(te[seen],w[seen]) if np.any(seen) else None,
         "edge_proj_seenY":wrmse((pe-te)[seen],w[seen]) if np.any(seen) else None,
         "edge_base_unseenY":wrmse(te[~seen],w[~seen]) if np.any(~seen) else None,
         "edge_proj_unseenY":wrmse((pe-te)[~seen],w[~seen]) if np.any(~seen) else None,
         "history":hist}
    print("RESULT",json.dumps(out,sort_keys=True),flush=True)
    Path(a.out).write_text(json.dumps(out,indent=2))
if __name__=="__main__":main()
