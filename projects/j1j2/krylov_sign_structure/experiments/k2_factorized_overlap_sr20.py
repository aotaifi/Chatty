#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as cs
import scipy.sparse.linalg as sla
import jax, jax.numpy as jnp
from flax import linen as nn
from jax.flatten_util import ravel_pytree

from finite_tau_matching_20site_exact import build_H,basis,D,N,NN,NNN

def normalized(v):
    v=np.asarray(v,float)
    return v/np.linalg.norm(v)

def adj(bonds):
    A=np.zeros((N,N),np.float32)
    for u,v in bonds:
        A[u,v]=A[v,u]=1
    A/=np.maximum(A.sum(1,keepdims=True),1)
    return jnp.asarray(A)
A1,A2=adj(NN),adj(NNN)

class PairGNN(nn.Module):
    hidden:int=48
    layers:int=4
    @nn.compact
    def __call__(self,xs,ys,z):
        h=nn.Dense(self.hidden)(jnp.stack([xs,ys,xs*ys],-1))
        for _ in range(self.layers):
            m1=jnp.einsum('ij,bjh->bih',A1,h)
            m2=jnp.einsum('ij,bjh->bih',A2,h)
            q=jnp.concatenate([h,m1,m2],-1)
            dh=nn.Dense(self.hidden)(nn.gelu(nn.Dense(self.hidden*2)(q)))
            h=nn.gelu(h+dh)
        q=jnp.concatenate([jnp.mean(h,1),jnp.mean(h*h,1),z],-1)
        q=nn.gelu(nn.Dense(64)(q))
        return nn.Dense(1)(q)[:,0]

def bits(states):
    s=np.asarray(states,np.uint32)
    return (2*((s[:,None]>>np.arange(N,dtype=np.uint32))&1).astype(np.float32)-1)
def make_wd(H,y):
    co=(H-sp.diags(H.diagonal())).tocoo()
    cost=np.where(np.abs(co.data)>0.375,1000.,1001.)
    W=sp.coo_matrix((cost,(co.row,co.col)),shape=H.shape).tocsr()
    return np.asarray(cs.shortest_path(W,directed=False,indices=[int(y)]))[0]

def globals_from_wd(wd,tau,idx):
    idx=np.asarray(idx,np.int32)
    d=(wd[idx]//1000).astype(np.float32)
    n2=np.rint(wd[idx]-1000*d).astype(np.float32)
    return np.column_stack([d/10,n2/10,np.full(len(idx),tau/.5,np.float32)]).astype(np.float32)

def wrms(e,w):
    return float(np.sqrt(np.sum(w*e*e)/np.sum(w)))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cache",required=True)
    ap.add_argument("--dt",type=float,default=.0125)
    ap.add_argument("--steps",type=int,default=20)
    ap.add_argument("--ns",type=int,default=1024)
    ap.add_argument("--trust",type=float,default=.002)
    ap.add_argument("--shift",type=float,default=.1)
    ap.add_argument("--cg-maxiter",type=int,default=120)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()

    H,_=build_H(.5)
    HO=H.copy(); HO.setdiag(0); HO.eliminate_zeros()
    c=np.load(a.cache)
    y=int(c["y"]); s=np.asarray(c["s"],float); logg=np.asarray(c["logg"],float)
    g=normalized(np.exp(np.clip(logg-logg.max(),-700,0)))
    psi=s*g
    h1=H@psi
    psi2=normalized(psi-a.dt*h1+.5*a.dt*a.dt*(H@h1))
    amp2=np.abs(psi2)
    s2=np.where(psi2>=0,1.0,-1.0)
    targ=np.log(np.maximum(amp2,1e-300))-np.log(np.maximum(g,1e-300))

    # Exact physical state is diagnostic only, never used in the SR force.
    e=np.zeros(D); e[y]=1.
    ex=normalized(sla.expm_multiply(-(0.5+a.dt)*H,e))

    wd=make_wd(H,y)
    tau=.5+a.dt
    Xall=bits(basis)
    Gall=globals_from_wd(wd,tau,np.arange(D))
    yb=bits(np.array([basis[y]]))[0]
    model=PairGNN()
    p0=model.init(jax.random.PRNGKey(20261003),
                  jnp.asarray(Xall[:4]),
                  jnp.asarray(np.repeat(yb[None,:],4,0)),
                  jnp.asarray(Gall[:4]))["params"]
    flat,unravel=ravel_pytree(p0)

    @jax.jit
    def apply_flat(z,x,yy,gg):
        return model.apply({"params":unravel(z)},x,yy,gg)

    @jax.jit
    def force_sr(z,xp,yp,gp,xq,yq,gq):
        def mp(v):
            return jnp.mean(apply_flat(v,xp,yp,gp))
        def mq(v):
            return jnp.mean(apply_flat(v,xq,yq,gq))
        return jax.grad(mq)(z)-jax.grad(mp)(z)

    @jax.jit
    def fisher_mv(z,v,xp,yp,gp):
        def lv(zz):
            return apply_flat(zz,xp,yp,gp)
        _,u=jax.jvp(lv,(z,),(v,))
        u=u-jnp.mean(u)
        _,pb=jax.vjp(lv,z)
        return pb(u/len(xp))[0]

    @jax.jit
    def tangent(z,v,xp,yp,gp):
        def lv(zz):
            return apply_flat(zz,xp,yp,gp)
        return jax.jvp(lv,(z,),(v,))[1]

    def eval_full(z,batch=8192):
        out=[]
        for lo in range(0,D,batch):
            hi=min(D,lo+batch)
            yy=np.repeat(yb[None,:],hi-lo,0)
            out.append(np.asarray(apply_flat(
                z,jnp.asarray(Xall[lo:hi]),jnp.asarray(yy),jnp.asarray(Gall[lo:hi]))))
        return np.concatenate(out)

    f0=eval_full(flat)
    rng=np.random.default_rng(20261003)
    # Fixed blind edge audit, sampled from the K2 target probability.
    pt=amp2*amp2; pt/=pt.sum()
    centers=rng.choice(D,4000,p=pt)
    src=[]; dst=[]; ww=[]
    for x in centers:
        lo,hi=HO.indptr[x],HO.indptr[x+1]
        js=HO.indices[lo:hi]; hs=np.abs(HO.data[lo:hi])
        src.extend([int(x)]*len(js)); dst.extend(js.tolist()); ww.extend(hs.tolist())
    src=np.asarray(src,np.int32); dst=np.asarray(dst,np.int32); ww=np.asarray(ww,float)
    target_edge=targ[dst]-targ[src]
    baseline_edge=wrms(target_edge,ww)
    rows=[]
    for k in range(a.steps+1):
        f=eval_full(flat)-f0
        loga=np.log(np.maximum(g,1e-300))+f
        m=loga.max()
        cur=np.exp(np.clip(loga-m,-80,0)); cur=normalized(cur)
        p=cur*cur; p/=p.sum()
        q=cur*amp2; q/=q.sum()
        tfid=float((cur@amp2)**2)
        pfid=float((s2*cur@ex)**2)
        edge=wrms((f[dst]-f[src])-target_edge,ww)
        row={"step":k,"target_fid":tfid,"physical_fid":pfid,
             "edge_rmse":edge,"edge_baseline":baseline_edge,
             "residual_rms_p":float(np.sqrt(np.sum(p*(f-np.sum(p*f))**2)))}
        if k==a.steps:
            rows.append(row)
            print("OVERLAP_SR",json.dumps(row,sort_keys=True),flush=True)
            break

        ip=rng.choice(D,a.ns,replace=True,p=p)
        iq=rng.choice(D,a.ns,replace=True,p=q)
        xp=jnp.asarray(Xall[ip]); gp=jnp.asarray(Gall[ip])
        xq=jnp.asarray(Xall[iq]); gq=jnp.asarray(Gall[iq])
        yp=jnp.asarray(np.repeat(yb[None,:],a.ns,0))
        yq=yp

        F=force_sr(flat,xp,yp,gp,xq,yq,gq)
        def A(v):
            return fisher_mv(flat,v,xp,yp,gp)+a.shift*v
        sol,_=jax.scipy.sparse.linalg.cg(A,F,tol=1e-5,atol=0.0,maxiter=a.cg_maxiter)
        sol.block_until_ready()
        rel=float(jnp.linalg.norm(A(sol)-F)/jnp.maximum(jnp.linalg.norm(F),1e-30))
        u=np.asarray(tangent(flat,sol,xp,yp,gp)).copy()
        u-=u.mean()
        trms=float(np.sqrt(np.mean(u*u)))
        eta=a.trust/max(trms,1e-30)
        row.update({"cg_relres":rel,"tangent_rms_per_eta":trms,"eta":float(eta)})
        rows.append(row)
        print("OVERLAP_SR",json.dumps(row,sort_keys=True),flush=True)
        flat=flat+eta*sol
    out={"M":100000,"y":y,"dt":a.dt,"steps":a.steps,"ns":a.ns,
         "trust":a.trust,"shift":a.shift,"pdim":int(flat.size),
         "k2_target_physical_fid":float((psi2@ex)**2),
         "start_physical_fid":float((psi@normalized(sla.expm_multiply(-.5*H,e)))**2),
         "baseline_edge_rmse":baseline_edge,"rows":rows}
    Path(a.out).write_text(json.dumps(out,indent=2))
    print("FINAL",json.dumps(rows[-1],sort_keys=True),flush=True)

if __name__=="__main__":
    main()
