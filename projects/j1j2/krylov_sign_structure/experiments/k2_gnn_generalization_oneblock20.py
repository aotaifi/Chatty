#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np
import jax,jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from finite_tau_matching_20site_exact import D,special_columns,basis
from finite_tau_ct_local_ratio_20site import H,distance_sign,normalized
from finite_tau_ct_smc_edgeratio_longbeta20 import reconstruct
from finite_tau_gnn_densityratio_oracle20 import PairGNN,bits,make_wd,globals_from_wd
from k2_sr_edge_tangent_oneblock20 import edge_set,wrmse

@jax.jit
def step(st,x,y,g,t):
    def lossf(p):
        f=st.apply_fn({"params":p},x,y,g)
        return jnp.mean((f-t)**2)
    l,gr=jax.value_and_grad(lossf)(st.params)
    return st.apply_gradients(grads=gr),l

def predict(model,params,idx,y,wd,tau,batch=4096):
    out=[]; yb=bits(np.array([basis[y]]))[0]
    for q in range(0,len(idx),batch):
        z=idx[q:q+batch]
        yy=np.repeat(yb[None,:],len(z),axis=0)
        gg=globals_from_wd(wd,tau,z)
        out.append(np.asarray(model.apply({"params":params},
                     jnp.asarray(bits(basis[z])),jnp.asarray(yy),jnp.asarray(gg))))
    return np.concatenate(out)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--M",type=int,default=100000)
    ap.add_argument("--dt",type=float,default=.0125)
    ap.add_argument("--ns",type=int,default=2000)
    ap.add_argument("--epochs",type=int,default=20)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    y=int(special_columns(np.random.default_rng(20260930),2)[1])
    wd=make_wd(H,[y])[0]; s=distance_sign(y)
    _,_,logg=reconstruct(y,s,a.M)
    g=normalized(np.exp(np.clip(logg-logg.max(),-700,0))); p=g*g;p/=p.sum()
    psi=s*g; h1=H@psi
    psi2=normalized(psi-a.dt*h1+.5*a.dt*a.dt*(H@h1)); amp2=np.abs(psi2)
    targ=np.log(np.maximum(amp2,1e-300))-np.log(np.maximum(g,1e-300))
    tau=.5+a.dt; rg=np.random.default_rng(2400001+a.M)
    tr=rg.choice(D,a.ns,p=p); va=rg.choice(D,a.ns,p=p)
    # center target under each split to remove normalization gauge
    tt=targ[tr]-np.mean(targ[tr]); vt=targ[va]-np.mean(targ[va])
    yb=bits(np.array([basis[y]]))[0]
    X=bits(basis[tr]); Y=np.repeat(yb[None,:],len(tr),axis=0); G=globals_from_wd(wd,tau,tr)
    VX=bits(basis[va]); VY=np.repeat(yb[None,:],len(va),axis=0); VG=globals_from_wd(wd,tau,va)

    model=PairGNN(hidden=48,layers=4)
    params=model.init(jax.random.PRNGKey(99),jnp.asarray(X[:4]),jnp.asarray(Y[:4]),jnp.asarray(G[:4]))["params"]
    st=TrainState.create(apply_fn=model.apply,params=params,tx=optax.adamw(5e-4,weight_decay=1e-6))
    best=np.inf;bestp=None;hist=[];bs=512
    for ep in range(a.epochs):
        order=rg.permutation(len(tr)); ls=[]
        for q in range(0,len(order),bs):
            z=order[q:q+bs]
            st,l=step(st,jnp.asarray(X[z]),jnp.asarray(Y[z]),jnp.asarray(G[z]),jnp.asarray(tt[z]))
            ls.append(float(l))
        pv=np.asarray(model.apply({"params":st.params},jnp.asarray(VX),jnp.asarray(VY),jnp.asarray(VG)))
        rm=float(np.sqrt(np.mean((pv-vt)**2)))
        hist.append([ep,float(np.sqrt(np.mean(ls))),rm])
        if rm<best:
            best=rm; bestp=jax.tree_util.tree_map(lambda x:np.asarray(x).copy(),st.params)
        if ep in (0,1,4,9,14,19,a.epochs-1):
            print("EPOCH",ep+1,"val_rmse",rm,flush=True)
    params=jax.tree_util.tree_map(jnp.asarray,bestp)

    allidx=np.arange(D,dtype=np.int32)
    f=predict(model,params,allidx,y,wd,tau)
    f-=np.sum(p*f)
    gp=normalized(g*np.exp(np.clip(f,-20,20)))

    src,dst,w=edge_set(va)
    pe=f[dst]-f[src]; te=targ[dst]-targ[src]
    train_seen=np.zeros(D,dtype=bool);train_seen[np.unique(tr)]=True
    seen=train_seen[dst]
    out={"M":a.M,"dt":a.dt,"ns":a.ns,"epochs":a.epochs,"best_val_rmse":best,
         "target_fid":float((gp@amp2)**2),
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
