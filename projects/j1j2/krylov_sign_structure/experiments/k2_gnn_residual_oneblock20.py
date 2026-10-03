#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np
import jax,jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from finite_tau_matching_20site_exact import build_H,basis,D,N,special_columns
from finite_tau_ct_local_ratio_20site import distance_sign,normalized
from finite_tau_ct_smc_edgeratio_longbeta20 import reconstruct
from finite_tau_gnn_densityratio_oracle20 import PairGNN,bits,make_wd,globals_from_wd

ROOT=Path("/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure")
H,_=build_H(.5); HO=H.copy(); HO.setdiag(0); HO.eliminate_zeros()

@jax.jit
def step(st,x,y,g,t):
    def lossf(p):
        f=st.apply_fn({"params":p},x,y,g)
        d=f-t
        return jnp.mean(d*d)
    loss,gr=jax.value_and_grad(lossf)(st.params)
    return st.apply_gradients(grads=gr),loss

def eval_ids(model,params,yb,wd,tau,ids):
    ids=np.asarray(ids,np.int32); out=[]
    for q in range(0,len(ids),4096):
        z=ids[q:q+4096]
        yy=np.repeat(yb[None,:],len(z),0)
        gg=globals_from_wd(wd,tau,z)
        out.append(np.asarray(model.apply({"params":params},jnp.asarray(bits(basis[z])),jnp.asarray(yy),jnp.asarray(gg))))
    return np.concatenate(out)

def wrms(e,w):
    return float(np.sqrt(np.sum(w*e*e)/np.sum(w)))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--M",type=int,default=100000)
    ap.add_argument("--dt",type=float,default=.0125)
    ap.add_argument("--ns",type=int,default=2000)
    ap.add_argument("--epochs",type=int,default=15)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()

    cache=np.load(ROOT/"results/reconstruct_M100000_y59279.npz")
    y=int(cache["y"]); s=cache["s"]; logg=cache["logg"]
    wd=make_wd(H,[y])[0]
    g=normalized(np.exp(np.clip(logg-logg.max(),-700,0))); p=g*g; p/=p.sum()
    psi=s*g; h1=H@psi; psi2=normalized(psi-a.dt*h1+.5*a.dt*a.dt*(H@h1))
    amp2=np.abs(psi2)
    targ=np.log(np.maximum(amp2,1e-300))-np.log(np.maximum(g,1e-300))

    rg=np.random.default_rng(2400001+a.M)
    tr=rg.choice(D,a.ns,p=p); va=rg.choice(D,a.ns,p=p)
    yb=bits(np.array([basis[y]]))[0]
    tau=.5+a.dt
    Xtr=bits(basis[tr]); Ytr=np.repeat(yb[None,:],a.ns,0); Gtr=globals_from_wd(wd,tau,tr)
    Xva=bits(basis[va]); Yva=np.repeat(yb[None,:],a.ns,0); Gva=globals_from_wd(wd,tau,va)
    # Gauge fix only for optimization; constant does not affect amplitudes after normalization.
    tt=(targ[tr]-np.mean(targ[tr])).astype(np.float32)
    tv=(targ[va]-np.mean(targ[va])).astype(np.float32)

    model=PairGNN()
    params=model.init(jax.random.PRNGKey(17),jnp.asarray(Xtr[:4]),jnp.asarray(Ytr[:4]),jnp.asarray(Gtr[:4]))["params"]
    st=TrainState.create(apply_fn=model.apply,params=params,tx=optax.adamw(1e-3,weight_decay=1e-6))
    bestv=1e99; best=None; bs=512
    for ep in range(a.epochs):
        order=rg.permutation(a.ns); ls=[]
        for q in range(0,a.ns,bs):
            z=order[q:q+bs]
            st,l=step(st,jnp.asarray(Xtr[z]),jnp.asarray(Ytr[z]),jnp.asarray(Gtr[z]),jnp.asarray(tt[z]))
            ls.append(float(l))
        pv=np.asarray(model.apply({"params":st.params},jnp.asarray(Xva),jnp.asarray(Yva),jnp.asarray(Gva)))
        vm=float(np.sqrt(np.mean((pv-tv)**2)))
        if vm<bestv:
            bestv=vm
            best=jax.tree_util.tree_map(lambda x:np.asarray(x),st.params)
        if ep in (0,1,2,4,9,a.epochs-1):
            print("EPOCH",ep+1,"train_rmse",float(np.sqrt(np.mean(ls))),"val_rmse",vm,flush=True)

    # Full correction and global fidelity.
    ids=np.arange(D,dtype=np.int32)
    f=eval_ids(model,best,yb,wd,tau,ids)
    f-=np.sum(p*f)
    gp=normalized(g*np.exp(np.clip(f,-20,20)))

    # Blind all-edge audit from independent exact-distribution validation centers.
    xs=rg.choice(D,2000,p=p)
    src=[];dst=[];ww=[]
    for x in xs:
        lo,hi=HO.indptr[x],HO.indptr[x+1]; js=HO.indices[lo:hi]; hs=np.abs(HO.data[lo:hi])
        src.extend([int(x)]*len(js)); dst.extend(js.tolist()); ww.extend(hs.tolist())
    src=np.asarray(src,np.int32);dst=np.asarray(dst,np.int32);ww=np.asarray(ww,float)
    pred=f[dst]-f[src]; tar=targ[dst]-targ[src]; err=pred-tar
    seenmask=np.zeros(D,dtype=bool); seenmask[np.unique(tr)]=True
    seen=seenmask[dst]
    out={"M":a.M,"dt":a.dt,"ns":a.ns,"epochs":a.epochs,
         "val_state_rmse":bestv,"target_fid":float((gp@amp2)**2),
         "edge_seenY_frac":float(seen.mean()),
         "edge_base_rmse":wrms(tar,ww),"edge_proj_rmse":wrms(err,ww),
         "edge_base_seenY":wrms(tar[seen],ww[seen]) if np.any(seen) else None,
         "edge_proj_seenY":wrms(err[seen],ww[seen]) if np.any(seen) else None,
         "edge_base_unseenY":wrms(tar[~seen],ww[~seen]) if np.any(~seen) else None,
         "edge_proj_unseenY":wrms(err[~seen],ww[~seen]) if np.any(~seen) else None}
    print("RESULT",json.dumps(out,sort_keys=True),flush=True)
    Path(a.out).write_text(json.dumps(out,indent=2))
    np.savez_compressed(str(Path(a.out).with_suffix(".npz")),f=f,targ=targ,g=g,p=p,psi2=psi2,amp2=amp2,train=tr,val=va)
if __name__=="__main__":main()
