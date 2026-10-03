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
def step(st,xi,xj,yb,gi,gj,t):
    def lossf(p):
        fi=st.apply_fn({"params":p},xi,yb,gi)
        fj=st.apply_fn({"params":p},xj,yb,gj)
        e=(fj-fi)-t
        return jnp.mean(e*e)
    loss,gr=jax.value_and_grad(lossf)(st.params)
    return st.apply_gradients(grads=gr),loss

def eval_ids(model,params,yb,wd,tau,ids):
    ids=np.asarray(ids,np.int32); out=[]
    for q in range(0,len(ids),4096):
        z=ids[q:q+4096]; yy=np.repeat(yb[None,:],len(z),0)
        gg=globals_from_wd(wd,tau,z)
        out.append(np.asarray(model.apply({"params":params},jnp.asarray(bits(basis[z])),jnp.asarray(yy),jnp.asarray(gg))))
    return np.concatenate(out)

def make_pairs(rg,p,targ,yb,wd,tau,n):
    ii=rg.choice(D,size=n,replace=True,p=p).astype(np.int32)
    jj=np.empty(n,np.int32)
    for q,x in enumerate(ii):
        lo,hi=HO.indptr[x],HO.indptr[x+1]; ns=HO.indices[lo:hi]; hs=np.abs(HO.data[lo:hi])
        jj[q]=int(rg.choice(ns,p=hs/hs.sum()))
    tt=(targ[jj]-targ[ii]).astype(np.float32)
    YY=np.repeat(yb[None,:],n,0)
    return (ii,jj,bits(basis[ii]),bits(basis[jj]),YY,
            globals_from_wd(wd,tau,ii),globals_from_wd(wd,tau,jj),tt)

def wrms(e,w):
    return float(np.sqrt(np.sum(w*e*e)/np.sum(w)))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--M",type=int,default=100000)
    ap.add_argument("--dt",type=float,default=.0125)
    ap.add_argument("--ntrain",type=int,default=30000)
    ap.add_argument("--nval",type=int,default=10000)
    ap.add_argument("--epochs",type=int,default=15)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()

    y=int(special_columns(np.random.default_rng(20260930),2)[1]); wd=make_wd(H,[y])[0]
    s=distance_sign(y); _,_,logg=reconstruct(y,s,a.M)
    g=normalized(np.exp(np.clip(logg-logg.max(),-700,0))); p=g*g; p/=p.sum()
    psi=s*g; h1=H@psi; psi2=normalized(psi-a.dt*h1+.5*a.dt*a.dt*(H@h1)); amp2=np.abs(psi2)
    targ=np.log(np.maximum(amp2,1e-300))-np.log(np.maximum(g,1e-300))
    tau=.5+a.dt; yb=bits(np.array([basis[y]]))[0]; rg=np.random.default_rng(2600001+a.M)

    tr=make_pairs(rg,p,targ,yb,wd,tau,a.ntrain)
    va=make_pairs(rg,p,targ,yb,wd,tau,a.nval)

    model=PairGNN()
    params=model.init(jax.random.PRNGKey(37),jnp.asarray(tr[2][:4]),jnp.asarray(tr[4][:4]),jnp.asarray(tr[5][:4]))["params"]
    st=TrainState.create(apply_fn=model.apply,params=params,tx=optax.adamw(1e-3,weight_decay=1e-6))
    bestv=1e99; best=None; bs=1024
    for ep in range(a.epochs):
        order=rg.permutation(a.ntrain); ls=[]
        for q in range(0,a.ntrain,bs):
            z=order[q:q+bs]
            st,l=step(st,*[jnp.asarray(v[z]) for v in tr[2:]])
            ls.append(float(l))
        fi=np.asarray(model.apply({"params":st.params},jnp.asarray(va[2]),jnp.asarray(va[4]),jnp.asarray(va[5])))
        fj=np.asarray(model.apply({"params":st.params},jnp.asarray(va[3]),jnp.asarray(va[4]),jnp.asarray(va[6])))
        vm=float(np.sqrt(np.mean((fj-fi-va[7])**2)))
        if vm<bestv:
            bestv=vm; best=jax.tree_util.tree_map(lambda x:np.asarray(x),st.params)
        if ep in (0,1,2,4,9,a.epochs-1):
            print("EPOCH",ep+1,"train_rmse",float(np.sqrt(np.mean(ls))),"val_edge_rmse",vm,flush=True)

    f=eval_ids(model,best,yb,wd,tau,np.arange(D,dtype=np.int32)); f-=np.sum(p*f)
    rawr=float(np.sqrt(np.sum(p*f*f))); eta=min(1.0,.05/max(rawr,1e-14)); f*=eta
    gp=normalized(g*np.exp(np.clip(f,-20,20)))

    # blind all-edge audit from independent centers
    xs=rg.choice(D,2000,p=p)
    src=[];dst=[];ww=[]
    for x in xs:
        lo,hi=HO.indptr[x],HO.indptr[x+1]; js=HO.indices[lo:hi]; hs=np.abs(HO.data[lo:hi])
        src.extend([int(x)]*len(js));dst.extend(js.tolist());ww.extend(hs.tolist())
    src=np.asarray(src,np.int32);dst=np.asarray(dst,np.int32);ww=np.asarray(ww,float)
    pred=f[dst]-f[src]; tar=targ[dst]-targ[src]; err=pred-tar
    seenmask=np.zeros(D,dtype=bool);seenmask[np.unique(np.concatenate([tr[0],tr[1]]))]=True
    seen=seenmask[dst]
    out={"M":a.M,"dt":a.dt,"ntrain":a.ntrain,"nval":a.nval,"epochs":a.epochs,
         "train_unique":int(seenmask.sum()),"train_frac_D":float(seenmask.mean()),
         "val_edge_rmse":bestv,"raw_f_rms_p":rawr,"eta":float(eta),
         "target_fid":float((gp@amp2)**2),"edge_seenY_frac":float(seen.mean()),
         "edge_base_rmse":wrms(tar,ww),"edge_proj_rmse":wrms(err,ww),
         "edge_base_seenY":wrms(tar[seen],ww[seen]) if np.any(seen) else None,
         "edge_proj_seenY":wrms(err[seen],ww[seen]) if np.any(seen) else None,
         "edge_base_unseenY":wrms(tar[~seen],ww[~seen]) if np.any(~seen) else None,
         "edge_proj_unseenY":wrms(err[~seen],ww[~seen]) if np.any(~seen) else None}
    print("RESULT",json.dumps(out,sort_keys=True),flush=True)
    Path(a.out).write_text(json.dumps(out,indent=2))
if __name__=="__main__": main()
