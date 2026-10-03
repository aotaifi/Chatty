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

def sample_neighbors(xs,K,rg):
    out=[]
    for x in xs:
        lo,hi=HO.indptr[x],HO.indptr[x+1]
        js=HO.indices[lo:hi]; hs=np.abs(HO.data[lo:hi]); pr=hs/hs.sum()
        out.extend(rg.choice(js,size=K,replace=True,p=pr).tolist())
    return np.asarray(out,np.int32)

@jax.jit
def step(st,x,y,g,t,w):
    def lossf(p):
        f=st.apply_fn({"params":p},x,y,g)
        e=f-t
        return jnp.sum(w*e*e)/jnp.sum(w)
    loss,gr=jax.value_and_grad(lossf)(st.params)
    return st.apply_gradients(grads=gr),loss

def eval_ids(model,params,yb,wd,tau,ids):
    ids=np.asarray(ids,np.int32); out=[]
    for q in range(0,len(ids),4096):
        z=ids[q:q+4096]; yy=np.repeat(yb[None,:],len(z),0)
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
    ap.add_argument("--K",type=int,default=4)
    ap.add_argument("--epochs",type=int,default=15)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()

    y=int(special_columns(np.random.default_rng(20260930),2)[1]); wd=make_wd(H,[y])[0]
    s=distance_sign(y); _,_,logg=reconstruct(y,s,a.M)
    g=normalized(np.exp(np.clip(logg-logg.max(),-700,0))); p=g*g; p/=p.sum()
    psi=s*g; h1=H@psi; psi2=normalized(psi-a.dt*h1+.5*a.dt*a.dt*(H@h1)); amp2=np.abs(psi2)
    targ=np.log(np.maximum(amp2,1e-300))-np.log(np.maximum(g,1e-300))
    tau=.5+a.dt; rg=np.random.default_rng(2500001+a.M+a.K)

    trx=rg.choice(D,a.ns,p=p); vax=rg.choice(D,a.ns,p=p)
    trn=sample_neighbors(trx,a.K,rg); van=sample_neighbors(vax,a.K,rg)
    tri=np.concatenate([trx,trn]); vai=np.concatenate([vax,van])
    # equal total weight to centers and neighbor cloud
    tw=np.concatenate([np.full(len(trx),.5/len(trx)),np.full(len(trn),.5/len(trn))]).astype(np.float32)
    vw=np.concatenate([np.full(len(vax),.5/len(vax)),np.full(len(van),.5/len(van))]).astype(np.float32)

    yb=bits(np.array([basis[y]]))[0]
    Xtr=bits(basis[tri]); Ytr=np.repeat(yb[None,:],len(tri),0); Gtr=globals_from_wd(wd,tau,tri)
    Xva=bits(basis[vai]); Yva=np.repeat(yb[None,:],len(vai),0); Gva=globals_from_wd(wd,tau,vai)
    # weighted gauge means
    mt=float(np.sum(tw*targ[tri])/np.sum(tw)); mv=float(np.sum(vw*targ[vai])/np.sum(vw))
    tt=(targ[tri]-mt).astype(np.float32); tv=(targ[vai]-mv).astype(np.float32)

    model=PairGNN()
    params=model.init(jax.random.PRNGKey(27),jnp.asarray(Xtr[:4]),jnp.asarray(Ytr[:4]),jnp.asarray(Gtr[:4]))["params"]
    st=TrainState.create(apply_fn=model.apply,params=params,tx=optax.adamw(1e-3,weight_decay=1e-6))
    bestv=1e99; best=None; bs=1024
    for ep in range(a.epochs):
        order=rg.permutation(len(tri)); ls=[]
        for q in range(0,len(order),bs):
            z=order[q:q+bs]
            st,l=step(st,jnp.asarray(Xtr[z]),jnp.asarray(Ytr[z]),jnp.asarray(Gtr[z]),jnp.asarray(tt[z]),jnp.asarray(tw[z]))
            ls.append(float(l))
        pv=np.asarray(model.apply({"params":st.params},jnp.asarray(Xva),jnp.asarray(Yva),jnp.asarray(Gva)))
        vm=float(np.sqrt(np.sum(vw*(pv-tv)**2)/np.sum(vw)))
        if vm<bestv:
            bestv=vm; best=jax.tree_util.tree_map(lambda x:np.asarray(x),st.params)
        if ep in (0,1,2,4,9,a.epochs-1):
            print("EPOCH",ep+1,"train_loss",float(np.mean(ls)),"val_mix_rmse",vm,flush=True)

    ids=np.arange(D,dtype=np.int32)
    f=eval_ids(model,best,yb,wd,tau,ids); f-=np.sum(p*f)
    rawr=float(np.sqrt(np.sum(p*f*f))); eta=min(1.0,.05/max(rawr,1e-14)); f*=eta
    gp=normalized(g*np.exp(np.clip(f,-20,20)))

    # blind edge audit from validation centers; seen means explicitly in train center+neighbor set
    src=[];dst=[];ww=[]
    for x in vax:
        lo,hi=HO.indptr[x],HO.indptr[x+1]; js=HO.indices[lo:hi]; hs=np.abs(HO.data[lo:hi])
        src.extend([int(x)]*len(js));dst.extend(js.tolist());ww.extend(hs.tolist())
    src=np.asarray(src,np.int32);dst=np.asarray(dst,np.int32);ww=np.asarray(ww,float)
    pred=f[dst]-f[src]; tar=targ[dst]-targ[src]; err=pred-tar
    sm=np.zeros(D,dtype=bool);sm[np.unique(tri)]=True; seen=sm[dst]
    out={"M":a.M,"dt":a.dt,"ns":a.ns,"K":a.K,"epochs":a.epochs,
         "train_unique":int(sm.sum()),"train_frac_D":float(sm.mean()),
         "val_mix_rmse":bestv,"raw_f_rms_p":rawr,"eta":float(eta),
         "target_fid":float((gp@amp2)**2),"edge_seenY_frac":float(seen.mean()),
         "edge_base_rmse":wrms(tar,ww),"edge_proj_rmse":wrms(err,ww),
         "edge_base_seenY":wrms(tar[seen],ww[seen]) if np.any(seen) else None,
         "edge_proj_seenY":wrms(err[seen],ww[seen]) if np.any(seen) else None,
         "edge_base_unseenY":wrms(tar[~seen],ww[~seen]) if np.any(~seen) else None,
         "edge_proj_unseenY":wrms(err[~seen],ww[~seen]) if np.any(~seen) else None}
    print("RESULT",json.dumps(out,sort_keys=True),flush=True)
    Path(a.out).write_text(json.dumps(out,indent=2))
if __name__=="__main__":main()
