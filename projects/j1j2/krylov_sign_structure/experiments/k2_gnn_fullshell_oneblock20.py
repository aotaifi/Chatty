#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np
import jax,jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from finite_tau_matching_20site_exact import build_H,basis,D,N
from finite_tau_ct_local_ratio_20site import normalized
from finite_tau_gnn_densityratio_oracle20 import PairGNN,bits,make_wd,globals_from_wd

ROOT=Path("/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure")
H,_=build_H(.5); HO=H.copy(); HO.setdiag(0); HO.eliminate_zeros()

def shell(xs):
    nb=[]
    for x in xs:
        lo,hi=HO.indptr[x],HO.indptr[x+1]
        nb.extend(HO.indices[lo:hi].tolist())
    return np.unique(np.asarray(nb,np.int32))

@jax.jit
def step(st,x,y,g,t,w):
    def lf(p):
        f=st.apply_fn({"params":p},x,y,g)
        e=f-t
        return jnp.sum(w*e*e)/jnp.sum(w)
    loss,gr=jax.value_and_grad(lf)(st.params)
    return st.apply_gradients(grads=gr),loss

def eval_ids(model,params,yb,wd,tau,ids):
    out=[]; ids=np.asarray(ids,np.int32)
    for q in range(0,len(ids),4096):
        z=ids[q:q+4096]; yy=np.repeat(yb[None,:],len(z),0)
        out.append(np.asarray(model.apply({"params":params},jnp.asarray(bits(basis[z])),jnp.asarray(yy),jnp.asarray(globals_from_wd(wd,tau,z)))))
    return np.concatenate(out)

def wrms(e,w):
    return float(np.sqrt(np.sum(w*e*e)/np.sum(w)))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ns",type=int,default=2000)
    ap.add_argument("--epochs",type=int,default=10)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()

    z=np.load(ROOT/"results/reconstruct_M100000_y59279.npz")
    y=int(z["y"]); s=z["s"]; logg=z["logg"]
    g=normalized(np.exp(np.clip(logg-logg.max(),-700,0))); p=g*g;p/=p.sum()
    wd=make_wd(H,[y])[0]; dt=.0125; tau=.5+dt
    psi=s*g; h1=H@psi; psi2=normalized(psi-dt*h1+.5*dt*dt*(H@h1)); amp2=np.abs(psi2)
    targ=np.log(np.maximum(amp2,1e-300))-np.log(np.maximum(g,1e-300))

    rg=np.random.default_rng(2800001)
    trx=rg.choice(D,a.ns,p=p); vax=rg.choice(D,a.ns,p=p)
    trn=shell(trx); van=shell(vax)
    tri=np.concatenate([trx,trn]); vai=np.concatenate([vax,van])
    # centers and full one-hop shell each carry half the objective.
    tw=np.concatenate([np.full(len(trx),.5/len(trx)),np.full(len(trn),.5/len(trn))]).astype(np.float32)
    vw=np.concatenate([np.full(len(vax),.5/len(vax)),np.full(len(van),.5/len(van))]).astype(np.float32)

    mt=float(np.sum(tw*targ[tri])/np.sum(tw)); mv=float(np.sum(vw*targ[vai])/np.sum(vw))
    tt=(targ[tri]-mt).astype(np.float32); tv=(targ[vai]-mv).astype(np.float32)
    yb=bits(np.array([basis[y]]))[0]
    Xtr=bits(basis[tri]); Ytr=np.repeat(yb[None,:],len(tri),0); Gtr=globals_from_wd(wd,tau,tri)
    Xva=bits(basis[vai]); Yva=np.repeat(yb[None,:],len(vai),0); Gva=globals_from_wd(wd,tau,vai)

    model=PairGNN()
    params=model.init(jax.random.PRNGKey(47),jnp.asarray(Xtr[:4]),jnp.asarray(Ytr[:4]),jnp.asarray(Gtr[:4]))["params"]
    st=TrainState.create(apply_fn=model.apply,params=params,tx=optax.adamw(1e-3,weight_decay=1e-6))
    bestv=1e99;best=None;bs=1024
    for ep in range(a.epochs):
        order=rg.permutation(len(tri)); ls=[]
        for q in range(0,len(order),bs):
            ix=order[q:q+bs]
            st,l=step(st,jnp.asarray(Xtr[ix]),jnp.asarray(Ytr[ix]),jnp.asarray(Gtr[ix]),jnp.asarray(tt[ix]),jnp.asarray(tw[ix]))
            ls.append(float(l))
        pv=np.asarray(model.apply({"params":st.params},jnp.asarray(Xva),jnp.asarray(Yva),jnp.asarray(Gva)))
        vm=float(np.sqrt(np.sum(vw*(pv-tv)**2)/np.sum(vw)))
        if vm<bestv: bestv=vm;best=jax.tree_util.tree_map(lambda x:np.asarray(x),st.params)
        print("EPOCH",ep+1,"val_mix_rmse",vm,flush=True)

    f=eval_ids(model,best,yb,wd,tau,np.arange(D,dtype=np.int32)); f-=np.sum(p*f)
    rawr=float(np.sqrt(np.sum(p*f*f))); eta=min(1.,.05/max(rawr,1e-14));f*=eta
    gp=normalized(g*np.exp(np.clip(f,-20,20)))

    # blind all-edge validation from independent physical centers
    src=[];dst=[];ww=[]
    for x in vax:
        lo,hi=HO.indptr[x],HO.indptr[x+1]; js=HO.indices[lo:hi];hs=np.abs(HO.data[lo:hi])
        src.extend([int(x)]*len(js));dst.extend(js.tolist());ww.extend(hs.tolist())
    src=np.asarray(src,np.int32);dst=np.asarray(dst,np.int32);ww=np.asarray(ww,float)
    pred=f[dst]-f[src];tar=targ[dst]-targ[src];err=pred-tar
    sm=np.zeros(D,bool);sm[np.unique(tri)]=True;seen=sm[dst]
    R=psi2/np.where(np.abs(psi)>1e-300,psi,np.sign(psi)*1e-300)
    out={"ns":a.ns,"epochs":a.epochs,"train_centers_unique":int(len(np.unique(trx))),
         "train_shell_unique":int(len(trn)),"train_frac_D":float(sm.mean()),
         "train_crossing_unique":int(np.sum(R[trn]<0)),
         "train_lowmargin_unique":int(np.sum(np.abs(R[trn])<.2)),
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
