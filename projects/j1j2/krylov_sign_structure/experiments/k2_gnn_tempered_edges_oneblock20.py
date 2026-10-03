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

ROOT=Path(__file__).resolve().parents[1]
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
    ids=np.asarray(ids,np.int32);out=[]
    for q in range(0,len(ids),4096):
        z=ids[q:q+4096]; yy=np.repeat(yb[None,:],len(z),0)
        gg=globals_from_wd(wd,tau,z)
        out.append(np.asarray(model.apply({"params":params},
            jnp.asarray(bits(basis[z])),jnp.asarray(yy),jnp.asarray(gg))))
    return np.concatenate(out)

def draw_centers(rg,p,q,n,mix):
    nm=int(round(n*mix)); np0=n-nm
    return np.concatenate([
        rg.choice(D,size=np0,replace=True,p=p),
        rg.choice(D,size=nm,replace=True,p=q)
    ]).astype(np.int32)

def make_pairs(rg,p,q,mix,targ,yb,wd,tau,n):
    ii=draw_centers(rg,p,q,n,mix)
    jj=np.empty(n,np.int32)
    for z,x in enumerate(ii):
        lo,hi=HO.indptr[x],HO.indptr[x+1]; ns=HO.indices[lo:hi]; hs=np.abs(HO.data[lo:hi])
        jj[z]=int(rg.choice(ns,p=hs/hs.sum()))
    tt=(targ[jj]-targ[ii]).astype(np.float32)
    YY=np.repeat(yb[None,:],n,0)
    return (ii,jj,bits(basis[ii]),bits(basis[jj]),YY,
            globals_from_wd(wd,tau,ii),globals_from_wd(wd,tau,jj),tt)

def wrms(e,w):
    return float(np.sqrt(np.sum(w*e*e)/np.sum(w)))

def audit(rg,dist,targ,f,n=2000):
    xs=rg.choice(D,n,p=dist)
    src=[];dst=[];ww=[]
    for x in xs:
        lo,hi=HO.indptr[x],HO.indptr[x+1]; js=HO.indices[lo:hi]; hs=np.abs(HO.data[lo:hi])
        src.extend([int(x)]*len(js));dst.extend(js.tolist());ww.extend(hs.tolist())
    src=np.asarray(src,np.int32);dst=np.asarray(dst,np.int32);ww=np.asarray(ww,float)
    pred=f[dst]-f[src]; tar=targ[dst]-targ[src]; err=pred-tar
    return {"base":wrms(tar,ww),"proj":wrms(err,ww),
            "nedge":int(len(src))}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--M",type=int,default=100000)
    ap.add_argument("--dt",type=float,default=.0125)
    ap.add_argument("--ntrain",type=int,default=30000)
    ap.add_argument("--nval",type=int,default=10000)
    ap.add_argument("--epochs",type=int,default=15)
    ap.add_argument("--alpha",type=float,default=.6)
    ap.add_argument("--mix",type=float,default=.5)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()

    y=int(special_columns(np.random.default_rng(20260930),2)[1]); wd=make_wd(H,[y])[0]
    s=distance_sign(y); _,_,logg=reconstruct(y,s,a.M)
    g=normalized(np.exp(np.clip(logg-logg.max(),-700,0))); p=g*g; p/=p.sum()
    q=np.power(np.maximum(p,1e-300),a.alpha); q/=q.sum()
    psi=s*g; h1=H@psi
    raw2=psi-a.dt*h1+.5*a.dt*a.dt*(H@h1)
    psi2=normalized(raw2); amp2=np.abs(psi2)
    targ=np.log(np.maximum(amp2,1e-300))-np.log(np.maximum(g,1e-300))
    Rraw=raw2/np.where(np.abs(psi)>1e-300,psi,np.where(psi>=0,1.,-1.)*1e-300)
    tau=.5+a.dt;yb=bits(np.array([basis[y]]))[0];rg=np.random.default_rng(3300001+a.M)

    tr=make_pairs(rg,p,q,a.mix,targ,yb,wd,tau,a.ntrain)
    va=make_pairs(rg,p,q,a.mix,targ,yb,wd,tau,a.nval)

    model=PairGNN()
    params=model.init(jax.random.PRNGKey(91),jnp.asarray(tr[2][:4]),jnp.asarray(tr[4][:4]),jnp.asarray(tr[5][:4]))["params"]
    st=TrainState.create(apply_fn=model.apply,params=params,tx=optax.adamw(1e-3,weight_decay=1e-6))
    bestv=1e99;best=None;bs=1024
    for ep in range(a.epochs):
        order=rg.permutation(a.ntrain);ls=[]
        for z0 in range(0,a.ntrain,bs):
            z=order[z0:z0+bs]
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
    rawr=float(np.sqrt(np.sum(p*f*f))); eta=min(1.,.05/max(rawr,1e-14));f*=eta
    gp=normalized(g*np.exp(np.clip(f,-20,20)))

    pa=audit(rg,p,targ,f); qa=audit(rg,q,targ,f)
    B=(Rraw<0)|(np.abs(Rraw)<.2)
    # exact weighted error over all boundary centers (small set on N=20)
    bx=np.flatnonzero(B)
    src=[];dst=[];ww=[]
    for x in bx:
        lo,hi=HO.indptr[x],HO.indptr[x+1];js=HO.indices[lo:hi];hs=np.abs(HO.data[lo:hi])
        src.extend([int(x)]*len(js));dst.extend(js.tolist());ww.extend(hs.tolist())
    if src:
        src=np.asarray(src,np.int32);dst=np.asarray(dst,np.int32);ww=np.asarray(ww,float)
        berr=wrms((f[dst]-f[src])-(targ[dst]-targ[src]),ww)
        bbase=wrms(targ[dst]-targ[src],ww)
    else: berr=bbase=None

    seen=np.zeros(D,bool);seen[np.unique(np.concatenate([tr[0],tr[1]]))]=True
    out={"M":a.M,"dt":a.dt,"alpha":a.alpha,"mix":a.mix,
         "ntrain":a.ntrain,"nval":a.nval,"epochs":a.epochs,
         "train_unique":int(seen.sum()),"train_frac_D":float(seen.mean()),
         "val_edge_rmse":bestv,"raw_f_rms_p":rawr,"eta":float(eta),
         "target_fid":float((gp@amp2)**2),
         "physical_audit":pa,"tempered_audit":qa,
         "boundary_n":int(B.sum()),"boundary_base_rmse":bbase,"boundary_proj_rmse":berr}
    print("RESULT",json.dumps(out,sort_keys=True),flush=True)
    Path(a.out).write_text(json.dumps(out,indent=2))
if __name__=="__main__":main()
