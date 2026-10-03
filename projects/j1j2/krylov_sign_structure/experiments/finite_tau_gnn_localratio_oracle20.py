#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import scipy.sparse.linalg as sla
import jax,jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from finite_tau_matching_20site_exact import build_H,basis,D,special_columns
from finite_tau_gnn_densityratio_oracle20 import PairGNN,bits,make_wd,globals_from_wd

ROOT=Path("/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure")
H,_=build_H(.5)
HO=H.copy(); HO.setdiag(0); HO.eliminate_zeros()
rng=np.random.default_rng(20261001)

@jax.jit
def step(st,xi,xj,yb,gi,gj,t):
    def lossf(p):
        fi=st.apply_fn({"params":p},xi,yb,gi)
        fj=st.apply_fn({"params":p},xj,yb,gj)
        return jnp.mean((fj-fi-t)**2)
    loss,gr=jax.value_and_grad(lossf)(st.params)
    return st.apply_gradients(grads=gr),loss

def oracle_sign_at(y,tau=.5,dt=.05):
    e=np.zeros(D);e[y]=1.; psi=e.copy()
    dsign=None
    wd=make_wd(H,[y])[0]
    d=(wd//1000).astype(np.int64)
    s=np.where(d%2==0,1,-1)
    for _ in range(int(round(tau/dt))):
        psi=np.asarray(sla.expm_multiply(-dt*H,psi),float)
        psi/=np.linalg.norm(psi)
        pred=s*np.abs(psi)-dt*(H@(s*np.abs(psi)))
        s=np.where(pred>=0,1,-1)
    return psi,s,wd

def make_pairs(y,psi,wd,n):
    a=np.abs(psi); p=a/a.sum()
    ii=rng.choice(D,size=n,replace=True,p=p)
    jj=np.empty(n,np.int32)
    for q,i in enumerate(ii):
        lo,hi=HO.indptr[i],HO.indptr[i+1]
        ns=HO.indices[lo:hi]
        jj[q]=int(ns[rng.integers(len(ns))])
    tar=np.log(np.maximum(a[jj],1e-30))-np.log(np.maximum(a[ii],1e-30))
    tar=np.clip(tar,-8,8).astype(np.float32)
    yb=np.repeat(bits(np.array([basis[y]])),n,axis=0)
    return (bits(basis[ii]),bits(basis[jj]),yb,
            globals_from_wd(wd,.5,ii),globals_from_wd(wd,.5,jj),tar)

def main():
    y=int(special_columns(np.random.default_rng(20260930),2)[1])
    psi,s,wd=oracle_sign_at(y)
    tr=make_pairs(y,psi,wd,30000); va=make_pairs(y,psi,wd,10000)
    model=PairGNN(); params=model.init(jax.random.PRNGKey(5),jnp.asarray(tr[0][:4]),jnp.asarray(tr[2][:4]),jnp.asarray(tr[3][:4]))["params"]
    st=TrainState.create(apply_fn=model.apply,params=params,tx=optax.adamw(1e-3,weight_decay=1e-6))
    bs=2048
    for ep in range(10):
        order=rng.permutation(len(tr[0])); ls=[]
        for q in range(0,len(order),bs):
            ix=order[q:q+bs]
            st,l=step(st,*[jnp.asarray(z[ix]) for z in tr]); ls.append(float(l))
        fi=np.asarray(model.apply({"params":st.params},jnp.asarray(va[0]),jnp.asarray(va[2]),jnp.asarray(va[3])))
        fj=np.asarray(model.apply({"params":st.params},jnp.asarray(va[1]),jnp.asarray(va[2]),jnp.asarray(va[4])))
        vr=float(np.sqrt(np.mean((fj-fi-va[5])**2)))
        print("EPOCH",ep+1,"train",float(np.mean(ls)),"val_ratio_rmse",vr,flush=True)

    ybit=bits(np.array([basis[y]]))[0]; out=[]
    for q in range(0,D,4096):
        ix=np.arange(q,min(q+4096,D))
        yy=np.repeat(ybit[None,:],len(ix),0)
        gg=globals_from_wd(wd,.5,ix)
        out.append(np.asarray(model.apply({"params":st.params},jnp.asarray(bits(basis[ix])),jnp.asarray(yy),jnp.asarray(gg))))
    f=np.concatenate(out); ah=np.exp(np.clip(f-f.max(),-60,0)); ah/=np.linalg.norm(ah)
    a=np.abs(psi); a/=np.linalg.norm(a)
    exn=np.asarray(sla.expm_multiply(-.05*H,psi),float); exn/=np.linalg.norm(exn)
    sn=np.where((s*ah-.05*(H@(s*ah)))>=0,1,-1)
    p=exn*exn
    err=float(np.sum(p[sn!=np.where(exn>=0,1,-1)]))
    fid=float(np.dot(ah,a)**2)
    row={"y":y,"amplitude_fidelity":fid,"k1_next_sign_error":err}
    print("RESULT",row,flush=True)
    (ROOT/"results/finite_tau_gnn_localratio_oracle20.json").write_text(json.dumps(row,indent=2))

if __name__=="__main__": main()
