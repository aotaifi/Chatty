#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp
import flax
import netket.jax as nkjax
from jax.flatten_util import ravel_pytree
from nqsmagic.models import ViT
from nqsmagic.utils import _logpsi_transl_2d
jax.config.update("jax_enable_x64", True)
N=64

def bits2x(ss):
    a=np.asarray(ss,np.uint64).reshape(-1,1)
    return 2*((a>>np.arange(N,dtype=np.uint64))&1).astype(np.float64)-1

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--a1",required=True)
    ap.add_argument("--freeze",required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--nsplit",type=int,default=4)
    a=ap.parse_args()

    model=ViT(num_layers=8,d_model=60,heads=10,L_eff=16,b=2,
              transl_invariant=True,two_dimensional=True)
    apply=nkjax.HashablePartial(_logpsi_transl_2d,model.apply,2)
    template=model.init(jax.random.PRNGKey(1234),jnp.zeros((1,N)))
    obj=flax.serialization.msgpack_restore(open(a.a1,"rb").read())
    variables=flax.serialization.from_state_dict(template,obj)
    p=variables["params"]; flat,unravel=ravel_pytree(p)

    z=np.load(a.freeze)
    base=z["base"].astype(np.uint64)
    train=z["train"].astype(np.uint64); val=z["val"].astype(np.uint64)
    el=z["start_el"].astype(float)
    itr=np.searchsorted(base,train); iva=np.searchsorted(base,val)
    Etr=el[itr]; Eva=el[iva]

    def grad_force(states,E):
        X=jnp.asarray(bits2x(states))
        coeff=jnp.asarray(2.0*(E-E.mean())/len(E))
        def f(pp):
            lv=jnp.real(apply({"params":pp},X))
            return jnp.dot(coeff,lv)
        _,g=jax.value_and_grad(f)(p)
        gf,_=ravel_pytree(g)
        return gf

    gtr=grad_force(train,Etr); gva=grad_force(val,Eva)
    ntr=float(jnp.linalg.norm(gtr)); nva=float(jnp.linalg.norm(gva))
    dot=float(jnp.vdot(gtr,gva).real)
    rawcos=dot/(ntr*nva)

    Xq=jnp.asarray(bits2x(train))
    def logvec(ff):
        return jnp.real(apply({"params":unravel(ff)},Xq))
    _,pb=jax.vjp(logvec,flat)
    def fisher(v):
        _,u=jax.jvp(logvec,(flat,),(v,))
        u=u-jnp.mean(u)
        return pb(u/len(train))[0]
    fisher=jax.jit(fisher)
    _=fisher(jnp.zeros_like(flat)).block_until_ready()
    sr=[]
    for shift in (1.0,0.1,0.01):
        def A(v): return fisher(v)+shift*v
        sol,info=jax.scipy.sparse.linalg.cg(A,gtr,tol=1e-5,atol=0.,maxiter=250)
        sol.block_until_ready()
        res=float(jnp.linalg.norm(A(sol)-gtr)/jnp.linalg.norm(gtr))
        sr.append(dict(shift=shift,cg_rel_resid=res,
                       train_dot=float(jnp.vdot(gtr,sol).real),
                       val_dot=float(jnp.vdot(gva,sol).real)))
    allS=np.concatenate([train,val]); allE=np.concatenate([Etr,Eva])
    splits=[]
    for k in range(a.nsplit):
        rg=np.random.default_rng(20261010+k)
        q=rg.permutation(len(allS)); ia=q[:len(allS)//2]; ib=q[len(allS)//2:]
        ga=grad_force(allS[ia],allE[ia]); gb=grad_force(allS[ib],allE[ib])
        na=float(jnp.linalg.norm(ga)); nb=float(jnp.linalg.norm(gb))
        co=float(jnp.vdot(ga,gb).real)/(na*nb)
        splits.append(dict(k=k,cos=co,norm_a=na,norm_b=nb,
                           meanEa=float(allE[ia].mean()),meanEb=float(allE[ib].mean())))

    def stats(E):
        d=np.abs(E-E.mean())
        return dict(mean=float(E.mean()),sd=float(E.std(ddof=1)),
                    q90=float(np.quantile(d,.90)),q99=float(np.quantile(d,.99)),
                    q999=float(np.quantile(d,.999)),max_abs_dev=float(d.max()))

    out=dict(ntrain=len(train),nval=len(val),
             raw_force_cos=rawcos,raw_dot=dot,norm_train=ntr,norm_val=nva,
             train_E=stats(Etr),val_E=stats(Eva),sr=sr,splits=splits)
    Path(a.out).write_text(json.dumps(out,indent=2))
    print("FROZEN_HFN_FORCE_ALIGNMENT",json.dumps(out),flush=True)

if __name__=="__main__":
    main()
