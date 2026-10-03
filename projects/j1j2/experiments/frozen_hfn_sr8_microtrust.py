#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np
import jax,jax.numpy as jnp
import flax
import netket.jax as nkjax
from jax.flatten_util import ravel_pytree
from nqsmagic.models import ViT
from nqsmagic.utils import _logpsi_transl_2d
jax.config.update("jax_enable_x64",True)
N=64
def bits2x(ss):
    a=np.asarray(ss,np.uint64).reshape(-1,1)
    return 2*((a>>np.arange(N,dtype=np.uint64))&1).astype(np.float64)-1

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--a1",required=True); ap.add_argument("--freeze",required=True)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    model=ViT(num_layers=8,d_model=60,heads=10,L_eff=16,b=2,
              transl_invariant=True,two_dimensional=True)
    apply=nkjax.HashablePartial(_logpsi_transl_2d,model.apply,2)
    template=model.init(jax.random.PRNGKey(1234),jnp.zeros((1,N)))
    obj=flax.serialization.msgpack_restore(open(a.a1,"rb").read())
    variables=flax.serialization.from_state_dict(template,obj)
    p=variables["params"]; flat,unravel=ravel_pytree(p)
    @jax.jit
    def evb(pp,x): return jnp.real(apply({"params":pp},x))
    def evals(pp,s,batch=2048):
        s=np.asarray(s,np.uint64); out=[]
        for i in range(0,len(s),batch):
            out.append(np.asarray(evb(pp,jnp.asarray(bits2x(s[i:i+batch])))))
        return np.concatenate(out)

    z=np.load(a.freeze)
    base=z["base"].astype(np.uint64); train=z["train"].astype(np.uint64); val=z["val"].astype(np.uint64)
    df=z["df"].astype(float); src=z["allowed_src"].astype(np.int32)
    dst=z["allowed_dst"].astype(np.uint64); rate=z["allowed_rate"].astype(float)
    start=z["start_el"].astype(float)
    itr=np.searchsorted(base,train); iva=np.searchsorted(base,val)
    Etr=start[itr]; Eva=start[iva]
    needed=np.unique(np.concatenate([base,dst]))
    pb=np.searchsorted(needed,base); pd=np.searchsorted(needed,dst)
    l1need=evals(p,needed); l1tr=evals(p,train); l1va=evals(p,val)

    def local_unique(pp):
        ln=evals(pp,needed); d=ln-l1need; out=df.copy()
        rr=rate*np.exp(np.clip(d[pd]-d[pb[src]],-40,40))
        np.add.at(out,src,-rr)
        return out

    def force(states,E):
        X=jnp.asarray(bits2x(states)); c=jnp.asarray(2*(E-E.mean())/len(E))
        def f(pp): return jnp.dot(c,jnp.real(apply({"params":pp},X)))
        _,g=jax.value_and_grad(f)(p)
        return ravel_pytree(g)[0]
    gt=force(train,Etr); gv=force(val,Eva)
    Xq=jnp.asarray(bits2x(train))
    def logvec(ff): return jnp.real(apply({"params":unravel(ff)},Xq))
    _,pbk=jax.vjp(logvec,flat)
    def fisher(v):
        _,u=jax.jvp(logvec,(flat,),(v,)); u=u-jnp.mean(u)
        return pbk(u/len(train))[0]
    fisher=jax.jit(fisher); _=fisher(jnp.zeros_like(flat)).block_until_ready()
    rows=[]
    for shift in (1.0,0.1,0.01):
        def A(v): return fisher(v)+shift*v
        sol,info=jax.scipy.sparse.linalg.cg(A,gt,tol=1e-5,atol=0.,maxiter=250)
        sol.block_until_ready()
        valdot=float(jnp.vdot(gv,sol).real); trdot=float(jnp.vdot(gt,sol).real)
        _,tan=jax.jvp(logvec,(flat,),(-sol,)); tan=np.asarray(tan,float)
        trms=float(np.std(tan))
        for trust in (5e-5,1e-4,2e-4,5e-4,1e-3):
            eta=trust/trms; p2=unravel(flat-eta*sol)
            eu=local_unique(p2); e2t=eu[itr]; e2v=eu[iva]
            l2t=evals(p2,train); l2v=evals(p2,val)
            wt=np.exp(2*(l2t-l1tr)-np.max(2*(l2t-l1tr))); wt/=wt.sum()
            wv=np.exp(2*(l2v-l1va)-np.max(2*(l2v-l1va))); wv/=wv.sum()
            Et=float(wt@e2t); Ev=float(wv@e2v)
            infl2=len(val)*wv*(e2v-Ev); infl1=Eva-Eva.mean()
            di=infl2-infl1; se=float(np.std(di,ddof=1)/np.sqrt(len(val)))
            obs=Ev-float(Eva.mean()); lin=-eta*valdot
            row=dict(shift=shift,trust=trust,eta=float(eta),
                     train_delta=Et-float(Etr.mean()),val_delta=obs,val_se=se,
                     val_z=obs/se if se else 0.,linear_val_pred=lin,
                     curvature_residual=obs-lin,
                     train_dot=trdot,val_dot=valdot,
                     train_ESS=float(1/(wt@wt)),val_ESS=float(1/(wv@wv)))
            rows.append(row); print("MICROTRUST",json.dumps(row),flush=True)
    out={"rows":rows}
    Path(a.out).write_text(json.dumps(out,indent=2))
if __name__=="__main__": main()
