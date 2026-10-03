#!/usr/bin/env python3
import argparse, json, math, sys
from pathlib import Path
import numpy as np
import scipy.sparse as sp

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import optax
from flax import serialization
from nqsmagic.models import ViT
from nqsmagic.utils import _logpsi_transl_2d

from closed_fn_krylov_exact4x4 import (
    build_H, ground, marshall_signs, canonical, basis,
    build_fixed_node, fixed_node_solve, projected_krylov_update,
    physical_energy,
)

L=4
N=16
B=2
D=len(basis)


def bits2x(ss):
    a=np.asarray(ss,np.uint32).reshape(-1,1)
    return 2.0*(((a>>np.arange(N,dtype=np.uint32))&1).astype(np.float64))-1.0


def exact_loop_snapshot(iteration):
    H,diag=build_H(0.5)
    H0,_=build_H(0.0)
    s=canonical(marshall_signs())
    _,psi=ground(H0)
    if np.dot(psi,s)<0: psi=-psi
    a=np.abs(psi); a/=np.linalg.norm(a)
    for _ in range(iteration):
        _,phi=fixed_node_solve(H,diag,a,s)
        snew,_,_,_,_=projected_krylov_update(H,diag,phi,s)
        a,s=phi,snew
    return H,diag,a,s


def weighted_center(z,w):
    return z-np.sum(w*z)


def np_candidate(a,r):
    r=np.asarray(r,float)
    r-=np.sum((a*a)*r)
    r=np.clip(r,-20,20)
    b=a*np.exp(r)
    b/=np.linalg.norm(b)
    return b


def frozen_energy_np(F,b):
    return float(b@(F@b)/(b@b))


def local_stats(F,b,E):
    el=np.asarray(F@b/np.maximum(b,1e-300),float)
    w=b*b; w/=w.sum()
    de=el-E
    var=float(np.sum(w*de*de))
    q=np.quantile(np.abs(de),[.5,.9,.99,.999]).tolist()
    # deterministic weighted resample for a simple tail-index diagnostic
    rng=np.random.default_rng(20261002)
    ii=rng.choice(len(b),size=200000,replace=True,p=w)
    z=np.sort(np.abs(de[ii]))
    k=max(50,int(.01*len(z)))
    top=z[-k:]
    u=max(z[-k-1],1e-15)
    hill=float(k/np.sum(np.log(np.maximum(top/u,1+1e-15)))) if np.all(top>0) else float("inf")
    return dict(var=var,abs_q=q,max_abs=float(np.max(np.abs(de))),hill_alpha_top1pct=hill)


def candidate_metrics(H,diag,F,a,s,phi,target_log,b,label):
    efrozen=frozen_energy_np(F,b)
    w=phi*phi; w/=w.sum()
    lr=np.log(np.maximum(b,1e-300))-np.log(np.maximum(phi,1e-300))
    lr=weighted_center(lr,w)
    log_rms=float(np.sqrt(np.sum(w*lr*lr)))
    fidelity=float((b@phi)**2)
    return dict(label=label,frozen_energy=efrozen,fidelity_phi=fidelity,
                phi_weighted_logratio_rms=log_rms,
                local=local_stats(F,b,efrozen))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--iteration",type=int,required=True)
    ap.add_argument("--steps",type=int,default=1000)
    ap.add_argument("--out",required=True)
    args=ap.parse_args()

    H,diag,a,s=exact_loop_snapshot(args.iteration)
    efn,phi=fixed_node_solve(H,diag,a,s)
    F=build_fixed_node(H,diag,a,s)
    efn_refresh,_=fixed_node_solve(H,diag,phi,s)
    Ebase=frozen_energy_np(F,a)

    wphi=phi*phi; wphi/=wphi.sum()
    target=np.log(np.maximum(phi,1e-300))-np.log(np.maximum(a,1e-300))
    target=weighted_center(target,wphi)
    target_rms=float(np.sqrt(np.sum(wphi*target*target)))
    exact_gain=float(Ebase-efn)
    rebuild_exact_gain=float(efn-efn_refresh)

    X=jnp.asarray(bits2x(basis))
    aj=jnp.asarray(a)
    diagj=jnp.asarray(F.diagonal())
    co=sp.triu(F-sp.diags(F.diagonal()),k=1).tocoo()
    ei=jnp.asarray(co.row.astype(np.int32)); ej=jnp.asarray(co.col.astype(np.int32))
    hv=jnp.asarray(co.data.astype(np.float64))
    wj=jnp.asarray(wphi)
    tj=jnp.asarray(target)

    model=ViT(num_layers=4,d_model=60,heads=10,L_eff=4,b=2,
              transl_invariant=True,two_dimensional=True)
    p0=model.init(jax.random.PRNGKey(20261002+args.iteration),X[:2])["params"]

    def fnet(p):
        return jnp.real(_logpsi_transl_2d(model.apply,B,{"params":p},X))

    f0=jax.lax.stop_gradient(fnet(p0))

    def residual(p):
        return fnet(p)-f0

    def amp_from_p(p):
        r=residual(p)
        # gauge-fix using base a^2; clipping is intentionally absent during training
        r=r-jnp.sum((aj*aj)*r)
        logb=jnp.log(jnp.maximum(aj,1e-300))+r
        logb=logb-0.5*jax.scipy.special.logsumexp(2*logb)
        return jnp.exp(logb),r

    def energy_loss(p):
        b,_=amp_from_p(p)
        num=jnp.sum(diagj*b*b)+2*jnp.sum(hv*b[ei]*b[ej])
        return num

    def oracle_loss(p):
        _,r=amp_from_p(p)
        rc=r-jnp.sum(wj*r)
        return jnp.sum(wj*(rc-tj)**2)

    def train(obj,name,lr):
        p=p0
        tx=optax.chain(optax.clip_by_global_norm(10.0),optax.adam(lr))
        st=tx.init(p)
        valgrad=jax.jit(jax.value_and_grad(obj))
        checkpoints={}
        logsteps={1,10,50,100,250,500,args.steps}
        for it in range(1,args.steps+1):
            loss,g=valgrad(p)
            upd,st=tx.update(g,st,p)
            p=optax.apply_updates(p,upd)
            if it in logsteps:
                b,r=amp_from_p(p)
                bn=np.asarray(b,float)
                rr=np.asarray(r,float)
                fr=float(energy_loss(p))
                orl=float(oracle_loss(p))
                fid=float((bn@phi)**2)
                lrerr=np.log(np.maximum(bn,1e-300))-np.log(np.maximum(phi,1e-300))
                lrerr=weighted_center(lrerr,wphi)
                row=dict(step=it,loss=float(loss),frozen_energy=fr,oracle_mse=orl,
                         fidelity_phi=fid,
                         phi_weighted_logratio_rms=float(np.sqrt(np.sum(wphi*lrerr*lrerr))),
                         residual_rms_phi=float(np.sqrt(np.sum(wphi*(rr-np.sum(wphi*rr))**2))))
                print("TRAIN",name,json.dumps(row,sort_keys=True),flush=True)
                if it in (250,args.steps):
                    checkpoints[it]=(bn, row)
        return p,checkpoints

    print("SETUP",json.dumps(dict(
        iteration=args.iteration,D=D,Ebase=Ebase,Efn=efn,Efn_after_exact_refresh=efn_refresh,
        exact_frozen_gain=exact_gain,exact_rebuild_gain=rebuild_exact_gain,
        target_logratio_rms=target_rms,
        guide_physical_E=float(physical_energy(H,a,s)),
        n_edges=int(len(co.data))),sort_keys=True),flush=True)

    pE,ckE=train(energy_loss,"frozen_energy",1e-4)
    pR,ckR=train(oracle_loss,"oracle_ratio",2e-4)

    results={}
    for name,cks in (("frozen_energy",ckE),("oracle_ratio",ckR)):
        results[name]={}
        for step,(b,row) in cks.items():
            met=candidate_metrics(H,diag,F,a,s,phi,target,b,f"{name}_{step}")
            erebuild,_=fixed_node_solve(H,diag,b,s)
            met["rebuilt_Efn"]=float(erebuild)
            met["frozen_gain_fraction"]=float((Ebase-met["frozen_energy"])/exact_gain) if exact_gain>0 else float("nan")
            met["rebuild_gain_fraction"]=float((efn-met["rebuilt_Efn"])/rebuild_exact_gain) if abs(rebuild_exact_gain)>1e-14 else float("nan")
            met["train_row"]=row
            results[name][str(step)]=met
            print("CANDIDATE",json.dumps(met,sort_keys=True),flush=True)

    base_met=candidate_metrics(H,diag,F,a,s,phi,target,a,"base")
    phi_met=candidate_metrics(H,diag,F,a,s,phi,target,phi,"exact_refresh")
    out=dict(
        iteration=args.iteration,
        setup=dict(D=D,Ebase=Ebase,Efn=efn,Efn_after_exact_refresh=efn_refresh,
                   exact_frozen_gain=exact_gain,exact_rebuild_gain=rebuild_exact_gain,
                   target_logratio_rms=target_rms,
                   guide_physical_E=float(physical_energy(H,a,s)),
                   n_edges=int(len(co.data))),
        base=base_met,exact_refresh=phi_met,results=results,
    )
    Path(args.out).write_text(json.dumps(out,indent=2))
    print("RESULT_FILE",args.out,flush=True)


if __name__=="__main__":
    main()
