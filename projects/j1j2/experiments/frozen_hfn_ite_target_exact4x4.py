#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import optax
from nqsmagic.models import ViT
from nqsmagic.utils import _logpsi_transl_2d

from closed_fn_krylov_exact4x4 import (
    build_H, ground, marshall_signs, canonical, basis,
    build_fixed_node, fixed_node_solve, projected_krylov_update,
)

L=4; N=16; B=2; D=len(basis)

def bits2x(ss):
    a=np.asarray(ss,np.uint32).reshape(-1,1)
    return 2.0*(((a>>np.arange(N,dtype=np.uint32))&1).astype(np.float64))-1.0

def snapshot(iteration):
    H,diag=build_H(.5); H0,_=build_H(0.0)
    s=canonical(marshall_signs())
    _,psi=ground(H0)
    if np.dot(psi,s)<0: psi=-psi
    a=np.abs(psi); a/=np.linalg.norm(a)
    for _ in range(iteration):
        _,phi=fixed_node_solve(H,diag,a,s)
        s,_,_,_,_=projected_krylov_update(H,diag,phi,s)
        a=phi
    return H,diag,a,s

def energy(F,b):
    return float(b@(F@b)/(b@b))

def optimal_dt(F,b):
    # Exact version of Ledinauskas-Anisimovas Eq. (11).
    Fb=F@b; F2b=F@Fb
    e1=float(b@Fb)
    e2=float(Fb@Fb)
    e3=float(Fb@F2b)
    sig2=max(e2-e1*e1,0.0)
    A=e2*e2-e1*e3
    Bc=e1*e2-e3
    cands=[]
    if abs(A)>1e-15:
        disc=max(Bc*Bc+4*A*sig2,0.0)
        root=np.sqrt(disc)
        cands.extend([(Bc+root)/(2*A),(Bc-root)/(2*A)])
    # retain positive finite stationary points
    cands=[float(x) for x in cands if np.isfinite(x) and x>0]
    if not cands:
        # robust fallback: tiny Euler step
        cands=[1e-3]
    def Et(dt):
        t=b-dt*Fb
        return energy(F,t)
    dt=min(cands,key=Et)

    # Our inner problem is positive-amplitude only. If the Euler target would
    # change sign, cap dt just below the first zero. Record this explicitly.
    el=Fb/np.maximum(b,1e-300)
    pos=el[el>0]
    cap=np.inf if len(pos)==0 else 0.95/float(np.max(pos))
    capped=False
    if dt>cap:
        dt=cap; capped=True
    t=b-dt*Fb
    if np.min(t)<=0:
        raise RuntimeError(("nonpositive target despite cap",dt,float(np.min(t))))
    t/=np.linalg.norm(t)
    return dt,t,dict(E1=e1,E2=e2,E3=e3,sigma2=sig2,dt_cap=cap,capped=capped,target_E=energy(F,t))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--iteration",type=int,default=10)
    ap.add_argument("--lr",type=float,required=True)
    ap.add_argument("--epochs",type=int,default=20)
    ap.add_argument("--energy-samples",type=int,default=100000)
    ap.add_argument("--max-inner",type=int,default=2000)
    ap.add_argument("--out",required=True)
    args=ap.parse_args()

    H,diag,a,s=snapshot(args.iteration)
    F=build_fixed_node(H,diag,a,s)
    efn,phi=fixed_node_solve(H,diag,a,s)
    efn2,_=fixed_node_solve(H,diag,phi,s)
    Ebase=energy(F,a)
    exact_gain=Ebase-efn
    rebuild_gain=efn-efn2
    wphi=phi*phi; wphi/=wphi.sum()

    X=jnp.asarray(bits2x(basis))
    aj=jnp.asarray(a)
    model=ViT(num_layers=4,d_model=60,heads=10,L_eff=4,b=2,
              transl_invariant=True,two_dimensional=True)
    p=model.init(jax.random.PRNGKey(20261006+args.iteration),X[:2])["params"]

    def fnet(pp):
        return jnp.real(_logpsi_transl_2d(model.apply,B,{"params":pp},X))
    f0=jax.lax.stop_gradient(fnet(p))
    def amp(pp):
        r=fnet(pp)-f0
        r=r-jnp.sum((aj*aj)*r)
        logb=jnp.log(jnp.maximum(aj,1e-300))+r
        logb=logb-0.5*jax.scipy.special.logsumexp(2*logb)
        return jnp.exp(logb)

    tx=optax.chain(optax.clip_by_global_norm(10.0),optax.adam(args.lr))
    st=tx.init(p)
    rows=[]

    for ep in range(1,args.epochs+1):
        b0=np.asarray(amp(p),float)
        E0=energy(F,b0)
        dt,target,meta=optimal_dt(F,b0)
        tj=jnp.asarray(target)

        def overlap_loss(pp):
            b=amp(pp)
            ov=jnp.dot(b,tj)
            # both normalized; keep normalized form for numerical robustness
            n2=jnp.dot(b,b)*jnp.dot(tj,tj)
            return -jnp.log(jnp.maximum((ov*ov)/n2,1e-300))
        vg=jax.jit(jax.value_and_grad(overlap_loss))
        sigmaE=float(np.sqrt(meta["sigma2"]/args.energy_samples))
        Ethr=E0-sigmaE
        reached=False
        inner_steps=0
        Ef=E0
        for k in range(1,args.max_inner+1):
            loss,g=vg(p)
            upd,st=tx.update(g,st,p)
            p=optax.apply_updates(p,upd)
            b=np.asarray(amp(p),float)
            Ef=energy(F,b)
            inner_steps=k
            if Ef < Ethr:
                reached=True
                break
        if not reached:
            print("ITE_STALL",ep,E0,Ethr,Ef,inner_steps,flush=True)

        erebuild,_=fixed_node_solve(H,diag,b,s)
        fid=float((b@phi)**2)
        lrerr=np.log(np.maximum(b,1e-300))-np.log(np.maximum(phi,1e-300))
        lrerr-=np.sum(wphi*lrerr)
        row=dict(epoch=ep,inner_steps=inner_steps,reached_threshold=reached,
                 lr=args.lr,dt=dt,sigmaE=sigmaE,energy_threshold=Ethr,
                 E_before=E0,E_after=Ef,delta=Ef-E0,
                 target_E=meta["target_E"],target_capped=meta["capped"],
                 dt_cap=meta["dt_cap"],last_loss=float(loss),
                 rebuilt_Efn=float(erebuild),
                 frozen_gain_fraction=float((Ebase-Ef)/exact_gain),
                 rebuild_gain_fraction=float((efn-erebuild)/rebuild_gain),
                 fidelity_phi=fid,
                 phi_logratio_rms=float(np.sqrt(np.sum(wphi*lrerr*lrerr))))
        rows.append(row)
        print("ITE",json.dumps(row,sort_keys=True),flush=True)
        if not reached:
            break

    b=np.asarray(amp(p),float)
    Efin=energy(F,b)
    erebuild,_=fixed_node_solve(H,diag,b,s)
    lrerr=np.log(np.maximum(b,1e-300))-np.log(np.maximum(phi,1e-300))
    lrerr-=np.sum(wphi*lrerr)
    out=dict(iteration=args.iteration,lr=args.lr,epochs=args.epochs,
             energy_samples=args.energy_samples,max_inner=args.max_inner,
             Ebase=Ebase,Efn=efn,Efn_after_exact_refresh=efn2,
             exact_frozen_gain=exact_gain,exact_rebuild_gain=rebuild_gain,
             final_frozen_energy=Efin,final_rebuilt_Efn=float(erebuild),
             frozen_gain_fraction=float((Ebase-Efin)/exact_gain),
             rebuild_gain_fraction=float((efn-erebuild)/rebuild_gain),
             fidelity_phi=float((b@phi)**2),
             phi_logratio_rms=float(np.sqrt(np.sum(wphi*lrerr*lrerr))),
             rows=rows)
    Path(args.out).write_text(json.dumps(out,indent=2))
    print("FINAL",json.dumps({k:v for k,v in out.items() if k!="rows"},sort_keys=True),flush=True)

if __name__=="__main__":
    main()
