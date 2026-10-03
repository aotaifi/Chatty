#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
import numpy as np
import scipy.sparse.linalg as spla
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from nqsmagic.models import ViT
from nqsmagic.utils import _logpsi_transl_2d

from closed_fn_krylov_exact4x4 import (
    build_H, ground, marshall_signs, canonical, basis,
    build_fixed_node, fixed_node_solve, projected_krylov_update,
)

L=4; N=16; B=2

def bits2x(ss):
    a=np.asarray(ss,np.uint32).reshape(-1,1)
    return 2.0*(((a>>np.arange(N,dtype=np.uint32))&1).astype(np.float64))-1.0

def snapshot(iteration):
    H,diag=build_H(0.5); H0,_=build_H(0.0)
    s=canonical(marshall_signs())
    _,psi=ground(H0)
    if np.dot(psi,s)<0: psi=-psi
    a=np.abs(psi); a/=np.linalg.norm(a)
    for _ in range(iteration):
        _,phi=fixed_node_solve(H,diag,a,s)
        s,_,_,_,_=projected_krylov_update(H,diag,phi,s)
        a=phi
    return H,diag,a,s

def energy_np(F,b):
    return float(b@(F@b)/(b@b))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--iteration",type=int,default=10)
    ap.add_argument("--shift",type=float,required=True)
    ap.add_argument("--trust",type=float,default=0.002)
    ap.add_argument("--steps",type=int,default=30)
    ap.add_argument("--chunk",type=int,default=512)
    ap.add_argument("--cg-tol",type=float,default=3e-5)
    ap.add_argument("--cg-maxiter",type=int,default=120)
    ap.add_argument("--out",required=True)
    args=ap.parse_args()

    H,diag,a,s=snapshot(args.iteration)
    efn,phi=fixed_node_solve(H,diag,a,s)
    F=build_fixed_node(H,diag,a,s)
    efn2,_=fixed_node_solve(H,diag,phi,s)
    Ebase=energy_np(F,a)
    exact_gain=Ebase-efn
    rebuild_gain=efn-efn2
    wphi=phi*phi; wphi/=wphi.sum()
    X=np.asarray(bits2x(basis))
    D=len(X)
    model=ViT(num_layers=4,d_model=60,heads=10,L_eff=4,b=2,
              transl_invariant=True,two_dimensional=True)
    p0=model.init(jax.random.PRNGKey(20261004+args.iteration),
                  jnp.asarray(X[:2]))["params"]
    flat,unravel=ravel_pytree(p0)
    npar=int(flat.size)

    def raw_f(z,x):
        return jnp.real(_logpsi_transl_2d(
            model.apply,B,{"params":unravel(z)},x))

    f_chunk=jax.jit(raw_f)

    @jax.jit
    def jvp_chunk(z,v,x):
        return jax.jvp(lambda zz: raw_f(zz,x),(z,),(v,))[1]

    @jax.jit
    def vjp_chunk(z,x,w):
        _,pb=jax.vjp(lambda zz: raw_f(zz,x),z)
        return pb(w)[0]

    bounds=[(i,min(i+args.chunk,D)) for i in range(0,D,args.chunk)]

    def eval_f(z):
        out=[]
        for lo,hi in bounds:
            out.append(np.asarray(f_chunk(z,jnp.asarray(X[lo:hi]))))
        return np.concatenate(out)
    f0=eval_f(flat)

    def amp_np(z):
        r=eval_f(z)-f0
        logb=np.log(np.maximum(a,1e-300))+r
        m=np.max(logb)
        b=np.exp(logb-m)
        b/=np.linalg.norm(b)
        return b,r

    def energy_grad(z):
        b,_=amp_np(z)
        Fb=F@b
        e=float(b@Fb)
        # exact derivative dE / d(log b_i); its sum is zero
        q=2.0*b*(Fb-e*b)
        g=np.zeros(npar,dtype=np.float64)
        for lo,hi in bounds:
            gg=vjp_chunk(
                z,jnp.asarray(X[lo:hi]),
                jnp.asarray(q[lo:hi]))
            g+=np.asarray(gg)
        return e,g,b

    def jv_all(z,v):
        vv=jnp.asarray(v)
        out=[]
        for lo,hi in bounds:
            out.append(np.asarray(jvp_chunk(
                z,vv,jnp.asarray(X[lo:hi]))))
        return np.concatenate(out)

    rows=[]
    print("SETUP",json.dumps(dict(D=D,npar=npar,chunk=args.chunk,
          Ebase=Ebase,Efn=efn,exact_gain=exact_gain)),flush=True)
    for it in range(1,args.steps+1):
        e,g,bcur=energy_grad(flat)
        pcur=bcur*bcur
        pcur/=pcur.sum()

        def fisher_mv(v):
            u=jv_all(flat,v)
            mu=float(np.dot(pcur,u))
            wc=pcur*(u-mu)
            out=np.zeros(npar,dtype=np.float64)
            for lo,hi in bounds:
                gg=vjp_chunk(
                    flat,jnp.asarray(X[lo:hi]),
                    jnp.asarray(wc[lo:hi]))
                out+=np.asarray(gg)
            return out+args.shift*np.asarray(v)

        A=spla.LinearOperator((npar,npar),matvec=fisher_mv,dtype=np.float64)
        sol,info=spla.cg(A,g,rtol=args.cg_tol,atol=0.0,
                         maxiter=args.cg_maxiter)
        relres=float(np.linalg.norm(fisher_mv(sol)-g)/
                     max(np.linalg.norm(g),1e-300))

        d=-sol
        dr=jv_all(flat,d)
        dr-=np.dot(pcur,dr)
        rms_per_eta=float(np.sqrt(np.dot(pcur,dr*dr)))
        eta=args.trust/max(rms_per_eta,1e-300)

        accepted=False; best=None
        for fac in [1.0,0.5,0.25,0.125,0.0625,0.03125,0.015625]:
            cand=flat+jnp.asarray(eta*fac*d)
            bc,_=amp_np(cand)
            ec=energy_np(F,bc)
            if np.isfinite(ec) and ec<e:
                best=(cand,ec,fac,bc); accepted=True; break
        if not accepted:
            row=dict(step=it,E=e,accepted=False,cg_info=int(info),
                     cg_relres=relres,rms_per_eta=rms_per_eta)
            rows.append(row); print("SR",json.dumps(row),flush=True)
            break

        flat,ec,fac,bn=best
        lr=np.log(np.maximum(bn,1e-300))-np.log(np.maximum(phi,1e-300))
        lr-=np.sum(wphi*lr)
        row=dict(step=it,E_before=e,E_after=ec,delta=ec-e,
                 accepted=True,shift=args.shift,trust=args.trust,
                 factor=fac,cg_info=int(info),cg_relres=relres,
                 rms_per_eta=rms_per_eta,
                 phi_fidelity=float((bn@phi)**2),
                 phi_logratio_rms=float(np.sqrt(np.sum(wphi*lr*lr))))
        rows.append(row)
        print("SR",json.dumps(row,sort_keys=True),flush=True)

    bfin,_=amp_np(flat)
    Efin=energy_np(F,bfin)
    erebuild,_=fixed_node_solve(H,diag,bfin,s)
    lr=np.log(np.maximum(bfin,1e-300))-np.log(np.maximum(phi,1e-300))
    lr-=np.sum(wphi*lr)
    out=dict(iteration=args.iteration,shift=args.shift,trust=args.trust,
             chunk=args.chunk,cg_tol=args.cg_tol,cg_maxiter=args.cg_maxiter,
             steps_requested=args.steps,
             steps_accepted=int(sum(bool(r.get("accepted")) for r in rows)),
             Ebase=Ebase,Efn=efn,Efn_after_exact_refresh=efn2,
             exact_frozen_gain=exact_gain,exact_rebuild_gain=rebuild_gain,
             final_frozen_energy=Efin,final_rebuilt_Efn=float(erebuild),
             frozen_gain_fraction=float((Ebase-Efin)/exact_gain),
             rebuild_gain_fraction=float((efn-erebuild)/rebuild_gain),
             fidelity_phi=float((bfin@phi)**2),
             phi_logratio_rms=float(np.sqrt(np.sum(wphi*lr*lr))),
             rows=rows)
    Path(args.out).write_text(json.dumps(out,indent=2))
    print("FINAL",json.dumps({k:v for k,v in out.items() if k!="rows"},
          sort_keys=True),flush=True)

if __name__=="__main__":
    main()
