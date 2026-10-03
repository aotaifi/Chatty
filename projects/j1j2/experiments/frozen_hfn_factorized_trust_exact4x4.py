#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from nqsmagic.models import ViT
from nqsmagic.utils import _logpsi_transl_2d

from closed_fn_krylov_exact4x4 import (
    build_H, ground, marshall_signs, canonical, basis,
    build_fixed_node, fixed_node_solve, projected_krylov_update,
    physical_energy,
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
    ap.add_argument("--trust",type=float,required=True)
    ap.add_argument("--steps",type=int,default=40)
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

    X=jnp.asarray(bits2x(basis))
    aj=jnp.asarray(a)
    dj=jnp.asarray(F.diagonal())
    co=sp.triu(F-sp.diags(F.diagonal()),k=1).tocoo()
    ei=jnp.asarray(co.row.astype(np.int32)); ej=jnp.asarray(co.col.astype(np.int32))
    hv=jnp.asarray(co.data.astype(np.float64))

    model=ViT(num_layers=4,d_model=60,heads=10,L_eff=4,b=2,
              transl_invariant=True,two_dimensional=True)
    p=model.init(jax.random.PRNGKey(20261003+args.iteration),X[:2])["params"]
    flat0,unravel=ravel_pytree(p)

    def fnet_flat(flat):
        pp=unravel(flat)
        return jnp.real(_logpsi_transl_2d(model.apply,B,{"params":pp},X))

    f0=jax.lax.stop_gradient(fnet_flat(flat0))

    def residual(flat):
        return fnet_flat(flat)-f0

    def amp(flat):
        r=residual(flat)
        r=r-jnp.sum((aj*aj)*r)
        logb=jnp.log(jnp.maximum(aj,1e-300))+r
        logb=logb-0.5*jax.scipy.special.logsumexp(2*logb)
        return jnp.exp(logb),r

    def loss(flat):
        b,_=amp(flat)
        return jnp.sum(dj*b*b)+2*jnp.sum(hv*b[ei]*b[ej])

    vg=jax.jit(jax.value_and_grad(loss))
    lossj=jax.jit(loss)
    ampj=jax.jit(amp)

    flat=flat0
    rows=[]
    factors=[1.0,0.5,0.25,0.125,0.0625,0.03125]
    for it in range(1,args.steps+1):
        e,g=vg(flat)
        e=float(e)
        gn=float(jnp.linalg.norm(g))
        if not np.isfinite(gn) or gn==0:
            break
        d=-g/gn
        # JVP calibrates parameter displacement to requested RMS delta log-amplitude.
        _,dr=jax.jvp(residual,(flat,),(d,))
        bcur,_=ampj(flat)
        wc=bcur*bcur; wc=wc/jnp.sum(wc)
        drc=dr-jnp.sum(wc*dr)
        slope=float(jnp.sqrt(jnp.sum(wc*drc*drc)))
        if not np.isfinite(slope) or slope==0:
            break
        alpha=args.trust/slope

        accepted=False; best=None
        for fac in factors:
            cand=flat+(alpha*fac)*d
            ec=float(lossj(cand))
            if np.isfinite(ec) and ec < e:
                accepted=True; best=(cand,ec,fac); break
        if not accepted:
            row=dict(step=it,E=e,grad_norm=gn,jvp_rms_per_unit=slope,
                     accepted=False)
            rows.append(row); print("TRUST",json.dumps(row),flush=True)
            break
        flat,ec,fac=best
        bn,rn=ampj(flat); bn=np.asarray(bn,float); rn=np.asarray(rn,float)
        lr=np.log(np.maximum(bn,1e-300))-np.log(np.maximum(phi,1e-300))
        lr-=np.sum(wphi*lr)
        row=dict(step=it,E_before=e,E_after=ec,delta=ec-e,
                 grad_norm=gn,jvp_rms_per_unit=slope,
                 factor=fac,requested_trust=args.trust,
                 phi_fidelity=float((bn@phi)**2),
                 phi_logratio_rms=float(np.sqrt(np.sum(wphi*lr*lr))),
                 residual_rms=float(np.sqrt(np.sum(wphi*(rn-np.sum(wphi*rn))**2))),
                 accepted=True)
        rows.append(row)
        if it<=5 or it%5==0 or it==args.steps:
            print("TRUST",json.dumps(row,sort_keys=True),flush=True)

    bfin,_=ampj(flat); bfin=np.asarray(bfin,float)
    Efin=energy_np(F,bfin)
    erebuild,_=fixed_node_solve(H,diag,bfin,s)
    lr=np.log(np.maximum(bfin,1e-300))-np.log(np.maximum(phi,1e-300))
    lr-=np.sum(wphi*lr)
    out=dict(iteration=args.iteration,trust=args.trust,steps_requested=args.steps,
             steps_accepted=int(sum(bool(r.get("accepted")) for r in rows)),
             Ebase=Ebase,Efn=efn,Efn_after_exact_refresh=efn2,
             exact_frozen_gain=exact_gain,exact_rebuild_gain=rebuild_gain,
             final_frozen_energy=Efin,
             final_rebuilt_Efn=float(erebuild),
             frozen_gain_fraction=float((Ebase-Efin)/exact_gain),
             rebuild_gain_fraction=float((efn-erebuild)/rebuild_gain),
             fidelity_phi=float((bfin@phi)**2),
             phi_logratio_rms=float(np.sqrt(np.sum(wphi*lr*lr))),
             rows=rows)
    Path(args.out).write_text(json.dumps(out,indent=2))
    print("FINAL",json.dumps({k:v for k,v in out.items() if k!="rows"},sort_keys=True),flush=True)

if __name__=="__main__":
    main()
