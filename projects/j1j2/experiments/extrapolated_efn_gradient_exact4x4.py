import json, sys
from pathlib import Path
import numpy as np

KDIR=Path("/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure/experiments")
sys.path.insert(0,str(KDIR))
from closed_fn_krylov_exact4x4 import (
    build_H, ground, marshall_signs, canonical, basis,
    fixed_node_solve,
)

H,diag=build_H(0.5)
H0,_=build_H(0.0)
D=len(basis)
s=canonical(marshall_signs())
_,psi_init=ground(H0)
if np.dot(psi_init,s)<0: psi_init=-psi_init
a=np.abs(psi_init); a/=np.linalg.norm(a)

def grad_from_measure(a,measure):
    g=np.zeros(D,float)
    for i in range(D):
        st,en=H.indptr[i],H.indptr[i+1]
        for q in range(st,en):
            j=int(H.indices[q]); hij=float(H.data[q])
            if j==i: continue
            kij=float(s[i])*hij*float(s[j])
            if kij>0:
                c=float(measure[i])*kij*a[j]/a[i]
                g[j]+=c; g[i]-=c
    return g
def fn_energy(a):
    return float(fixed_node_solve(H,diag,a,s)[0])

def apply_dlog(a,d):
    z=np.asarray(d,float)
    z-=np.sum((a*a)*z)
    b=a*np.exp(np.clip(z,-20,20))
    b/=np.linalg.norm(b)
    return b

def trust_scale(d,a,rms_target=0.05,max_abs=0.50):
    q=a*a
    z=np.asarray(d,float).copy()
    z-=np.sum(q*z)
    rms=np.sqrt(np.sum(q*z*z))
    if rms>0: z*=rms_target/rms
    m=np.max(np.abs(z))
    if m>max_abs: z*=max_abs/m
    return z

def metrics(g,gtrue):
    n=np.linalg.norm(g); nt=np.linalg.norm(gtrue)
    return dict(
        cos=float((g@gtrue)/(max(n,1e-300)*max(nt,1e-300))),
        relerr=float(np.linalg.norm(g-gtrue)/max(nt,1e-300)),
        norm_ratio=float(n/max(nt,1e-300)),
    )

def step_report(a,g,e0):
    q=np.maximum(a*a,1e-300)
    z=trust_scale(-g/q,a)
    b=apply_dlog(a,z)
    return dict(
        E_after=fn_energy(b),
        delta=float(fn_energy(b)-e0),
        rms=float(np.sqrt(np.sum((a*a)*z*z))),
        maxabs=float(np.max(np.abs(z))),
    )
rows=[]
guides=[]
for k in range(6):
    e,phi=fixed_node_solve(H,diag,a,s)
    phi=np.abs(phi); phi/=np.linalg.norm(phi)
    pure=phi*phi; pure/=pure.sum()
    mixed=a*phi; mixed/=mixed.sum()
    var=a*a; var/=var.sum()

    gp=grad_from_measure(a,pure)
    gm=grad_from_measure(a,mixed)
    gv=grad_from_measure(a,var)
    ge=2*gm-gv

    row=dict(
        refresh_iter=k,
        E_FN=float(e),
        overlap=float(a@phi),
        pure_vs_mixed_TV=float(.5*np.abs(pure-mixed).sum()),
        pure_vs_extrap_TV=float(.5*np.abs(pure-(2*mixed-var)).sum()),
        mixed=metrics(gm,gp),
        variational=metrics(gv,gp),
        extrapolated=metrics(ge,gp),
        exact_step=step_report(a,gp,float(e)),
        mixed_step=step_report(a,gm,float(e)),
        extrapolated_step=step_report(a,ge,float(e)),
    )
    rows.append(row)
    guides.append((a.copy(),phi.copy(),pure.copy(),mixed.copy(),var.copy(),gp.copy()))
    print("EXTRAP_ROW",json.dumps(row,sort_keys=True),flush=True)

    # Same half-refresh map used in the MLE interpretation.
    a=np.sqrt(np.maximum(a*phi,1e-300))
    a/=np.linalg.norm(a)

# Sampling-only test of the estimator bias/variance at iter 0 and 2.
rng=np.random.default_rng(20261001)
sampling=[]
for k in (0,2):
    a,phi,pure,mixed,var,gp=guides[k]
    for M in (128,512,2048,8192):
        mm=[]; me=[]
        for rep in range(100):
            sm=rng.choice(D,size=M,p=mixed)
            sv=rng.choice(D,size=M,p=var)
            hm=np.bincount(sm,minlength=D).astype(float)/M
            hv=np.bincount(sv,minlength=D).astype(float)/M
            gm=grad_from_measure(a,hm)
            ge=2*gm-grad_from_measure(a,hv)
            mm.append(metrics(gm,gp))
            me.append(metrics(ge,gp))
        sampling.append(dict(
            refresh_iter=k,M=M,
            mixed_cos_mean=float(np.mean([x["cos"] for x in mm])),
            mixed_cos_sd=float(np.std([x["cos"] for x in mm])),
            mixed_relerr_mean=float(np.mean([x["relerr"] for x in mm])),
            extrap_cos_mean=float(np.mean([x["cos"] for x in me])),
            extrap_cos_sd=float(np.std([x["cos"] for x in me])),
            extrap_relerr_mean=float(np.mean([x["relerr"] for x in me])),
        ))
        print("SAMPLE_ROW",json.dumps(sampling[-1],sort_keys=True),flush=True)

out=dict(rows=rows,sampling=sampling,
         identity="p_pure = 2 p_mixed - p_var + O((phi-a)^2)")
opath=Path("/Users/aliotaifi/Chatty/projects/j1j2/results/extrapolated_efn_gradient_exact4x4.json")
opath.write_text(json.dumps(out,indent=2))
print("WROTE",opath,flush=True)
