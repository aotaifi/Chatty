import json, sys
from pathlib import Path
import numpy as np

KDIR=Path("/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure/experiments")
sys.path.insert(0,str(KDIR))
from closed_fn_krylov_exact4x4 import (
    build_H, ground, marshall_signs, canonical, basis,
    build_fixed_node, fixed_node_solve, physical_energy,
)

H,diag=build_H(0.5)
H0,_=build_H(0.0)
E0,psi0=ground(H)
s=canonical(marshall_signs())
if np.dot(psi0,s)<0: psi0=-psi0
atrue=np.abs(psi0); atrue/=np.linalg.norm(atrue)
strue=canonical(np.where(psi0>=0,1,-1).astype(np.int8))
ptrue=atrue**2
_,psi_init=ground(H0)
if np.dot(psi_init,s)<0: psi_init=-psi_init
a=np.abs(psi_init); a/=np.linalg.norm(a)

def efn_grad_logamp(a,s):
    e,phi=fixed_node_solve(H,diag,a,s)
    g=np.zeros_like(a)
    for i in range(len(a)):
        st,en=H.indptr[i],H.indptr[i+1]
        for q in range(st,en):
            j=int(H.indices[q]); hij=float(H.data[q])
            if j==i: continue
            kij=float(s[i])*hij*float(s[j])
            if kij>0:
                c=(phi[i]**2)*kij*(a[j]/a[i])
                g[j]+=c
                g[i]-=c
    return e,phi,g

def fn_energy(a,s):
    return fixed_node_solve(H,diag,a,s)[0]

def apply_dlog(a,d):
    z=np.asarray(d,float)
    z=z-np.sum((a*a)*z)
    b=a*np.exp(np.clip(z,-20,20))
    b/=np.linalg.norm(b)
    return b

def trust_scale(d,a,rms_target=0.05,max_abs=0.50):
    p=a*a; z=np.asarray(d,float); z-=np.sum(p*z)
    rms=np.sqrt(np.sum(p*z*z))
    if rms>0: z*=rms_target/rms
    m=np.max(np.abs(z))
    if m>max_abs: z*=max_abs/m
    return z

def local_field_stats(a,s):
    off=H@ (a*s) - diag*(a*s)
    sloc=np.where(off<=0,1,-1).astype(np.int8)
    sloc=canonical(sloc)
    flags=sloc!=s
    wrong=s!=strue
    return dict(
        flags=int(flags.sum()), wrong=int(wrong.sum()),
        true_flags=int(np.sum(flags & wrong)),
        false_flags=int(np.sum(flags & ~wrong)),
        caught_wrong_weight=float(np.sum(ptrue[flags & wrong])),
        total_wrong_weight=float(np.sum(ptrue[wrong])),
    )

def direction_report(name,a,s,d,e_before):
    z=trust_scale(d,a)
    b=apply_dlog(a,z)
    e=fn_energy(b,s)
    out=dict(
        name=name, E_before=float(e_before), E_after=float(e),
        delta=float(e-e_before),
        rms_dlog=float(np.sqrt(np.sum((a*a)*(z-np.sum((a*a)*z))**2))),
        max_abs_dlog=float(np.max(np.abs(z))),
        physical_E_after=float(physical_energy(H,b,s)),
        local_field=local_field_stats(b,s),
    )
    return out,b

e,phi,g=efn_grad_logamp(a,s)
rng=np.random.default_rng(20261001)
dtest=rng.normal(size=len(a)); dtest-=np.sum((a*a)*dtest)
eps=1e-5
ep=fn_energy(apply_dlog(a,eps*dtest),s)
em=fn_energy(apply_dlog(a,-eps*dtest),s)
fd=(ep-em)/(2*eps)
dot=float(g@dtest)

p=a*a
plain=-g
nat=-g/np.maximum(p,1e-300)
plain_out,ap=direction_report("plain_logamp_gradient",a,s,plain,e)
nat_out,an=direction_report("full_fisher_natural_gradient",a,s,nat,e)

refresh=[]
ar=a.copy()
for k in range(5):
    er,phir,gr=efn_grad_logamp(ar,s)
    refresh.append(dict(
        k=k,E_FN=float(er),grad_l2=float(np.linalg.norm(gr)),
        grad_nat_rms=float(np.sqrt(np.sum((ar*ar)*(gr/np.maximum(ar*ar,1e-300))**2))),
        local_field=local_field_stats(ar,s),
    ))
    ar=phir.copy(); ar/=np.linalg.norm(ar)

out=dict(
    E0=float(E0), initial_E_FN=float(e),
    initial_physical_E=float(physical_energy(H,a,s)),
    sign_overlap=float(abs(np.sum(ptrue*s*strue))),
    finite_difference=dict(fd=float(fd),dot=float(dot),abs_err=float(abs(fd-dot))),
    gradient_sum=float(np.sum(g)),
    plain=plain_out,natural=nat_out,
    refresh=refresh,
)
print("EFN_HF_SR4",json.dumps(out,sort_keys=True),flush=True)
Path("/Users/aliotaifi/Chatty/projects/j1j2/results/efn_hf_sr_exact4x4.json").write_text(json.dumps(out,indent=2))
