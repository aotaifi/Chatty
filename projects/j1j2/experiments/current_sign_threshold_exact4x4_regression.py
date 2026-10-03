import sys, math, json
import numpy as np
from pathlib import Path
KDIR=Path("/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure/experiments")
sys.path.insert(0,str(KDIR))
from closed_fn_krylov_exact4x4 import build_H, ground, marshall_signs, canonical, projected_krylov_update, physical_energy

H,diag=build_H(0.5); H0,_=build_H(0.0)
_,psi0=ground(H); sM=canonical(marshall_signs())
if np.dot(psi0,sM)<0: psi0=-psi0
_,pi=ground(H0)
if np.dot(pi,sM)<0: pi=-pi
a=np.abs(pi); a/=np.linalg.norm(a); s=sM.copy()

# Make one amplitude improvement so current-sign field is nontrivial.
psi=a*s
el=np.asarray((H@psi)/psi,float)
p=a*a; E=float(np.sum(p*el)); cen=el-E
rms=float(np.sqrt(np.sum(p*cen*cen)))
dlog=-0.05*cen/rms; dlog-=float(np.sum(p*dlog))
a=a*np.exp(dlog); a/=np.linalg.norm(a)

# Exact result.
sex,t_exact,E_exact,r,G=projected_krylov_update(H,diag,a,s)

# Accumulator equivalent to 8x8 implementation, with full basis weights a^2.
states=np.arange(len(a),dtype=int); w=a*a; w/=w.sum()
order=np.argsort(r,kind="mergesort")
gid=np.empty(len(a),np.int32); g=0; ref=r[order[0]]; gid[order[0]]=0
for jj in order[1:]:
    v=r[jj]
    if abs(v-ref)>1e-10+1e-10*max(abs(v),abs(ref),1.0):
        g+=1; ref=v
    gid[jj]=g
G2=g+1
base=0.; diff=np.zeros(G2)
for i,wi in zip(states,w):
    base += wi*diag[i]
    st,en=H.indptr[i],H.indptr[i+1]
    for q in range(st,en):
        j=int(H.indices[q]); hij=float(H.data[q])
        if j==i: continue
        c=wi*hij*(s[j]/s[i])*(a[j]/a[i])
        base += c
        gi,gj=gid[i],gid[j]
        if gi!=gj:
            lo,hi=min(gi,gj),max(gi,gj)
            diff[lo] += -2*c
            diff[hi] += +2*c
cand=np.r_[base,base+np.cumsum(diff)]
ib=int(np.argmin(cand)); k=ib-1
if k<0: t_acc=float(np.min(r)-1)
elif k>=G2-1: t_acc=float(np.max(r)+1)
else:
    t_acc=.5*(np.max(r[gid==k])+np.min(r[gid==k+1]))
sacc=canonical((s*np.where(r<=t_acc,1,-1)).astype(np.int8))
Eacc=physical_energy(H,a,sacc)
out={"t_exact":t_exact,"t_acc":t_acc,"E_exact":E_exact,"E_acc":Eacc,
     "same_signs":bool(np.array_equal(sex,sacc)),"G_exact":G,"G_acc":G2,
     "base":base,"E_current":physical_energy(H,a,s)}
print("CURSIGN4_REGRESSION",json.dumps(out),flush=True)
assert np.array_equal(sex,sacc)
assert abs(E_exact-Eacc)<1e-10
