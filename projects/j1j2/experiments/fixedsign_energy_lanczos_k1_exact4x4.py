import sys,json
from pathlib import Path
import numpy as np
import scipy.linalg as la
KDIR=Path("/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure/experiments")
sys.path.insert(0,str(KDIR))
from closed_fn_krylov_exact4x4 import build_H,ground,marshall_signs,canonical,physical_energy,sign_hash

BASE_TRUST=.05; MAXITER=120
H,diag=build_H(.5);H0,_=build_H(0.)
E0,psi0=ground(H);sM=canonical(marshall_signs())
if np.dot(psi0,sM)<0:psi0=-psi0
atrue=np.abs(psi0);atrue/=np.linalg.norm(atrue)
strue=canonical(np.where(psi0>=0,1,-1).astype(np.int8));ptrue=atrue**2
_,pi=ground(H0)
if np.dot(pi,sM)<0:pi=-pi
a=np.abs(pi);a/=np.linalg.norm(a);s=sM.copy()
hist=[]
for it in range(MAXITER+1):
    psi=a*s; E=physical_energy(H,a,s)
    O=float(abs(np.sum(ptrue*s*strue)));F=float(abs(np.dot(a,atrue))**2)
    rec={"it":it,"E":E,"Eerr":E-E0,"O":O,"F":F,"sign":sign_hash(s)}
    hist.append(rec)
    if it in {0,1,2,3,5,10,20,50,75,100,120} or O>1-1e-12:
        print("LK4",json.dumps(rec),flush=True)
    if O>1-1e-12 and E-E0<1e-6:break
    # one trust-gated exact fixed-sign natural-gradient / imaginary-time amplitude step
    el=np.asarray((H@psi)/psi,float);p=a*a;p/=p.sum();cen=el-p@el
    rms=float(np.sqrt(p@(cen*cen)));trust=BASE_TRUST;accepted=False
    while trust>=1e-6:
        dlog=-trust*cen/max(rms,1e-14);dlog-=p@dlog
        at=a*np.exp(np.clip(dlog,-5,5));at/=np.linalg.norm(at)
        Et=physical_energy(H,at,s)
        if Et<E-1e-12:accepted=True;break
        trust*=.5
    if not accepted:break
    # 2D Rayleigh-Ritz in span{psi,H psi}; extract vector c0 psi + c1 Hpsi = -c1 (T-H)psi.
    v0=at*s;v1=H@v0
    S=np.array([[v0@v0,v0@v1],[v1@v0,v1@v1]],float)
    Hv1=H@v1
    K=np.array([[v0@v1,v0@Hv1],[v1@v1,v1@Hv1]],float)
    ew,ev=la.eigh(K,S)
    cc=ev[:,0]
    if abs(cc[1])<1e-14: T=np.inf; sn=s.copy()
    else:
        T=float(-cc[0]/cc[1])
        r=np.asarray((H@v0)/v0,float)
        sn=canonical((s*np.where(T-r>=0,1,-1)).astype(np.int8))
    # sign-only acceptance gate: don't accept a sign update that raises fixed-amplitude energy
    Es=physical_energy(H,at,sn)
    if Es>Et+1e-12: sn=s.copy(); Es=Et
    hist[-1].update({"trust":trust,"T_lanczos":T,"Eamp":Et,"Esign":Es,"ritz_E":float(ew[0])})
    a,s=at,sn
Path("/Users/aliotaifi/Chatty/projects/j1j2/results/fixedsign_energy_lanczos_k1_exact4x4.json").write_text(json.dumps({"E0":E0,"history":hist,"final":hist[-1]},indent=2))
