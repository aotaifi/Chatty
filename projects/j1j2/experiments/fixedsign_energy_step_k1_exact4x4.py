import sys, json
from pathlib import Path
import numpy as np

KDIR=Path("/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure/experiments")
sys.path.insert(0,str(KDIR))
from closed_fn_krylov_exact4x4 import build_H, ground, marshall_signs, canonical
from closed_fn_krylov_exact4x4 import projected_krylov_update, physical_energy, sign_hash

TRUST=0.05
MAXITER=100
H,diag=build_H(0.5)
H0,_=build_H(0.0)
E0,psi0=ground(H)
sM=canonical(marshall_signs())
if np.dot(psi0,sM)<0: psi0=-psi0
atrue=np.abs(psi0); atrue/=np.linalg.norm(atrue)
strue=canonical(np.where(psi0>=0,1,-1).astype(np.int8))
ptrue=atrue**2
_,psi_init=ground(H0)
if np.dot(psi_init,sM)<0: psi_init=-psi_init
a=np.abs(psi_init); a/=np.linalg.norm(a); s=sM.copy()

hist=[]
for it in range(MAXITER+1):
    E=physical_energy(H,a,s)
    O=float(abs(np.sum(ptrue*s*strue)))
    F=float(abs(np.dot(a,atrue))**2)
    rec={"it":it,"E":E,"Eerr":E-E0,"O":O,"F":F,"sign":sign_hash(s)}
    hist.append(rec)
    if it in {0,1,2,3,5,10,20,50,100} or O>1-1e-12:
        print("E4",json.dumps(rec),flush=True)
    if O>1-1e-12 and F>1-1e-8: break
    psi=a*s
    el=np.asarray((H@psi)/np.where(np.abs(psi)>1e-300,psi,1e-300),float)
    p=a*a; p/=p.sum()
    centered=el-float(np.sum(p*el))
    rms=float(np.sqrt(np.sum(p*centered*centered)))
    dlog=-TRUST*centered/max(rms,1e-14)
    dlog-=float(np.sum(p*dlog))
    anew=a*np.exp(np.clip(dlog,-5,5)); anew/=np.linalg.norm(anew)
    snew,t,eth,r,G=projected_krylov_update(H,diag,anew,s)
    a,s=anew,snew

out={"trust_rms":TRUST,"E0":E0,"history":hist,"final":hist[-1]}
Path("/Users/aliotaifi/Chatty/projects/j1j2/results/fixedsign_energy_step_k1_exact4x4.json").write_text(json.dumps(out,indent=2))
