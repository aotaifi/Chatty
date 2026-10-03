import sys, json
from pathlib import Path
import numpy as np
KDIR=Path("/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure/experiments")
sys.path.insert(0,str(KDIR))
from closed_fn_krylov_exact4x4 import build_H, ground, marshall_signs, canonical, physical_energy, sign_hash

BASE_TRUST=0.05; MAXITER=120; ALPHA=1.2

def wkmeans(r,w):
    w=np.asarray(w,float); w/=w.sum()
    o=np.argsort(r); rr=r[o]; ww=w[o]; c=np.cumsum(ww)
    def q(p): return float(np.interp(p,c,rr))
    lo,hi=q(.1),q(.9); a=np.clip(r,lo,hi)
    m1,m2=q(.3),q(.8)
    for _ in range(100):
        cut=.5*(m1+m2); lab=a>cut
        if lab.all() or (~lab).all(): break
        n1=float(np.sum(w[~lab]*a[~lab])/np.sum(w[~lab]))
        n2=float(np.sum(w[lab]*a[lab])/np.sum(w[lab]))
        if abs(n1-m1)+abs(n2-m2)<1e-12:
            m1,m2=n1,n2; break
        m1,m2=n1,n2
    return .5*(m1+m2)

H,diag=build_H(.5); H0,_=build_H(0.0)
E0,psi0=ground(H); sM=canonical(marshall_signs())
if np.dot(psi0,sM)<0: psi0=-psi0
atrue=np.abs(psi0); atrue/=np.linalg.norm(atrue)
strue=canonical(np.where(psi0>=0,1,-1).astype(np.int8)); ptrue=atrue**2
_,pi=ground(H0)
if np.dot(pi,sM)<0: pi=-pi
a=np.abs(pi); a/=np.linalg.norm(a); s=sM.copy()
hist=[]
for it in range(MAXITER+1):
    E=physical_energy(H,a,s); O=float(abs(np.sum(ptrue*s*strue))); F=float(abs(np.dot(a,atrue))**2)
    rec={"it":it,"E":E,"Eerr":E-E0,"O":O,"F":F,"sign":sign_hash(s)}
    hist.append(rec)
    if it in {0,1,2,3,5,10,20,50,75,100,120} or O>1-1e-12:
        print("KM4",json.dumps(rec),flush=True)
    if O>1-1e-12 and E-E0<1e-6: break
    psi=a*s; el=np.asarray((H@psi)/np.where(np.abs(psi)>1e-300,psi,1e-300),float)
    p=a*a; p/=p.sum(); cen=el-float(np.sum(p*el)); rms=float(np.sqrt(np.sum(p*cen*cen)))
    trust=BASE_TRUST; accepted=False
    while trust>=1e-6:
        dlog=-trust*cen/max(rms,1e-14); dlog-=float(np.sum(p*dlog))
        at=a*np.exp(np.clip(dlog,-5,5)); at/=np.linalg.norm(at)
        Et=physical_energy(H,at,s)
        if Et<E-1e-12: accepted=True; break
        trust*=.5
    if not accepted: break
    psi_t=at*s; r=np.asarray((H@psi_t)/np.where(np.abs(psi_t)>1e-300,psi_t,1e-300),float)
    q=np.power(np.maximum(at,1e-300),ALPHA); q/=q.sum()
    t=wkmeans(r,q)
    sn=canonical((s*np.where(r<=t,1,-1)).astype(np.int8))
    Es=physical_energy(H,at,sn)
    if Es>Et: sn=s.copy(); Es=Et
    a,s=at,sn
    hist[-1]["trust"]=trust; hist[-1]["T"]=t; hist[-1]["Eamp"]=Et; hist[-1]["Esign"]=Es

out={"base_trust":BASE_TRUST,"alpha":ALPHA,"E0":E0,"history":hist,"final":hist[-1]}
Path("/Users/aliotaifi/Chatty/projects/j1j2/results/fixedsign_energy_trust_kmeans_k1_exact4x4.json").write_text(json.dumps(out,indent=2))
