import json,time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

LX,LY=6,3
N=LX*LY
OUT=Path("/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure/results")
site=lambda x,y:(x%LX)+LX*(y%LY)
BONDS=[]
for y in range(LY):
    for x in range(LX):
        i=site(x,y)
        BONDS += [(i,site(x+1,y)),(i,site(x,y+1)),(i,site(x+1,y+1))]
basis=np.array([s for s in range(1<<N) if s.bit_count()==N//2],np.uint32)
D=len(basis);idx={int(s):i for i,s in enumerate(basis)}
print("BASIS",D,"BONDS",len(BONDS),flush=True)

rr=[];cc=[];vv=[];diag=np.zeros(D)
for bi,s0 in enumerate(basis):
    s=int(s0);e=0.
    for u,v in BONDS:
        if ((s>>u)&1)==((s>>v)&1):e+=.25
        else:
            e-=.25
            t=s^(1<<u)^(1<<v)
            rr.append(bi);cc.append(idx[t]);vv.append(.5)
    diag[bi]=e
rr.extend(range(D));cc.extend(range(D));vv.extend(diag)
H=sp.coo_matrix((vv,(rr,cc)),shape=(D,D)).tocsr()
print("H nnz",H.nnz,flush=True)

t0=time.time()
ev,V=sla.eigsh(H,k=3,which="SA",tol=2e-11,maxiter=80000)
o=np.argsort(ev);ev=ev[o];V=V[:,o]
vec=np.asarray(V[:,0],float)
if vec[np.argmax(np.abs(vec))]<0:vec=-vec
E0=float(ev[0]);gap=float(ev[1]-ev[0]);gap2=float(ev[2]-ev[0])
a=np.abs(vec);p=a*a;p/=p.sum();truth=np.where(vec>=0,1,-1).astype(np.int8)
print("ED",ev.tolist(),"sec",time.time()-t0,flush=True)

def gauge(mask):
    return np.array([1 if ((int(s)&mask).bit_count()%2)==0 else -1 for s in basis],np.int8)

masks={
 "maxcut_x":sum(1<<site(x,y) for y in range(LY) for x in range(LX) if x%2==0),
 "parity_y":sum(1<<site(x,y) for y in range(LY) for x in range(LX) if y%2==0),
 "parity_xy":sum(1<<site(x,y) for y in range(LY) for x in range(LX) if (x+y)%2==0),
}
BASE={k:gauge(m) for k,m in masks.items()}

def energy_opt(s0,r):
    W=H.copy().tolil();W.setdiag(0);W=W.tocsr()
    W=sp.diags(a)@W@sp.diags(a)
    dE=float(np.sum(diag*a*a))
    s=(-s0).astype(np.int8)
    E=dE+float(s@(W@s));h=np.asarray(W@s).ravel()
    order=np.argsort(r)
    best=(float(E),float(r[order[0]]-1),s.copy())
    for ii in order:
        old=int(s[ii]);E += -4*old*h[ii];s[ii]=-old
        st,en=W.indptr[ii],W.indptr[ii+1]
        js=W.indices[st:en];ws=W.data[st:en]
        h[js] += -2*old*ws
        if E<best[0]-1e-13:best=(float(E),float(r[ii]),s.copy())
    return best

def oracle(s0,r):
    order=np.argsort(r);s=(-s0).astype(np.int8)
    z=float(np.sum(p*s*truth));best=(abs(z),float(r[order[0]]-1),s.copy())
    for ii in order:
        old=int(s[ii]);s[ii]=-old
        z += p[ii]*(-2*old)*truth[ii]
        if abs(z)>best[0]+1e-15:best=(abs(z),float(r[ii]),s.copy())
    return best

def score(name,s0):
    psi0=a*s0;r=(H@psi0)/np.where(np.abs(psi0)>1e-300,psi0,1e-300)
    E1,t,s1=energy_opt(s0,r)
    Oo,to,so=oracle(s0,r)
    Ebase=float(psi0@(H@psi0))
    O0=float(abs(np.sum(p*s0*truth)));O1=float(abs(np.sum(p*s1*truth)))
    h=(H@(a*s1))-diag*(a*s1)
    P=float(np.sum(p*((s1*h)<=1e-12)))
    changed=float(np.sum(p*(s1!=s0)))
    Eoracle=float((a*so)@(H@(a*so)))
    return dict(baseline=name,O_S_0=O0,O_S_1=O1,oracle_O_S_1=float(Oo),
                E0=E0,fixed_amp_energy_error0_per_site=(Ebase-E0)/N,
                fixed_amp_energy_error1_per_site=(E1-E0)/N,
                oracle_fixed_amp_error1_per_site=(Eoracle-E0)/N,
                P_stable_1=P,changed_weight=changed,
                threshold_energyopt=float(t),threshold_oracle=float(to))

rows=[]
for name,s0 in BASE.items():
    q=score(name,s0);rows.append(q);print("TRI",json.dumps(q),flush=True)

best0=min(rows,key=lambda x:x["fixed_amp_energy_error0_per_site"])
best1=min(rows,key=lambda x:x["fixed_amp_energy_error1_per_site"])
result=dict(method={
 "model":"nearest-neighbor spin-1/2 triangular-lattice Heisenberg AF",
 "cluster":"6x3 periodic triangular Bravais torus, Sz=0 exact diagonalization",
 "baselines":"three simple two-color gauges; maxcut_x cuts two of the three local bond directions",
 "threshold":"fixed-amplitude energy-minimizing t; no hidden signs",
 "oracle":"best sign-overlap threshold only as capacity diagnostic"
},spectrum=ev.tolist(),gap=gap,gap2=gap2,results=rows,
best_baseline_before=best0["baseline"],best_baseline_after=best1["baseline"])
with open(OUT/"triangular_exact_test.json","w") as f:json.dump(result,f,indent=2)
print("BEST",best0["baseline"],"->",best1["baseline"],flush=True)
