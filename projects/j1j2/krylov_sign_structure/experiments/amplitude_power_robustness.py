import json, csv, hashlib
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

L=4; N=16
CASES=[(0.5,"marshall"),(0.6,"marshall"),(1.0,"stripe_x")]
QGRID=[0.40,0.45,0.49]
TRIALS=32
MAXITER=40
SEED=20260928
OUT=Path("/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure/results")
site=lambda x,y:(x%L)+L*(y%L)
NN=[]; NNN=[]
for y in range(L):
    for x in range(L):
        i=site(x,y)
        NN += [(i,site(x+1,y)),(i,site(x,y+1))]
        NNN += [(i,site(x+1,y+1)),(i,site(x+1,y-1))]
basis=np.array([s for s in range(1<<N) if s.bit_count()==N//2],np.uint32)
D=len(basis); pos={int(s):i for i,s in enumerate(basis)}

def build_H(J2):
    rr=[];cc=[];vv=[];diag=np.zeros(D)
    for bi,s0 in enumerate(basis):
        s=int(s0); e=0.
        for bonds,J in ((NN,1.0),(NNN,J2)):
            if J==0: continue
            for u,v in bonds:
                if ((s>>u)&1)==((s>>v)&1): e+=0.25*J
                else:
                    e-=0.25*J
                    t=s^(1<<u)^(1<<v)
                    rr.append(bi);cc.append(pos[t]);vv.append(0.5*J)
        diag[bi]=e
    rr.extend(range(D));cc.extend(range(D));vv.extend(diag.tolist())
    return sp.coo_matrix((vv,(rr,cc)),shape=(D,D)).tocsr(),diag

def canonical(s,anchor):
    s=np.asarray(s,np.int8).copy()
    if s[anchor]<0:s=-s
    return s

def overlap(p,s,t):
    return float(abs(np.sum(p*s*t)))

def group_ids(r,atol=1e-11,rtol=1e-11):
    order=np.argsort(r,kind="mergesort")
    rs=r[order]
    scale=np.maximum(np.maximum(np.abs(rs[1:]),np.abs(rs[:-1])),1.0)
    br=np.empty(D,dtype=bool); br[0]=True
    br[1:]=np.abs(rs[1:]-rs[:-1]) > (atol+rtol*scale)
    gs=np.cumsum(br,dtype=np.int32)-1
    gid=np.empty(D,dtype=np.int32); gid[order]=gs
    return order,gid,int(gs[-1])+1

def prep_edges(H,a,diag):
    co=sp.triu(H-sp.diags(diag),k=1).tocoo()
    return co.row.astype(np.int32),co.col.astype(np.int32),(co.data*a[co.row]*a[co.col]).astype(float)

def update(H,a,p,truth,s,anchor,ei,ej,wij):
    psi=a*s
    r=(H@psi)/np.where(a>1e-300,psi,1.0)
    order,gid,G=group_ids(r)

    # candidate below min(r): c=-s. Edge term in c^T W c is 2*w_ij*c_i*c_j.
    c0=-s
    val=2.0*wij*c0[ei]*c0[ej]
    e0=float(np.sum((a*a)*H.diagonal()) + np.sum(val))

    gi=gid[ei]; gj=gid[ej]
    lo=np.minimum(gi,gj); hi=np.maximum(gi,gj)
    mask=lo<hi
    # Exact vectorized energy sweep over all thresholds.
    delta=(np.bincount(lo[mask],weights=-2.0*val[mask],minlength=G)
           +np.bincount(hi[mask],weights=+2.0*val[mask],minlength=G))
    energies=e0+np.cumsum(delta)
    k=int(np.argmin(np.r_[e0,energies]))-1

    # oracle overlap on exactly the same threshold family, diagnostic only.
    raw0=float(np.sum(p*c0*truth))
    contrib=np.bincount(gid,weights=(2.0*p*s*truth),minlength=G)
    raw=np.r_[raw0,raw0+np.cumsum(contrib)]
    oracle=float(np.max(np.abs(raw)))

    if k<0:
        sn=c0.copy()
    else:
        sn=c0.copy()
        sn[gid<=k]*=-1
    sn=canonical(sn,anchor)
    return sn,float(np.min(np.r_[e0,energies])),oracle,G

def corrupt(truth,p,anchor,q,rng):
    # Random-order weighted subset, stopping at the cumulative weight nearest q.
    inds=np.array([i for i in range(D) if i!=anchor],dtype=np.int32)
    rng.shuffle(inds)
    cs=np.cumsum(p[inds])
    j=int(np.searchsorted(cs,q))
    opts=[]
    if j<len(inds): opts.append(j+1)
    if j>0: opts.append(j)
    if not opts: opts=[1]
    n=min(opts,key=lambda n:abs(float(cs[n-1])-q))
    flip=inds[:n]
    s=truth.copy();s[flip]*=-1
    return s,float(np.sum(p[flip]))


GAMMAS=[0.0,0.25,0.5,0.75,0.9,1.0,1.1,1.25]
MAXITER=30

def baseline_sign(kind):
    if kind=="marshall":
        m=sum(1<<site(x,y) for y in range(L) for x in range(L) if (x+y)%2==0)
    elif kind=="stripe_x":
        m=sum(1<<site(x,y) for y in range(L) for x in range(L) if x%2==0)
    else: raise ValueError(kind)
    return np.array([1 if ((int(st)&m).bit_count()%2)==0 else -1 for st in basis],np.int8)

rows=[]
for J2,base in CASES:
    print("CASE",J2,base,flush=True)
    H,diag=build_H(J2)
    ev,V=sla.eigsh(H,k=2,which="SA",tol=1e-12,maxiter=100000)
    j=int(np.argmin(ev)); E0=float(ev[j]); vec=np.asarray(V[:,j],float)
    anchor=int(np.argmax(np.abs(vec)))
    if vec[anchor]<0: vec=-vec
    atrue=np.abs(vec); ptrue=atrue*atrue; ptrue/=ptrue.sum()
    truth=np.where(vec>=0,1,-1).astype(np.int8)
    sbase=canonical(baseline_sign(base),anchor)
    for gamma in GAMMAS:
        # Power-tempered amplitude profile. gamma=1 exact, gamma=0 uniform over the Sz=0 basis.
        floor=max(float(np.max(atrue))*1e-14,1e-300)
        a=np.maximum(atrue,floor)**gamma
        a=a/np.linalg.norm(a)
        p=a*a; p/=p.sum()
        ei,ej,wij=prep_edges(H,a,diag)
        s=sbase.copy(); seen=set(); exact_hit=False
        traj=[]
        for it in range(MAXITER+1):
            O=overlap(ptrue,s,truth)
            traj.append(O)
            if O>1-1e-12:
                exact_hit=True
                break
            if it==MAXITER: break
            key=np.packbits((s>0).astype(np.uint8)).tobytes()
            if key in seen: break
            seen.add(key)
            sn,Ebest,oracle,G=update(H,a,p,truth,s,anchor,ei,ej,wij)
            if np.array_equal(sn,s): break
            s=sn
        psi_approx=a*s
        Eapprox=float(psi_approx@(H@psi_approx))
        row=dict(J2=J2,baseline=base,gamma=gamma,O_initial=float(traj[0]),
                 O_final=float(traj[-1]),max_O=float(max(traj)),iterations=len(traj)-1,
                 exact_signs_reached=exact_hit,approx_energy=float(Eapprox),
                 approx_energy_error_per_site=float((Eapprox-E0)/N),
                 trajectory=traj)
        rows.append(row)
        print("G",gamma,"O",row["O_initial"],"->",row["O_final"],"max",row["max_O"],
              "it",row["iterations"],"exact",exact_hit,"Eerr",row["approx_energy_error_per_site"],flush=True)

with open(OUT/"amplitude_power_robustness.json","w") as f:
    json.dump(dict(method={
      "system":"4x4 periodic square J1-J2, Sz=0 exact diagonalization",
      "approx_amplitudes":"a_gamma proportional to max(|psi_GS|,floor)^gamma; gamma=1 exact, gamma=0 uniform",
      "start":"Marshall for J2=0.5,0.6; stripe_x for J2=1.0",
      "update":"iterated local-energy/Krylov sign map, threshold minimizing fixed-a_gamma energy",
      "score":"physical sign overlap evaluated with exact ground-state probability weights",
      "max_iterations":MAXITER
    },rows=rows),f,indent=2)
flat=[]
for r in rows:
    q={k:v for k,v in r.items() if k!="trajectory"}; flat.append(q)
with open(OUT/"amplitude_power_robustness.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(flat[0].keys()));w.writeheader();w.writerows(flat)
print("DONE",flush=True)
