import json, csv, hashlib
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

L=4; N=16
CASES=[(0.5,"marshall"),(0.6,"marshall"),(1.0,"stripe_x")]
QGRID=[0.35,0.38,0.40,0.42,0.44,0.45,0.46,0.48,0.49]
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

rows=[]; summaries=[]
master=np.random.default_rng(SEED)
for J2,base in CASES:
    print("CASE",J2,base,flush=True)
    H,diag=build_H(J2)
    ev,V=sla.eigsh(H,k=2,which="SA",tol=1e-12,maxiter=100000)
    j=int(np.argmin(ev)); E0=float(ev[j]); vec=np.asarray(V[:,j],float)
    anchor=int(np.argmax(np.abs(vec)))
    if vec[anchor]<0: vec=-vec
    a=np.abs(vec); p=a*a; p/=p.sum()
    truth=np.where(vec>=0,1,-1).astype(np.int8)
    ei,ej,wij=prep_edges(H,a,diag)
    for q in QGRID:
        suc=[]; its=[]; finals=[]; actuals=[]; oracle1=[]
        for tr in range(TRIALS):
            rng=np.random.default_rng(int(master.integers(0,2**63-1)))
            s,qact=corrupt(truth,p,anchor,q,rng)
            actuals.append(qact)
            initialO=overlap(p,s,truth)
            seen=set(); success=False; nit=0; first_or=None
            for it in range(MAXITER+1):
                O=overlap(p,s,truth)
                if O>1-1e-12:
                    success=True; nit=it; break
                key=np.packbits((s>0).astype(np.uint8)).tobytes()
                if key in seen:
                    nit=it; break
                seen.add(key)
                sn,Ebest,oracle,G=update(H,a,p,truth,s,anchor,ei,ej,wij)
                if first_or is None:first_or=oracle
                if np.array_equal(sn,s):
                    nit=it; break
                s=sn; nit=it+1
            finalO=overlap(p,s,truth)
            suc.append(success);its.append(nit);finals.append(finalO);oracle1.append(first_or if first_or is not None else 1.0)
            rows.append(dict(J2=J2,baseline=base,q_target=q,q_actual=qact,trial=tr,
                             O_initial=initialO,success_exact=success,iterations=nit,O_final=finalO,
                             oracle_O_after_first=oracle1[-1]))
        summary=dict(J2=J2,baseline=base,q_target=q,q_actual_mean=float(np.mean(actuals)),
                     O_initial_mean=float(np.mean([1-2*x for x in actuals])),
                     success_fraction=float(np.mean(suc)),
                     median_iterations=float(np.median([v for v,ok in zip(its,suc) if ok])) if any(suc) else None,
                     O_final_mean=float(np.mean(finals)),
                     O_final_min=float(np.min(finals)),
                     oracle_first_mean=float(np.mean(oracle1)))
        summaries.append(summary)
        print("Q",q,"success",summary["success_fraction"],"O0",summary["O_initial_mean"],
              "Ofinal",summary["O_final_mean"],"medit",summary["median_iterations"],flush=True)

with open(OUT/"basin_radius_refined.json","w") as f:
    json.dump(dict(method={
      "system":"4x4 periodic square J1-J2, Sz=0 exact diagonalization",
      "amplitudes":"exact ground-state |psi| fixed",
      "initialization":"exact signs with random configurations flipped until target physical probability weight q; anchor excluded to fix global gauge",
      "update":"iterated r_k=(H a s_k)/(a s_k), label-free threshold minimizing exact fixed-amplitude energy",
      "success":"exact ground-state signs up to fixed global gauge within O_S>1-1e-12",
      "trials_per_q":TRIALS,"max_iterations":MAXITER,"seed":SEED
    },summaries=summaries,rows=rows),f,indent=2)
fields=list(rows[0].keys())
with open(OUT/"basin_radius_refined.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
print("DONE",flush=True)
