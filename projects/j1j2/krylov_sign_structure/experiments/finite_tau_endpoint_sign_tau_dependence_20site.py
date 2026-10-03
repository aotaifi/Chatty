#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as cs
import scipy.sparse.linalg as sla
from finite_tau_matching_20site_exact import build_H,D,special_columns

ROOT=Path("/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure")
H,diag=build_H(.5)
A=(H-sp.diags(diag)).copy().tocsr()
A.data=np.ones_like(A.data)

rng=np.random.default_rng(20260930)
ys=special_columns(rng,8)
dist=np.asarray(cs.shortest_path(A,directed=False,unweighted=True,indices=ys))
sgn_dist=np.where((dist.astype(np.int64)%2)==0,1.0,-1.0)

taus=np.array([0.005,0.01,0.015,0.02,0.025,0.03,0.04,0.05,0.075,0.1,
               0.15,0.25,0.5,0.75,1.0,1.5,2.0,3.0,4.0,6.0,8.0])
G=np.zeros((D,len(ys))); G[ys,np.arange(len(ys))]=1.0
rows=[]; first_fail=None; first_weighted=None; crossings={}; prevS=None; prevG=None; prev_tau=0.0
for tau in taus:
    dt=float(tau-prev_tau)
    G=np.asarray(sla.expm_multiply(-dt*H,G),float)
    # Positive column rescaling only; preserves all signs and normalized mismatch metrics.
    norms=np.linalg.norm(G,axis=0); G/=norms[None,:]
    S=G*sgn_dist.T
    neg_total=0; wm=[]
    for k,y in enumerate(ys):
        g=G[:,k]; s=S[:,k]; scale=float(np.max(np.abs(g)))
        robust=np.abs(g)>max(1e-15,1e-12*scale)
        neg=robust & (s<0)
        wmis=float(np.sum(g[neg]**2)/np.sum(g*g))
        neg_total+=int(neg.sum()); wm.append(wmis)
        row={"tau":float(tau),"y":int(y),"robust_pairs":int(robust.sum()),
             "negative_pairs":int(neg.sum()),"weighted_mismatch":wmis,
             "min_S_robust":float(np.min(s[robust])) if robust.any() else None,
             "min_signed_margin_rel":float(np.min(s[robust]/scale)) if robust.any() else None}
        rows.append(row)
        if neg.any() and first_fail is None:
            inds=np.flatnonzero(neg)
            # choose largest-magnitude violating endpoint, not an arbitrary tail state
            x=int(inds[np.argmax(np.abs(g[inds]))])
            first_fail={"tau":float(tau),"x":x,"y":int(y),"G":float(g[x]),
                        "d":int(dist[k,x]),"S":float(s[x]),"rel_amp":float(abs(g[x])/scale),
                        "column_weight":float(g[x]*g[x]/np.sum(g*g))}
        if wmis>1e-8 and first_weighted is None:
            first_weighted={"tau":float(tau),"y":int(y),"weighted_mismatch":wmis}
    if prevS is not None:
        for k,y in enumerate(ys):
            gp=prevG[:,k]; gc=G[:,k]; sp=prevS[:,k]; sc=S[:,k]
            thp=max(1e-15,1e-12*np.max(np.abs(gp))); thc=max(1e-15,1e-12*np.max(np.abs(gc)))
            m=(np.abs(gp)>thp)&(np.abs(gc)>thc)&(sp*sc<0)
            inds=np.flatnonzero(m)
            for x in inds[:50]:
                key=f"{int(y)}:{int(x)}"
                if key not in crossings:
                    crossings[key]={"y":int(y),"x":int(x),"tau_lo":float(prev_tau),"tau_hi":float(tau),
                                   "S_lo":float(sp[x]),"S_hi":float(sc[x])}
    print("TAU",float(tau),"neg_total",neg_total,"weighted_mean",float(np.mean(wm)),
          "weighted_max",float(np.max(wm)),flush=True)
    prevS=S.copy(); prevG=G.copy(); prev_tau=float(tau)
print("EIGSH",flush=True)
evals,evecs=sla.eigsh(H,k=2,which="SA",tol=1e-9,maxiter=10000)
o=np.argsort(evals); evals=evals[o]; psi=evecs[:,o[0]]
gs=[]; ampmax=float(np.max(np.abs(psi)))
for k,y in enumerate(ys):
    robust=np.abs(psi)>max(1e-15,1e-12*ampmax)
    sy=1.0 if psi[int(y)]>=0 else -1.0
    gs_sign=np.where(psi*sy>=0,1.0,-1.0)
    bad=robust & (gs_sign!=sgn_dist[k])
    w=psi*psi; w/=w.sum()
    gs.append({"y":int(y),"robust_pairs":int(robust.sum()),"mismatch_pairs":int(bad.sum()),
               "weighted_mismatch":float(w[bad].sum()),"psi_y":float(psi[int(y)])})
    print("GS",gs[-1],flush=True)

# Compare the last propagated columns directly with ground-state sign products.
last=[]
for k,y in enumerate(ys):
    g=G[:,k]; robust=np.abs(g)>max(1e-15,1e-12*np.max(np.abs(g)))
    sy=1.0 if psi[int(y)]>=0 else -1.0
    gs_sign=np.where(psi*sy>=0,1.0,-1.0)
    prop_sign=np.where(g>=0,1.0,-1.0)
    bad=robust & (prop_sign!=gs_sign)
    last.append({"y":int(y),"tau":float(taus[-1]),"mismatch_pairs_vs_gs":int(bad.sum()),
                 "weighted_mismatch_vs_gs":float(np.sum(g[bad]**2)/np.sum(g*g))})

out={"D":D,"ys":[int(y) for y in ys],"taus":taus.tolist(),"pairs_per_tau":int(D*len(ys)),
     "rows":rows,"first_fail":first_fail,"first_weighted":first_weighted,
     "crossings":list(crossings.values())[:500],"eigenvalues":[float(x) for x in evals],
     "ground_state":gs,"tau_max_vs_ground_state":last}
path=ROOT/"results/finite_tau_endpoint_sign_tau_dependence_20site.json"
path.write_text(json.dumps(out,indent=2))
print("FIRST_FAIL",first_fail,flush=True)
print("FIRST_WEIGHTED",first_weighted,flush=True)
print("NCROSS",len(crossings),flush=True)
print("EVALS",evals.tolist(),flush=True)
print("WROTE",path,flush=True)
