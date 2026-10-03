#!/usr/bin/env python3
import json
import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as cs
from scipy.optimize import linear_sum_assignment
from closed_fn_krylov_exact4x4 import build_H, basis, site

SEED=1234; NCOLS=32; NSAMP=200
H,diag=build_H(0.5)
A=(H-sp.diags(diag)).copy(); A.data=np.ones_like(A.data)
rng=np.random.default_rng(SEED)
ys=np.array(rng.choice(np.arange(H.shape[0]),size=NCOLS,replace=False),dtype=int)
dist_cfg=np.asarray(cs.shortest_path(A,directed=False,unweighted=True,indices=ys)).T

# Physical-site graph with both J1 and J2 exchange edges.
S=16
adj=np.zeros((S,S),dtype=int)
for a in range(S):
    ax,ay=a%4,a//4
    for b in range(S):
        bx,by=b%4,b//4
        dx=min((ax-bx)%4,(bx-ax)%4); dy=min((ay-by)%4,(by-ay)%4)
        if (dx,dy) in ((1,0),(0,1),(1,1)): adj[a,b]=1
site_dist=cs.shortest_path(sp.csr_matrix(adj),directed=False,unweighted=True)
def matching_distance(sa,sb):
    a=int(basis[sa]); b=int(basis[sb])
    src=[i for i in range(S) if ((b>>i)&1) and not ((a>>i)&1)]
    dst=[i for i in range(S) if ((a>>i)&1) and not ((b>>i)&1)]
    if not src: return 0
    C=site_dist[np.ix_(src,dst)]
    r,c=linear_sum_assignment(C)
    return int(round(float(C[r,c].sum())))

eq=[]; parity=[]
examples=[]
for j,y in enumerate(ys):
    # Uniform samples plus states with smallest exact distance.
    xs=list(rng.choice(np.arange(H.shape[0]),size=NSAMP,replace=False))
    xs+=list(np.argsort(dist_cfg[:,j])[:NSAMP])
    for x in xs:
        dm=matching_distance(int(x),int(y)); dc=int(dist_cfg[x,j])
        eq.append(dm==dc); parity.append((dm-dc)%2==0)
        if dm!=dc and len(examples)<10:
            examples.append([int(x),int(y),dc,dm])
out={
 "n_pairs":len(eq),
 "distance_exact_fraction":float(np.mean(eq)),
 "parity_exact_fraction":float(np.mean(parity)),
 "mismatch_examples":examples
}
path="/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure/results/finite_tau_matching_distance_exact4x4.json"
open(path,"w").write(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
print("WROTE",path)
