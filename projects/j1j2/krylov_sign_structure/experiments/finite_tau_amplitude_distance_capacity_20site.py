#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as cs
import scipy.sparse.linalg as sla
from finite_tau_matching_20site_exact import build_H,basis,D,special_columns,NN,NNN

ROOT=Path("/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure")
H,diag=build_H(.5)
rng=np.random.default_rng(20260930)
ys=special_columns(rng,2)

co=(H-sp.diags(diag)).tocoo()
mask=co.row!=co.col
r=co.row[mask]; c=co.col[mask]; v=np.abs(co.data[mask])
# Lexicographic shortest path: event count first, then number of J2 events.
cost=np.where(v>0.375,1000.0,1001.0)
W=sp.coo_matrix((cost,(r,c)),shape=H.shape).tocsr()
wd=np.asarray(cs.shortest_path(W,directed=False,indices=ys))
d=(wd//1000).astype(np.int16)
n2=(wd-1000*d).round().astype(np.int16)

E=np.zeros((D,len(ys))); E[ys,np.arange(len(ys))]=1.0
f1=np.zeros(D,dtype=np.int16); f2=np.zeros(D,dtype=np.int16)
for u,v in NN: f1 += (((basis>>u)^(basis>>v))&1).astype(np.int16)
for u,v in NNN: f2 += (((basis>>u)^(basis>>v))&1).astype(np.int16)

def proj_fidelity(a,labels):
    vals,inv=np.unique(labels,return_inverse=True)
    cnt=np.bincount(inv)
    sm=np.bincount(inv,weights=a)
    proj=sm[inv]/cnt[inv]
    return float(np.dot(proj,proj)),len(vals)

rows=[]
for tau in (0.05,0.1,0.25,0.5):
    G=np.asarray(sla.expm_multiply(-tau*H,E),float)
    for k,y in enumerate(ys):
        a=np.abs(G[:,k]); a/=np.linalg.norm(a)
        fd,nd=proj_fidelity(a,d[k])
        pair=d[k].astype(np.int32)*100+n2[k].astype(np.int32)
        fpair,npair=proj_fidelity(a,pair)
        ecode=np.rint(8.0*diag).astype(np.int32)+100
        triple=pair.astype(np.int64)*1000+ecode.astype(np.int64)
        ftri,ntri=proj_fidelity(a,triple)
        rich=triple.astype(np.int64)*10000+f1.astype(np.int64)*100+f2.astype(np.int64)
        frich,nrich=proj_fidelity(a,rich)
        row={"tau":tau,"y":int(y),"distance_fidelity":fd,"distance_bins":nd,
             "distance_n2_fidelity":fpair,"distance_n2_bins":npair,
             "distance_n2_diag_fidelity":ftri,"distance_n2_diag_bins":ntri,
             "rich_fidelity":frich,"rich_bins":nrich}
        rows.append(row)
        print(row,flush=True)

out={"D":D,"ys":[int(x) for x in ys],"rows":rows}
path=ROOT/"results/finite_tau_amplitude_distance_capacity_20site.json"
path.write_text(json.dumps(out,indent=2))
print("WROTE",path)
