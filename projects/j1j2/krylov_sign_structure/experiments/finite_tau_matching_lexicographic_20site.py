#!/usr/bin/env python3
import json
import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as cs
from scipy.optimize import linear_sum_assignment
from finite_tau_matching_20site_exact import build_H,basis,D,N,NN,NNN,special_columns

H,diag=build_H(.5)
HO=(H-sp.diags(diag)).tocoo()
edgecost=np.where(np.abs(HO.data)>0.375,1000.0,1001.0)
CW=sp.coo_matrix((edgecost,(HO.row,HO.col)),shape=H.shape).tocsr()

rr=[];cc=[];vv=[]
for u,v in NN:
    rr += [u,v]; cc += [v,u]; vv += [1000.,1000.]
for u,v in NNN:
    rr += [u,v]; cc += [v,u]; vv += [1001.,1001.]
SW=sp.coo_matrix((vv,(rr,cc)),shape=(N,N)).tocsr()
SD=cs.shortest_path(SW,directed=False)

rng=np.random.default_rng(20260930)
ys=special_columns(rng,4)
CD=np.asarray(cs.shortest_path(CW,directed=False,indices=ys))
ok=[]; examples=[]
for k,y in enumerate(ys):
    xs=list(rng.choice(np.arange(D),size=500,replace=False))
    xs += list(np.argsort(CD[k])[:500])
    by=int(basis[y])
    for x in xs:
        bx=int(basis[x])
        src=[i for i in range(N) if ((by>>i)&1) and not ((bx>>i)&1)]
        dst=[i for i in range(N) if ((bx>>i)&1) and not ((by>>i)&1)]
        if src:
            C=SD[np.ix_(src,dst)]
            a,b=linear_sum_assignment(C)
            mc=int(round(float(C[a,b].sum())))
        else: mc=0
        ccfg=int(round(float(CD[k,x])))
        good=mc==ccfg; ok.append(good)
        if not good and len(examples)<10: examples.append([int(x),int(y),ccfg,mc])

out={"n_pairs":len(ok),"lexicographic_exact_fraction":float(np.mean(ok)),
     "mismatch_examples":examples}
path="/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure/results/finite_tau_matching_lexicographic_20site.json"
open(path,"w").write(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
