#!/usr/bin/env python3
import json
from functools import lru_cache
from pathlib import Path
import numpy as np
from finite_tau_ct_local_ratio_20site import H,normalized

z=np.load("../results/reconstruct_M100000_y59279.npz")
s0=z["s"].astype(float); logg=z["logg"].astype(float)
g=normalized(np.exp(np.clip(logg-logg.max(),-700,0)))
logg=np.log(np.maximum(g,1e-300))
p=g*g;p/=p.sum()
rg=np.random.default_rng(440001)
starts=[int(rg.choice(len(g),p=p)) for _ in range(3)]
dt=.0125
diag=H.diagonal()
HO=H.copy();HO.setdiag(0);HO.eliminate_zeros()

def run(x0,kmax=4):
    touched=[set() for _ in range(kmax+1)]
    @lru_cache(None)
    def st(k,x):
        touched[k].add(int(x))
        if k==0:
            return float(logg[x]),float(s0[x])
        lx,sx=st(k-1,x)
        lo,hi=HO.indptr[x],HO.indptr[x+1]
        js=HO.indices[lo:hi]; hs=HO.data[lo:hi]
        e1=float(diag[x])
        for y,h in zip(js,hs):
            ly,sy=st(k-1,int(y))
            e1+=float(h)*(sy/sx)*np.exp(np.clip(ly-lx,-700,700))
        r=1-dt*e1
        return lx+np.log(max(abs(r),1e-300)), sx*(1. if r>=0 else -1.)
    rows=[]
    for k in range(1,kmax+1):
        try:
            st(k,x0)
            rows.append({"k":k,"per_level":[len(touched[j]) for j in range(k+1)],
                         "total_cached":int(st.cache_info().currsize)})
        except RecursionError:
            rows.append({"k":k,"error":"RecursionError"})
            break
    return rows

out={"dt":dt,"starts":starts,"runs":[{"x":x,"rows":run(x)} for x in starts]}
Path("../results/diag_recursive_k1_cost20.json").write_text(json.dumps(out,indent=2))
print("RESULT",json.dumps(out,sort_keys=True),flush=True)
