#!/usr/bin/env python3
import json, numpy as np
import jax,jax.numpy as jnp
from flax import serialization
import scipy.sparse.linalg as sla

from finite_tau_matching_20site_exact import build_H,basis,D,N
from finite_tau_gnn_densityratio_oracle20 import PairGNN,bits,make_wd,globals_from_wd
from finite_tau_gnn_second_block20 import ys,train_y,wd

ROOT="/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure"
H,_=build_H(.5); HO=H.copy(); HO.setdiag(0); HO.eliminate_zeros()
model=PairGNN()
dummy=model.init(jax.random.PRNGKey(0),jnp.zeros((1,N)),jnp.zeros((1,N)),jnp.zeros((1,3)))["params"]
params=serialization.from_bytes(dummy,open(ROOT+"/results/finite_tau_gnn_second_block20.mpack","rb").read())
dat=np.load(ROOT+"/results/finite_tau_gnn_second_block20_samples.npz")
pos=dat["pos"].astype(np.int32); q=dat["q"].astype(np.int32)
apply=jax.jit(lambda x,y,g:model.apply({"params":params},x,y,g))

def eval_ids(y,k,ids,tau=.2):
    ids=np.asarray(ids,np.int32); yb=bits(np.array([basis[y]]))[0]
    out=[]
    for a in range(0,len(ids),4096):
        z=ids[a:a+4096]; yy=np.repeat(yb[None,:],len(z),0)
        out.append(np.asarray(apply(jnp.asarray(bits(basis[z])),jnp.asarray(yy),jnp.asarray(globals_from_wd(wd[k],tau,z)))))
    return np.concatenate(out)

# exact amplitudes at tau=.2
E=np.zeros((D,len(ys)));E[ys,np.arange(len(ys))]=1.
G=np.asarray(sla.expm_multiply(-.2*H,E),float);G/=np.linalg.norm(G,axis=0,keepdims=True)

rows=[]
for k,y in enumerate(ys):
    # use the saved training positive support only where available; for test endpoints,
    # classify seen relative to union of all training endpoint supports as a conservative proxy.
    if k<len(train_y):
        support=np.unique(np.concatenate([pos[k],q[k]]))
    else:
        support=np.unique(np.concatenate([pos.reshape(-1),q.reshape(-1)]))
    seenmask=np.zeros(D,dtype=bool); seenmask[support]=True
    # validation centers sampled from exact |psi|^2 so this diagnostic is independent of training draws
    a=np.abs(G[:,k]); p=a*a; p/=p.sum(); rg=np.random.default_rng(310000+k)
    xs=rg.choice(D,2000,p=p)
    src=[];dst=[];ww=[]
    for x in xs:
        lo,hi=HO.indptr[x],HO.indptr[x+1]; js=HO.indices[lo:hi]; hs=np.abs(HO.data[lo:hi])
        src.extend([int(x)]*len(js));dst.extend(js.tolist());ww.extend(hs.tolist())
    src=np.asarray(src,np.int32);dst=np.asarray(dst,np.int32);ww=np.asarray(ww,float)
    ids=np.unique(np.concatenate([src,dst]))
    predall=eval_ids(int(y),k,ids)
    mp={int(z):float(v) for z,v in zip(ids,predall)}
    pred=np.array([mp[int(v)]-mp[int(u)] for u,v in zip(src,dst)])
    targ=np.log(np.maximum(a[dst],1e-30))-np.log(np.maximum(a[src],1e-30))
    err=pred-targ; sy=seenmask[dst]
    def wrms(z,w):
        return float(np.sqrt(np.sum(w*z*z)/np.sum(w))) if len(z) else None
    row={"split":"train" if k<len(train_y) else "test","y":int(y),
         "seenY_frac":float(sy.mean()),"base_rmse":wrms(targ,ww),"proj_rmse":wrms(err,ww),
         "seen_rmse":wrms(err[sy],ww[sy]) if np.any(sy) else None,
         "unseen_rmse":wrms(err[~sy],ww[~sy]) if np.any(~sy) else None}
    rows.append(row);print(json.dumps(row,sort_keys=True),flush=True)
open(ROOT+"/results/diag_gnn_second_block_edges20.json","w").write(json.dumps(rows,indent=2))
