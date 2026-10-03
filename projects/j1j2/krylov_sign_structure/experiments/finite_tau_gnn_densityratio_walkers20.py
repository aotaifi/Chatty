#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import scipy.sparse.linalg as sla
import jax,jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from flax import serialization
from finite_tau_matching_20site_exact import build_H,basis,D,N,special_columns
from finite_tau_gnn_densityratio_oracle20 import PairGNN,bits,make_wd,globals_from_wd,eval_full,DRState,step

ROOT=Path("/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure")
rng=np.random.default_rng(20260931)
dat=np.load(ROOT/'results/finite_tau_gfmc_uniform_block20.npz')
ys=dat['ys'].astype(int).tolist(); train_y=dat['train_y'].astype(int).tolist()
rep1=dat['rep1'].astype(np.int32);rep2=dat['rep2'].astype(np.int32); tau=float(dat['beta'])
H,_=build_H(.5);wd=make_wd(H,ys)

def dataset(pos,seed):
    rg=np.random.default_rng(seed);XX=[];YY=[];GG=[];LL=[];CC=[]
    M=pos.shape[1]
    for k,y in enumerate(train_y):
        ip=pos[k];iq=rg.integers(0,D,size=M,dtype=np.int32);idx=np.concatenate([ip,iq])
        XX.append(bits(basis[idx]));yb=bits(np.array([basis[y]]))[0];YY.append(np.repeat(yb[None,:],2*M,0))
        GG.append(globals_from_wd(wd[k],tau,idx))
        LL.append(np.concatenate([np.ones(M,np.float32),np.zeros(M,np.float32)]));CC.append(np.full(2*M,k,np.int32))
    return tuple(np.concatenate(z) for z in (XX,YY,GG,LL,CC))
tr=dataset(rep1,9901);va=dataset(rep2,9902)
print("TRAIN",len(tr[0]),"VAL",len(va[0]),flush=True)
model=PairGNN();params=model.init(jax.random.PRNGKey(21),jnp.asarray(tr[0][:4]),jnp.asarray(tr[1][:4]),jnp.asarray(tr[2][:4]))['params']
st=DRState.create(apply_fn=model.apply,params=params,tx=optax.adamw(1e-3,weight_decay=1e-6),bias=jnp.zeros(len(train_y)))
bs=4096;hist=[];bestv=1e9;best=None
for ep in range(10):
    order=rng.permutation(len(tr[0]));ls=[]
    for i in range(0,len(order),bs):
        ix=order[i:i+bs];st,l=step(st,*[jnp.asarray(z[ix]) for z in tr]);ls.append(float(l))
    vl=[]
    for i in range(0,len(va[0]),bs):
        F=np.asarray(model.apply({'params':st.params},jnp.asarray(va[0][i:i+bs]),jnp.asarray(va[1][i:i+bs]),jnp.asarray(va[2][i:i+bs])))
        z=F+np.asarray(st.bias)[va[4][i:i+bs]];lab=va[3][i:i+bs]
        vl.append(np.mean(np.logaddexp(0,z)-lab*z))
    v=float(np.mean(vl));hist.append([ep,float(np.mean(ls)),v]);print("EPOCH",hist[-1],flush=True)
    if v<bestv:bestv=v;best=(jax.tree_util.tree_map(lambda x:np.asarray(x),st.params),np.asarray(st.bias),ep)
params=best[0]
(ROOT/'results/finite_tau_gnn_densityratio_walkers20.mpack').write_bytes(serialization.to_bytes(params))
# exact transient only for scoring after walker-only learning
E=np.zeros((D,len(ys)));E[ys,np.arange(len(ys))]=1
G=np.asarray(sla.expm_multiply(-tau*H,E),float);G/=np.linalg.norm(G,axis=0,keepdims=True)
rows=[]
for k,y in enumerate(ys):
    fid=eval_full(model,params,y,wd[k],tau,G[:,k])
    row={'split':'train' if k<4 else 'test','y':int(y),'fidelity':fid}
    rows.append(row);print("RESULT",row,flush=True)
out={'tau':tau,'M':int(dat['M']),'best_epoch':int(best[2]),'best_val_bce':float(bestv),'history':hist,'rows':rows}
path=ROOT/'results/finite_tau_gnn_densityratio_walkers20.json';path.write_text(json.dumps(out,indent=2))
print("WROTE",path,flush=True)
