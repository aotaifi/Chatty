#!/usr/bin/env python3
import numpy as np, scipy.sparse as sp, scipy.sparse.csgraph as cs, scipy.sparse.linalg as sla
import jax,jax.numpy as jnp
from flax import serialization
from finite_tau_matching_20site_exact import build_H,basis,D,N,special_columns
from finite_tau_gnn_densityratio_oracle20 import PairGNN,bits,make_wd,globals_from_wd
ROOT="/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure"
rng=np.random.default_rng(20260930); H,_=build_H(.5); ys=special_columns(rng,6); wd=make_wd(H,ys); tau=.1
E=np.zeros((D,len(ys)));E[ys,np.arange(len(ys))]=1
G=np.asarray(sla.expm_multiply(-tau*H,E),float);G/=np.linalg.norm(G,axis=0,keepdims=True)
model=PairGNN(); dummy=model.init(jax.random.PRNGKey(0),jnp.zeros((1,N)),jnp.zeros((1,N)),jnp.zeros((1,3)))['params']
params=serialization.from_bytes(dummy,open(ROOT+"/results/finite_tau_gnn_densityratio_oracle20.mpack","rb").read())
apply=jax.jit(lambda p,x,y,g:model.apply({'params':p},x,y,g))
Fs=[]
for k,y in enumerate(ys):
 yb=bits(np.array([basis[y]]))[0]; arr=[]
 for i in range(0,D,4096):
  j=min(D,i+4096);idx=np.arange(i,j);yy=np.repeat(yb[None,:],j-i,0)
  arr.append(np.asarray(apply(params,jnp.asarray(bits(basis[idx])),jnp.asarray(yy),jnp.asarray(globals_from_wd(wd[k],tau,idx)))))
 Fs.append(np.concatenate(arr))
alphas=np.linspace(.5,1.5,101)
def fid(F,t,a):
 z=np.exp(np.clip(a*F-np.max(a*F),-80,0));z/=np.linalg.norm(z)
 q=np.abs(t);q/=np.linalg.norm(q);return float(np.dot(z,q)**2)
trainmean=np.array([np.mean([fid(Fs[k],G[:,k],a) for k in range(4)]) for a in alphas])
a=alphas[np.argmax(trainmean)]
print("TRAIN_OPT_ALPHA",a,"MEAN",trainmean.max())
for k,y in enumerate(ys):
 vals=np.array([fid(Fs[k],G[:,k],z) for z in alphas])
 print("COL",k,y,"fid_at_trainalpha",fid(Fs[k],G[:,k],a),"oracle_alpha",alphas[np.argmax(vals)],"oracle_fid",vals.max())
