#!/usr/bin/env python3
import numpy as np
from scipy.special import logsumexp
import scipy.sparse.linalg as sla
import jax,jax.numpy as jnp
from flax import serialization
from finite_tau_matching_20site_exact import build_H,basis,D,N,special_columns
from finite_tau_gnn_densityratio_oracle20 import PairGNN,bits,make_wd,globals_from_wd
ROOT="/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure";tau=.2
rng=np.random.default_rng(20260930);H,_=build_H(.5);ys=special_columns(rng,6);wd=make_wd(H,ys)
model=PairGNN();dum=model.init(jax.random.PRNGKey(0),jnp.zeros((1,N)),jnp.zeros((1,N)),jnp.zeros((1,3)))['params']
pa=serialization.from_bytes(dum,open(ROOT+'/results/finite_tau_gnn_second_block20.mpack','rb').read())
apply=jax.jit(lambda p,x,y,g:model.apply({'params':p},x,y,g))
E=np.zeros((D,6));E[ys,np.arange(6)]=1
G=np.asarray(sla.expm_multiply(-tau*H,E),float);G/=np.linalg.norm(G,axis=0,keepdims=True)
lams=(.25,.5,.75,1.0,1.5);epss=(1e-5,1e-4,1e-3,1e-2,.05,.1)
for k,y in enumerate(ys):
 yb=bits(np.array([basis[y]]))[0];F=[]
 for i in range(0,D,4096):
  j=min(D,i+4096);idx=np.arange(i,j);yy=np.repeat(yb[None,:],j-i,0)
  F.append(np.asarray(apply(pa,jnp.asarray(bits(basis[idx])),jnp.asarray(yy),jnp.asarray(globals_from_wd(wd[k],tau,idx)))))
 F=np.concatenate(F);d=(wd[k]//1000).astype(float)
 p=np.abs(G[:,k])**2;p/=p.sum();lp=np.log(np.maximum(p,1e-300));lg=2*F
 base_ess=float(np.exp(-logsumexp(2*lp-(lg-logsumexp(lg)))))
 print('COL',k,y,'BASE',base_ess)
 for lam in lams:
  # broad distance-only amplitude, anchored to learned amplitude at y
  B=F[y]-lam*d;lb=2*B
  best=None
  for e in epss:
   lraw=np.logaddexp(np.log1p(-e)+lg,np.log(e)+lb);lq=lraw-logsumexp(lraw);q=np.exp(lq)
   ess=float(np.exp(-logsumexp(2*lp-lq)));fid=float(np.dot(np.sqrt(q),np.sqrt(p))**2)
   if best is None or ess>best[0]:best=(ess,fid,e)
  print(' lam',lam,'best_ess',best[0],'fid',best[1],'eps',best[2])
