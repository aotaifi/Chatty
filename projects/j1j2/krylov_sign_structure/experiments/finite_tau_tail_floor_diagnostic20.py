#!/usr/bin/env python3
import math,numpy as np
from scipy.special import logsumexp,gammaln
import scipy.sparse.linalg as sla
import jax,jax.numpy as jnp
from flax import serialization
from finite_tau_matching_20site_exact import build_H,basis,D,N,special_columns
from finite_tau_gnn_densityratio_oracle20 import PairGNN,bits,make_wd,globals_from_wd
ROOT="/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure";tau=.2
rng=np.random.default_rng(20260930);H,_=build_H(.5);ys=special_columns(rng,6);wd=make_wd(H,ys)
model=PairGNN();dum=model.init(jax.random.PRNGKey(0),jnp.zeros((1,N)),jnp.zeros((1,N)),jnp.zeros((1,3)))['params']
p=serialization.from_bytes(dum,open(ROOT+'/results/finite_tau_gnn_second_block20.mpack','rb').read())
apply=jax.jit(lambda pa,x,y,g:model.apply({'params':pa},x,y,g))
E=np.zeros((D,6));E[ys,np.arange(6)]=1
G=np.asarray(sla.expm_multiply(-tau*H,E),float);G/=np.linalg.norm(G,axis=0,keepdims=True)
epss=[0,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,.1]
for k,y in enumerate(ys):
 yb=bits(np.array([basis[y]]))[0];F=[]
 for i in range(0,D,4096):
  j=min(D,i+4096);idx=np.arange(i,j);yy=np.repeat(yb[None,:],j-i,0)
  F.append(np.asarray(apply(p,jnp.asarray(bits(basis[idx])),jnp.asarray(yy),jnp.asarray(globals_from_wd(wd[k],tau,idx)))))
 F=np.concatenate(F); d=(wd[k]//1000).astype(float);n2=np.rint(wd[k]-1000*d)
 # shortest-path leading-order amplitude, anchored to the learned amplitude at x=y
 B=F[y] + d*np.log(tau/2.0)-gammaln(d+1)-n2*np.log(2.0)
 lg=2*F; lb=2*B
 target=np.abs(G[:,k]);target/=np.linalg.norm(target);lp=np.log(np.maximum(target*target,1e-300))
 print('COL',k,y,'split','train' if k<4 else 'test')
 for e in epss:
  if e==0:lraw=lg
  else:lraw=np.logaddexp(np.log1p(-e)+lg,np.log(e)+lb)
  lq=lraw-logsumexp(lraw);q=np.exp(lq)
  ess=float(np.exp(-logsumexp(2*lp-lq)))
  amp=np.sqrt(q);fid=float(np.dot(amp,target)**2)
  pp=target*target; hole=float(np.sum(pp[q<.01*pp]))
  print('eps',e,'fid',fid,'ess',ess,'hole01',hole)
