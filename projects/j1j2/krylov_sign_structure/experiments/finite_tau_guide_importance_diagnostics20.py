#!/usr/bin/env python3
import numpy as np
from scipy.special import logsumexp
import scipy.sparse.linalg as sla
import jax,jax.numpy as jnp
from flax import serialization
from finite_tau_matching_20site_exact import build_H,basis,D,N,special_columns
from finite_tau_gnn_densityratio_oracle20 import PairGNN,bits,make_wd,globals_from_wd

ROOT="/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure"
rng=np.random.default_rng(20260930);H,_=build_H(.5);ys=special_columns(rng,6);wd=make_wd(H,ys)
model=PairGNN();dummy=model.init(jax.random.PRNGKey(0),jnp.zeros((1,N)),jnp.zeros((1,N)),jnp.zeros((1,3)))['params']
files={.1:'finite_tau_gnn_densityratio_walkers20.mpack',.2:'finite_tau_gnn_second_block20.mpack'}
params={t:serialization.from_bytes(dummy,open(ROOT+'/results/'+f,'rb').read()) for t,f in files.items()}
apply=jax.jit(lambda p,x,y,g:model.apply({'params':p},x,y,g))
E=np.zeros((D,len(ys)));E[ys,np.arange(len(ys))]=1
for tau in (.1,.2):
 G=np.asarray(sla.expm_multiply(-tau*H,E),float);G/=np.linalg.norm(G,axis=0,keepdims=True)
 print('TAU',tau)
 for k,y in enumerate(ys):
  yb=bits(np.array([basis[y]]))[0];F=[]
  for i in range(0,D,4096):
   j=min(D,i+4096);idx=np.arange(i,j);yy=np.repeat(yb[None,:],j-i,0)
   F.append(np.asarray(apply(params[tau],jnp.asarray(bits(basis[idx])),jnp.asarray(yy),jnp.asarray(globals_from_wd(wd[k],tau,idx)))))
  F=np.concatenate(F); lg=F-logsumexp(2*F)/2
  a=np.abs(G[:,k]);a/=np.linalg.norm(a);la=np.log(np.maximum(a,1e-300))
  logq=2*lg;logp=2*la
  logew2=logsumexp(2*logp-logq)
  ess=float(np.exp(-logew2))
  p=a*a;q=np.exp(logq)
  kl=float(np.sum(p*(logp-logq)))
  fid=float(np.dot(np.exp(lg),a)**2)
  hole1=float(np.sum(p[q<.1*p]));hole01=float(np.sum(p[q<.01*p]))
  print({'split':'train' if k<4 else 'test','y':int(y),'fid':fid,'ess_fraction':ess,'KL_pq':kl,
         'p_mass_q_lt_0.1p':hole1,'p_mass_q_lt_0.01p':hole01})
