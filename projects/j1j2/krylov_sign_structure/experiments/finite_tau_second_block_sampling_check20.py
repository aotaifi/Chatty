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
S=np.load(ROOT+'/results/finite_tau_gnn_second_block20_samples.npz')
model=PairGNN();dum=model.init(jax.random.PRNGKey(0),jnp.zeros((1,N)),jnp.zeros((1,N)),jnp.zeros((1,3)))['params']
pa=serialization.from_bytes(dum,open(ROOT+'/results/finite_tau_gnn_densityratio_walkers20.mpack','rb').read())
apply=jax.jit(lambda p,x,y,g:model.apply({'params':p},x,y,g))
E=np.zeros((D,6));E[ys,np.arange(6)]=1
G=np.asarray(sla.expm_multiply(-.2*H,E),float);G/=np.linalg.norm(G,axis=0,keepdims=True)
def tv_bins(samples,labels,exact):
 B=int(labels.max())+1; emp=np.bincount(labels[samples],minlength=B).astype(float);emp/=emp.sum()
 ex=np.bincount(labels,weights=exact,minlength=B);ex/=ex.sum()
 return .5*np.abs(emp-ex).sum()
for k,y in enumerate(ys[:4]):
 yb=bits(np.array([basis[y]]))[0];F=[]
 for i in range(0,D,4096):
  j=min(D,i+4096);idx=np.arange(i,j);yy=np.repeat(yb[None,:],j-i,0)
  F.append(np.asarray(apply(pa,jnp.asarray(bits(basis[idx])),jnp.asarray(yy),jnp.asarray(globals_from_wd(wd[k],.1,idx)))))
 F=np.concatenate(F);a=np.abs(G[:,k])
 q=np.exp(2*F-logsumexp(2*F));f=np.exp(F+np.log(np.maximum(a,1e-300))-logsumexp(F+np.log(np.maximum(a,1e-300))))
 d=(wd[k]//1000).astype(int);n2=np.rint(wd[k]-1000*d).astype(int);raw=d*100+n2;_,lab=np.unique(raw,return_inverse=True)
 print({'y':y,'q_tv_rep1':tv_bins(S['q'][k],lab,q),'q_tv_rep2':tv_bins(S['qv'][k],lab,q),
        'f_tv_rep1':tv_bins(S['pos'][k],lab,f),'f_tv_rep2':tv_bins(S['posv'][k],lab,f)})
