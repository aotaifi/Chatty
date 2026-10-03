import math
import numpy as np
import jax,jax.numpy as jnp
import netket as nk, flax
import netket.jax as nkjax
from nqsmagic.models import ViT
from nqsmagic.utils import _logpsi_transl_2d

L=8;N=64;J2=.5
z0=np.load('/Users/aliotaifi/j1j2_vit_bench/krylov_scaling_8x8_a1.20_tr4096_va2048.npz')
TR=z0['train_states'][:256].astype(np.uint64); VA=z0['val_states'][:128].astype(np.uint64)
WT=z0['iwtrain'][:256].astype(float); WV=z0['iwval'][:128].astype(float); WT/=WT.sum();WV/=WV.sum()
print('NEEL',float(np.sum(WT*z0['rtrain'][:256])),float(np.sum(WV*z0['rval'][:128])),flush=True)

hi=nk.hilbert.Spin(s=.5,N=N)
graph=nk.graph.Hypercube(length=L,n_dim=2,pbc=True,max_neighbor_order=2)
model=ViT(num_layers=8,d_model=60,heads=10,L_eff=16,b=2,transl_invariant=True,two_dimensional=True)
apply=nkjax.HashablePartial(_logpsi_transl_2d,model.apply,2)
sam=nk.sampler.MetropolisExchange(hi,graph=graph,d_max=2,n_chains=16,sweep_size=N)
v=nk.vqs.MCState(sampler=sam,apply_fun=apply,n_samples=16,variables=model.init(jax.random.PRNGKey(1234),jnp.zeros((1,N))),n_discard_per_chain=10)
with open('vit_J2=0.50_N=8x8_k=0.mpack','rb') as f:obj=flax.serialization.msgpack_restore(f.read())
v.variables=flax.serialization.from_state_dict(v.variables,obj)
def bits2x(ss):
 a=np.asarray(ss,np.uint64).reshape(-1,1);return 2*((a>>np.arange(N,dtype=np.uint64))&1).astype(float)-1
def evalz(ss,b=4096):
 X=bits2x(ss);return np.concatenate([np.asarray(v.log_value(jnp.asarray(X[i:i+b]))) for i in range(0,len(X),b)])
q=lambda x,y:(x%L)+L*(y%L)
B=[]
for y in range(L):
 for x in range(L):
  i=q(x,y);B += [(i,q(x+1,y),1.),(i,q(x,y+1),1.),(i,q(x+1,y+1),J2),(i,q(x+1,y-1),J2)]
masks={
 'stripe_x':sum(1<<(x+L*y) for y in range(L) for x in range(L) if x%2==0),
 'stripe_y':sum(1<<(x+L*y) for y in range(L) for x in range(L) if y%2==0)}
def sg(s,m):return 1 if ((int(s)&m).bit_count()%2)==0 else -1
ALL=np.concatenate([TR,VA]);met=[];ne=[]
for s0 in ALL:
 s=int(s0);d=0.;ls=[]
 for i,j,J in B:
  same=((s>>i)&1)==((s>>j)&1);d+=J*(.25 if same else -.25)
  if not same:
   yy=s^(1<<i)^(1<<j);ls.append((yy,J));ne.append(yy)
 met.append((d,ls))
U=np.unique(np.concatenate([ALL,np.asarray(ne,np.uint64)]));Z=evalz(U);lm={int(s):float(z.real) for s,z in zip(U,Z)}
print('CACHE',len(U),flush=True)
for name,m in masks.items():
 r=[]
 for s0,(d,ls) in zip(ALL,met):
  s=int(s0);sx=sg(s,m);lx=lm[s];rr=d
  for yy,J in ls:rr += .5*J*(sg(yy,m)/sx)*math.exp(lm[int(yy)]-lx)
  r.append(rr)
 r=np.asarray(r);rt=r[:len(TR)];rv=r[len(TR):]
 print(name,float(np.sum(WT*rt)),float(np.sum(WV*rv)),flush=True)
