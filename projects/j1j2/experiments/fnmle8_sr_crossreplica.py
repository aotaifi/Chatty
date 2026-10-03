import os,json,math,time
import numpy as np
import jax,jax.numpy as jnp
import netket as nk, flax
import netket.jax as nkjax
from nqsmagic.models import ViT
from nqsmagic.utils import _logpsi_transl_2d

jax.config.update("jax_enable_x64", True)
L=8;N=64;B=2
BASE_CK=os.environ.get("BASE_CK","vit_J2=0.50_N=8x8_k=0.mpack")
REP1=os.environ.get("REP1","gfmc_8x8_grscale_M128_seed10501.npz")
REP2=os.environ.get("REP2","gfmc_8x8_grscale_M128_seed10502.npz")
GUIDE=os.environ.get("GUIDE","krylov_phys8_a2_fixedT.npz")
OUT=os.environ.get("OUT","fnmle8_sr_crossreplica.json")

model=ViT(num_layers=8,d_model=60,heads=10,L_eff=16,b=2,transl_invariant=True,two_dimensional=True)
apply=nkjax.HashablePartial(_logpsi_transl_2d,model.apply,2)
template=model.init(jax.random.PRNGKey(1234),jnp.zeros((1,N)))
with open(BASE_CK,"rb") as f: obj=flax.serialization.msgpack_restore(f.read())
variables=flax.serialization.from_state_dict(template,obj)
p0=variables["params"]

def bits2x(ss):
 a=np.asarray(ss,np.uint64).reshape(-1,1)
 return 2*((a>>np.arange(N,dtype=np.uint64))&1).astype(np.float64)-1
r1=bits2x(np.load(REP1)["mixed"].astype(np.uint64))
r2=bits2x(np.load(REP2)["mixed"].astype(np.uint64))
q=bits2x(np.load(GUIDE)["states"].astype(np.uint64))

def mean_logamp(p,X):
 return jnp.mean(jnp.real(apply({"params":p},X)))
def mean_grad(p,X,chunk=256):
 total=None; n=len(X)
 for i in range(0,n,chunk):
  xb=jnp.asarray(X[i:i+chunk])
  _,g=jax.value_and_grad(mean_logamp)(p,xb)
  w=len(xb)/n
  g=jax.tree_util.tree_map(lambda z:z*w,g)
  total=g if total is None else jax.tree_util.tree_map(lambda a,b:a+b,total,g)
 return total
def sub2(a,b): return jax.tree_util.tree_map(lambda x,y:2*(x-y),a,b)
def dot(a,b):
 return float(sum(jnp.vdot(x,y).real for x,y in zip(jax.tree_util.tree_leaves(a),jax.tree_util.tree_leaves(b))))
def norm(a): return math.sqrt(max(dot(a,a),0.0))

print("GRAD q",flush=True); gq=mean_grad(p0,q)
print("GRAD f1",flush=True); gf1=mean_grad(p0,r1)
print("GRAD f2",flush=True); gf2=mean_grad(p0,r2)
g1=sub2(gq,gf1); g2=sub2(gq,gf2)
d12=dot(g1,g2); n1=norm(g1); n2=norm(g2)
print("EUCLIDEAN",json.dumps(dict(dot=d12,n1=n1,n2=n2,cos=d12/(n1*n2))),flush=True)

# Build current-guide variational state only for the Fisher/QGT.
hi=nk.hilbert.Spin(s=.5,N=N)
graph=nk.graph.Hypercube(length=L,n_dim=2,pbc=True,max_neighbor_order=2)
sampler=nk.sampler.MetropolisExchange(hi,graph=graph,d_max=2,n_chains=32,sweep_size=N)
v=nk.vqs.MCState(sampler=sampler,apply_fun=apply,n_samples=4096,
    variables=template,n_discard_per_chain=10,seed=24680)
v.variables=variables
_ = v.sample()
rows=[]
for shift in [1.0,0.3,0.1,0.03,0.01,0.003,0.001]:
 t=time.time()
 sr=nk.optimizer.SR(diag_shift=shift,solver_restart=True)
 try:
  delta=sr(v,g1)
  d1=dot(g1,delta); d2=dot(g2,delta); nd=norm(delta)
  row=dict(diag_shift=shift,train_deriv_gain=d1,val_deriv_gain=d2,
           delta_norm=nd,cos_g1_delta=d1/(n1*nd),sec=time.time()-t)
  rows.append(row); print("SR",json.dumps(row),flush=True)
 except Exception as e:
  row=dict(diag_shift=shift,error=repr(e),sec=time.time()-t)
  rows.append(row); print("SRERR",json.dumps(row),flush=True)

json.dump({"euclidean":dict(dot=d12,n1=n1,n2=n2,cos=d12/(n1*n2)),"rows":rows},open(OUT,"w"),indent=2)
