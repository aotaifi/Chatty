import os,json,math,time
import numpy as np
import jax,jax.numpy as jnp
import flax
import netket.jax as nkjax
from jax.flatten_util import ravel_pytree
from nqsmagic.models import ViT
from nqsmagic.utils import _logpsi_transl_2d

jax.config.update("jax_enable_x64", True)
L=8; N=64
BASE_CK="vit_J2=0.50_N=8x8_k=0.mpack"
REP1="gfmc_8x8_grscale_M128_seed10501.npz"
REP2="gfmc_8x8_grscale_M128_seed10502.npz"
GUIDE="krylov_phys8_a2_fixedT.npz"
OUT=os.environ.get("OUT","fnmle8_sr_matrixfree.json")
NS=int(os.environ.get("SR_NS","512"))

model=ViT(num_layers=8,d_model=60,heads=10,L_eff=16,b=2,transl_invariant=True,two_dimensional=True)
apply=nkjax.HashablePartial(_logpsi_transl_2d,model.apply,2)
template=model.init(jax.random.PRNGKey(1234),jnp.zeros((1,N)))
with open(BASE_CK,"rb") as f: obj=flax.serialization.msgpack_restore(f.read())
variables=flax.serialization.from_state_dict(template,obj)
p0=variables["params"]
flat0,unravel=ravel_pytree(p0)
print("NPARAM",flat0.size,flush=True)

def bits2x(ss):
 a=np.asarray(ss,np.uint64).reshape(-1,1)
 return 2*((a>>np.arange(N,dtype=np.uint64))&1).astype(np.float64)-1
r1=bits2x(np.load(REP1)["mixed"].astype(np.uint64))
r2=bits2x(np.load(REP2)["mixed"].astype(np.uint64))
qall=bits2x(np.load(GUIDE)["states"].astype(np.uint64))

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
def pdot(a,b): return float(sum(jnp.vdot(x,y).real for x,y in zip(jax.tree_util.tree_leaves(a),jax.tree_util.tree_leaves(b))))
def pnorm(a): return math.sqrt(max(pdot(a,a),0.0))

print("GRADS",flush=True)
gq=mean_grad(p0,qall)
gf1=mean_grad(p0,r1)
gf2=mean_grad(p0,r2)
g1=sub2(gq,gf1); g2=sub2(gq,gf2)
g1f,_=ravel_pytree(g1); g2f,_=ravel_pytree(g2)
eu=float(jnp.vdot(g1f,g2f).real); n1=float(jnp.linalg.norm(g1f)); n2=float(jnp.linalg.norm(g2f))
print("EU",json.dumps({"dot":eu,"cos":eu/(n1*n2),"n1":n1,"n2":n2}),flush=True)

# deterministic spread through the stored guide sample rather than first contiguous block
idx=np.linspace(0,len(qall)-1,NS,dtype=int)
Xq=jnp.asarray(qall[idx])

def logvec(flat):
 p=unravel(flat)
 return jnp.real(apply({"params":p},Xq))

# Matrix-free empirical Fisher S v = J^T P J v / Ns.
# VJP at theta0 is reused; centering P removes the constant log-amplitude/gauge mode.
print("BUILD_VJP ns",NS,flush=True)
_,pullback=jax.vjp(logvec,flat0)

def fisher0(v):
 _,u=jax.jvp(logvec,(flat0,),(v,))
 u=u-jnp.mean(u)
 return pullback(u/NS)[0]

# JIT only the matvec; no NetKet sharding machinery.
fisher0=jax.jit(fisher0)
# warmup
_ = fisher0(jnp.zeros_like(flat0)).block_until_ready()
print("WARM",flush=True)

rows=[]
for shift in [1.0,0.3,0.1,0.03,0.01,0.003,0.001]:
 t=time.time()
 def A(v): return fisher0(v)+shift*v
 try:
  sol,info=jax.scipy.sparse.linalg.cg(A,g1f,tol=1e-5,atol=0.0,maxiter=200)
  sol.block_until_ready()
  d1=float(jnp.vdot(g1f,sol).real)
  d2=float(jnp.vdot(g2f,sol).real)
  ns=float(jnp.linalg.norm(sol))
  # residual diagnostic
  res=float(jnp.linalg.norm(A(sol)-g1f)/jnp.linalg.norm(g1f))
  row={"diag_shift":shift,"train_deriv_gain":d1,"val_deriv_gain":d2,
       "delta_norm":ns,"rel_resid":res,"info":None if info is None else str(info),
       "sec":time.time()-t}
  rows.append(row); print("SRMF",json.dumps(row),flush=True)
 except Exception as e:
  row={"diag_shift":shift,"error":repr(e),"sec":time.time()-t}
  rows.append(row); print("ERR",json.dumps(row),flush=True)

json.dump({"ns":NS,"euclidean":{"dot":eu,"cos":eu/(n1*n2),"n1":n1,"n2":n2},"rows":rows},open(OUT,"w"),indent=2)
