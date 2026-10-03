import os,json,math
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
OUT=os.environ.get("OUT","fnmle8_fullbatch_linesearch.json")
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

def eval_real(p,X,batch=1024):
 out=[]; vars={"params":p}
 for i in range(0,len(X),batch):
  out.append(np.asarray(apply(vars,jnp.asarray(X[i:i+batch]))).real)
 return np.concatenate(out)
l0r1=eval_real(p0,r1); l0r2=eval_real(p0,r2); l0q=eval_real(p0,q)

# full-batch sampled KL gradient at theta0: 2(E_q log a - E_f log a)
# chunked accumulation to avoid giant activation memory.
def mean_logamp(p,X):
 # X is jax array, this call may be large; use exact chunks outside grad with tree sum.
 return jnp.mean(jnp.real(apply({"params":p},X)))

# accumulate gradients of means by chunks
def mean_grad(p,X,chunk=256):
 total=None; n=len(X)
 for i in range(0,n,chunk):
  xb=jnp.asarray(X[i:i+chunk])
  val,g=jax.value_and_grad(mean_logamp)(p,xb)
  w=len(xb)/n
  g=jax.tree_util.tree_map(lambda z:z*w,g)
  total=g if total is None else jax.tree_util.tree_map(lambda a,b:a+b,total,g)
 return total
print("GRAD q",flush=True)
gq=mean_grad(p0,q)
print("GRAD f",flush=True)
gf=mean_grad(p0,r1)
g=jax.tree_util.tree_map(lambda a,b:2*(a-b),gq,gf)
# Adam first-step direction per unit learning rate.
eps=1e-8
d=jax.tree_util.tree_map(lambda x:-x/(jnp.sqrt(x*x)+eps),g)
# also raw normalized gradient direction for cross-check
norm=float(np.sqrt(sum(float(jnp.vdot(x,x).real) for x in jax.tree_util.tree_leaves(g))))
draw=jax.tree_util.tree_map(lambda x:-x/(norm+1e-30),g)
print("GRADNORM",norm,flush=True)

def logmeanexp(x):
 x=np.asarray(x,float); m=x.max(); return float(m+np.log(np.mean(np.exp(x-m))))
def metrics(p):
 lr1=eval_real(p,r1);lr2=eval_real(p,r2);lq=eval_real(p,q)
 dr1=lr1-l0r1;dr2=lr2-l0r2;dq=lq-l0q
 z=logmeanexp(2*dq)
 ww=np.exp(2*dq-np.max(2*dq))
 return dict(train_gain=float(2*dr1.mean()-z),val_gain=float(2*dr2.mean()-z),
             logZ=float(z),ess=float(ww.sum()**2/(ww@ww)),rms=float(np.std(dq)),
             mean_dq=float(dq.mean()))
rows=[]
for typ,dirn,scales in [
 ("adam1",d,[1e-9,3e-9,1e-8,3e-8,1e-7,3e-7,1e-6,3e-6,1e-5,2e-5]),
 ("rawnorm",draw,[1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1])]:
 for s in scales:
  p=jax.tree_util.tree_map(lambda a,b:a+s*b,p0,dirn)
  m=metrics(p);m.update(direction=typ,scale=s);rows.append(m)
  print("LINE",json.dumps(m),flush=True)
json.dump({"gradnorm":norm,"rows":rows},open(OUT,"w"),indent=2)
