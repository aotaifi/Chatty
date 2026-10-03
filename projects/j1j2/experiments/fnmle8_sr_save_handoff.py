import os,json,math,time
import numpy as np
import jax,jax.numpy as jnp
import flax
import netket.jax as nkjax
from jax.flatten_util import ravel_pytree
from nqsmagic.models import ViT
from nqsmagic.utils import _logpsi_transl_2d

jax.config.update("jax_enable_x64", True)
L=8; N=64; ALPHA=1.2
ETA=float(os.environ.get("SR_ETA","0.01"))
SHIFT=float(os.environ.get("SR_SHIFT","1.0"))
NS=int(os.environ.get("SR_NS","512"))
BASE_CK="vit_J2=0.50_N=8x8_k=0.mpack"
REP1="gfmc_8x8_grscale_M128_seed10501.npz"
REP2="gfmc_8x8_grscale_M128_seed10502.npz"
GUIDE="krylov_phys8_a2_fixedT.npz"
THRESH="krylov_scaling_8x8_a1.20_tr4096_va2048.npz"
OUT_CK="vit8_fnmle_sr_eta001.mpack"
OUT_NPZ="vit8_fnmle_sr_handoff.npz"
OUT_JSON="vit8_fnmle_sr_handoff.json"

model=ViT(num_layers=8,d_model=60,heads=10,L_eff=16,b=2,transl_invariant=True,two_dimensional=True)
apply=nkjax.HashablePartial(_logpsi_transl_2d,model.apply,2)
template=model.init(jax.random.PRNGKey(1234),jnp.zeros((1,N)))
with open(BASE_CK,"rb") as f: obj=flax.serialization.msgpack_restore(f.read())
variables=flax.serialization.from_state_dict(template,obj)
p0=variables["params"]; flat0,unravel=ravel_pytree(p0)

def bits2x(ss):
 a=np.asarray(ss,np.uint64).reshape(-1,1)
 return 2*((a>>np.arange(N,dtype=np.uint64))&1).astype(np.float64)-1
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
def eval_real(p,X,batch=1024):
 out=[]
 for i in range(0,len(X),batch):
  out.append(np.asarray(apply({"params":p},jnp.asarray(X[i:i+batch]))).real)
 return np.concatenate(out)
def logmeanexp(x):
 x=np.asarray(x,float); m=x.max(); return float(m+np.log(np.mean(np.exp(x-m))))

r1=bits2x(np.load(REP1)["mixed"].astype(np.uint64))
r2=bits2x(np.load(REP2)["mixed"].astype(np.uint64))
pdat=np.load(GUIDE); pool_states=pdat["states"].astype(np.uint64); q=bits2x(pool_states)
tdat=np.load(THRESH); threshold_states=tdat["train_states"].astype(np.uint64)
xt=bits2x(threshold_states)

print("SR_SAVE gradients",flush=True)
gq=mean_grad(p0,q); gf1=mean_grad(p0,r1)
g1=jax.tree_util.tree_map(lambda a,b:2*(a-b),gq,gf1)
g1f,_=ravel_pytree(g1)

idx=np.linspace(0,len(q)-1,NS,dtype=int)
Xq=jnp.asarray(q[idx])
def logvec(flat):
 return jnp.real(apply({"params":unravel(flat)},Xq))
_,pullback=jax.vjp(logvec,flat0)
def fisher(v):
 _,u=jax.jvp(logvec,(flat0,),(v,))
 u=u-jnp.mean(u)
 return pullback(u/NS)[0]
fisher=jax.jit(fisher)
_ = fisher(jnp.zeros_like(flat0)).block_until_ready()
def A(v): return fisher(v)+SHIFT*v
sol,info=jax.scipy.sparse.linalg.cg(A,g1f,tol=1e-5,atol=0.0,maxiter=200)
sol.block_until_ready()
res=float(jnp.linalg.norm(A(sol)-g1f)/jnp.linalg.norm(g1f))
pnew=unravel(flat0-ETA*sol)
vnew=dict(variables); vnew["params"]=pnew
with open(OUT_CK,"wb") as f: f.write(flax.serialization.to_bytes(vnew))
print("SR_SAVE checkpoint",OUT_CK,"res",res,flush=True)

# Reproduce held-out MLE gate.
l0q=eval_real(p0,q); lnq=eval_real(pnew,q); dq=lnq-l0q
z=logmeanexp(2*dq)
l0r1=eval_real(p0,r1); lnr1=eval_real(pnew,r1)
l0r2=eval_real(p0,r2); lnr2=eval_real(pnew,r2)
train_gain=float(2*np.mean(lnr1-l0r1)-z)
val_gain=float(2*np.mean(lnr2-l0r2)-z)
ww=np.exp(2*dq-np.max(2*dq)); guide_ess=float(ww.sum()**2/(ww@ww))

# Correct threshold measure: threshold states are already a0^alpha samples.
# Reweight only by (a1/a0)^alpha. Do NOT multiply by stored iwtrain.
l0t=eval_real(p0,xt); lnt=eval_real(pnew,xt); dt=lnt-l0t
tw=np.exp(ALPHA*dt-np.max(ALPHA*dt)); tw/=tw.sum()
# Physical initialization pool was drawn from a0^2.
pw=np.exp(2*dq-np.max(2*dq)); pw/=pw.sum()
np.savez_compressed(OUT_NPZ,
 threshold_states=threshold_states,threshold_weights=tw,
 pool_states=pool_states,pool_weights=pw,
 delta_threshold=dt,delta_pool=dq,
 r0_threshold=tdat["rtrain"].astype(float),
 r0_pool=pdat["r"].astype(float) if "r" in pdat.files else np.array([]))
out=dict(eta=ETA,diag_shift=SHIFT,fisher_ns=NS,cg_rel_resid=res,
 train_ll_gain=train_gain,val_ll_gain=val_gain,guide_ESS=guide_ess,
 delta_pool_rms=float(np.std(dq)),delta_threshold_rms=float(np.std(dt)),
 threshold_ESS=float(1/(tw@tw)),pool_ESS=float(1/(pw@pw)))
print("SR_HANDOFF",json.dumps(out),flush=True)
json.dump(out,open(OUT_JSON,"w"),indent=2)
