import json,math
import numpy as np
import jax,jax.numpy as jnp
import flax
import netket.jax as nkjax
from jax.flatten_util import ravel_pytree
from nqsmagic.models import ViT
from nqsmagic.utils import _logpsi_transl_2d

jax.config.update("jax_enable_x64",True)
N=64;ALPHA=1.2;SHIFT=1.0
BASE_CK="vit_J2=0.50_N=8x8_k=0.mpack"
REP1="gfmc_8x8_grscale_M128_seed10501.npz"
REP2="gfmc_8x8_grscale_M128_seed10502.npz"
GUIDE="krylov_phys8_a2_fixedT.npz"
THR="krylov_scaling_8x8_a1.20_tr4096_va2048.npz"
OUT_CK="vit8_fnmle_sr_pooled.mpack"
OUT_HAND="vit8_fnmle_sr_pooled_handoff.npz"
OUT_JSON="vit8_fnmle_sr_pooled_build.json"

model=ViT(num_layers=8,d_model=60,heads=10,L_eff=16,b=2,
          transl_invariant=True,two_dimensional=True)
apply=nkjax.HashablePartial(_logpsi_transl_2d,model.apply,2)
template=model.init(jax.random.PRNGKey(1234),jnp.zeros((1,N)))
with open(BASE_CK,"rb") as f:obj=flax.serialization.msgpack_restore(f.read())
variables=flax.serialization.from_state_dict(template,obj)
p0=variables["params"];flat0,unravel=ravel_pytree(p0)

def bits2x(ss):
 a=np.asarray(ss,np.uint64).reshape(-1,1)
 return 2*((a>>np.arange(N,dtype=np.uint64))&1).astype(np.float64)-1
r1=bits2x(np.load(REP1)["mixed"].astype(np.uint64))
r2=bits2x(np.load(REP2)["mixed"].astype(np.uint64))
pool_states=np.load(GUIDE)["states"].astype(np.uint64)
q=bits2x(pool_states)

def mean_logamp(p,X):return jnp.mean(jnp.real(apply({"params":p},X)))
def mean_grad(p,X,chunk=256):
 total=None;n=len(X)
 for i in range(0,n,chunk):
  xb=jnp.asarray(X[i:i+chunk]);_,g=jax.value_and_grad(mean_logamp)(p,xb)
  w=len(xb)/n;g=jax.tree_util.tree_map(lambda z:z*w,g)
  total=g if total is None else jax.tree_util.tree_map(lambda a,b:a+b,total,g)
 return total
def sub2(a,b):return jax.tree_util.tree_map(lambda x,y:2*(x-y),a,b)
def avg(a,b):return jax.tree_util.tree_map(lambda x,y:.5*(x+y),a,b)
def eval_real(p,X,batch=1024):
 out=[]
 for i in range(0,len(X),batch):
  out.append(np.asarray(apply({"params":p},jnp.asarray(X[i:i+batch]))).real)
 return np.concatenate(out)
def logmeanexp(x):
 x=np.asarray(x,float);m=x.max();return float(m+np.log(np.mean(np.exp(x-m))))

print("POOL_GRADS",flush=True)
gq=mean_grad(p0,q);gf1=mean_grad(p0,r1);gf2=mean_grad(p0,r2)
g1=sub2(gq,gf1);g2=sub2(gq,gf2);gp=avg(g1,g2)
g1f,_=ravel_pytree(g1);g2f,_=ravel_pytree(g2);gpf,_=ravel_pytree(gp)
print("POOL_EU",json.dumps({"cos12":float(jnp.vdot(g1f,g2f).real/(jnp.linalg.norm(g1f)*jnp.linalg.norm(g2f))),
 "n1":float(jnp.linalg.norm(g1f)),"n2":float(jnp.linalg.norm(g2f)),
 "npool":float(jnp.linalg.norm(gpf))}),flush=True)
idx=np.linspace(0,len(q)-1,512,dtype=int);Xq=jnp.asarray(q[idx])
def logvec(flat):return jnp.real(apply({"params":unravel(flat)},Xq))
_,pullback=jax.vjp(logvec,flat0)
def fisher(v):
 _,u=jax.jvp(logvec,(flat0,),(v,));u=u-jnp.mean(u)
 return pullback(u/len(Xq))[0]
fisher=jax.jit(fisher);_=fisher(jnp.zeros_like(flat0)).block_until_ready()
def A(v):return fisher(v)+SHIFT*v
sol,info=jax.scipy.sparse.linalg.cg(A,gpf,tol=1e-5,atol=0,maxiter=200)
sol.block_until_ready()
res=float(jnp.linalg.norm(A(sol)-gpf)/jnp.linalg.norm(gpf))
cross1=float(jnp.vdot(g1f,sol).real);cross2=float(jnp.vdot(g2f,sol).real)
print("POOL_SR",json.dumps({"resid":res,"rep1_deriv":cross1,"rep2_deriv":cross2}),flush=True)

l0q=eval_real(p0,q);l0r1=eval_real(p0,r1);l0r2=eval_real(p0,r2)
rows=[]
for eta in [0.001,0.003,0.006,0.01,0.015,0.02]:
 p=unravel(flat0-eta*sol)
 lq=eval_real(p,q);lr1=eval_real(p,r1);lr2=eval_real(p,r2)
 dq=lq-l0q;z=logmeanexp(2*dq);ww=np.exp(2*dq-np.max(2*dq))
 row={"eta":eta,"gain1":float(2*np.mean(lr1-l0r1)-z),
      "gain2":float(2*np.mean(lr2-l0r2)-z),
      "gain_avg":float(np.mean([2*np.mean(lr1-l0r1)-z,2*np.mean(lr2-l0r2)-z])),
      "ess":float(ww.sum()**2/(ww@ww)),"rms":float(np.std(dq))}
 rows.append(row);print("POOL_LINE",json.dumps(row),flush=True)
valid=[x for x in rows if x["gain1"]>0 and x["gain2"]>0 and x["ess"]>1000 and x["rms"]<.5]
if not valid:raise RuntimeError("no pooled SR step transfers to both replicas")
best=max(valid,key=lambda x:min(x["gain1"],x["gain2"]))
ETA=best["eta"];p1=unravel(flat0-ETA*sol)
with open(OUT_CK,"wb") as f:f.write(flax.serialization.msgpack_serialize({"params":p1}))
thr=np.load(THR);tr=thr["train_states"].astype(np.uint64)
l0tr=eval_real(p0,bits2x(tr));l1tr=eval_real(p1,bits2x(tr))
l1q=eval_real(p1,q);dtr=l1tr-l0tr;dp=l1q-l0q
wt=np.exp(ALPHA*dtr-np.max(ALPHA*dtr));wt/=wt.sum()
wp=np.exp(2*dp-np.max(2*dp));wp/=wp.sum()
np.savez_compressed(OUT_HAND,threshold_states=tr,threshold_weights=wt,
 pool_states=pool_states,pool_weights=wp,delta_logamp_threshold=dtr,
 delta_logamp_pool=dp,r0_threshold=thr["rtrain"].astype(float),
 alpha=ALPHA,eta=ETA,shift=SHIFT)
out={"shift":SHIFT,"eta":ETA,"sr_resid":res,"rep1_deriv":cross1,"rep2_deriv":cross2,
 "selected":best,"lines":rows,"threshold_ess":float(1/(wt@wt)),
 "pool_ess":float(1/(wp@wp))}
json.dump(out,open(OUT_JSON,"w"),indent=2)
print("POOL_BUILD",json.dumps(out),flush=True)
