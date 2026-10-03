import os,json,math
import numpy as np
import jax,jax.numpy as jnp
import flax
import netket.jax as nkjax
from jax.flatten_util import ravel_pytree
from nqsmagic.models import ViT
from nqsmagic.utils import _logpsi_transl_2d
jax.config.update("jax_enable_x64", True)

N=64; ALPHA=1.2; SHIFT=1.0; NS=512
BASE_CK="vit_J2=0.50_N=8x8_k=0.mpack"
CUR_CK="vit8_fnmle_sr_eta001.mpack"
REP1="fnmle_sr_k1fn_seed13001.npz"
REP2="fnmle_sr_k1fn_seed13002.npz"
GUIDE="krylov_phys8_a2_fixedT.npz"
THRESH="krylov_scaling_8x8_a1.20_tr4096_va2048.npz"
HAND1="vit8_fnmle_sr_handoff.npz"
OUT_CK="vit8_fnmle_sr_iter2.mpack"
OUT_NPZ="vit8_fnmle_sr_iter2_handoff.npz"
OUT_JSON="vit8_fnmle_sr_iter2_train.json"

model=ViT(num_layers=8,d_model=60,heads=10,L_eff=16,b=2,transl_invariant=True,two_dimensional=True)
apply=nkjax.HashablePartial(_logpsi_transl_2d,model.apply,2)
template=model.init(jax.random.PRNGKey(1234),jnp.zeros((1,N)))
def loadv(path):
 obj=flax.serialization.msgpack_restore(open(path,'rb').read())
 return flax.serialization.from_state_dict(template,obj)
vb=loadv(BASE_CK); vc=loadv(CUR_CK)
pb=vb["params"]; pc=vc["params"]; flatc,unravel=ravel_pytree(pc)

def bits2x(ss):
 a=np.asarray(ss,np.uint64).reshape(-1,1)
 return 2*((a>>np.arange(N,dtype=np.uint64))&1).astype(np.float64)-1
def mean_logamp(p,X):
 return jnp.mean(jnp.real(apply({"params":p},X)))
def mean_grad(p,X,weights=None,chunk=256):
 n=len(X); total=None
 if weights is None: weights=np.ones(n)/n
 else:
  weights=np.asarray(weights,float); weights=weights/weights.sum()
 for i in range(0,n,chunk):
  xb=jnp.asarray(X[i:i+chunk]); wb=jnp.asarray(weights[i:i+chunk])
  def loss(pp): return jnp.sum(wb*jnp.real(apply({"params":pp},xb)))
  _,g=jax.value_and_grad(loss)(p)
  total=g if total is None else jax.tree_util.tree_map(lambda a,b:a+b,total,g)
 return total
def eval_real(p,X,batch=1024):
 out=[]
 for i in range(0,len(X),batch):
  out.append(np.asarray(apply({"params":p},jnp.asarray(X[i:i+batch]))).real)
 return np.concatenate(out)
def logsumexp_weighted(logv,w):
 logv=np.asarray(logv,float);w=np.asarray(w,float);w=w/w.sum()
 m=np.max(logv); return float(m+np.log(np.sum(w*np.exp(logv-m))))

r1s=np.load(REP1)["mixed"].astype(np.uint64); r2s=np.load(REP2)["mixed"].astype(np.uint64)
r1=bits2x(r1s); r2=bits2x(r2s)
pdat=np.load(GUIDE); pool_states=pdat["states"].astype(np.uint64); q=bits2x(pool_states)
tdat=np.load(THRESH); threshold_states=tdat["train_states"].astype(np.uint64); xt=bits2x(threshold_states)
h1=np.load(HAND1); w1=np.asarray(h1["pool_weights"],float); w1/=w1.sum()

print("ITER2_GRADS",len(r1),len(r2),"qESS",1/(w1@w1),flush=True)
gq=mean_grad(pc,q,w1); gf1=mean_grad(pc,r1); gf2=mean_grad(pc,r2)
g1=jax.tree_util.tree_map(lambda a,b:2*(a-b),gq,gf1)
g2=jax.tree_util.tree_map(lambda a,b:2*(a-b),gq,gf2)
g1f,_=ravel_pytree(g1); g2f,_=ravel_pytree(g2)
eu=float(jnp.vdot(g1f,g2f).real); n1=float(jnp.linalg.norm(g1f)); n2=float(jnp.linalg.norm(g2f))
print("ITER2_EU",json.dumps({"dot":eu,"cos":eu/(n1*n2),"n1":n1,"n2":n2}),flush=True)

rng=np.random.default_rng(20261001)
idx=rng.choice(len(q),size=NS,replace=True,p=w1)
Xq=jnp.asarray(q[idx])
def logvec(flat): return jnp.real(apply({"params":unravel(flat)},Xq))
_,pullback=jax.vjp(logvec,flatc)
def fisher(v):
 _,u=jax.jvp(logvec,(flatc,),(v,))
 u=u-jnp.mean(u)
 return pullback(u/NS)[0]
fisher=jax.jit(fisher); _=fisher(jnp.zeros_like(flatc)).block_until_ready()
def A(v): return fisher(v)+SHIFT*v
sol,info=jax.scipy.sparse.linalg.cg(A,g1f,tol=1e-5,atol=0.,maxiter=200)
sol.block_until_ready()
res=float(jnp.linalg.norm(A(sol)-g1f)/jnp.linalg.norm(g1f))
dtrain=float(jnp.vdot(g1f,sol).real); dval=float(jnp.vdot(g2f,sol).real)
print("ITER2_SR_DERIV",json.dumps({"train":dtrain,"val":dval,"res":res}),flush=True)

lcq=eval_real(pc,q); lcr1=eval_real(pc,r1); lcr2=eval_real(pc,r2)
rows=[]; candidates=[]
for eta in [1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2,2e-2,3e-2]:
 p=unravel(flatc-eta*sol)
 dq=eval_real(p,q)-lcq
 z=logsumexp_weighted(2*dq,w1)
 tg=float(2*np.mean(eval_real(p,r1)-lcr1)-z)
 vg=float(2*np.mean(eval_real(p,r2)-lcr2)-z)
 wt=w1*np.exp(2*dq-np.max(2*dq));wt/=wt.sum()
 ess=float(1/(wt@wt))
 row={"eta":eta,"train_gain":tg,"val_gain":vg,"pool_ESS":ess,"delta_pool_rms":float(np.std(dq))}
 rows.append(row); print("ITER2_LINE",json.dumps(row),flush=True)
 if vg>0 and ess>=2000: candidates.append((vg,eta,p,row))
if not candidates:
 json.dump({"status":"FAIL_NO_VALID_STEP","eu_cos":eu/(n1*n2),"sr_val_deriv":dval,"rows":rows},open(OUT_JSON,'w'),indent=2)
 raise SystemExit("no validation-positive healthy-ESS SR step")
_,eta,pbest,brow=max(candidates,key=lambda x:x[0])

# Absolute a0-based reweights for corrected threshold and physical pool.
lbq=eval_real(pb,q); lnq=eval_real(pbest,q); dq0=lnq-lbq
lbt=eval_real(pb,xt); lnt=eval_real(pbest,xt); dt0=lnt-lbt
pw=np.exp(2*dq0-np.max(2*dq0));pw/=pw.sum()
tw=np.exp(ALPHA*dt0-np.max(ALPHA*dt0));tw/=tw.sum()
vnew=dict(vc); vnew["params"]=pbest
open(OUT_CK,'wb').write(flax.serialization.to_bytes(vnew))
np.savez_compressed(OUT_NPZ,
 threshold_states=threshold_states,threshold_weights=tw,
 pool_states=pool_states,pool_weights=pw,
 delta_threshold=dt0,delta_pool=dq0,
 r0_threshold=tdat["rtrain"].astype(float),
 r0_pool=pdat["r"].astype(float) if "r" in pdat.files else np.array([]))
out={"status":"PASS","eta":eta,"diag_shift":SHIFT,"fisher_ns":NS,
 "eu_cos":eu/(n1*n2),"sr_train_deriv":dtrain,"sr_val_deriv":dval,"cg_rel_resid":res,
 "best":brow,"threshold_ESS":float(1/(tw@tw)),"pool_ESS_absolute":float(1/(pw@pw)),
 "rows":rows}
print("ITER2_TRAIN_RESULT",json.dumps(out),flush=True)
json.dump(out,open(OUT_JSON,'w'),indent=2)
