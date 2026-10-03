import os,json
import numpy as np
import jax,jax.numpy as jnp
import flax
import netket.jax as nkjax
from jax.flatten_util import ravel_pytree
from nqsmagic.models import ViT
from nqsmagic.utils import _logpsi_transl_2d

jax.config.update("jax_enable_x64", True)
N=64; ALPHA=1.2
ETA=float(os.environ.get("SR_ETA","0.01"))
SHIFT=float(os.environ.get("SR_SHIFT","1.0"))
BASE_CK="vit_J2=0.50_N=8x8_k=0.mpack"
REP1="gfmc_8x8_grscale_M128_seed10501.npz"
REP2="gfmc_8x8_grscale_M128_seed10502.npz"
GUIDE="krylov_phys8_a2_fixedT.npz"
THR="krylov_scaling_8x8_a1.20_tr4096_va2048.npz"
OUT_CK="vit8_fnmle_sr_eta001.mpack"
OUT_HAND="vit8_fnmle_sr_eta001_handoff.npz"
OUT_JSON="vit8_fnmle_sr_eta001_build.json"

model=ViT(num_layers=8,d_model=60,heads=10,L_eff=16,b=2,
          transl_invariant=True,two_dimensional=True)
apply=nkjax.HashablePartial(_logpsi_transl_2d,model.apply,2)
template=model.init(jax.random.PRNGKey(1234),jnp.zeros((1,N)))
with open(BASE_CK,"rb") as f:
    obj=flax.serialization.msgpack_restore(f.read())
variables=flax.serialization.from_state_dict(template,obj)
p0=variables["params"]; flat0,unravel=ravel_pytree(p0)
def bits2x(ss):
    a=np.asarray(ss,np.uint64).reshape(-1,1)
    return 2*((a>>np.arange(N,dtype=np.uint64))&1).astype(np.float64)-1

r1=bits2x(np.load(REP1)["mixed"].astype(np.uint64))
r2=bits2x(np.load(REP2)["mixed"].astype(np.uint64))
qall_states=np.load(GUIDE)["states"].astype(np.uint64)
qall=bits2x(qall_states)

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

def sub2(a,b):
    return jax.tree_util.tree_map(lambda x,y:2*(x-y),a,b)

def eval_real(p,X,batch=1024):
    out=[]
    for i in range(0,len(X),batch):
        out.append(np.asarray(apply({"params":p},jnp.asarray(X[i:i+batch]))).real)
    return np.concatenate(out)
def logmeanexp(x):
    x=np.asarray(x,float); m=x.max()
    return float(m+np.log(np.mean(np.exp(x-m))))

print("PROD_GRADS",flush=True)
gq=mean_grad(p0,qall); gf1=mean_grad(p0,r1); gf2=mean_grad(p0,r2)
g1=sub2(gq,gf1); g2=sub2(gq,gf2)
g1f,_=ravel_pytree(g1); g2f,_=ravel_pytree(g2)
cos=float(jnp.vdot(g1f,g2f).real/(jnp.linalg.norm(g1f)*jnp.linalg.norm(g2f)))
print("PROD_EU_COS",cos,flush=True)

NS=512
idx=np.linspace(0,len(qall)-1,NS,dtype=int)
Xq=jnp.asarray(qall[idx])
def logvec(flat):
    return jnp.real(apply({"params":unravel(flat)},Xq))
_,pullback=jax.vjp(logvec,flat0)
def fisher(v):
    _,u=jax.jvp(logvec,(flat0,),(v,))
    u=u-jnp.mean(u)
    return pullback(u/NS)[0]
fisher=jax.jit(fisher)
_=fisher(jnp.zeros_like(flat0)).block_until_ready()
def A(v):
    return fisher(v)+SHIFT*v

print("PROD_SR_SOLVE",flush=True)
sol,info=jax.scipy.sparse.linalg.cg(A,g1f,tol=1e-5,atol=0.0,maxiter=200)
sol.block_until_ready()
res=float(jnp.linalg.norm(A(sol)-g1f)/jnp.linalg.norm(g1f))
dtrain=float(jnp.vdot(g1f,sol).real)
dval=float(jnp.vdot(g2f,sol).real)
flat1=flat0-ETA*sol; p1=unravel(flat1)
print("PROD_SR",json.dumps({"shift":SHIFT,"eta":ETA,"resid":res,
      "train_deriv":dtrain,"val_deriv":dval}),flush=True)

l0r1=eval_real(p0,r1); l0r2=eval_real(p0,r2); l0q=eval_real(p0,qall)
l1r1=eval_real(p1,r1); l1r2=eval_real(p1,r2); l1q=eval_real(p1,qall)
dq=l1q-l0q; z=logmeanexp(2*dq)
ww=np.exp(2*dq-np.max(2*dq))
train_gain=float(2*np.mean(l1r1-l0r1)-z)
val_gain=float(2*np.mean(l1r2-l0r2)-z)
ess=float(ww.sum()**2/(ww@ww)); rms=float(np.std(dq))
print("PROD_GATE",json.dumps({"train_gain":train_gain,"val_gain":val_gain,
      "guide_ess":ess,"dloga_rms":rms}),flush=True)
if not (val_gain>0 and ess>1000 and res<1e-3):
    raise RuntimeError("SR production gate failed")

with open(OUT_CK,"wb") as f:
    f.write(flax.serialization.msgpack_serialize({"params":p1}))

thr=np.load(THR); tr=thr["train_states"].astype(np.uint64)
ltr0=eval_real(p0,bits2x(tr)); ltr1=eval_real(p1,bits2x(tr))
dtr=ltr1-ltr0; dp=l1q-l0q
wt=np.exp(ALPHA*dtr-np.max(ALPHA*dtr)); wt/=wt.sum()
wp=np.exp(2*dp-np.max(2*dp)); wp/=wp.sum()
tess=float(1/(wt@wt)); pess=float(1/(wp@wp))
np.savez_compressed(OUT_HAND,threshold_states=tr,threshold_weights=wt,
    pool_states=qall_states,pool_weights=wp,delta_logamp_threshold=dtr,
    delta_logamp_pool=dp,r0_threshold=thr["rtrain"].astype(float),
    alpha=ALPHA,eta=ETA,shift=SHIFT)
summary={"euclidean_cos":cos,"shift":SHIFT,"eta":ETA,"sr_resid":res,
    "train_deriv":dtrain,"val_deriv":dval,"train_gain":train_gain,
    "val_gain":val_gain,"guide_ess":ess,"delta_logamp_rms":rms,
    "threshold_ess":tess,"pool_ess":pess}
json.dump(summary,open(OUT_JSON,"w"),indent=2)
print("PROD_BUILD",json.dumps(summary),flush=True)
