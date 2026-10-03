import json
import numpy as np
import jax, jax.numpy as jnp
from flax import serialization
import optax
from nqsmagic.models import ViT
from nqsmagic.utils import _logpsi_transl_2d
jax.config.update("jax_enable_x64",True)
L=8;N=64;B=2
ROOT="/Users/aliotaifi/Chatty/projects/j1j2/results/8x8_krylov_3471544"
CK="/Users/aliotaifi/j1j2_vit_bench/vit_J2=0.50_N=8x8_k=0.mpack"
OUT="/Users/aliotaifi/Chatty/projects/j1j2/results/warmstart_vit_head_ratio_gate_8x8.json"
OUTCK="/Users/aliotaifi/Chatty/projects/j1j2/results/warmstart_vit_head_ratio_8x8.mpack"
model=ViT(num_layers=8,d_model=60,heads=10,L_eff=16,b=2,transl_invariant=True,two_dimensional=True)
template=model.init(jax.random.PRNGKey(1),jnp.zeros((1,N)))
variables=serialization.from_state_dict(template,serialization.msgpack_restore(open(CK,'rb').read()))
basep=variables["params"]; head=basep["output"]
def b2x(s):
 a=np.asarray(s,np.uint64).reshape(-1,1); return 2*((a>>np.arange(N,dtype=np.uint64))&1).astype(float)-1
def full(h):
 p=dict(basep);p["output"]=h;return p
def la_h(h,x): return jnp.real(_logpsi_transl_2d(model.apply,B,{"params":full(h)},x))
la_jit=jax.jit(la_h)
def ev(h,s,b=1024):
 X=b2x(s);o=[]
 for i in range(0,len(X),b):o.append(np.asarray(la_jit(h,jnp.asarray(X[i:i+b]))))
 return np.concatenate(o)
def auc(y,s):
 o=np.argsort(s,kind='mergesort');r=np.empty(len(s));r[o]=np.arange(1,len(s)+1);n1=y.sum();n0=len(y)-n1
 return float((r[y==1].sum()-n1*(n1+1)/2)/(n1*n0))

r1=np.load(f"{ROOT}/gfmc_8x8_grscale_M128_seed10501.npz")["mixed"].astype(np.uint64)
r2=np.load(f"{ROOT}/gfmc_8x8_grscale_M128_seed10502.npz")["mixed"].astype(np.uint64)
q=np.load(f"{ROOT}/krylov_phys8_a2_fixedT.npz")["states"].astype(np.uint64)
# split by stored FN snapshot blocks: first half train, second half validation
h1=(len(r1)//256)*128; h2=(len(r2)//256)*128
rp=np.r_[r1[:h1],r2[:h2]]; rv=np.r_[r1[h1:],r2[h2:]]
rng=np.random.default_rng(20260930);perm=rng.permutation(len(q));qp=q[perm[:2048]];qv=q[perm[2048:]]
base_head=basep["output"]
brp=ev(base_head,rp);brv=ev(base_head,rv);bqp=ev(base_head,qp);bqv=ev(base_head,qv)
bias=jnp.array(0.0);tx=optax.adam(5e-5);st=tx.init((head,bias))
@jax.jit
def step(h,b,st,xp,xn,bp,bn):
 def lossfn(hb):
  hh,bb=hb;dp=la_h(hh,xp)-bp;dn=la_h(hh,xn)-bn
  zp=2*dp+bb;zn=2*dn+bb
  return .5*(jnp.mean(jax.nn.softplus(-zp))+jnp.mean(jax.nn.softplus(zn)))
 loss,g=jax.value_and_grad(lossfn)((h,b));up,st=tx.update(g,st,(h,b));nh,nb=optax.apply_updates((h,b),up)
 return nh,nb,st,loss
def met(h,b,rs,qs,br,bq):
 dr=ev(h,rs)-br;dq=ev(h,qs)-bq;y=np.r_[np.ones(len(dr)),np.zeros(len(dq))];sc=np.r_[2*dr+float(b),2*dq+float(b)]
 return dict(auc=auc(y,sc),gap=float(dr.mean()-dq.mean()),dr_sd=float(dr.std()),dq_sd=float(dq.std()),bias=float(b))

rows=[]
for it in range(1,121):
 ip=rng.integers(0,len(rp),128);iq=rng.integers(0,len(qp),128)
 head,bias,st,loss=step(head,bias,st,jnp.asarray(b2x(rp[ip])),jnp.asarray(b2x(qp[iq])),jnp.asarray(brp[ip]),jnp.asarray(bqp[iq]))
 if it in (1,5,10,20,40,80,120):
  tr=met(head,bias,rp,qp,brp,bqp);va=met(head,bias,rv,qv,brv,bqv)
  row={"step":it,"loss":float(loss),"train":tr,"val":va};rows.append(row)
  print("HEAD_GATE",json.dumps(row,sort_keys=True),flush=True)
pnew=dict(basep);pnew["output"]=head;vnew=dict(variables);vnew["params"]=pnew
open(OUTCK,'wb').write(serialization.msgpack_serialize(serialization.to_state_dict(vnew)))
json.dump({"rows":rows,"ntrain_fn":len(rp),"nval_fn":len(rv),"ntrain_q":len(qp),"nval_q":len(qv)},open(OUT,'w'),indent=2)
print("SAVED",OUTCK,OUT,flush=True)
