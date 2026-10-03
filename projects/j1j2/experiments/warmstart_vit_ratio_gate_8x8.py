import json
import numpy as np
import jax, jax.numpy as jnp
from flax import serialization
import optax
from nqsmagic.models import ViT
from nqsmagic.utils import _logpsi_transl_2d

jax.config.update("jax_enable_x64", True)
L=8; N=64; B=2
ROOT="/Users/aliotaifi/Chatty/projects/j1j2/results/8x8_krylov_3471544"
CK="/Users/aliotaifi/j1j2_vit_bench/vit_J2=0.50_N=8x8_k=0.mpack"
OUT="/Users/aliotaifi/Chatty/projects/j1j2/results/warmstart_vit_ratio_gate_8x8.json"
OUTCK="/Users/aliotaifi/Chatty/projects/j1j2/results/warmstart_vit_ratio_rep1_8x8.mpack"

model=ViT(num_layers=8,d_model=60,heads=10,L_eff=16,b=2,transl_invariant=True,two_dimensional=True)
template=model.init(jax.random.PRNGKey(1234),jnp.zeros((1,N)))
with open(CK,"rb") as f: obj=serialization.msgpack_restore(f.read())
variables=serialization.from_state_dict(template,obj)
params0=variables["params"]; params=params0

def bits2x(ss):
    a=np.asarray(ss,np.uint64).reshape(-1,1)
    return 2.0*(((a>>np.arange(N,dtype=np.uint64))&1).astype(np.float64))-1.0

def logamp(p,x):
    return jnp.real(_logpsi_transl_2d(model.apply,B,{"params":p},x))
logamp_jit=jax.jit(logamp)

def eval_logamp(p,s,batch=1024):
    X=bits2x(s); out=[]
    for i in range(0,len(X),batch):
        out.append(np.asarray(logamp_jit(p,jnp.asarray(X[i:i+batch]))))
    return np.concatenate(out)

def auc(y,s):
    y=np.asarray(y,int);s=np.asarray(s,float)
    order=np.argsort(s,kind="mergesort");r=np.empty(len(s),float);r[order]=np.arange(1,len(s)+1)
    n1=y.sum();n0=len(y)-n1
    return float((r[y==1].sum()-n1*(n1+1)/2)/(n1*n0))

r1=np.load(f"{ROOT}/gfmc_8x8_grscale_M128_seed10501.npz")["mixed"].astype(np.uint64)
r2=np.load(f"{ROOT}/gfmc_8x8_grscale_M128_seed10502.npz")["mixed"].astype(np.uint64)
q=np.load(f"{ROOT}/krylov_phys8_a2_fixedT.npz")["states"].astype(np.uint64)
rng=np.random.default_rng(20260930)
perm=rng.permutation(len(q)); qtrain=q[perm[:2048]]; qval=q[perm[2048:]]

base_r1=eval_logamp(params0,r1); base_r2=eval_logamp(params0,r2)
base_qt=eval_logamp(params0,qtrain); base_qv=eval_logamp(params0,qval)

bias=jnp.array(0.0)
tx=optax.adam(2e-5)
ost=tx.init((params,bias))

@jax.jit
def step(p,b,st,xp,xn,bp,bn):
    def lossfn(pb):
        pp,bb=pb
        dp=logamp(pp,xp)-bp
        dn=logamp(pp,xn)-bn
        zp=2.0*dp+bb; zn=2.0*dn+bb
        loss=.5*(jnp.mean(jax.nn.softplus(-zp))+jnp.mean(jax.nn.softplus(zn)))
        return loss
    loss,g=jax.value_and_grad(lossfn)((p,b))
    up,st=tx.update(g,st,(p,b))
    npb=optax.apply_updates((p,b),up)
    return npb[0],npb[1],st,loss

def metrics(p,b):
    nr=eval_logamp(p,r2); nq=eval_logamp(p,qval)
    dr=nr-base_r2; dq=nq-base_qv
    score=np.r_[2*dr+float(b),2*dq+float(b)]
    y=np.r_[np.ones(len(dr)),np.zeros(len(dq))]
    return dict(
      auc=auc(y,score),
      gap=float(dr.mean()-dq.mean()),
      bias=float(b),
      dr_q=np.quantile(dr,[.01,.1,.5,.9,.99]).tolist(),
      dq_q=np.quantile(dq,[.01,.1,.5,.9,.99]).tolist())

rows=[]
for it in range(1,81):
    ip=rng.integers(0,len(r1),size=128)
    iq=rng.integers(0,len(qtrain),size=128)
    xp=jnp.asarray(bits2x(r1[ip])); xn=jnp.asarray(bits2x(qtrain[iq]))
    bp=jnp.asarray(base_r1[ip]); bn=jnp.asarray(base_qt[iq])
    params,bias,ost,loss=step(params,bias,ost,xp,xn,bp,bn)
    if it in (1,5,10,20,40,80):
        m=metrics(params,bias);m.update(step=it,loss=float(loss))
        rows.append(m); print("VIT_RATIO_GATE",json.dumps(m,sort_keys=True),flush=True)

vars_new=dict(variables);vars_new["params"]=params
state=serialization.to_state_dict(vars_new)
with open(OUTCK,"wb") as f:f.write(serialization.msgpack_serialize(state))
with open(OUT,"w") as f:json.dump({"rows":rows,"lr":2e-5,"batch":128,
   "train_rep1":len(r1),"val_rep2":len(r2),"qtrain":len(qtrain),"qval":len(qval)},f,indent=2)
print("SAVED",OUTCK,OUT,flush=True)
