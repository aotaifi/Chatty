import os, json, math
import numpy as np
import jax, jax.numpy as jnp
import flax
from flax import serialization
import optax
from nqsmagic.models import ViT
from nqsmagic.utils import _logpsi_transl_2d

jax.config.update("jax_enable_x64", True)
L=8; N=64; J2=.5; B=2
ROOT="/Users/aliotaifi/Chatty/projects/j1j2/results/8x8_krylov_3471544"
CK="/Users/aliotaifi/j1j2_vit_bench/vit_J2=0.50_N=8x8_k=0.mpack"
OUT="/Users/aliotaifi/Chatty/projects/j1j2/results/warmstart_vit_mle_gate_8x8.json"
OUTCK="/Users/aliotaifi/Chatty/projects/j1j2/results/warmstart_vit_mle_rep1_8x8.mpack"

model=ViT(num_layers=8,d_model=60,heads=10,L_eff=16,b=2,transl_invariant=True,two_dimensional=True)
template=model.init(jax.random.PRNGKey(1234),jnp.zeros((1,N)))
with open(CK,"rb") as f: obj=serialization.msgpack_restore(f.read())
variables=serialization.from_state_dict(template,obj)
params=variables["params"]

def bits2x(ss):
    a=np.asarray(ss,np.uint64).reshape(-1,1)
    return 2.0*(((a>>np.arange(N,dtype=np.uint64))&1).astype(np.float64))-1.0

def logamp(p,x):
    return jnp.real(_logpsi_transl_2d(model.apply,B,{"params":p},x))

logamp_jit=jax.jit(logamp)

def eval_logamp(p,states,batch=1024):
    X=bits2x(states); out=[]
    for i in range(0,len(X),batch):
        out.append(np.asarray(logamp_jit(p,jnp.asarray(X[i:i+batch]))))
    return np.concatenate(out)

def bonds():
    nn=[]; nnn=[]; q=lambda x,y:(x%L)+L*(y%L)
    for y in range(L):
        for x in range(L):
            i=q(x,y)
            nn += [(i,q(x+1,y)),(i,q(x,y+1))]
            nnn += [(i,q(x+1,y+1)),(i,q(x+1,y-1))]
    return nn,nnn
NN,NNN=bonds(); ALL=NN+NNN

def mh_round(p,ss,la,rng,nround=4):
    ss=list(map(int,ss))
    for _ in range(nround):
        cand=[]; valid=[]
        for q in ss:
            i,j=ALL[int(rng.integers(len(ALL)))]
            ok=((q>>i)^(q>>j))&1
            cand.append(q^(1<<i)^(1<<j) if ok else q); valid.append(bool(ok))
        lb=eval_logamp(p,np.asarray(cand,np.uint64),batch=1024)
        ac=(np.log(rng.random(len(ss))) < np.minimum(0.0,2.0*(lb-la))) & np.asarray(valid)
        for k in np.where(ac)[0]: ss[k]=cand[k]
        la=np.where(ac,lb,la)
    return np.asarray(ss,np.uint64),la

def auc(y,s):
    y=np.asarray(y,int);s=np.asarray(s,float)
    order=np.argsort(s,kind="mergesort");r=np.empty(len(s),float);r[order]=np.arange(1,len(s)+1)
    n1=y.sum();n0=len(y)-n1
    return float((r[y==1].sum()-n1*(n1+1)/2)/(n1*n0))

def gate_metrics(p,rep2,qval,base_rep2,base_q):
    nr=eval_logamp(p,rep2); nq=eval_logamp(p,qval)
    dr=nr-base_rep2; dq=nq-base_q
    y=np.r_[np.ones(len(dr)),np.zeros(len(dq))]
    sc=np.r_[dr,dq]
    return {
      "auc_delta_rep2_vs_q":auc(y,sc),
      "delta_rep2_q":np.quantile(dr,[.01,.1,.5,.9,.99]).tolist(),
      "delta_q_q":np.quantile(dq,[.01,.1,.5,.9,.99]).tolist(),
      "delta_gap_mean":float(dr.mean()-dq.mean()),
    }

r1=np.load(f"{ROOT}/gfmc_8x8_grscale_M128_seed10501.npz")["mixed"].astype(np.uint64)
r2=np.load(f"{ROOT}/gfmc_8x8_grscale_M128_seed10502.npz")["mixed"].astype(np.uint64)
q=np.load(f"{ROOT}/krylov_phys8_a2_fixedT.npz")["states"].astype(np.uint64)
rng=np.random.default_rng(20260930)
perm=rng.permutation(len(q)); qtrain=q[perm[:2048]]; qval=q[perm[2048:]]
base_r2=eval_logamp(params,r2); base_qval=eval_logamp(params,qval)

nchains=128
neg=qtrain[rng.choice(len(qtrain),nchains,replace=False)].copy()
neg_la=eval_logamp(params,neg)
tx=optax.adam(1e-4)
ost=tx.init(params)

@jax.jit
def step(p,st,xpos,xneg):
    def lossfn(pp):
        # score-function MLE surrogate; samples are stop-gradient draws
        lp=logamp(pp,xpos); ln=logamp(pp,xneg)
        return 2.0*(jnp.mean(ln)-jnp.mean(lp))
    loss,g=jax.value_and_grad(lossfn)(p)
    up,st=tx.update(g,st,p)
    return optax.apply_updates(p,up),st,loss

rows=[]
for it in range(1,41):
    neg,neg_la=mh_round(params,neg,neg_la,rng,nround=4)
    pos=r1[rng.integers(0,len(r1),size=128)]
    params,ost,loss=step(params,ost,jnp.asarray(bits2x(pos)),jnp.asarray(bits2x(neg)))
    # refresh log amplitudes after parameter move for persistent chains
    neg_la=eval_logamp(params,neg)
    if it in (1,5,10,20,40):
        m=gate_metrics(params,r2,qval,base_r2,base_qval)
        m.update(step=it,loss=float(loss))
        rows.append(m)
        print("MLE_GATE",json.dumps(m,sort_keys=True),flush=True)

vars_new=variables.copy({"params":params})
state=serialization.to_state_dict(vars_new)
with open(OUTCK,"wb") as f: f.write(serialization.msgpack_serialize(state))
with open(OUT,"w") as f: json.dump({"rows":rows,"train_rep1":len(r1),"val_rep2":len(r2),
    "qtrain":len(qtrain),"qval":len(qval),"batch":128,"mh_rounds_per_step":4,"lr":1e-4},f,indent=2)
print("SAVED",OUTCK,OUT,flush=True)
