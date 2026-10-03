import os,time,json,math
import numpy as np
import jax,jax.numpy as jnp
import netket as nk,flax,optax
import netket.jax as nkjax
from nqsmagic.models import ViT
from nqsmagic.utils import _logpsi_transl_2d

L=8;N=64
BATCH=int(os.environ.get("MLE_BATCH","512"))
STEPS=int(os.environ.get("MLE_STEPS","160"))
LR=float(os.environ.get("MLE_LR","2e-5"))
SEED=int(os.environ.get("MLE_SEED","20260930"))
EVAL_EVERY=int(os.environ.get("MLE_EVAL_EVERY","10"))
BASE_CK=os.environ.get("BASE_CK","vit_J2=0.50_N=8x8_k=0.mpack")
REP1=os.environ.get("REP1","gfmc_8x8_grscale_M128_seed10501.npz")
REP2=os.environ.get("REP2","gfmc_8x8_grscale_M128_seed10502.npz")
GUIDE_POOL=os.environ.get("GUIDE_POOL","krylov_phys8_a2_fixedT.npz")
OUT_CK=os.environ.get("OUT_CK","vit8_fnmle_rep1_best.mpack")
OUT_JSON=os.environ.get("OUT_JSON","vit8_fnmle_rep1_metrics.json")

def bits2x(ss):
    a=np.asarray(ss,np.uint64).reshape(-1,1)
    return 2*((a>>np.arange(N,dtype=np.uint64))&1).astype(np.float32)-1

hi=nk.hilbert.Spin(s=.5,N=N)
graph=nk.graph.Hypercube(length=L,n_dim=2,pbc=True,max_neighbor_order=2)
model=ViT(num_layers=8,d_model=60,heads=10,L_eff=16,b=2,transl_invariant=True,two_dimensional=True)
apply=nkjax.HashablePartial(_logpsi_transl_2d,model.apply,2)
sampler=nk.sampler.MetropolisExchange(hi,graph=graph,d_max=2,n_chains=16,sweep_size=N)
v=nk.vqs.MCState(sampler=sampler,apply_fun=apply,n_samples=BATCH,
    variables=model.init(jax.random.PRNGKey(1234),jnp.zeros((1,N))),n_discard_per_chain=10)
with open(BASE_CK,"rb") as f: obj=flax.serialization.msgpack_restore(f.read())
v.variables=flax.serialization.from_state_dict(v.variables,obj)
params=v.parameters
train_bits=np.load(REP1)["mixed"].astype(np.uint64)
val_bits=np.load(REP2)["mixed"].astype(np.uint64)
guide_bits=np.load(GUIDE_POOL)["states"].astype(np.uint64)
trainX=bits2x(train_bits);valX=bits2x(val_bits);guideX=bits2x(guide_bits)
rng=np.random.default_rng(SEED)

def eval_real(p,X,batch=4096):
    out=[]
    vars={"params":p}
    for i in range(0,len(X),batch):
        out.append(np.asarray(apply(vars,jnp.asarray(X[i:i+batch]))).real)
    return np.concatenate(out)

log0_train=eval_real(params,trainX)
log0_val=eval_real(params,valX)
log0_guide=eval_real(params,guideX)

def logmeanexp(x):
    x=np.asarray(x,float);m=float(np.max(x))
    return m+math.log(float(np.mean(np.exp(x-m))))

def metrics(p,step):
    lt=eval_real(p,trainX);lv=eval_real(p,valX);lq=eval_real(p,guideX)
    dt=lt-log0_train;dv=lv-log0_val;dq=lq-log0_guide
    logz=logmeanexp(2*dq)
    gain_t=float(2*np.mean(dt)-logz)
    gain_v=float(2*np.mean(dv)-logz)
    ww=np.exp(2*dq-np.max(2*dq))
    ess=float((ww.sum()**2)/(ww@ww))
    return dict(step=int(step),train_ll_gain=gain_t,val_ll_gain=gain_v,
        logZ_ratio=float(logz),guide_IS_ESS=ess,
        delta_logamp_rms=float(np.std(dq)),
        delta_logamp_q=np.quantile(dq,[.01,.1,.5,.9,.99]).tolist())
def loss_fn(p,pos,neg):
    vars={"params":p}
    lp=jnp.real(apply(vars,pos))
    ln=jnp.real(apply(vars,neg))
    # Stochastic gradient of KL(f_target || p_theta), with negative samples
    # treated as fixed samples from current p_theta.
    return 2.0*(jnp.mean(ln)-jnp.mean(lp))

loss_grad=jax.jit(jax.value_and_grad(loss_fn))
tx=optax.chain(optax.clip_by_global_norm(1.0),optax.adamw(LR,weight_decay=1e-6))
ost=tx.init(params)

def save_ck(p,path):
    state=flax.serialization.to_state_dict({"params":p})
    with open(path,"wb") as f:f.write(flax.serialization.msgpack_serialize(state))

history=[]
m0=metrics(params,0);history.append(m0)
best_gain=m0["val_ll_gain"];best_step=0
save_ck(params,OUT_CK)
print("MLE8_EVAL",json.dumps(m0),flush=True)
t0=time.time()
for step in range(1,STEPS+1):
    ids=rng.integers(0,len(trainX),size=BATCH)
    pos=jnp.asarray(trainX[ids])
    v.variables={"params":params}
    neg=np.asarray(v.sample()).reshape(-1,N).astype(np.float32)
    if len(neg)!=BATCH:
        if len(neg)>BATCH: neg=neg[:BATCH]
        else:
            jj=rng.integers(0,len(neg),size=BATCH);neg=neg[jj]
    loss,gr=loss_grad(params,pos,jnp.asarray(neg))
    upd,ost=tx.update(gr,ost,params)
    params=optax.apply_updates(params,upd)
    if step==1 or step%EVAL_EVERY==0 or step==STEPS:
        mm=metrics(params,step);mm["stoch_loss"]=float(loss);mm["sec"]=time.time()-t0
        history.append(mm);print("MLE8_EVAL",json.dumps(mm),flush=True)
        if mm["val_ll_gain"]>best_gain:
            best_gain=mm["val_ll_gain"];best_step=step;save_ck(params,OUT_CK)
        # Stop if held-out likelihood has clearly rolled over after a useful update.
        if step>=40 and mm["val_ll_gain"] < best_gain-0.01:
            print("MLE8_EARLY_STOP",step,best_step,best_gain,flush=True);break

summary={"best_step":int(best_step),"best_val_ll_gain":float(best_gain),
         "n_train":int(len(train_bits)),"n_val":int(len(val_bits)),
         "n_guide":int(len(guide_bits)),"batch":BATCH,"lr":LR,"history":history}
json.dump(summary,open(OUT_JSON,"w"),indent=2)
print("MLE8_RESULT",json.dumps({k:v for k,v in summary.items() if k!="history"}),flush=True)
