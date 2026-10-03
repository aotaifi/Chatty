import json
import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn, serialization
import optax
from scipy.optimize import minimize_scalar

jax.config.update("jax_enable_x64", False)

L=8; N=64; WIDTH=16
ROOT="/Users/aliotaifi/Chatty/projects/j1j2/results/8x8_krylov_3471544"
OUT="/Users/aliotaifi/Chatty/projects/j1j2/results"

def bits_to_spins(states):
    a=np.asarray(states,np.uint64).reshape(-1,1)
    x=2.0*(((a>>np.arange(N,dtype=np.uint64))&1).astype(np.float32))-1.0
    return x.reshape(-1,L,L,1)

def auc_score(y,score):
    y=np.asarray(y,np.int8); score=np.asarray(score,float)
    order=np.argsort(score,kind="mergesort")
    ranks=np.empty(len(score),float); ranks[order]=np.arange(1,len(score)+1)
    n1=int(y.sum()); n0=len(y)-n1
    return float((ranks[y==1].sum()-n1*(n1+1)/2.0)/(n1*n0))
class ResidualCNN(nn.Module):
    width:int=WIDTH
    @nn.compact
    def __call__(self,x):
        for _ in range(3):
            x=jnp.pad(x,((0,0),(1,1),(1,1),(0,0)),mode="wrap")
            x=nn.Conv(self.width,kernel_size=(3,3),padding="VALID")(x)
            x=nn.gelu(x)
        x=jnp.mean(x,axis=(1,2))
        x=nn.Dense(self.width)(x); x=nn.gelu(x)
        return nn.Dense(1)(x)[:,0]

def load_data():
    r1=np.load(f"{ROOT}/gfmc_8x8_grscale_M128_seed10501.npz")["mixed"]
    r2=np.load(f"{ROOT}/gfmc_8x8_grscale_M128_seed10502.npz")["mixed"]
    q=np.load(f"{ROOT}/krylov_phys8_a2_fixedT.npz")["states"]
    rng=np.random.default_rng(20260930); qi=rng.permutation(len(q))
    return r1,r2,q[qi[:len(q)//2]],q[qi[len(q)//2:]],q
def train_model(f,q,steps=400,batch=256,lr=2e-3,wd=1e-4,seed=0):
    xf=jnp.asarray(bits_to_spins(f)); xq=jnp.asarray(bits_to_spins(q))
    model=ResidualCNN(WIDTH)
    params=model.init(jax.random.PRNGKey(seed),xf[:2])["params"]
    tx=optax.adamw(lr,weight_decay=wd); state=tx.init(params)
    rng=np.random.default_rng(seed+54321)
    def loss_fn(p,bf,bq):
        lf=model.apply({"params":p},bf); lq=model.apply({"params":p},bq)
        return .5*(jnp.mean(jax.nn.softplus(-lf))+jnp.mean(jax.nn.softplus(lq)))
    @jax.jit
    def step(p,s,bf,bq):
        loss,g=jax.value_and_grad(loss_fn)(p,bf,bq)
        u,s=tx.update(g,s,p)
        return optax.apply_updates(p,u),s,loss
    for i in range(steps):
        fi=rng.integers(0,len(xf),size=batch); qi=rng.integers(0,len(xq),size=batch)
        params,state,loss=step(params,state,xf[fi],xq[qi])
    return model,params,float(loss)

def logits(model,params,states,batch=4096):
    x=bits_to_spins(states); out=[]
    for i in range(0,len(x),batch):
        out.append(np.asarray(model.apply({"params":params},jnp.asarray(x[i:i+batch]))))
    return np.concatenate(out)
def calibrate(train_f,train_q,test_f,test_q,seed):
    model,p,loss=train_model(train_f,train_q,seed=seed)
    lf=logits(model,p,test_f); lq=logits(model,p,test_q)
    y=np.concatenate([np.ones(len(lf)),np.zeros(len(lq))])
    score=np.concatenate([lf,lq]); auc=auc_score(y,score)
    def ce(eta):
        return float(.5*(np.mean(np.logaddexp(0,-eta*lf))+np.mean(np.logaddexp(0,eta*lq))))
    opt=minimize_scalar(ce,bounds=(0.0,2.0),method="bounded")
    return dict(auc=float(auc),eta=float(opt.x),ce1=ce(1.0),ce_cal=float(opt.fun),
                train_loss=float(loss),
                lf_q=np.quantile(lf,[.01,.1,.5,.9,.99]).tolist(),
                lq_q=np.quantile(lq,[.01,.1,.5,.9,.99]).tolist())

def main():
    r1,r2,q1,q2,q=load_data()
    c12=calibrate(r1,q1,r2,q2,11); print("CAL12",c12,flush=True)
    c21=calibrate(r2,q2,r1,q1,22); print("CAL21",c21,flush=True)
    eta=float(np.clip(.5*(c12["eta"]+c21["eta"]),0.0,1.0))
    f=np.concatenate([r1,r2])
    model,params,loss=train_model(f,q,steps=500,seed=33)
    raw_f=logits(model,params,f); raw_q=logits(model,params,q)
    meta={"width":WIDTH,"eta":eta,"cross_12":c12,"cross_21":c21,
          "pooled_train_loss":loss,
          "raw_f_quantiles":np.quantile(raw_f,[.01,.1,.5,.9,.99]).tolist(),
          "raw_q_quantiles":np.quantile(raw_q,[.01,.1,.5,.9,.99]).tolist()}
    payload={"params":jax.tree_util.tree_map(np.asarray,params),
             "width":np.asarray(WIDTH),"eta":np.asarray(eta,dtype=np.float32)}
    with open(f"{OUT}/fn_residual_cnn_8x8.mpack","wb") as fp:
        fp.write(serialization.msgpack_serialize(payload))
    with open(f"{OUT}/fn_residual_cnn_8x8.json","w") as fp:
        json.dump(meta,fp,indent=2)
    print("FINAL_MODEL eta",eta,"loss",loss,
          "fq",meta["raw_f_quantiles"],"qq",meta["raw_q_quantiles"],flush=True)

if __name__=="__main__":
    main()
