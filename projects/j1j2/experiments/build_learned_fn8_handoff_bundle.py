import sys, numpy as np, jax, jax.numpy as jnp
from flax import linen as nn, serialization
L=8;N=64
ROOT="/Users/aliotaifi/Chatty/projects/j1j2/results/8x8_krylov_3471544"
CK="/Users/aliotaifi/Chatty/projects/j1j2/results/fn_residual_cnn_8x8.mpack"
OUT="/Users/aliotaifi/Chatty/projects/j1j2/results/learned_fn_handoff_8x8.npz"
class ResidualCNN(nn.Module):
    width:int=16
    @nn.compact
    def __call__(self,x):
        for _ in range(3):
            x=jnp.pad(x,((0,0),(1,1),(1,1),(0,0)),mode="wrap")
            x=nn.Conv(self.width,(3,3),padding="VALID")(x); x=nn.gelu(x)
        x=jnp.mean(x,axis=(1,2)); x=nn.Dense(self.width)(x); x=nn.gelu(x)
        return nn.Dense(1)(x)[:,0]
def b2x(s):
    a=np.asarray(s,np.uint64).reshape(-1,1)
    x=2*(((a>>np.arange(N,dtype=np.uint64))&1).astype(np.float32))-1
    return x.reshape(-1,L,L,1)
with open(CK,"rb") as f: pay=serialization.msgpack_restore(f.read())
eta=float(np.asarray(pay["eta"])); width=int(np.asarray(pay["width"]))
p=jax.tree_util.tree_map(jnp.asarray,pay["params"]); m=ResidualCNN(width)
def pred(s):
    x=b2x(s); out=[]
    for i in range(0,len(x),4096):
        out.append(np.asarray(m.apply({"params":p},jnp.asarray(x[i:i+4096]))))
    return np.concatenate(out)
z=np.load(f"{ROOT}/krylov_scaling_8x8_a1.20_tr4096_va2048.npz")
pool=np.load(f"{ROOT}/krylov_phys8_a2_fixedT.npz")["states"].astype(np.uint64)
gt=pred(z["train_states"]); gv=pred(z["val_states"]); gp=pred(pool)
wt=z["iwtrain"].astype(float)*np.exp(2*eta*gt); wt/=wt.sum()
wv=z["iwval"].astype(float)*np.exp(2*eta*gv); wv/=wv.sum()
wp=np.exp(2*eta*gp); wp/=wp.sum()
nt=len(z["train_states"]); nv=len(z["val_states"]); frac=nt/(nt+nv)
th_states=np.concatenate([z["train_states"],z["val_states"]])
th_weights=np.concatenate([frac*wt,(1-frac)*wv]); th_weights/=th_weights.sum()
np.savez_compressed(OUT,train_states=z["train_states"],train_weights=wt,
    val_states=z["val_states"],val_weights=wv,threshold_states=th_states,
    threshold_weights=th_weights,pool_states=pool,pool_weights=wp,
    train_g=gt,val_g=gv,pool_g=gp,eta=eta)
print("BUNDLE",OUT,"eta",eta,"ESS_train",1/(wt@wt),"ESS_val",1/(wv@wv),
      "ESS_threshold",1/(th_weights@th_weights),"ESS_pool",1/(wp@wp),
      "gq",np.quantile(gt,[.01,.1,.5,.9,.99]).tolist())
