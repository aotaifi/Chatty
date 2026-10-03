import os
import numpy as np
import jax
import jax.numpy as jnp
import netket as nk
import flax
from flax import linen as nn, serialization
import netket.jax as nkjax
from nqsmagic.models import ViT
from nqsmagic.utils import _logpsi_transl_2d

L=8; N=64

class ResidualCNN(nn.Module):
    width:int=16
    @nn.compact
    def __call__(self,x):
        for _ in range(3):
            x=jnp.pad(x,((0,0),(1,1),(1,1),(0,0)),mode="wrap")
            x=nn.Conv(self.width,kernel_size=(3,3),padding="VALID")(x)
            x=nn.gelu(x)
        x=jnp.mean(x,axis=(1,2))
        x=nn.Dense(self.width)(x); x=nn.gelu(x)
        return nn.Dense(1)(x)[:,0]
def bits2spins(states):
    a=np.asarray(states,np.uint64).reshape(-1,1)
    x=2.0*(((a>>np.arange(N,dtype=np.uint64))&1).astype(np.float32))-1.0
    return x.reshape(-1,L,L,1)

class Backend:
    def __init__(self,residual_checkpoint):
        base=os.environ.get("BASE_VIT_CHECKPOINT","vit_J2=0.50_N=8x8_k=0.mpack")
        hi=nk.hilbert.Spin(s=.5,N=N)
        graph=nk.graph.Hypercube(length=L,n_dim=2,pbc=True,max_neighbor_order=2)
        model=ViT(num_layers=8,d_model=60,heads=10,L_eff=16,b=2,
                  transl_invariant=True,two_dimensional=True)
        apply=nkjax.HashablePartial(_logpsi_transl_2d,model.apply,2)
        sampler=nk.sampler.MetropolisExchange(hi,graph=graph,d_max=2,n_chains=16,sweep_size=N)
        self.v0=nk.vqs.MCState(sampler=sampler,apply_fun=apply,n_samples=16,
            variables=model.init(jax.random.PRNGKey(1234),jnp.zeros((1,N))),
            n_discard_per_chain=10)
        with open(base,"rb") as f: obj=flax.serialization.msgpack_restore(f.read())
        self.v0.variables=flax.serialization.from_state_dict(self.v0.variables,obj)
        with open(residual_checkpoint,"rb") as f:
            pay=serialization.msgpack_restore(f.read())
        self.eta=float(np.asarray(pay["eta"]))
        self.width=int(np.asarray(pay["width"]))
        self.res_model=ResidualCNN(self.width)
        self.res_params=jax.tree_util.tree_map(jnp.asarray,pay["params"])
        self.res_apply=jax.jit(lambda x: self.res_model.apply({"params":self.res_params},x))

    def log_amplitude_bits(self,states):
        states=np.asarray(states,np.uint64)
        X=bits2spins(states)
        flat=X.reshape(len(X),N)
        out=[]
        for i in range(0,len(X),4096):
            base=np.asarray(self.v0.log_value(jnp.asarray(flat[i:i+4096]))).real
            g=np.asarray(self.res_apply(jnp.asarray(X[i:i+4096])))
            out.append(base+self.eta*g)
        return np.concatenate(out)

def load(checkpoint):
    if not checkpoint: raise ValueError("residual checkpoint path required")
    return Backend(checkpoint)
