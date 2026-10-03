import numpy as np
import jax, jax.numpy as jnp
from flax import linen as nn
import optax

L=6; N=36
D=np.load("global_rayleigh_1024.npz")
Xb=D["Xb"].astype(np.uint64); Yb=D["Yb"].astype(np.uint64)
C=D["C"].astype(np.float32); diag=D["diag"].astype(np.float32)
hx=D["hx"].astype(np.int8); hy=D["hy"].astype(np.int8)
bits=np.arange(N,dtype=np.uint64)
X=(2*((Xb[:,None]>>bits[None,:])&1)-1).astype(np.float32)
Y=(2*((Yb[:,:,None]>>bits[None,None,:])&1)-1).astype(np.float32)

Aidx=np.array([x+L*y for y in range(L) for x in range(L) if (x+y)%2==0],dtype=np.int32)

def marshall_np(x):
    q=(x[:,Aidx]>0).sum(axis=1)%2
    return np.where(q==0,1.0,-1.0).astype(np.float32)

class SignCNN(nn.Module):
    @nn.compact
    def __call__(self,x):
        z=x.reshape((-1,L,L,1))
        for ch in (16,16):
            z=jnp.pad(z,((0,0),(1,1),(1,1),(0,0)),mode="wrap")
            z=nn.Conv(ch,(3,3),padding="VALID")(z)
            z=nn.tanh(z)
        z=jnp.mean(z,axis=(1,2))
        z=nn.tanh(nn.Dense(16)(z))
        g=nn.Dense(1,
            kernel_init=nn.initializers.normal(0.01),
            bias_init=nn.initializers.constant(0.5))(z).squeeze(-1)
        soft=jnp.tanh(g)
        hard=jnp.where(soft>=0,1.0,-1.0)
        corr=soft+jax.lax.stop_gradient(hard-soft)
        nup=jnp.sum(x[:,Aidx]>0,axis=1)
        m=jnp.where((nup%2)==0,1.0,-1.0)
        return m*corr, hard

model=SignCNN()
params=model.init(jax.random.PRNGKey(1),jnp.asarray(X[:2]))
tx=optax.adam(3e-3)
state=tx.init(params)

@jax.jit
def step(params,state,xb,yb,cb):
    def lossfn(p):
        fx,_=model.apply(p,xb)
        fy,_=model.apply(p,yb.reshape((-1,N)))
        fy=fy.reshape((xb.shape[0],-1))
        # sign-dependent off-diagonal Rayleigh numerator; f is hard +/-1 forward.
        loc=jnp.sum(cb*fy,axis=1)
        return jnp.mean(fx*loc)
    loss,gr=jax.value_and_grad(lossfn)(params)
    upd,state=tx.update(gr,state,params)
    params=optax.apply_updates(params,upd)
    return params,state,loss

@jax.jit
def predict(params,x):
    return model.apply(params,x)[0]

rng=np.random.default_rng(20260927)
perm=rng.permutation(len(X)); train=perm[:768]; val=perm[768:]

def metrics(params,ids):
    fx=np.asarray(predict(params,jnp.asarray(X[ids])))
    fy=np.asarray(predict(params,jnp.asarray(Y[ids].reshape((-1,N))))).reshape((len(ids),-1))
    E=float(np.mean(diag[ids])+np.mean(fx*np.sum(C[ids]*fy,axis=1)))
    ov=float(abs(np.mean(np.sign(fx)*hx[ids])))
    corr=np.sign(fx)*marshall_np(X[ids])
    frac=float(np.mean(corr<0))
    return E,ov,frac

def oracle_energy(ids):
    return float(np.mean(diag[ids])+np.mean(hx[ids]*np.sum(C[ids]*hy[ids],axis=1)))

print("ORACLE_E_VAL",oracle_energy(val),flush=True)
e0,o0,f0=metrics(params,val)
print("INIT_MARSHALL","E",e0,"O",o0,"corrneg",f0,flush=True)

bs=16
for t in range(801):
    ids=rng.choice(train,size=bs,replace=False)
    params,state,loss=step(params,state,jnp.asarray(X[ids]),
        jnp.asarray(Y[ids]),jnp.asarray(C[ids]))
    if t%50==0:
        et,ot,ft=metrics(params,train[:256])
        ev,ov,fv=metrics(params,val)
        print("STEP",t,"loss",float(loss),
              "trainE",et,"trainO",ot,"valE",ev,"valO",ov,
              "val_corrneg",fv,flush=True)

# save params
from flax.serialization import to_bytes
open("global_sign_cnn.msgpack","wb").write(to_bytes(params))
print("SAVED global_sign_cnn.msgpack",flush=True)
