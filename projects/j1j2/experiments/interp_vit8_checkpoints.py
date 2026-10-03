import os
os.environ["JAX_PLATFORMS"]="cpu"
import jax, jax.numpy as jnp, flax
from nqsmagic.models import ViT
L=8;N=64
model=ViT(num_layers=8,d_model=60,heads=10,L_eff=16,b=2,transl_invariant=True,two_dimensional=True)
t=model.init(jax.random.PRNGKey(1234),jnp.zeros((1,N)))
def load(p):
 obj=flax.serialization.msgpack_restore(open(p,'rb').read())
 return flax.serialization.from_state_dict(t,obj)
v1=load('a1.mpack'); v2=load('a2.mpack')
for eta in (0.0003,0.001):
 f=eta/0.003
 p=jax.tree_util.tree_map(lambda x,y:x+f*(y-x),v1['params'],v2['params'])
 v=dict(v1);v['params']=p
 name=f'a_eta{eta:.4f}.mpack'.replace('.','p')
 # fix extension
 name=name.replace('mpack','')+'mpack'
 open(name,'wb').write(flax.serialization.to_bytes(v))
 print(eta,name)
