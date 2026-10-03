#!/usr/bin/env python3
import json, numpy as np, scipy.sparse as sp, jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp, optax
from nqsmagic.models import ViT
from nqsmagic.utils import _logpsi_transl_2d
from closed_fn_krylov_exact4x4 import build_H,ground,marshall_signs,canonical,basis,build_fixed_node,fixed_node_solve,projected_krylov_update
L=4;N=16;B=2
def bits2x(ss):
 a=np.asarray(ss,np.uint32).reshape(-1,1); return 2.0*(((a>>np.arange(N,dtype=np.uint32))&1).astype(float))-1
def snap(k):
 H,d=build_H(.5); H0,_=build_H(0.); s=canonical(marshall_signs()); _,p=ground(H0)
 if np.dot(p,s)<0:p=-p
 a=np.abs(p);a/=np.linalg.norm(a)
 for _ in range(k):
  _,ph=fixed_node_solve(H,d,a,s); s,_,_,_,_=projected_krylov_update(H,d,ph,s); a=ph
 return H,d,a,s
def E(F,b): return float(b@(F@b)/(b@b))
H,d,a,s=snap(10);F=build_fixed_node(H,d,a,s);Fb=F@a
e1=float(a@Fb);e2=float(Fb@Fb);e3=float(Fb@(F@Fb));sig=e2-e1*e1;A=e2*e2-e1*e3;Bc=e1*e2-e3
disc=max(Bc*Bc+4*A*sig,0); roots=[(Bc+np.sqrt(disc))/(2*A),(Bc-np.sqrt(disc))/(2*A)]
roots=[x for x in roots if np.isfinite(x) and x>0]
dt=min(roots,key=lambda x:E(F,a-x*Fb)); t=a-dt*Fb;t/=np.linalg.norm(t)
X=jnp.asarray(bits2x(basis));aj=jnp.asarray(a)
model=ViT(num_layers=4,d_model=60,heads=10,L_eff=4,b=2,transl_invariant=True,two_dimensional=True)
p0=model.init(jax.random.PRNGKey(20261016),X[:2])["params"]
def fn(p): return jnp.real(_logpsi_transl_2d(model.apply,B,{"params":p},X))
f0=jax.lax.stop_gradient(fn(p0))
def amp(p):
 r=fn(p)-f0; r=r-jnp.sum((aj*aj)*r); z=jnp.log(aj)+r; z-=.5*jax.scipy.special.logsumexp(2*z); return jnp.exp(z)
tj=jnp.asarray(t)
def loss(p):
 b=amp(p); ov=jnp.dot(b,tj); return -jnp.log(jnp.maximum(ov*ov/(jnp.dot(b,b)*jnp.dot(tj,tj)),1e-300))
vg=jax.jit(jax.value_and_grad(loss))
for lr in [1e-3,2e-4,5e-5,1e-5]:
 p=p0; st=optax.adam(lr).init(p); rows=[]
 for k in range(1,21):
  l,g=vg(p);u,st=optax.adam(lr).update(g,st,p);p=optax.apply_updates(p,u)
  b=np.asarray(amp(p),float); rows.append([k,float(l),E(F,b),float((b@t)**2)])
 print("TRACE",json.dumps({"lr":lr,"E0":E(F,a),"Et":E(F,t),"dt":dt,"rows":rows}),flush=True)
