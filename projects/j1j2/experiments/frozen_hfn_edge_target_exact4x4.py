#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import jax
jax.config.update("jax_enable_x64",True)
import jax.numpy as jnp
import optax
from nqsmagic.models import ViT
from nqsmagic.utils import _logpsi_transl_2d
from closed_fn_krylov_exact4x4 import build_H,ground,marshall_signs,canonical,basis,build_fixed_node,fixed_node_solve,projected_krylov_update

N=16; B=2
def bits2x(ss):
 a=np.asarray(ss,np.uint32).reshape(-1,1)
 return 2.0*(((a>>np.arange(N,dtype=np.uint32))&1).astype(float))-1.0

def snapshot(it):
 H,d=build_H(.5); H0,_=build_H(0.0); s=canonical(marshall_signs())
 _,p=ground(H0)
 if np.dot(p,s)<0:p=-p
 a=np.abs(p); a/=np.linalg.norm(a)
 for _ in range(it):
  _,phi=fixed_node_solve(H,d,a,s)
  s,_,_,_,_=projected_krylov_update(H,d,phi,s); a=phi
 return H,d,a,s

def en(M,b): return float(b@(M@b)/(b@b))
def target_step(F,b):
 Fb=F@b; F2b=F@Fb; e1=float(b@Fb); e2=float(Fb@Fb); e3=float(Fb@F2b)
 A=e2*e2-e1*e3; C=e1*e2-e3; sig=max(e2-e1*e1,0.); cand=[]
 if abs(A)>1e-15:
  q=np.sqrt(max(C*C+4*A*sig,0.))
  cand=[x for x in ((C+q)/(2*A),(C-q)/(2*A)) if np.isfinite(x) and x>0]
 if not cand:cand=[1e-3]
 dt=min(cand,key=lambda z:en(F,b-z*Fb))
 el=Fb/np.maximum(b,1e-300); pos=el[el>0]
 if len(pos): dt=min(dt,.95/float(pos.max()))
 t=b-dt*Fb; t/=np.linalg.norm(t)
 return dt,t

def edges(M,a):
 co=sp.triu(M-sp.diags(M.diagonal()),k=1).tocoo()
 m=np.abs(co.data)>1e-15; i=co.row[m].astype(np.int32); j=co.col[m].astype(np.int32); v=np.abs(co.data[m])
 w=v*a[i]*a[j]; w/=w.sum()
 return i,j,w

def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--mode",choices=["physical","frozen"],required=True)
 ap.add_argument("--lr",type=float,default=1e-4); ap.add_argument("--steps",type=int,default=1000); ap.add_argument("--out",required=True); args=ap.parse_args()
 H,d,a,s=snapshot(10); F=build_fixed_node(H,d,a,s); efn,phi=fixed_node_solve(H,d,a,s); efn2,_=fixed_node_solve(H,d,phi,s)
 E0=en(F,a); exact=E0-efn; outer=efn-efn2
 dt,t=target_step(F,a); Et=en(F,t); et2,_=fixed_node_solve(H,d,t,s)
 w=a*a; w/=w.sum(); delta=np.log(np.maximum(t,1e-300))-np.log(np.maximum(a,1e-300)); delta-=w@delta
 ip,jp,wp=edges(H,a); iff,jff,wff=edges(F,a)
 ii,jj,we=(ip,jp,wp) if args.mode=="physical" else (iff,jff,wff)
 X=jnp.asarray(bits2x(basis)); aj=jnp.asarray(a); wj=jnp.asarray(w); dj=jnp.asarray(delta)
 iij=jnp.asarray(ii); jjj=jnp.asarray(jj); wej=jnp.asarray(we)
 ipj=jnp.asarray(ip); jpj=jnp.asarray(jp); wpj=jnp.asarray(wp)
 ifj=jnp.asarray(iff); jfj=jnp.asarray(jff); wfj=jnp.asarray(wff)
 model=ViT(num_layers=4,d_model=60,heads=10,L_eff=4,b=2,transl_invariant=True,two_dimensional=True)
 p=model.init(jax.random.PRNGKey(20261008),X[:2])["params"]
 def f(pp): return jnp.real(_logpsi_transl_2d(model.apply,B,{"params":pp},X))
 f0=jax.lax.stop_gradient(f(p))
 def residual(pp):
  r=f(pp)-f0; return r-jnp.sum(wj*r)
 def amp(pp):
  r=residual(pp); z=jnp.log(jnp.maximum(aj,1e-300))+r; z-=.5*jax.scipy.special.logsumexp(2*z); return jnp.exp(z)
 def edge_mse_from(r,i,j,ww):
  q=(r[j]-r[i])-(dj[j]-dj[i]); return jnp.sum(ww*q*q)
 def loss(pp): return edge_mse_from(residual(pp),iij,jjj,wej)
 def point_mse(pp):
  z=residual(pp)-dj; return jnp.sum(wj*z*z)
 vg=jax.jit(jax.value_and_grad(loss)); tx=optax.chain(optax.clip_by_global_norm(10.),optax.adam(args.lr)); st=tx.init(p)
 rows=[]; marks={1,10,50,100,250,500,args.steps}
 for k in range(1,args.steps+1):
  l,g=vg(p); u,st=tx.update(g,st,p); p=optax.apply_updates(p,u)
  if k in marks:
   b=np.asarray(amp(p),float); E=en(F,b); er,_=fixed_node_solve(H,d,b,s); r=residual(p)
   row=dict(step=k,train_edge_mse=float(loss(p)),point_mse=float(point_mse(p)),
            physical_edge_mse=float(edge_mse_from(r,ipj,jpj,wpj)),frozen_edge_mse=float(edge_mse_from(r,ifj,jfj,wfj)),
            frozen_E=E,target_fidelity=float((b@t)**2),rebuilt_Efn=float(er),
            exact_frozen_fraction=float((E0-E)/exact),exact_rebuild_fraction=float((efn-er)/outer))
   rows.append(row); print("EDGE",json.dumps(row,sort_keys=True),flush=True)
 b=np.asarray(amp(p),float); E=en(F,b); er,_=fixed_node_solve(H,d,b,s); r=residual(p)
 out=dict(mode=args.mode,lr=args.lr,steps=args.steps,dt=dt,n_train_edges=int(len(ii)),
          Ebase=E0,target_E=Et,target_rebuilt_Efn=float(et2),
          target_exact_frozen_fraction=float((E0-Et)/exact),target_exact_rebuild_fraction=float((efn-et2)/outer),
          final_E=E,final_rebuilt_Efn=float(er),final_exact_frozen_fraction=float((E0-E)/exact),
          final_exact_rebuild_fraction=float((efn-er)/outer),final_point_mse=float(point_mse(p)),
          final_physical_edge_mse=float(edge_mse_from(r,ipj,jpj,wpj)),final_frozen_edge_mse=float(edge_mse_from(r,ifj,jfj,wfj)),rows=rows)
 Path(args.out).write_text(json.dumps(out,indent=2)); print("FINAL",json.dumps({k:v for k,v in out.items() if k!="rows"},sort_keys=True),flush=True)
if __name__=="__main__": main()
