import math,time,json
import numpy as np
import jax,jax.numpy as jnp
import netket as nk, flax
import netket.jax as nkjax
from nqsmagic.models import ViT
from nqsmagic.utils import _logpsi_transl_2d

L=6; N=36; J2=.5
hi=nk.hilbert.Spin(s=.5,N=N)
g=nk.graph.Hypercube(length=L,n_dim=2,pbc=True,max_neighbor_order=2)
model=ViT(num_layers=4,d_model=60,heads=10,L_eff=9,b=2,transl_invariant=True,two_dimensional=True)
apply=nkjax.HashablePartial(_logpsi_transl_2d,model.apply,2)
sam0=nk.sampler.MetropolisExchange(hi,graph=g,d_max=2,n_chains=16,sweep_size=N)
v0=nk.vqs.MCState(sampler=sam0,apply_fun=apply,n_samples=16,
 variables=model.init(jax.random.PRNGKey(1234),jnp.zeros((1,N))),n_discard_per_chain=10)
with open('vit_J2=0.50_N=6x6_k=0.mpack','rb') as f:
 o=flax.serialization.msgpack_restore(f.read())
v0.variables=flax.serialization.from_state_dict(v0.variables,o['variables'])
print("LOADED",v0.n_parameters,flush=True)

def evalz(X,batch=2048):
 X=np.asarray(X,float); out=[]
 for i in range(0,len(X),batch):
  out.append(np.asarray(v0.log_value(jnp.asarray(X[i:i+batch]))))
 return np.concatenate(out)

def bits2x(ss):
 a=np.asarray(ss,np.uint64).reshape(-1,1)
 return 2*((a>>np.arange(N,dtype=np.uint64))&1).astype(float)-1

def bonds():
 nn=[]; nnn=[]; q=lambda x,y:(x%L)+L*(y%L)
 for y in range(L):
  for x in range(L):
   i=q(x,y); nn += [(i,q(x+1,y)),(i,q(x,y+1))]
   nnn += [(i,q(x+1,y+1)),(i,q(x+1,y-1))]
 return nn,nnn
NN,NNN=bonds(); ALL=[(i,j,1.0,0) for i,j in NN]+[(i,j,J2,1) for i,j in NNN]

logcache={}
localcache={}
neval=0

def ensure_log(states):
 global neval
 uu=np.unique(np.asarray(states,np.uint64))
 miss=[int(x) for x in uu if int(x) not in logcache]
 if miss:
  z=evalz(bits2x(np.asarray(miss,np.uint64))).real
  neval += len(miss)
  for x,l in zip(miss,z): logcache[x]=float(l)

def physdiag(s):
 e=0.
 for i,j in NN: e += .25 if (((s>>i)&1)==((s>>j)&1)) else -.25
 for i,j in NNN: e += J2*(.25 if (((s>>i)&1)==((s>>j)&1)) else -.25)
 return e

def ensure_local(states):
 miss=[int(x) for x in np.unique(np.asarray(states,np.uint64)) if int(x) not in localcache]
 if not miss: return
 ensure_log(miss)
 metas=[]; alln=[]
 for x in miss:
  ls=[]
  for i,j,J,kind in ALL:
   if ((x>>i)^(x>>j))&1:
    y=x^(1<<i)^(1<<j); ls.append((y,J,kind)); alln.append(y)
  metas.append(ls)
 ensure_log(alln)
 for x,ls in zip(miss,metas):
  lx=logcache[x]; d=physdiag(x); an=[]; ar=[]
  for y,J,kind in ls:
   rat=math.exp(logcache[y]-lx)
   if kind==0: # NN: Marshall signs opposite => allowed
    an.append(y); ar.append(.5*J*rat)
   else:       # NNN: same Marshall sign => sign-flip potential
    d += .5*J*rat
  ar=np.asarray(ar,float); an=np.asarray(an,np.uint64)
  el=d-ar.sum()
  localcache[x]=(float(d),float(el),an,ar)

def sample_q(n,seed,nchains=64,burn=120,between=6):
 rg=np.random.default_rng(seed); ss=[]
 for _ in range(nchains):
  pos=rg.choice(N,N//2,replace=False); x=0
  for i in pos:x|=1<<int(i)
  ss.append(x)
 ensure_log(ss); la=np.array([logcache[x] for x in ss])
 out=[]; acc=0; props=0
 rounds=math.ceil(n/nchains)
 for it in range(burn+between*rounds):
  cand=[]
  for x in ss:
   i,j,J,k=ALL[int(rg.integers(len(ALL)))]
   cand.append(x^(1<<i)^(1<<j) if (((x>>i)^(x>>j))&1) else x)
  ensure_log(cand); lb=np.array([logcache[x] for x in cand])
  ok=np.array([cand[k]!=ss[k] for k in range(nchains)])
  ac=(np.log(rg.random(nchains))<np.minimum(0,2*(lb-la))) & ok
  acc+=int(ac.sum()); props+=nchains
  for k in np.where(ac)[0]: ss[k]=cand[k]
  la=np.where(ac,lb,la)
  if it>=burn and (it-burn+1)%between==0: out.extend(ss)
 print("Q_SAMPLE",n,"acc",acc/props,flush=True)
 return np.asarray(out[:n],np.uint64)

def systematic(w,rng,M):
 c=np.cumsum(w); c[-1]=1
 u=rng.random()/M+np.arange(M)/M
 return np.searchsorted(c,u,'right')

def gfmc(M=32,beta_target=.9,burn_beta=.3,tau_max=.025,seed=777):
 rg=np.random.default_rng(seed)
 pool=np.load('energy_krylov_vs_vit_6x6_indep.npz')['states'].astype(np.uint64)
 walkers=pool[rg.choice(len(pool),M,replace=False)].copy()
 beta=0.; it=0; snaps=[]; Es=[]; taus=[]
 t0=time.time()
 while beta<beta_target:
  ensure_local(walkers)
  dat=[localcache[int(x)] for x in walkers]
  diag=np.array([d[0] for d in dat]); el=np.array([d[1] for d in dat])
  Eref=float(el.mean())
  mx=max(0.,float(np.max(diag-Eref)))
  tau=min(tau_max,0.85/mx if mx>0 else tau_max)
  nxt=np.empty(M,np.uint64); bw=np.empty(M)
  for k,(x,(d,e,nei,rate)) in enumerate(zip(walkers,dat)):
   stay=1-tau*(d-Eref)
   ws=tau*rate; total=stay+ws.sum()
   if stay<0 or total<=0: raise RuntimeError(("bad GFMC coeff",stay,total,tau,d,Eref))
   u=rg.random()*total
   if u<stay: y=x
   else:
    j=np.searchsorted(np.cumsum(ws),u-stay,'right')
    y=int(nei[min(j,len(nei)-1)])
   nxt[k]=y; bw[k]=total
  bw/=bw.sum()
  walkers=nxt[systematic(bw,rg,M)]
  beta+=tau; it+=1
  if beta>=burn_beta and it%3==0:
   ensure_local(walkers)
   ee=np.mean([localcache[int(x)][1] for x in walkers])
   Es.append(float(ee)); taus.append(tau); snaps.append(walkers.copy())
  if it%10==0 or it==1:
   print("GFMC",it,"beta",beta,"tau",tau,"Eref",Eref,
         "unique",len(np.unique(walkers)),"localCache",len(localcache),
         "logCache",len(logcache),"neval",neval,"sec",time.time()-t0,flush=True)
 mixed=np.concatenate(snaps) if snaps else walkers.copy()
 print("GFMC_RESULT M",M,"iters",it,"beta",beta,
       "Emean",float(np.mean(Es)),"EsnapshotSE",float(np.std(Es)/np.sqrt(len(Es))),
       "Elast",float(Es[-1]),"mixedN",len(mixed),
       "uniqueMixed",len(np.unique(mixed)),"neval",neval,flush=True)
 np.savez_compressed("gfmc_6x6_marshall_matched32.npz",mixed=mixed,Es=np.asarray(Es),taus=np.asarray(taus))
 return mixed,Es

gfmc()
