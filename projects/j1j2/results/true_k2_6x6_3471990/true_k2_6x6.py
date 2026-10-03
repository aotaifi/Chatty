import math, json, time
import numpy as np
import jax, jax.numpy as jnp
import netket as nk, flax
import netket.jax as nkjax
from nqsmagic.models import ViT
from nqsmagic.utils import _logpsi_transl_2d

L=6; N=36; J2=.5; ALPHA=1.2
T1=-14.985799779143964

hi=nk.hilbert.Spin(s=.5,N=N)
g=nk.graph.Hypercube(length=L,n_dim=2,pbc=True,max_neighbor_order=2)
model=ViT(num_layers=4,d_model=60,heads=10,L_eff=9,b=2,transl_invariant=True,two_dimensional=True)
apply=nkjax.HashablePartial(_logpsi_transl_2d,model.apply,2)
sam=nk.sampler.MetropolisExchange(hi,graph=g,d_max=2,n_chains=16,sweep_size=N)
v=nk.vqs.MCState(sampler=sam,apply_fun=apply,n_samples=16,
 variables=model.init(jax.random.PRNGKey(1234),jnp.zeros((1,N))),n_discard_per_chain=10)
with open('vit_J2=0.50_N=6x6_k=0.mpack','rb') as f:
 v=flax.serialization.from_bytes(v,f.read())
print('LOADED',v.n_parameters,'backend',jax.default_backend(),flush=True)

def evalz(X,batch=4096):
 X=np.asarray(X,float); out=[]
 for i in range(0,len(X),batch):
  out.append(np.asarray(v.log_value(jnp.asarray(X[i:i+batch]))))
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
NN,NNN=bonds(); ALL=[(i,j,1.,-1.) for i,j in NN]+[(i,j,J2,1.) for i,j in NNN]
A_MASK=sum(1<<(x+L*y) for y in range(L) for x in range(L) if (x+y)%2==0)
def mar(s): return 1 if ((int(s)&A_MASK).bit_count()%2)==0 else -1
def diag(s):
 e=0.
 for i,j,J,mr in ALL:
  e += J*(.25 if (((s>>i)&1)==((s>>j)&1)) else -.25)
 return e
def neigh(s):
 return [(s^(1<<i)^(1<<j),J,mr) for i,j,J,mr in ALL if ((s>>i)^(s>>j))&1]

zc={}; r0c={}; r1c={}; neval=0
def ensure_z(ss):
 global neval
 miss=[int(x) for x in np.unique(np.asarray(ss,np.uint64)) if int(x) not in zc]
 if miss:
  z=evalz(bits2x(np.asarray(miss,np.uint64))); neval+=len(miss)
  for x,a in zip(miss,z): zc[x]=complex(a)

def ensure_r0(ss):
 miss=[int(x) for x in np.unique(np.asarray(ss,np.uint64)) if int(x) not in r0c]
 if not miss:return
 alln=[]; metas=[]
 for x in miss:
  ls=neigh(x); metas.append(ls); alln.extend(y for y,J,mr in ls)
 ensure_z(miss+alln)
 for x,ls in zip(miss,metas):
  lx=zc[x].real; r=diag(x)
  for y,J,mr in ls:
   r += .5*J*mr*math.exp(zc[y].real-lx)
  r0c[x]=float(r)

def ensure_r1(ss):
 miss=[int(x) for x in np.unique(np.asarray(ss,np.uint64)) if int(x) not in r1c]
 if not miss:return
 alln=[]; metas=[]
 for x in miss:
  ls=neigh(x); metas.append(ls); alln.extend(y for y,J,mr in ls)
 ensure_r0(miss+alln)
 for x,ls in zip(miss,metas):
  den=T1-r0c[x]
  lx=zc[x].real; r=diag(x)
  if abs(den)<1e-13:
   r1c[x]=np.nan; continue
  for y,J,mr in ls:
   num=T1-r0c[y]
   r += .5*J*mr*math.exp(zc[y].real-lx)*(num/den)
  r1c[x]=float(r)

def wquant(x,w,p):
 o=np.argsort(x); x=np.asarray(x)[o]; w=np.asarray(w,float)[o]
 c=np.cumsum(w); c/=c[-1]
 return float(np.interp(p,c,x))
def wkmeans_t(x,w,lo=.1,hi=.9):
 x=np.asarray(x,float); w=np.asarray(w,float); w/=w.sum()
 qlo=wquant(x,w,lo); qhi=wquant(x,w,hi)
 xc=np.clip(x,qlo,qhi)
 m1=wquant(xc,w,.3); m2=wquant(xc,w,.8)
 for _ in range(200):
  cut=.5*(m1+m2); left=xc<=cut
  if not left.any() or left.all(): break
  n1=np.sum(w[left]*xc[left])/np.sum(w[left])
  n2=np.sum(w[~left]*xc[~left])/np.sum(w[~left])
  if abs(n1-m1)+abs(n2-m2)<1e-12: break
  m1,m2=n1,n2
 return .5*(m1+m2),m1,m2,qlo,qhi

d=np.load('krylov_scaling_6x6_a1.20_tr4096_va2048.npz')
tr=d['train_states'].astype(np.uint64); va=d['val_states'].astype(np.uint64)
phys=np.load('energy_krylov_vs_vit_6x6_indep.npz')['states'].astype(np.uint64)
allc=np.concatenate([tr,va,phys])
t0=time.time(); ensure_r1(allc)
print('R1_DONE sec',time.time()-t0,'zcache',len(zc),'r0cache',len(r0c),'r1cache',len(r1c),'neval',neval,flush=True)

r0tr=np.array([r0c[int(x)] for x in tr]); r1tr=np.array([r1c[int(x)] for x in tr])
r0va=np.array([r0c[int(x)] for x in va]); r1va=np.array([r1c[int(x)] for x in va])
r0p=np.array([r0c[int(x)] for x in phys]); r1p=np.array([r1c[int(x)] for x in phys])
assert np.isfinite(r1tr).all() and np.isfinite(r1va).all() and np.isfinite(r1p).all()

# Existing alpha=1.2 samples are from a0^alpha. Reweight to |psi1|^alpha.
wf_tr=np.abs(T1-r0tr)**ALPHA; wf_tr/=wf_tr.sum()
wf_va=np.abs(T1-r0va)**ALPHA; wf_va/=wf_va.sum()
print('PSI1_ALPHA_ESS',1/np.sum(wf_tr*wf_tr),1/np.sum(wf_va*wf_va),flush=True)

with open('krylov_scaling_6x6_a1.20_tr4096_va2048.json') as f: phi=float(json.load(f)['phi'])
ensure_z(allc)
def hidden(ss):
 return np.array([1 if np.cos(zc[int(x)].imag-phi)>=0 else -1 for x in ss],np.int8)
def signs(ss,r0,r1,t2):
 sm=np.array([mar(x) for x in ss],np.int8)
 s1=sm*np.where(T1-r0>=0,1,-1)
 s2=s1*np.where(t2-r1>=0,1,-1)
 return s1,s2

# Physical a0^2 weights on alpha sample are stored in iw*. Use those only for diagnostics.
iwtr=np.asarray(d['iwtrain'],float); iwtr/=iwtr.sum()
iwva=np.asarray(d['iwval'],float); iwva/=iwva.sum()
htr=hidden(tr); hva=hidden(va); hp=hidden(phys)

rows=[]
for lo,hi in ((.01,.99),(.05,.95),(.1,.9)):
 t2,m1,m2,qlo,qhi=wkmeans_t(r1tr,wf_tr,lo,hi)
 s1t,s2t=signs(tr,r0tr,r1tr,t2); s1v,s2v=signs(va,r0va,r1va,t2); s1p,s2p=signs(phys,r0p,r1p,t2)
 row=dict(lo=lo,hi=hi,t2=float(t2),c1=float(m1),c2=float(m2),qlo=float(qlo),qhi=float(qhi),
  change_train_phys=float(np.sum(iwtr*(s1t!=s2t))),
  change_val_phys=float(np.sum(iwva*(s1v!=s2v))),
  change_phys_unw=float(np.mean(s1p!=s2p)),
  O1_train=float(abs(np.sum(iwtr*s1t*htr))),O2_train=float(abs(np.sum(iwtr*s2t*htr))),
  O1_val=float(abs(np.sum(iwva*s1v*hva))),O2_val=float(abs(np.sum(iwva*s2v*hva))),
  O1_phys=float(abs(np.mean(s1p*hp))),O2_phys=float(abs(np.mean(s2p*hp))))
 rows.append(row); print('K2',row,flush=True)

print('R0_Q',np.quantile(r0tr,[0,.01,.1,.5,.9,.99,1]).tolist(),flush=True)
print('R1_Q',np.quantile(r1tr,[0,.01,.1,.5,.9,.99,1]).tolist(),flush=True)
np.savez_compressed('true_k2_6x6.npz',
 train_states=tr,val_states=va,phys_states=phys,
 r0train=r0tr,r1train=r1tr,r0val=r0va,r1val=r1va,r0phys=r0p,r1phys=r1p,
 wf_train=wf_tr,wf_val=wf_va,rows=np.array([[r[k] for k in ['lo','hi','t2','change_train_phys','change_val_phys','change_phys_unw','O1_train','O2_train','O1_val','O2_val','O1_phys','O2_phys']] for r in rows],float))
with open('true_k2_6x6.json','w') as f: json.dump(rows,f,indent=2)
