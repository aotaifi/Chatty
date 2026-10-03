import math,json
import numpy as np
import jax,jax.numpy as jnp
import netket as nk, flax
import netket.jax as nkjax
from nqsmagic.models import ViT
from nqsmagic.utils import _logpsi_transl_2d

L=8;N=64;J2=.6
z0=np.load('frustration_scan_8x8_J2_0.60.npz')
TR=z0['tr_states'].astype(np.uint64); VA=z0['va_states'].astype(np.uint64)
WT=z0['wt'].astype(float); WV=z0['wv'].astype(float); WT/=WT.sum(); WV/=WV.sum()
YMtr=z0['Yt'].astype(int); YMva=z0['Yv'].astype(int)

hi=nk.hilbert.Spin(s=.5,N=N)
graph=nk.graph.Hypercube(length=L,n_dim=2,pbc=True,max_neighbor_order=2)
model=ViT(num_layers=8,d_model=60,heads=10,L_eff=16,b=2,transl_invariant=True,two_dimensional=True)
apply=nkjax.HashablePartial(_logpsi_transl_2d,model.apply,2)
sam=nk.sampler.MetropolisExchange(hi,graph=graph,d_max=2,n_chains=16,sweep_size=N)
v=nk.vqs.MCState(sampler=sam,apply_fun=apply,n_samples=16,
 variables=model.init(jax.random.PRNGKey(1234),jnp.zeros((1,N))),n_discard_per_chain=10)
with open('vit_J2=0.60_N=8x8_k=0.mpack','rb') as f: obj=flax.serialization.msgpack_restore(f.read())
v.variables=flax.serialization.from_state_dict(v.variables,obj)

def bits2x(ss):
 a=np.asarray(ss,np.uint64).reshape(-1,1)
 return 2*((a>>np.arange(N,dtype=np.uint64))&1).astype(float)-1
def evalz(states,batch=4096):
 X=bits2x(states);out=[]
 for i in range(0,len(X),batch): out.append(np.asarray(v.log_value(jnp.asarray(X[i:i+batch]))))
 return np.concatenate(out)

q=lambda x,y:(x%L)+L*(y%L)
B=[]
for y in range(L):
 for x in range(L):
  i=q(x,y)
  B += [(i,q(x+1,y),1.0),(i,q(x,y+1),1.0),
        (i,q(x+1,y+1),J2),(i,q(x+1,y-1),J2)]

masks={
 'marshall':sum(1<<(x+L*y) for y in range(L) for x in range(L) if (x+y)%2==0),
 'stripe_x':sum(1<<(x+L*y) for y in range(L) for x in range(L) if x%2==0),
 'stripe_y':sum(1<<(x+L*y) for y in range(L) for x in range(L) if y%2==0)}
def refsign(s,mask): return 1 if ((int(s)&mask).bit_count()%2)==0 else -1

# one shared amplitude cache for both stripe gauges
ALL=np.concatenate([TR,VA]); metas=[]; neigh=[]
for s0 in ALL:
 s=int(s0);ls=[];d=0.
 for i,j,J in B:
  same=((s>>i)&1)==((s>>j)&1)
  d += J*(.25 if same else -.25)
  if not same:
   yy=s^(1<<i)^(1<<j);ls.append((yy,J));neigh.append(yy)
 metas.append((d,ls))
U=np.unique(np.concatenate([ALL,np.asarray(neigh,np.uint64)]))
ZU=evalz(U); logmap={int(s):float(z.real) for s,z in zip(U,ZU)}
print('CACHE',len(ALL),len(U),flush=True)

def r_for(mask):
 out=np.empty(len(ALL))
 for k,(s0,(d,ls)) in enumerate(zip(ALL,metas)):
  s=int(s0);sx=refsign(s,mask);lx=logmap[s];r=d
  for yy,J in ls:
   rat=math.exp(logmap[int(yy)]-lx)
   r += .5*J*(refsign(yy,mask)/sx)*rat
  out[k]=r
 return out[:len(TR)],out[len(TR):]

def wq(x,w,p):
 o=np.argsort(x);xx=x[o];ww=w[o];c=np.cumsum(ww);c/=c[-1]
 return float(np.interp(p,c,xx))
def kmeans_t(x):
 w=np.ones(len(x))/len(x);lo=wq(x,w,.05);hi=wq(x,w,.95);a=np.clip(x,lo,hi)
 m1=wq(a,w,.3);m2=wq(a,w,.8)
 for _ in range(100):
  cut=.5*(m1+m2);lab=a>cut
  if lab.all() or (~lab).all():break
  n1=a[~lab].mean();n2=a[lab].mean()
  if abs(n1-m1)+abs(n2-m2)<1e-10:break
  m1,m2=n1,n2
 return .5*(m1+m2)
def oracle(score,y,w):
 best=(-9,None,1)
 wy=w*y
 for orient in (1,-1):
  s=orient*score;o=np.argsort(s);so=s[o];z=wy[o]
  ovs=-z.sum()+2*np.cumsum(z);k=int(np.argmax(ovs));ov=float(ovs[k])
  t=float(so[k]+1e-12) if k==len(so)-1 else float(.5*(so[k]+so[k+1]))
  if ov>best[0]:best=(ov,t,orient)
 return best

out={}
for name in ('stripe_x','stripe_y'):
 mask=masks[name]
 # hidden absolute sign = Marshall-relative target * Marshall reference
 Htr=np.array([YMtr[i]*refsign(s,masks['marshall']) for i,s in enumerate(TR)])
 Hva=np.array([YMva[i]*refsign(s,masks['marshall']) for i,s in enumerate(VA)])
 Ytr=np.array([Htr[i]*refsign(s,mask) for i,s in enumerate(TR)])
 Yva=np.array([Hva[i]*refsign(s,mask) for i,s in enumerate(VA)])
 rtr,rva=r_for(mask)
 t=kmeans_t(rtr);p_tr=np.where(rtr<=t,1,-1);p_va=np.where(rva<=t,1,-1)
 O0tr=abs(float(np.sum(WT*Ytr)));O0va=abs(float(np.sum(WV*Yva)))
 O1tr=abs(float(np.sum(WT*p_tr*Ytr)));O1va=abs(float(np.sum(WV*p_va*Yva)))
 oo,tt,orient=oracle(rtr,Ytr,WT)
 pvo=np.where(orient*rva<=tt,1,-1)
 Ovo=abs(float(np.sum(WV*pvo*Yva)))
 out[name]=dict(T=float(t),O0_train=O0tr,O0_val=O0va,O1_train=O1tr,O1_val=O1va,
   Eref_train=float(np.sum(WT*rtr)),Eref_val=float(np.sum(WV*rva)),
   oracle_train=float(abs(oo)),oracle_val=Ovo,oracle_t=float(tt),oracle_orient=int(orient),
   flip_train=float(np.sum(WT[p_tr<0])),flip_val=float(np.sum(WV[p_va<0])))
 print('RESULT',name,json.dumps(out[name],sort_keys=True),flush=True)
with open('stripe_reuse_8x8_J08.json','w') as f:json.dump(out,f,indent=2)
