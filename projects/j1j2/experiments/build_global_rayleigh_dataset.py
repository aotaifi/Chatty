import math,time
import numpy as np
import scipy.sparse as sp
import jax, jax.numpy as jnp
import netket as nk, flax
import netket.jax as nkjax
from nqsmagic.models import ViT
from nqsmagic.utils import _logpsi_transl_2d

L=6; N=36; J2=.5; B=2
rng=np.random.default_rng(20260927)
hi=nk.hilbert.Spin(s=.5,N=N)
g=nk.graph.Hypercube(length=L,n_dim=2,pbc=True,max_neighbor_order=2)
model=ViT(num_layers=4,d_model=60,heads=10,L_eff=9,b=2,transl_invariant=True,two_dimensional=True)
apply=nkjax.HashablePartial(_logpsi_transl_2d,model.apply,2)
sam0=nk.sampler.MetropolisExchange(hi,graph=g,d_max=2,n_chains=6000,sweep_size=N)
v0=nk.vqs.MCState(sampler=sam0,apply_fun=apply,n_samples=6000,
 variables=model.init(jax.random.PRNGKey(1234),jnp.zeros((1,N))),n_discard_per_chain=10)
with open('vit_J2=0.50_N=6x6_k=0.mpack','rb') as f: v0=flax.serialization.from_bytes(v0,f.read())

def evalz(X,batch=2048):
    X=np.asarray(X,float); out=[]
    for i in range(0,len(X),batch): out.append(np.asarray(v0.log_value(jnp.asarray(X[i:i+batch]))))
    return np.concatenate(out)

def bonds():
    nn=[]; nnn=[]
    q=lambda x,y:(x%L)+L*(y%L)
    for y in range(L):
      for x in range(L):
        i=q(x,y); nn += [(i,q(x+1,y)),(i,q(x,y+1))]
        nnn += [(i,q(x+1,y+1)),(i,q(x+1,y-1))]
    return nn,nnn
NN,NNN=bonds()
def neigh(s):
    for bb,J in ((NN,1.0),(NNN,J2)):
      for i,j in bb:
        if ((s>>i)^(s>>j))&1: yield s^(1<<i)^(1<<j),J
def bits2x(ss):
    a=np.asarray(ss,dtype=np.uint64).reshape(-1,1)
    return 2*((a>>np.arange(N,dtype=np.uint64))&1).astype(float)-1
def x2bits(x):
    s=0
    for i,q in enumerate(x):
      if q>0:s|=1<<i
    return s
def grow(seed,K,p=.5):
    S={seed}
    def ch(s): return [t for t,_ in neigh(s) if t not in S and rng.random()<=p]
    fr=set(ch(seed))
    while len(S)<K and fr:
      a=list(fr);rng.shuffle(a);new=set()
      for s in a:
        if s in S:continue
        S.add(s)
        if len(S)>=K:break
        new.update(ch(s))
      fr=new-S
    return S
def extend(S,hops):
    S=set(S);fr=set(S)
    for _ in range(hops):
      new=set()
      for s in fr:new.update(t for t,_ in neigh(s))
      new-=S;S|=new;fr=new
    return S
def edges(arr,la):
    pos={int(s):i for i,s in enumerate(arr)};u=[];v=[];lw=[]
    for i,s in enumerate(arr):
      for t,J in neigh(int(s)):
        j=pos.get(t)
        if j is not None and i<j:
          u.append(i);v.append(j);lw.append(math.log(.5*J)+la[i]+la[j])
    u=np.asarray(u,np.int32);v=np.asarray(v,np.int32);lw=np.asarray(lw)
    w=np.exp(lw-lw.max()) if len(lw) else np.array([])
    return u,v,w
def solve(n,u,v,w):
    order=np.argsort(w)[::-1];par=np.arange(n,dtype=np.int32);rk=np.zeros(n,np.int8);px=np.zeros(n,np.int8)
    def find(x):
      y=x;p=0
      while par[y]!=y:p^=int(px[y]);y=int(par[y])
      return y,p
    for q in order:
      i=int(u[q]);j=int(v[q]);ri,pi=find(i);rj,pj=find(j)
      if ri==rj:continue
      z=pi^pj^1
      if rk[ri]<rk[rj]:par[ri]=rj;px[ri]=z
      else:
        par[rj]=ri;px[rj]=z
        if rk[ri]==rk[rj]:rk[ri]+=1
    s=np.ones(n,np.int8)
    for i in range(n): _,p=find(i);s[i]=-1 if p else 1
    A=sp.coo_matrix((np.r_[w,w],(np.r_[u,v],np.r_[v,u])),shape=(n,n)).tocsr();h=A@s.astype(float)
    for _ in range(30):
      changed=0
      for i in range(n):
        if s[i]*h[i]>1e-15:
          old=int(s[i]);s[i]=-old;changed+=1
          a,b=A.indptr[i],A.indptr[i+1];h[A.indices[a:b]]+=-2*old*A.data[a:b]
      if not changed:break
    return s
def hidden(z,la):
    w=np.exp(2*(la-la.max()));phi=.5*np.angle(np.sum(w*np.exp(2j*z.imag)))
    rel=np.angle(np.exp(1j*(z.imag-phi)));return np.where(np.cos(rel)>=0,1,-1).astype(np.int8),float(np.average(np.abs(np.sin(rel)),weights=w))
def overlap(p,t,la):
    w=np.exp(2*(la-la.max()));w/=w.sum();return abs(np.sum(w*p*t))


def sample_q(ns=1024,nc=32,burn=100,between=5,alpha=2.0):
  rounds=(ns+nc-1)//nc
  ss=[]
  for _ in range(nc):
    pos=rng.choice(N,N//2,replace=False); q=0
    for i in pos:q|=1<<int(i)
    ss.append(q)
  la=evalz(bits2x(ss)).real; out=[]
  for it in range(burn+between*rounds):
    cand=[]
    for q in ss:
      up=[i for i in range(N) if (q>>i)&1]; dn=[i for i in range(N) if not((q>>i)&1)]
      i=int(rng.choice(up)); j=int(rng.choice(dn)); cand.append(q^(1<<i)^(1<<j))
    lb=evalz(bits2x(cand)).real
    acc=np.log(rng.random(nc))<np.minimum(0,alpha*(lb-la))
    for k in np.where(acc)[0]: ss[k]=cand[k]
    la=np.where(acc,lb,la)
    if it>=burn and (it-burn+1)%between==0: out.extend(ss)
  return np.asarray(out[:ns],dtype=np.uint64)

Xb=sample_q()
zx=evalz(bits2x(Xb)); lax=zx.real
allb=[(i,j,1.0) for i,j in NN]+[(i,j,J2) for i,j in NNN]
M=len(allb)
Yb=np.repeat(Xb[:,None],M,axis=1)
coef=np.zeros((len(Xb),M),dtype=np.float64)
active=[]
for r,s in enumerate(Xb):
  for k,(i,j,J) in enumerate(allb):
    if ((int(s)>>i)^(int(s)>>j))&1:
      y=int(s)^(1<<i)^(1<<j)
      Yb[r,k]=np.uint64(y); coef[r,k]=.5*J; active.append((r,k,y))
ay=np.asarray([q[2] for q in active],dtype=np.uint64)
zy=evalz(bits2x(ay))
# common binary-phase reference across x and active neighbors
m=max(float(np.max(lax)),float(np.max(zy.real)))
cc=np.sum(np.exp(2*(lax-m))*np.exp(2j*zx.imag))+np.sum(np.exp(2*(zy.real-m))*np.exp(2j*zy.imag))
phi=.5*np.angle(cc)
hx=np.where(np.cos(np.angle(np.exp(1j*(zx.imag-phi))))>=0,1,-1).astype(np.int8)
hy=np.ones((len(Xb),M),dtype=np.int8)
ratio=np.zeros((len(Xb),M),dtype=np.float64)
for q,(r,k,y) in enumerate(active):
  ratio[r,k]=np.exp(zy.real[q]-lax[r])
  hy[r,k]=1 if np.cos(np.angle(np.exp(1j*(zy.imag[q]-phi))))>=0 else -1
# diagonal SzSz
X=bits2x(Xb)
diag=np.zeros(len(Xb),dtype=np.float64)
for i,j in NN: diag += .25*X[:,i]*X[:,j]
for i,j in NNN: diag += .25*J2*X[:,i]*X[:,j]
C=coef*ratio
np.savez('global_rayleigh_1024.npz',Xb=Xb,Yb=Yb,C=C,diag=diag,hx=hx,hy=hy,phi=np.array(phi))
print('GLOBAL_DATA','ns',len(Xb),'bonds',M,'active',len(active),'phi',phi,'diagmean',diag.mean(),flush=True)
