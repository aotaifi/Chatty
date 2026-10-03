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

# flattened seeds: direct symmetric exchange Metropolis, p ~ |psi|^0.1
def flat_seeds(nc=8,burn=80,between=12,alpha=.1):
  ss=[]
  for _ in range(nc):
    pos=rng.choice(N,N//2,replace=False); q=0
    for i in pos:q|=1<<int(i)
    ss.append(q)
  z=evalz(bits2x(ss)); la=z.real
  out=[]
  for it in range(burn+between*6):
    cand=[]
    for q in ss:
      up=[i for i in range(N) if (q>>i)&1]; dn=[i for i in range(N) if not((q>>i)&1)]
      i=int(rng.choice(up));j=int(rng.choice(dn));cand.append(q^(1<<i)^(1<<j))
    z2=evalz(bits2x(cand));lb=z2.real
    acc=np.log(rng.random(nc))<np.minimum(0,alpha*(lb-la))
    for k in np.where(acc)[0]:ss[k]=cand[k]
    la=np.where(acc,lb,la)
    if it>=burn and (it-burn+1)%between==0: out.extend(ss)
  return out
seedbits=flat_seeds()
print('FLAT_SEEDS',len(seedbits),flush=True)

def run(K,hops,reps=6):
  ans=[]
  for r in range(reps):
    t0=time.time();core=grow(seedbits[r%len(seedbits)],K);ext=extend(core,hops)
    arr=np.array(sorted(ext),dtype=np.uint64);z=evalz(bits2x(arr));la=z.real
    tru,leak=hidden(z,la);u,v,w=edges(arr,la);pred=solve(len(arr),u,v,w)
    pos={int(s):i for i,s in enumerate(arr)};ci=np.array([pos[s] for s in core])
    O=overlap(pred[ci],tru[ci],la[ci])
    ans.append((O,len(arr),len(w),leak,time.time()-t0))
    print('RUN',K,hops,r,'O',O,'ext',len(arr),'edges',len(w),'leak',leak,'sec',ans[-1][-1],flush=True)
  a=np.array([q[0] for q in ans]); print('SUMMARY',K,hops,'medianO',np.median(a),'worstO',a.min(),'median_ext',np.median([q[1] for q in ans]),flush=True)

for K in (50,100,200,400): run(K,1,6)
