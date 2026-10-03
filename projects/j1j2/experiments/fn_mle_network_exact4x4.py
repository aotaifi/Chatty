import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from math import comb

L=4; N=L*L; J2=0.5
def bonds():
    nn=[]; nnn=[]; q=lambda x,y:(x%L)+L*(y%L)
    for y in range(L):
        for x in range(L):
            i=q(x,y)
            nn += [(i,q(x+1,y)),(i,q(x,y+1))]
            nnn += [(i,q(x+1,y+1)),(i,q(x+1,y-1))]
    return nn,nnn
NN,NNN=bonds()
BONDS=[(i,j,1.0) for i,j in NN]+[(i,j,J2) for i,j in NNN]

states=np.array([s for s in range(1<<N) if s.bit_count()==N//2],dtype=np.uint32)
idx={int(s):i for i,s in enumerate(states)}
D=len(states)
print("DIM",D,"expected",comb(N,N//2),flush=True)

rows=[]; cols=[]; vals=[]
for a,s0 in enumerate(states):
    s=int(s0); diag=0.0
    for i,j,J in BONDS:
        anti=((s>>i)^(s>>j))&1
        diag += J*(-0.25 if anti else 0.25)
        if anti:
            t=s^(1<<i)^(1<<j); b=idx[t]
            rows.append(a); cols.append(b); vals.append(0.5*J)
    rows.append(a); cols.append(a); vals.append(diag)
H=sp.coo_matrix((vals,(rows,cols)),shape=(D,D)).tocsr()
print("H nnz",H.nnz,flush=True)

A_mask=sum(1<<(x+L*y) for y in range(L) for x in range(L) if (x+y)%2==0)
sM=np.array([1.0 if ((int(s)&A_mask).bit_count()%2)==0 else -1.0 for s in states])

def ground(A,v0=None):
    e,v=sla.eigsh(A,k=1,which="SA",v0=v0,tol=1e-11,maxiter=10000)
    x=v[:,0]
    return float(e[0]),x/np.linalg.norm(x)

E0,psi0=ground(H)
# fix global phase/sign to maximize Marshall overlap
if np.dot(psi0,sM)<0: psi0=-psi0
a0=np.abs(psi0)
strue=np.where(psi0>=0,1.0,-1.0)
ptr=a0*a0; ptr/=ptr.sum()
print("EXACT E0",E0,"MarshallO",abs(np.sum(ptr*sM*strue)),flush=True)

def r_marshall(a):
    psi=a*sM
    return np.asarray(H@psi/psi)

def weighted_kmeans_threshold(r,w):
    m1,m2=np.quantile(r,[0.3,0.8])
    for _ in range(200):
        cut=0.5*(m1+m2); lo=r<=cut
        if not np.any(lo) or np.all(lo): break
        n1=np.sum(w[lo]*r[lo])/np.sum(w[lo])
        n2=np.sum(w[~lo]*r[~lo])/np.sum(w[~lo])
        if abs(n1-m1)+abs(n2-m2)<1e-13: break
        m1,m2=n1,n2
    return 0.5*(m1+m2),(m1,m2)

def sign_overlap(s):
    return abs(float(np.sum(ptr*s*strue)))

def energy(a,s):
    psi=a*s; psi=psi/np.linalg.norm(psi)
    return float(psi@(H@psi))

def build_fn(a,s):
    # Standard lattice fixed-node (gamma=0):
    # keep H_ab only when guide signs are opposite; put removed same-sign
    # contribution H_ab psi_b/psi_a onto the diagonal.
    rr=[]; cc=[]; vv=[]
    diag=H.diagonal().astype(float).copy()
    for ia,s0 in enumerate(states):
        x=int(s0)
        for i,j,J in BONDS:
            if not (((x>>i)^(x>>j))&1): continue
            y=x^(1<<i)^(1<<j); ib=idx[y]
            if s[ia]*s[ib] < 0:
                rr.append(ia);cc.append(ib);vv.append(0.5*J)
            else:
                diag[ia] += 0.5*J*a[ib]/a[ia]
    rr.extend(range(D));cc.extend(range(D));vv.extend(diag.tolist())
    return sp.coo_matrix((vv,(rr,cc)),shape=(D,D)).tocsr()

def fn_solve(a,s):
    Hfn=build_fn(a,s)
    E,v=ground(Hfn,v0=a*s)
    # orient and enforce expected fixed-node sign gauge
    if np.dot(v,a*s)<0: v=-v
    sf=np.where(v>=0,1.0,-1.0)
    mismatch=np.mean(sf!=s)
    anew=np.abs(v); anew/=np.linalg.norm(anew)
    return E,anew,mismatch

def krylov_sign(a):
    q=np.power(np.maximum(a,1e-300),1.2); q/=q.sum()
    r=r_marshall(a)
    t,cent=weighted_kmeans_threshold(r,q)
    corr=np.where(r<=t,1.0,-1.0)
    return sM*corr,t,cent,r

def report_direct(name,a):
    sk,t,c,r=krylov_sign(a)
    print("DIRECT",name,"t",t,"centers",c,
          "O",sign_overlap(sk),"E",energy(a,sk),
          "flip_phys",float(np.sum(ptr*(sk!=sM))),
          "r_q",np.quantile(r,[.01,.1,.5,.9,.99]).tolist(),flush=True)
    return sk


import json
import jax
import jax.numpy as jnp
from flax import linen as nn
import optax

def bits2x(ss):
    a=np.asarray(ss,np.uint32).reshape(-1,1)
    x=2.0*(((a>>np.arange(N,dtype=np.uint32))&1).astype(np.float32))-1.0
    return x.reshape(-1,L,L,1)

class AmpMLP(nn.Module):
    width:int=48
    @nn.compact
    def __call__(self,x):
        x=x.reshape((x.shape[0],-1))
        x=nn.Dense(self.width)(x); x=nn.gelu(x)
        x=nn.Dense(self.width)(x); x=nn.gelu(x)
        return nn.Dense(1)(x)[:,0]

X=jnp.asarray(bits2x(states))
model=AmpMLP()
params=model.init(jax.random.PRNGKey(123),X[:2])["params"]

def train_to(ptarget,params,steps,lr,label):
    ptarget=jnp.asarray(np.asarray(ptarget,np.float32))
    tx=optax.adamw(lr,weight_decay=1e-6)
    st=tx.init(params)
    def lossfn(p):
        u=model.apply({"params":p},X)
        logp=2*u-jax.scipy.special.logsumexp(2*u)
        return -jnp.sum(ptarget*logp)
    @jax.jit
    def step(p,s):
        loss,g=jax.value_and_grad(lossfn)(p)
        upd,s=tx.update(g,s,p)
        return optax.apply_updates(p,upd),s,loss
    for it in range(steps):
        params,st,loss=step(params,st)
        if it in (0,49,99,199,399,steps-1):
            print("MLE_STEP",label,it+1,float(loss),flush=True)
    return params,float(loss)

def amp_from(p):
    u=np.asarray(model.apply({"params":p},X),float)
    z=2*u
    z-=z.max()
    prob=np.exp(z)
    prob/=prob.sum()
    return np.sqrt(prob),prob

def ov(a,b):
    a=a/np.linalg.norm(a)
    b=b/np.linalg.norm(b)
    return float(abs(a@b))

ag=a0.copy()
ag/=np.linalg.norm(ag)
Efn,afn,fnmis=fn_solve(ag,sM)
f=ag*afn
f/=f.sum()
ah=np.sqrt(ag*afn)
ah/=np.linalg.norm(ah)
q=ag*ag
q/=q.sum()
print("TARGET","E0",E0,"Efn",Efn,"fnmis",fnmis,
      "guide_to_fn",ov(ag,afn),"ideal_half_to_fn",ov(ah,afn),flush=True)

params,preloss=train_to(q,params,400,2e-3,"guide")
apre,ppre=amp_from(params)
print("PRETRAIN","fid_guide",ov(apre,ag)**2,
      "KL_q_model",float(np.sum(q*np.log(np.maximum(q,1e-300)/np.maximum(ppre,1e-300)))),flush=True)

rng=np.random.default_rng(20260930)
nsamp=10000
ii=rng.choice(D,size=nsamp,replace=True,p=f)
counts=np.bincount(ii,minlength=D).astype(float)
phat=counts/counts.sum()
params2,finloss=train_to(phat,params,500,8e-4,"walkers10k")
alearn,plearn=amp_from(params2)

sk_ideal,t_ideal,_,_=krylov_sign(ah)
sk_learn,t_learn,_,_=krylov_sign(alearn)
fid_half=ov(alearn,ah)**2
fid_fn=ov(alearn,afn)**2
kl_f=float(np.sum(f*np.log(np.maximum(f,1e-300)/np.maximum(plearn,1e-300))))
disc=float(np.sum(ptr*(sk_learn!=sk_ideal)))
out=dict(E0=E0,Efn=Efn,nsamp=nsamp,
         pretrain_fid_guide=ov(apre,ag)**2,
         learned_fid_ideal_half=fid_half,
         learned_fid_FN=fid_fn,
         KL_exact_mixed_to_model=kl_f,
         ideal_half_E=energy(ah,sk_ideal),
         learned_E=energy(alearn,sk_learn),
         ideal_half_sign_overlap=sign_overlap(sk_ideal),
         learned_sign_overlap=sign_overlap(sk_learn),
         learned_vs_ideal_K1_disagree_trueW=disc,
         T_ideal=t_ideal,T_learn=t_learn,
         walker_unique=int(np.count_nonzero(counts)))
print("MLE_RESULT",json.dumps(out,sort_keys=True),flush=True)
json.dump(out,open("/Users/aliotaifi/Chatty/projects/j1j2/results/fn_mle_network_exact4x4.json","w"),indent=2)
