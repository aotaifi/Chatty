import math, json, numpy as np
import jax, jax.numpy as jnp
import netket as nk, flax
import netket.jax as nkjax
from nqsmagic.models import ViT
from nqsmagic.utils import _logpsi_transl_2d

L=8; N=64; ALPHA_REF=1.2
TARGETS=[.52,.54,.56,.58]
REF=.50

dat=np.load('frustration_scan_8x8_J2_0.50.npz')
tr=dat['tr_states'].astype(np.uint64); va=dat['va_states'].astype(np.uint64)
states=np.concatenate([tr,va]); ntr=len(tr)

hi=nk.hilbert.Spin(s=.5,N=N)
graph=nk.graph.Hypercube(length=L,n_dim=2,pbc=True,max_neighbor_order=2)
model=ViT(num_layers=8,d_model=60,heads=10,L_eff=16,b=2,transl_invariant=True,two_dimensional=True)
apply=nkjax.HashablePartial(_logpsi_transl_2d,model.apply,2)
sam=nk.sampler.MetropolisExchange(hi,graph=graph,d_max=2,n_chains=16,sweep_size=N)
v=nk.vqs.MCState(sampler=sam,apply_fun=apply,n_samples=16,
    variables=model.init(jax.random.PRNGKey(1234),jnp.zeros((1,N))),n_discard_per_chain=10)
template=v.variables

def set_ck(j2):
    with open(f'vit_J2={j2:.2f}_N=8x8_k=0.mpack','rb') as f:
        obj=flax.serialization.msgpack_restore(f.read())
    v.variables=flax.serialization.from_state_dict(template,obj)

def bits2x(ss):
    a=np.asarray(ss,np.uint64).reshape(-1,1)
    return 2*((a>>np.arange(N,dtype=np.uint64))&1).astype(float)-1

def evalz(ss,batch=4096):
    X=bits2x(ss); out=[]
    for i in range(0,len(X),batch):
        out.append(np.asarray(v.log_value(jnp.asarray(X[i:i+batch]))))
    return np.concatenate(out)

q=lambda x,y:(x%L)+L*(y%L)
NN=[]; NNN=[]
for y in range(L):
    for x in range(L):
        i=q(x,y)
        NN += [(i,q(x+1,y)),(i,q(x,y+1))]
        NNN += [(i,q(x+1,y+1)),(i,q(x+1,y-1))]
ALL=[(i,j,0) for i,j in NN]+[(i,j,1) for i,j in NNN]
A_MASK=sum(1<<(x+L*y) for y in range(L) for x in range(L) if (x+y)%2==0)

# Shared neighbor union for the fixed reference sample.
neighbors=[]
union=set(map(int,states))
for s0 in states:
    s=int(s0); row=[]
    for i,j,typ in ALL:
        if ((s>>i)^(s>>j))&1:
            y=s^(1<<i)^(1<<j); row.append((y,typ)); union.add(y)
    neighbors.append(row)
union=np.asarray(sorted(union),np.uint64)
uidx={int(s):i for i,s in enumerate(union)}
center_idx=np.asarray([uidx[int(s)] for s in states],int)
print('SHARED ncenter',len(states),'union',len(union),flush=True)

# Reference proposal log density q ~ a_ref^ALPHA_REF.
set_ck(REF)
zref=evalz(states)
lref=zref.real.copy()

def normw(logw):
    x=np.asarray(logw,float); x-=x.max(); w=np.exp(x); w/=w.sum(); return w
def ov(p,y,w): return float(np.sum(w*p*y))
def fit_threshold(score,y,w):
    s0=np.asarray(score,float); y=np.asarray(y,int); w=np.asarray(w,float); w=w/w.sum()
    best=(-2.0,None,None)
    for orient in (1,-1):
        s=orient*s0; ix=np.argsort(s); p=-np.ones(len(s),int)
        c=ov(p,y,w); t=float(s[ix[0]]-1.0)
        if c>best[0]: best=(c,t,orient)
        for k,j in enumerate(ix):
            p[j]=1; c=ov(p,y,w)
            t=float(s[j]+1.0) if k==len(ix)-1 else float(.5*(s[j]+s[ix[k+1]]))
            if c>best[0]: best=(c,t,orient)
    return best
def apply(score,t,o): return np.where(o*np.asarray(score)<=t,1,-1)
def marshall(ss):
    return np.asarray([1 if ((int(s)&A_MASK).bit_count()%2)==0 else -1 for s in ss],int)

for J2 in TARGETS:
    set_ck(J2)
    zu=evalz(union)
    zc=zu[center_idx]; lc=zc.real
    # importance from reference |a_ref|^1.2 sample to target |a|^2
    logw=2*lc-ALPHA_REF*lref
    wt=normw(logw[:ntr]); wv=normw(logw[ntr:])
    esst=1/np.sum(wt*wt); essv=1/np.sum(wv*wv)

    D1=np.zeros(len(states)); D2=np.zeros(len(states)); F1=np.zeros(len(states)); F2=np.zeros(len(states))
    for b,s0 in enumerate(states):
        s=int(s0); lx=lc[b]
        for i,j in NN: D1[b]+=.25 if (((s>>i)&1)==((s>>j)&1)) else -.25
        for i,j in NNN: D2[b]+=.25 if (((s>>i)&1)==((s>>j)&1)) else -.25
        for y,typ in neighbors[b]:
            rat=math.exp(zu[uidx[int(y)]].real-lx)
            if typ==0: F1[b]+=-.5*rat
            else: F2[b]+=.5*rat

    # Binary hidden target, diagnostics only.
    wall=np.concatenate([wt,wv]); wall/=wall.sum()
    phi=.5*np.angle(np.sum(wall*np.exp(2j*zc.imag)))
    h=np.where(np.cos(np.angle(np.exp(1j*(zc.imag-phi))))>=0,1,-1)
    y=h*marshall(states)
    # orient global sign using train
    if np.sum(wt*y[:ntr])<0: y=-y

    r=D1+J2*D2+F1+J2*F2
    rt,rv=r[:ntr],r[ntr:]; yt,yv=y[:ntr],y[ntr:]
    otr,t,o=fit_threshold(rt,yt,wt); pval=apply(rv,t,o)
    r_val=ov(pval,yv,wv)

    # Off-diagonal physical coefficient and best lambda.
    rows=[]
    for lam in np.arange(-.25,2.0001,.05):
        st=F1[:ntr]+lam*F2[:ntr]; sv=F1[ntr:]+lam*F2[ntr:]
        a,tt,oo=fit_threshold(st,yt,wt)
        rows.append((float(lam),float(a),ov(apply(sv,tt,oo),yv,wv)))
    best=max(rows,key=lambda q:q[1])
    phys=min(rows,key=lambda q:abs(q[0]-J2))

    out=dict(J2=J2,ESS_train=float(esst),ESS_val=float(essv),
      MarshallO_train=ov(np.ones(ntr,int),yt,wt),MarshallO_val=ov(np.ones(len(yv),int),yv,wv),
      r_oracle_train=float(otr),r_trainfit_val=float(r_val),r_threshold=float(t),r_orient=int(o),
      lambda_best_train=best[0],lambda_best_trainO=best[1],lambda_best_valO=best[2],
      lambda_physical=phys[0],lambda_physical_trainO=phys[1],lambda_physical_valO=phys[2])
    print('CROSSOVER',json.dumps(out,sort_keys=True),flush=True)
    with open(f'frustration_crossover_reweight_8x8_J2_{J2:.2f}.json','w') as f: json.dump(out,f,indent=2)
