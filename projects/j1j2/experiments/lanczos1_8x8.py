import math,time,json,hashlib
import numpy as np
import jax,jax.numpy as jnp
import netket as nk, flax
import netket.jax as nkjax
from nqsmagic.models import ViT
from nqsmagic.utils import _logpsi_transl_2d

L=8;N=64;J2=.5;B=2;ALPHA=.8
SEED_T=202609282;SEED_V=202609283
rng=np.random.default_rng(SEED_T)
hi=nk.hilbert.Spin(s=.5,N=N)
g=nk.graph.Hypercube(length=L,n_dim=2,pbc=True,max_neighbor_order=2)
model=ViT(num_layers=8,d_model=60,heads=10,L_eff=16,b=2,transl_invariant=True,two_dimensional=True)
apply=nkjax.HashablePartial(_logpsi_transl_2d,model.apply,2)
sam0=nk.sampler.MetropolisExchange(hi,graph=g,d_max=2,n_chains=16,sweep_size=N)
v0=nk.vqs.MCState(sampler=sam0,apply_fun=apply,n_samples=16,
 variables=model.init(jax.random.PRNGKey(1234),jnp.zeros((1,N))),n_discard_per_chain=10)
with open('vit_J2=0.50_N=8x8_k=0.mpack','rb') as f:
 obj=flax.serialization.msgpack_restore(f.read())
v0.variables=flax.serialization.from_state_dict(v0.variables,obj)
print('LOADED nparams',v0.n_parameters,flush=True)

def evalz(X,batch=1024):
 X=np.asarray(X,float);out=[]
 for i in range(0,len(X),batch):
  out.append(np.asarray(v0.log_value(jnp.asarray(X[i:i+batch]))))
 return np.concatenate(out)

def bonds():
 nn=[];nnn=[];q=lambda x,y:(x%L)+L*(y%L)
 for y in range(L):
  for x in range(L):
   i=q(x,y);nn += [(i,q(x+1,y)),(i,q(x,y+1))]
   nnn += [(i,q(x+1,y+1)),(i,q(x+1,y-1))]
 return nn,nnn
NN,NNN=bonds();ALL=[(i,j,1.,-1.) for i,j in NN]+[(i,j,J2,1.) for i,j in NNN]

def bits2x(ss):
 a=np.asarray(ss,dtype=np.uint64).reshape(-1,1)
 return 2*((a>>np.arange(N,dtype=np.uint64))&1).astype(float)-1

def sample(seed,nchains,nround,burn=160,between=8):
 rg=np.random.default_rng(seed);ss=[]
 for _ in range(nchains):
  pos=rg.choice(N,N//2,replace=False);q=0
  for i in pos:q|=1<<int(i)
  ss.append(q)
 la=evalz(bits2x(ss)).real
 out=[];acc=0;props=0
 for it in range(burn+between*nround):
  cand=[];valid=[]
  for q in ss:
   i,j,J,mr=ALL[int(rg.integers(len(ALL)))]
   ok=((q>>i)^(q>>j))&1
   cand.append(q^(1<<i)^(1<<j) if ok else q);valid.append(bool(ok))
  lb=evalz(bits2x(cand)).real
  ac=(np.log(rg.random(nchains))<np.minimum(0,ALPHA*(lb-la))) & np.asarray(valid)
  acc+=int(ac.sum());props+=nchains
  for k in np.where(ac)[0]:ss[k]=cand[k]
  la=np.where(ac,lb,la)
  if it>=burn and (it-burn+1)%between==0:out.extend(ss)
 print('SAMPLE',seed,'n',len(out),'accept',acc/props,flush=True)
 return np.asarray(out,np.uint64)

def diag_one(s):
 e=0.
 for bb,J in ((NN,1.0),(NNN,J2)):
  for i,j in bb:e += J*(.25 if (((s>>i)&1)==((s>>j)&1)) else -.25)
 return e

def local_rM(states,label):
 states=np.asarray(states,np.uint64);X=bits2x(states)
 z=evalz(X);la=z.real
 flat=[];meta=[];counts=[]
 for bi,s0 in enumerate(states):
  s=int(s0);cnt=0
  for i,j,J,mr in ALL:
   if ((s>>i)^(s>>j))&1:
    flat.append(s^(1<<i)^(1<<j));meta.append((bi,J,mr));cnt+=1
  counts.append(cnt)
 unq,inv=np.unique(np.asarray(flat,np.uint64),return_inverse=True)
 print(label,'central',len(states),'neighbor_occ',len(flat),'unique_neighbors',len(unq),
       'mean_degree',float(np.mean(counts)),flush=True)
 lu=evalz(bits2x(unq)).real
 r=np.asarray([diag_one(int(s)) for s in states],float)
 pos=0
 for jj,(bi,J,mr) in zip(inv,meta):
  r[bi] += .5*J*mr*math.exp(lu[jj]-la[bi])
 lw=(2-ALPHA)*la;lw-=lw.max();iw=np.exp(lw);iw/=iw.sum()
 ess=1/np.sum(iw*iw)
 print(label,'r_quant',np.quantile(r,[0,.01,.1,.25,.5,.75,.9,.99,1]).tolist(),
       'ESS',float(ess),'ESSfrac',float(ess/len(states)),flush=True)
 return dict(states=states,z=z,la=la,r=r,iw=iw,ess=ess)

trstates=sample(SEED_T,16,64) # 1024
vastates=sample(SEED_V,16,32) # 512
TR=local_rM(trstates,'TRAIN');VA=local_rM(vastates,'VAL')
np.savez_compressed('lanczos1_8x8_data.npz',train_states=trstates,val_states=vastates,
                    rtrain=TR['r'],rval=VA['r'],iwtrain=TR['iw'],iwval=VA['iw'])

# Hidden signs are diagnostics only.
# Estimate one binary global phase from the union with physical importance weights.
def binary_phi(dat):
 w=dat['iw'];z=dat['z'];return .5*np.angle(np.sum(w*np.exp(2j*z.imag)))
phit=binary_phi(TR);phiv=binary_phi(VA);phi=.5*np.angle(np.exp(2j*phit)+np.exp(2j*phiv))
def hidden(dat):
 rel=np.angle(np.exp(1j*(dat['z'].imag-phi)))
 h=np.where(np.cos(rel)>=0,1.,-1.)
 leak=float(np.sum(dat['iw']*np.abs(np.sin(rel))))
 return h,leak

A_mask=sum((1<<((x)+L*y)) for y in range(L) for x in range(L) if (x+y)%2==0)
def marshall_bits(states):
 # N_A=32 even, parity of down-A equals parity of up-A.
 return np.asarray([1 if ((int(s)&A_mask).bit_count()%2)==0 else -1 for s in states],np.int8)
mt=marshall_bits(TR['states']);mv=marshall_bits(VA['states'])
ht,leakt=hidden(TR);hv,leakv=hidden(VA)
Yt=ht*mt;Yv=hv*mv
print('PHASE phi',phi,'trainLeak',leakt,'valLeak',leakv,flush=True)
print('HIDDEN_DIAG train_nonM_unw',float(np.mean(Yt<0)),'val_nonM_unw',float(np.mean(Yv<0)),
      'MarshallO_train',float(abs(np.sum(TR['iw']*Yt))),
      'MarshallO_val',float(abs(np.sum(VA['iw']*Yv))),flush=True)

def apply_thresh(t):
 pt=np.where(TR['r']<=t,1,-1);pv=np.where(VA['r']<=t,1,-1)
 return dict(t=float(t),trainAcc=float(np.mean(pt==Yt)),valAcc=float(np.mean(pv==Yv)),
             trainO=float(abs(np.sum(TR['iw']*pt*Yt))),valO=float(abs(np.sum(VA['iw']*pv*Yv))),
             predFlipTrain=float(np.mean(pt<0)),predFlipVal=float(np.mean(pv<0)))

results={}
# Robust 2-means, no labels.
for lo,hi in ((.001,.999),(.01,.99),(.05,.95),(.1,.9)):
 a=np.clip(TR['r'],np.quantile(TR['r'],lo),np.quantile(TR['r'],hi))
 m1,m2=np.quantile(a,[.3,.8])
 for _ in range(100):
  cut=(m1+m2)/2;lab=a>cut
  if lab.all() or (~lab).all():break
  n1=a[~lab].mean();n2=a[lab].mean()
  if abs(n1-m1)+abs(n2-m2)<1e-10:break
  m1,m2=n1,n2
 t=(m1+m2)/2
 key=f'kmeans_{lo}_{hi}';results[key]=apply_thresh(t)
 print('UNSUP',key,'centers',float(m1),float(m2),results[key],flush=True)

# Otsu, no labels.
for lo,hi in ((.001,.999),(.01,.99),(.05,.95),(.1,.9)):
 a=np.sort(TR['r'][(TR['r']>=np.quantile(TR['r'],lo))&(TR['r']<=np.quantile(TR['r'],hi))])
 cs=np.cumsum(a);n=len(a);tot=cs[-1];best=(-1,None)
 for k in range(10,n-10):
  w0=k/n;w1=1-w0;m0=cs[k-1]/k;m1=(tot-cs[k-1])/(n-k)
  b=w0*w1*(m0-m1)**2
  if b>best[0]:best=(b,(a[k-1]+a[k])/2)
 key=f'otsu_{lo}_{hi}';results[key]=apply_thresh(best[1]);print('UNSUP',key,results[key],flush=True)

# Oracle threshold only as upper/capacity diagnostic.
vals=np.unique(TR['r']);mids=np.r_[vals[0]-1,(vals[:-1]+vals[1:])/2,vals[-1]+1]
best=(-1,None)
for t in mids:
 p=np.where(TR['r']<=t,1,-1);ac=np.mean(p==Yt)
 if ac>best[0]:best=(ac,t)
oracle=apply_thresh(best[1]);print('ORACLE_THRESHOLD',oracle,flush=True)

with open('lanczos1_8x8_result.json','w') as f:
 json.dump(dict(phi=float(phi),trainLeak=leakt,valLeak=leakv,
   MarshallO_train=float(abs(np.sum(TR['iw']*Yt))),MarshallO_val=float(abs(np.sum(VA['iw']*Yv))),
   hiddenNonM_train=float(np.mean(Yt<0)),hiddenNonM_val=float(np.mean(Yv<0)),
   unsupervised=results,oracle=oracle),f,indent=2)
