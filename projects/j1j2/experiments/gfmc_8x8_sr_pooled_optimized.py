import math,time,json
import numpy as np
import jax,jax.numpy as jnp
import netket as nk, flax
import netket.jax as nkjax
from nqsmagic.models import ViT
from nqsmagic.utils import _logpsi_transl_2d

L=8;N=64;J2=.5
T=float(json.load(open('vit8_fnmle_sr_pooled_node.json'))['T1'])
hi=nk.hilbert.Spin(s=.5,N=N)
g=nk.graph.Hypercube(length=L,n_dim=2,pbc=True,max_neighbor_order=2)
model=ViT(num_layers=8,d_model=60,heads=10,L_eff=16,b=2,transl_invariant=True,two_dimensional=True)
apply=nkjax.HashablePartial(_logpsi_transl_2d,model.apply,2)
sam0=nk.sampler.MetropolisExchange(hi,graph=g,d_max=2,n_chains=16,sweep_size=N)
v0=nk.vqs.MCState(sampler=sam0,apply_fun=apply,n_samples=16,
 variables=model.init(jax.random.PRNGKey(1234),jnp.zeros((1,N))),n_discard_per_chain=10)
with open('vit8_fnmle_sr_pooled.mpack','rb') as f:
 obj=flax.serialization.msgpack_restore(f.read())
v0.variables=flax.serialization.from_state_dict(v0.variables,obj)
print("LOADED_8x8",v0.n_parameters,"T",T,flush=True)

def evalz(X,batch=4096):
 out=[];X=np.asarray(X,float)
 for i in range(0,len(X),batch):out.append(np.asarray(v0.log_value(jnp.asarray(X[i:i+batch]))))
 return np.concatenate(out)
def bits2x(ss):
 a=np.asarray(ss,np.uint64).reshape(-1,1)
 return 2*((a>>np.arange(N,dtype=np.uint64))&1).astype(float)-1
def bonds():
 nn=[];nnn=[];q=lambda x,y:(x%L)+L*(y%L)
 for y in range(L):
  for x in range(L):
   i=q(x,y);nn += [(i,q(x+1,y)),(i,q(x,y+1))]
   nnn += [(i,q(x+1,y+1)),(i,q(x+1,y-1))]
 return nn,nnn
NN,NNN=bonds();ALL=[(i,j,1.,-1.) for i,j in NN]+[(i,j,J2,1.) for i,j in NNN]
A_MASK=sum(1<<(x+L*y) for y in range(L) for x in range(L) if (x+y)%2==0)
def marshall(s):return 1 if ((int(s)&A_MASK).bit_count()%2)==0 else -1
def diag_energy(s):
 e=0.
 for i,j,J,mr in ALL:e += J*(.25 if (((s>>i)&1)==((s>>j)&1)) else -.25)
 return e
def neigh(s):
 out=[]
 for i,j,J,mr in ALL:
  if ((s>>i)^(s>>j))&1:out.append((s^(1<<i)^(1<<j),J,mr))
 return out

logc={};rc={};signc={};localc={};neval=0
def ensure_log(ss):
 global neval
 miss=[int(x) for x in np.unique(np.asarray(ss,np.uint64)) if int(x) not in logc]
 if miss:
  z=evalz(bits2x(np.asarray(miss,np.uint64))).real;neval+=len(miss)
  for x,v in zip(miss,z):logc[x]=float(v)
def ensure_r(ss):
 miss=[int(x) for x in np.unique(np.asarray(ss,np.uint64)) if int(x) not in rc]
 if not miss:return
 alln=[]
 meta=[]
 for x in miss:
  ls=neigh(x);meta.append(ls);alln.extend(y for y,J,mr in ls)
 ensure_log(miss+alln)
 for x,ls in zip(miss,meta):
  lx=logc[x];r=diag_energy(x)
  for y,J,mr in ls:r += .5*J*mr*math.exp(logc[y]-lx)
  rc[x]=float(r);signc[x]=marshall(x)*(1 if r<=T else -1)
def ensure_local(ss):
 miss=[int(x) for x in np.unique(np.asarray(ss,np.uint64)) if int(x) not in localc]
 if not miss:return
 alln=[];met=[]
 for x in miss:
  ls=neigh(x);met.append(ls);alln.extend(y for y,J,mr in ls)
 ensure_r(miss+alln)
 for x,ls in zip(miss,met):
  lx=logc[x];sx=signc[x];d=diag_energy(x);ys=[];rates=[]
  for y,J,mr in ls:
   rat=math.exp(logc[y]-lx)
   if sx*signc[y]<0:
    ys.append(y);rates.append(.5*J*rat)
   else:d += .5*J*rat
  rates=np.asarray(rates,float);ys=np.asarray(ys,np.uint64)
  localc[x]=(float(d),float(d-rates.sum()),ys,rates)

def systematic(w,rng,M):
 c=np.cumsum(w);c[-1]=1
 return np.searchsorted(c,rng.random()/M+np.arange(M)/M,'right')

def run(M=128,beta_target=1.4,burn_beta=.5,tau_max=.025,seed=777,outfile=None):
 rg=np.random.default_rng(seed)
 hand=np.load('vit8_fnmle_sr_pooled_handoff.npz')
 pool=hand['pool_states'].astype(np.uint64)
 pw=hand['pool_weights'].astype(float); pw/=pw.sum()
 walkers=pool[rg.choice(len(pool),M,replace=False,p=pw)].copy()
 beta=0.;it=0;Es=[];snaps=[];t0=time.time()
 while beta<beta_target:
  ensure_local(walkers);dat=[localc[int(x)] for x in walkers]
  diag=np.array([z[0] for z in dat]);el=np.array([z[1] for z in dat]);Eref=float(el.mean())
  mx=max(0.,float(np.max(diag-Eref)));tau=min(tau_max,0.8/mx if mx>0 else tau_max)
  nxt=np.empty(M,np.uint64);bw=np.empty(M)
  for k,(x,(d,e,ys,rate)) in enumerate(zip(walkers,dat)):
   stay=1-tau*(d-Eref);ws=tau*rate;tot=stay+ws.sum()
   if stay<0 or tot<=0:raise RuntimeError(("bad",stay,tot,tau,d,Eref))
   u=rg.random()*tot
   if u<stay:y=x
   else:
    j=np.searchsorted(np.cumsum(ws),u-stay,'right');y=int(ys[min(j,len(ys)-1)])
   nxt[k]=y;bw[k]=tot
  bw/=bw.sum();walkers=nxt[systematic(bw,rg,M)]
  beta+=tau;it+=1
  if beta>=burn_beta:
   ensure_local(walkers);ee=float(np.mean([localc[int(x)][1] for x in walkers]))
   Es.append(ee);snaps.append(walkers.copy())
  if it==1 or it%5==0:
   print("PROG",it,"beta",beta,"tau",tau,"E",Eref,"uniq",len(np.unique(walkers)),
         "rc",len(rc),"local",len(localc),"neval",neval,"sec",time.time()-t0,flush=True)
 mixed=np.concatenate(snaps)
 print("RESULT Emean",np.mean(Es),"tail",np.mean(Es[-min(8,len(Es)):]),
       "naiveSE",np.std(Es,ddof=1)/np.sqrt(len(Es)) if len(Es)>1 else np.nan,
       "nE",len(Es),"mixed",len(mixed),"unique",len(np.unique(mixed)),
       "neval",neval,"sec",time.time()-t0,flush=True)
 np.savez_compressed(outfile or f'gfmc_8x8_krylov_collect_seed{seed}.npz',mixed=mixed,Es=np.asarray(Es),threshold=T)
 return mixed,Es

def weighted_quantile(x,w,p):
    o=np.argsort(x); xx=x[o]; ww=w[o]; c=np.cumsum(ww); c/=c[-1]
    return np.interp(p,c,xx)

def fit_g(mixed,edges,qmass,pc=.5):
    ensure_r(mixed)
    rr=np.array([rc[int(x)] for x in mixed])
    b=np.clip(np.searchsorted(edges,rr,side="right")-1,0,3)
    cc=np.bincount(b,minlength=4).astype(float)
    fm=(cc+pc)/(cc.sum()+4*pc)
    gg=np.log(fm/qmass); gg-=np.sum(qmass*gg)
    return gg,cc,rr


if __name__=="__main__":
    allrows=[]
    for seed in (10501,10502):
        mix,Es=run(M=128,beta_target=1.2,burn_beta=.4,tau_max=.025,seed=seed,
                   outfile=f"sr_pool_fn_seed{seed}.npz")
        E=np.asarray(Es,float)
        row=[seed,float(E.mean()),float(E[-min(8,len(E)):].mean()),len(E),len(mix),len(np.unique(mix))]
        allrows.append(row); print("SR_POOL_REP",row,flush=True)
    e=np.array([r[1] for r in allrows]); B=-31.8854241096
    out={"T1":T,"E_replicas":e.tolist(),"Emean":float(e.mean()),
         "between_replica_SE":float(abs(e[0]-e[1])/2),
         "delta_vs_baseline":float(e.mean()-B),
         "tail8":[r[2] for r in allrows],"rows":allrows,"amp_eval":neval}
    json.dump(out,open("sr_pool_fn_summary.json","w"),indent=2)
    print("SR_POOL_SUMMARY",json.dumps(out),flush=True)
