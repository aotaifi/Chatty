import json
from pathlib import Path
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sla
LX,LY=6,3;N=LX*LY;MAXITER=30;TRIALS=8
Q=[0.001,0.01,0.05,0.10,0.20,0.30,0.35,0.40,0.45,0.49];SEED=20260928
OUT=Path("/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure/results")
site=lambda x,y:(x%LX)+LX*(y%LY)
B=[]
for y in range(LY):
 for x in range(LX):
  i=site(x,y);B += [(i,site(x+1,y)),(i,site(x,y+1)),(i,site(x+1,y+1))]
basis=np.array([s for s in range(1<<N) if s.bit_count()==N//2],np.uint32);D=len(basis);idx={int(s):i for i,s in enumerate(basis)}
rr=[];cc=[];vv=[];diag=np.zeros(D)
for bi,s0 in enumerate(basis):
 s=int(s0);e=0.
 for u,v in B:
  if ((s>>u)&1)==((s>>v)&1):e+=.25
  else:e-=.25;t=s^(1<<u)^(1<<v);rr.append(bi);cc.append(idx[t]);vv.append(.5)
 diag[bi]=e
rr.extend(range(D));cc.extend(range(D));vv.extend(diag.tolist());H=sp.coo_matrix((vv,(rr,cc)),shape=(D,D)).tocsr()
ev,V=sla.eigsh(H,k=1,which="SA",tol=2e-11,maxiter=100000);v=V[:,0];anchor=int(np.argmax(abs(v)));v=v if v[anchor]>0 else -v
a=np.abs(v);p=a*a;p/=p.sum();truth=np.where(v>=0,1,-1).astype(np.int8)
co=sp.triu(H-sp.diags(diag),k=1).tocoo();ei=co.row.astype(np.int32);ej=co.col.astype(np.int32);w=co.data*a[ei]*a[ej]
def canon(s):
 s=s.copy()
 if s[anchor]<0:s=-s
 return s
def O(s):return float(abs(np.sum(p*s*truth)))
def gids(r):
 o=np.argsort(r,kind="mergesort");rs=r[o];sc=np.maximum(np.maximum(np.abs(rs[1:]),np.abs(rs[:-1])),1.)
 br=np.empty(D,bool);br[0]=1;br[1:]=np.abs(rs[1:]-rs[:-1])>(1e-11+1e-11*sc);gs=np.cumsum(br,dtype=np.int32)-1
 g=np.empty(D,np.int32);g[o]=gs;return g,int(gs[-1])+1
def upd(s):
 psi=a*s;r=(H@psi)/np.where(a>1e-300,psi,1.);g,G=gids(r);c0=-s;val=2*w*c0[ei]*c0[ej]
 e0=float(np.sum(a*a*diag)+np.sum(val));gi=g[ei];gj=g[ej];lo=np.minimum(gi,gj);hi=np.maximum(gi,gj);m=lo<hi
 de=np.bincount(lo[m],weights=-2*val[m],minlength=G)+np.bincount(hi[m],weights=2*val[m],minlength=G);es=e0+np.cumsum(de);k=int(np.argmin(np.r_[e0,es]))-1
 sn=c0.copy()
 if k>=0:sn[g<=k]*=-1
 return canon(sn)
def corrupt(q,rng):
 inds=np.arange(D,dtype=np.int32);inds=inds[inds!=anchor];rng.shuffle(inds);cs=np.cumsum(p[inds]);j=int(np.searchsorted(cs,q));opts=[]
 if j<len(inds):opts.append(j+1)
 if j>0:opts.append(j)
 n=min(opts,key=lambda n:abs(float(cs[n-1])-q));flip=inds[:n];s=truth.copy();s[flip]*=-1
 return s,float(np.sum(p[flip]))
R=[];master=np.random.default_rng(SEED)
for q in Q:
 suc=[];its=[];fis=[];acts=[]
 for tr in range(TRIALS):
  s,qa=corrupt(q,np.random.default_rng(int(master.integers(0,2**63-1))));acts.append(qa);seen=set();ok=False;nit=0
  for it in range(MAXITER+1):
   if O(s)>1-1e-12:ok=True;nit=it;break
   key=np.packbits(s>0).tobytes()
   if key in seen:nit=it;break
   seen.add(key);sn=upd(s)
   if np.array_equal(sn,s):nit=it;break
   s=sn;nit=it+1
  suc.append(ok);its.append(nit);fis.append(O(s))
 row=dict(q=q,q_actual=float(np.mean(acts)),O0=float(np.mean([1-2*x for x in acts])),success=float(np.mean(suc)),
          median_it=float(np.median([n for n,o in zip(its,suc) if o])) if any(suc) else None,Ofinal=float(np.mean(fis)))
 R.append(row);print("Q",q,"success",row["success"],"O0",row["O0"],"Ofinal",row["Ofinal"],"med",row["median_it"],flush=True)
Path(OUT/"triangular_basin_exact.json").write_text(json.dumps(R,indent=2));print("DONE",flush=True)
