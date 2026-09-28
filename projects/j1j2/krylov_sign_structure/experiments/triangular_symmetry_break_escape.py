import json
from pathlib import Path
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sla

LX,LY=6,3;N=LX*LY;MAXITER=60;TRIALS=12
Q=[1e-4,5e-4,1e-3,5e-3,1e-2,2e-2,5e-2,1e-1]
SEED=20260928
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

mask=sum(1<<site(x,y) for y in range(LY) for x in range(LX) if y%2==0)
base=np.array([1 if ((int(s)&mask).bit_count()%2)==0 else -1 for s in basis],np.int8)

def canon(s):
 s=s.copy()
 if s[anchor]<0:s=-s
 return s
base=canon(base)
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

# inversion P:(x,y)->(-x,2-y) and config permutation
mp=[site(-x,2-y) for y in range(LY) for x in range(LX)]
perm=np.empty(D,np.int32)
for bi,s0 in enumerate(basis):
 s=int(s0);t=0
 for i,j in enumerate(mp):
  if (s>>i)&1:t|=1<<j
 perm[bi]=idx[t]
def sym_expect(s):
 psi=a*s;tv=np.empty_like(psi);tv[perm]=psi
 c=float(np.dot(psi,tv)/np.dot(psi,psi))
 res=float(np.linalg.norm(tv-c*psi)/np.linalg.norm(psi))
 return c,res

def perturb(q,rng):
 inds=np.array([i for i in range(D) if i!=anchor],np.int32);rng.shuffle(inds);cs=np.cumsum(p[inds]);j=int(np.searchsorted(cs,q));opts=[]
 if j<len(inds):opts.append(j+1)
 if j>0:opts.append(j)
 n=min(opts,key=lambda n:abs(float(cs[n-1])-q));flip=inds[:n];s=base.copy();s[flip]*=-1
 return canon(s),float(np.sum(p[flip]))

master=np.random.default_rng(SEED);rows=[];summ=[]
print("BASE O",O(base),"sym",sym_expect(base),flush=True)
for q in Q:
 suc=[];its=[];ofs=[];qacts=[];syms=[]
 for tr in range(TRIALS):
  s,qa=perturb(q,np.random.default_rng(int(master.integers(0,2**63-1))));qacts.append(qa);syms.append(sym_expect(s)[1]);seen=set();ok=False
  for it in range(MAXITER+1):
   ov=O(s)
   if ov>1-1e-12:ok=True;break
   key=np.packbits(s>0).tobytes()
   if key in seen:break
   seen.add(key);sn=upd(s)
   if np.array_equal(sn,s):break
   s=sn
  suc.append(ok);its.append(it);ofs.append(O(s))
  rows.append(dict(q=q,trial=tr,q_actual=qa,initial_O=O(perturb(q,np.random.default_rng(0))[0]) if False else None,
                   initial_sym_res=syms[-1],success=ok,iterations=it,final_O=ofs[-1]))
 sm=dict(q=q,q_actual=float(np.mean(qacts)),success=float(np.mean(suc)),
         median_it=float(np.median([i for i,o in zip(its,suc) if o])) if any(suc) else None,
         final_O_mean=float(np.mean(ofs)),initial_sym_res_mean=float(np.mean(syms)))
 summ.append(sm);print("Q",q,sm,flush=True)
OUT.joinpath("triangular_symmetry_break_escape.json").write_text(json.dumps(dict(summary=summ,rows=rows),indent=2))
