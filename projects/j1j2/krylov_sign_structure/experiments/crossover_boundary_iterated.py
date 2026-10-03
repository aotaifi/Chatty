import json
from pathlib import Path
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sla
L=4;N=16;MAXITER=60
CASES=[(0.645,"marshall"),(0.645,"stripe_x"),(0.650,"marshall"),(0.650,"stripe_x"),(0.655,"marshall"),(0.655,"stripe_x")]
OUT=Path("/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure/results")
site=lambda x,y:(x%L)+L*(y%L)
NN=[];NNN=[]
for y in range(L):
  for x in range(L):
    i=site(x,y);NN += [(i,site(x+1,y)),(i,site(x,y+1))];NNN += [(i,site(x+1,y+1)),(i,site(x+1,y-1))]
basis=np.array([s for s in range(1<<N) if s.bit_count()==N//2],np.uint32);D=len(basis);pos={int(s):i for i,s in enumerate(basis)}
def build(J2):
 rr=[];cc=[];vv=[];diag=np.zeros(D)
 for bi,s0 in enumerate(basis):
  s=int(s0);e=0.
  for bonds,J in ((NN,1.),(NNN,J2)):
   for u,v in bonds:
    if ((s>>u)&1)==((s>>v)&1):e+=.25*J
    else:
     e-=.25*J;t=s^(1<<u)^(1<<v);rr.append(bi);cc.append(pos[t]);vv.append(.5*J)
  diag[bi]=e
 rr.extend(range(D));cc.extend(range(D));vv.extend(diag.tolist())
 return sp.coo_matrix((vv,(rr,cc)),shape=(D,D)).tocsr(),diag
def gauge(kind):
 if kind=="marshall":m=sum(1<<site(x,y) for y in range(L) for x in range(L) if (x+y)%2==0)
 else:m=sum(1<<site(x,y) for y in range(L) for x in range(L) if x%2==0)
 return np.array([1 if ((int(s)&m).bit_count()%2)==0 else -1 for s in basis],np.int8)
def canon(s,a):
 s=s.copy()
 if s[a]<0:s=-s
 return s
def gids(r):
 o=np.argsort(r,kind="mergesort");rs=r[o];sc=np.maximum(np.maximum(np.abs(rs[1:]),np.abs(rs[:-1])),1.)
 br=np.empty(D,bool);br[0]=1;br[1:]=np.abs(rs[1:]-rs[:-1])>(1e-11+1e-11*sc);gs=np.cumsum(br,dtype=np.int32)-1
 g=np.empty(D,np.int32);g[o]=gs;return g,int(gs[-1])+1
def prep(H,a,d):
 co=sp.triu(H-sp.diags(d),k=1).tocoo();return co.row.astype(np.int32),co.col.astype(np.int32),co.data*a[co.row]*a[co.col]
def upd(H,a,p,truth,s,anchor,ei,ej,w):
 psi=a*s;r=(H@psi)/np.where(a>1e-300,psi,1.);g,G=gids(r);c0=-s
 val=2*w*c0[ei]*c0[ej];e0=float(np.sum(a*a*H.diagonal())+np.sum(val));gi=g[ei];gj=g[ej];lo=np.minimum(gi,gj);hi=np.maximum(gi,gj);m=lo<hi
 d=np.bincount(lo[m],weights=-2*val[m],minlength=G)+np.bincount(hi[m],weights=2*val[m],minlength=G)
 es=e0+np.cumsum(d);k=int(np.argmin(np.r_[e0,es]))-1
 sn=c0.copy()
 if k>=0:sn[g<=k]*=-1
 return canon(sn,anchor),float(np.min(np.r_[e0,es]))
def O(p,s,t):return float(abs(np.sum(p*s*t)))
rows=[]
for J2,base in CASES:
 H,d=build(J2);ev,V=sla.eigsh(H,k=1,which="SA",tol=1e-12,maxiter=100000);v=V[:,0];anchor=int(np.argmax(abs(v)));v=v if v[anchor]>0 else -v
 a=np.abs(v);p=a*a;p/=p.sum();truth=np.where(v>=0,1,-1).astype(np.int8);ei,ej,w=prep(H,a,d);s=canon(gauge(base),anchor);seen={}
 print("CASE",J2,base,"O0",O(p,s,truth),flush=True)
 for it in range(MAXITER+1):
  ov=O(p,s,truth);key=np.packbits(s>0).tobytes();rows.append(dict(J2=J2,baseline=base,it=it,O=ov))
  if it<=10 or it%5==0:print("ITER",it,"O",ov,flush=True)
  if ov>1-1e-12:print("EXACT",it,flush=True);break
  if key in seen:print("CYCLE",seen[key],it,flush=True);break
  seen[key]=it;sn,E=upd(H,a,p,truth,s,anchor,ei,ej,w)
  if np.array_equal(sn,s):print("FIXED",it,flush=True);break
  s=sn
Path(OUT/"crossover_boundary_iterated.json").write_text(json.dumps(rows,indent=2))
