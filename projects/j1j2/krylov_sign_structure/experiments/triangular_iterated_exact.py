import json
from pathlib import Path
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sla
LX,LY=6,3;N=LX*LY;MAXITER=40
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
  else:
   e-=.25;t=s^(1<<u)^(1<<v);rr.append(bi);cc.append(idx[t]);vv.append(.5)
 diag[bi]=e
rr.extend(range(D));cc.extend(range(D));vv.extend(diag.tolist())
H=sp.coo_matrix((vv,(rr,cc)),shape=(D,D)).tocsr()
ev,V=sla.eigsh(H,k=2,which="SA",tol=2e-11,maxiter=100000);j=int(np.argmin(ev));v=V[:,j];anchor=int(np.argmax(abs(v)));v=v if v[anchor]>0 else -v
a=np.abs(v);p=a*a;p/=p.sum();truth=np.where(v>=0,1,-1).astype(np.int8)
co=sp.triu(H-sp.diags(diag),k=1).tocoo();ei=co.row.astype(np.int32);ej=co.col.astype(np.int32);w=co.data*a[ei]*a[ej]
def gauge(mask):return np.array([1 if ((int(s)&mask).bit_count()%2)==0 else -1 for s in basis],np.int8)
masks={
 "maxcut_x":sum(1<<site(x,y) for y in range(LY) for x in range(LX) if x%2==0),
 "parity_y":sum(1<<site(x,y) for y in range(LY) for x in range(LX) if y%2==0),
 "parity_xy":sum(1<<site(x,y) for y in range(LY) for x in range(LX) if (x+y)%2==0)}
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
 psi=a*s;r=(H@psi)/np.where(a>1e-300,psi,1.);g,G=gids(r);c0=-s
 val=2*w*c0[ei]*c0[ej];e0=float(np.sum(a*a*diag)+np.sum(val));gi=g[ei];gj=g[ej];lo=np.minimum(gi,gj);hi=np.maximum(gi,gj);m=lo<hi
 de=np.bincount(lo[m],weights=-2*val[m],minlength=G)+np.bincount(hi[m],weights=2*val[m],minlength=G);es=e0+np.cumsum(de);k=int(np.argmin(np.r_[e0,es]))-1
 sn=c0.copy()
 if k>=0:sn[g<=k]*=-1
 # oracle capacity on same scalar coordinate
 raw0=float(np.sum(p*c0*truth));con=np.bincount(g,weights=2*p*s*truth,minlength=G);oracle=float(np.max(np.abs(np.r_[raw0,raw0+np.cumsum(con)])))
 return canon(sn),oracle,float(np.min(np.r_[e0,es])),G
rows=[];summary=[]
for name,m in masks.items():
 s=canon(gauge(m));seen={};print("CASE",name,"O0",O(s),flush=True)
 term="maxiter"
 for it in range(MAXITER+1):
  ov=O(s);rows.append(dict(baseline=name,it=it,O=ov))
  if it<=12 or it%5==0:print("ITER",it,"O",ov,flush=True)
  if ov>1-1e-12:term="exact";print("EXACT",it,flush=True);break
  key=np.packbits(s>0).tobytes()
  if key in seen:term=f"cycle:{seen[key]}->{it}";print("CYCLE",seen[key],it,flush=True);break
  seen[key]=it;sn,oracle,E,G=upd(s);rows[-1].update(oracle_next=oracle,n_groups=G)
  if np.array_equal(sn,s):term="fixed";print("FIXED",it,"oracle",oracle,flush=True);break
  s=sn
 summary.append(dict(baseline=name,terminal=term,iterations=it,final_O=O(s)))
Path(OUT/"triangular_iterated_exact.json").write_text(json.dumps(dict(summary=summary,rows=rows),indent=2))
print("SUMMARY",json.dumps(summary),flush=True)
