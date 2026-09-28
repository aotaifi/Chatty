import json
from pathlib import Path
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sla

L=4;N=16;MAXITER=30
TARGETS=[0.5,1.0]
AMP_SOURCES=[0.0,0.2,0.4,0.5,0.6,0.8,1.0]
OUT=Path("/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure/results")
site=lambda x,y:(x%L)+L*(y%L)
NN=[];NNN=[]
for y in range(L):
 for x in range(L):
  i=site(x,y);NN += [(i,site(x+1,y)),(i,site(x,y+1))];NNN += [(i,site(x+1,y+1)),(i,site(x+1,y-1))]
basis=np.array([s for s in range(1<<N) if s.bit_count()==N//2],np.uint32);D=len(basis);idx={int(s):i for i,s in enumerate(basis)}

def build(J2):
 rr=[];cc=[];vv=[];diag=np.zeros(D)
 for bi,s0 in enumerate(basis):
  s=int(s0);e=0.
  for bonds,J in ((NN,1.),(NNN,J2)):
   for u,v in bonds:
    if ((s>>u)&1)==((s>>v)&1):e+=.25*J
    else:e-=.25*J;t=s^(1<<u)^(1<<v);rr.append(bi);cc.append(idx[t]);vv.append(.5*J)
  diag[bi]=e
 rr.extend(range(D));cc.extend(range(D));vv.extend(diag.tolist())
 return sp.coo_matrix((vv,(rr,cc)),shape=(D,D)).tocsr(),diag

def ground(J2):
 H,d=build(J2);ev,V=sla.eigsh(H,k=1,which="SA",tol=1e-12,maxiter=100000);v=np.asarray(V[:,0],float);anchor=int(np.argmax(abs(v)));v=v if v[anchor]>0 else -v
 a=np.abs(v);a/=np.linalg.norm(a);truth=np.where(v>=0,1,-1).astype(np.int8)
 return H,d,float(ev[0]),a,truth,anchor

cache={}
for j in sorted(set(TARGETS+AMP_SOURCES)):
 print("ED",j,flush=True);cache[j]=ground(j)

def mask(kind):
 if kind=="marshall":return sum(1<<site(x,y) for y in range(L) for x in range(L) if (x+y)%2==0)
 return sum(1<<site(x,y) for y in range(L) for x in range(L) if x%2==0)
def gauge(kind):
 m=mask(kind);return np.array([1 if ((int(s)&m).bit_count()%2)==0 else -1 for s in basis],np.int8)
def canon(s,anchor):
 s=s.copy()
 if s[anchor]<0:s=-s
 return s
def overlap(p,s,t):return float(abs(np.sum(p*s*t)))
def gids(r):
 o=np.argsort(r,kind="mergesort");rs=r[o];sc=np.maximum(np.maximum(np.abs(rs[1:]),np.abs(rs[:-1])),1.)
 br=np.empty(D,bool);br[0]=1;br[1:]=np.abs(rs[1:]-rs[:-1])>(1e-11+1e-11*sc);gs=np.cumsum(br,dtype=np.int32)-1
 g=np.empty(D,np.int32);g[o]=gs;return g,int(gs[-1])+1

def run(target,asrc,base):
 H,d,E0,atrue,truth,anchor=cache[target]
 _,_,_,a,_,_=cache[asrc]
 ptrue=atrue*atrue
 # score amplitude mismatch
 amp_overlap=float(np.dot(a,atrue)); amp_fidelity=amp_overlap**2
 l1_prob=float(0.5*np.sum(np.abs(a*a-atrue*atrue)))
 co=sp.triu(H-sp.diags(d),k=1).tocoo();ei=co.row.astype(np.int32);ej=co.col.astype(np.int32);w=co.data*a[ei]*a[ej]
 def upd(s):
  psi=a*s;r=(H@psi)/np.where(a>1e-300,psi,1.);g,G=gids(r);c0=-s;val=2*w*c0[ei]*c0[ej]
  e0=float(np.sum(a*a*d)+np.sum(val));gi=g[ei];gj=g[ej];lo=np.minimum(gi,gj);hi=np.maximum(gi,gj);m=lo<hi
  de=np.bincount(lo[m],weights=-2*val[m],minlength=G)+np.bincount(hi[m],weights=2*val[m],minlength=G);es=e0+np.cumsum(de);k=int(np.argmin(np.r_[e0,es]))-1
  sn=c0.copy()
  if k>=0:sn[g<=k]*=-1
  return canon(sn,anchor),float(np.min(np.r_[e0,es]))
 s=canon(gauge(base),anchor);seen={};traj=[]
 for it in range(MAXITER+1):
  O=overlap(ptrue,s,truth)
  # target-H energy with approximate amplitudes; true sign score uses target probability
  E=float((a*s)@(H@(a*s)))
  traj.append(dict(it=it,O_true_sign=O,E_fixed_approx_amp=E))
  if O>1-1e-12:term="exact_signs";break
  key=np.packbits(s>0).tobytes()
  if key in seen:term=f"cycle:{seen[key]}->{it}";break
  seen[key]=it;sn,_=upd(s)
  if np.array_equal(sn,s):term="fixed";break
  s=sn
 else:term="maxiter"
 return dict(target=target,amp_source=asrc,baseline=base,amp_fidelity=amp_fidelity,prob_TV=l1_prob,
             terminal=term,iterations=it,final_O=traj[-1]["O_true_sign"],trajectory=traj)

rows=[]
for target in TARGETS:
 base="marshall" # deliberately keep the same simple baseline
 for src in AMP_SOURCES:
  q=run(target,src,base);rows.append(q)
  print("RESULT",target,src,"F",q["amp_fidelity"],"TV",q["prob_TV"],q["terminal"],q["iterations"],q["final_O"],flush=True)

OUT.joinpath("amplitude_mismatch_square.json").write_text(json.dumps(rows,indent=2))
