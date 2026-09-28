import json
from pathlib import Path
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sla
OUT=Path("/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure/results")
def audit_model(LX,LY,bonds,kind,J2=None):
 N=LX*LY; site=lambda x,y:(x%LX)+LX*(y%LY)
 basis=np.array([s for s in range(1<<N) if s.bit_count()==N//2],np.uint32);D=len(basis);idx={int(s):i for i,s in enumerate(basis)}
 rr=[];cc=[];vv=[];diag=np.zeros(D)
 for bi,s0 in enumerate(basis):
  s=int(s0);e=0.
  for u,v,J in bonds:
   if ((s>>u)&1)==((s>>v)&1):e+=.25*J
   else:e-=.25*J;t=s^(1<<u)^(1<<v);rr.append(bi);cc.append(idx[t]);vv.append(.5*J)
  diag[bi]=e
 rr.extend(range(D));cc.extend(range(D));vv.extend(diag.tolist());H=sp.coo_matrix((vv,(rr,cc)),shape=(D,D)).tocsr()
 ev,V=sla.eigsh(H,k=1,which="SA",tol=2e-11,maxiter=100000);v=V[:,0];anchor=int(np.argmax(abs(v)));v=v if v[anchor]>0 else -v
 a=np.abs(v);a/=np.linalg.norm(a);truth=np.where(v>=0,1,-1).astype(np.int8)
 def trans_perm(dx,dy):
  perm=np.empty(D,np.int32)
  for bi,s0 in enumerate(basis):
   s=int(s0);t=0
   for y in range(LY):
    for x in range(LX):
     i=site(x,y)
     if (s>>i)&1:t |= 1<<site(x+dx,y+dy)
   perm[bi]=idx[t]
  return perm
 px=trans_perm(1,0);py=trans_perm(0,1)
 def gauge(mask):return np.array([1 if ((int(s)&mask).bit_count()%2)==0 else -1 for s in basis],np.int8)
 gauges={}
 if kind=="tri":
  gauges={
   "truth":truth,
   "maxcut_x":gauge(sum(1<<site(x,y) for y in range(LY) for x in range(LX) if x%2==0)),
   "parity_y":gauge(sum(1<<site(x,y) for y in range(LY) for x in range(LX) if y%2==0)),
   "parity_xy":gauge(sum(1<<site(x,y) for y in range(LY) for x in range(LX) if (x+y)%2==0))}
 else:
  gauges={
   "truth":truth,
   "marshall":gauge(sum(1<<site(x,y) for y in range(LY) for x in range(LX) if (x+y)%2==0)),
   "stripe_x":gauge(sum(1<<site(x,y) for y in range(LY) for x in range(LX) if x%2==0))}
 def sym(psi,perm):
  tv=np.empty_like(psi);tv[perm]=psi
  c=float(np.dot(psi,tv)/np.dot(psi,psi))
  res=float(np.linalg.norm(tv-c*psi)/np.linalg.norm(psi))
  return c,res
 rows=[]
 for name,s in gauges.items():
  psi=a*s
  cx,rx=sym(psi,px);cy,ry=sym(psi,py)
  ov=float(abs(np.sum((a*a)*s*truth)))
  rows.append(dict(name=name,O_truth=ov,Tx_expect=cx,Tx_residual=rx,Ty_expect=cy,Ty_residual=ry))
 return dict(model=kind,J2=J2,E0=float(ev[0]),D=D,amp_Tx=sym(a,px),amp_Ty=sym(a,py),rows=rows)

# triangular 6x3
LX,LY=6,3;site=lambda x,y:(x%LX)+LX*(y%LY);tb=[]
for y in range(LY):
 for x in range(LX):
  i=site(x,y)
  for j in (site(x+1,y),site(x,y+1),site(x+1,y+1)):tb.append((i,j,1.0))
tri=audit_model(LX,LY,tb,"tri")
# square 4x4 at J2=.8 and 1
sq=[]
for J2 in (0.8,1.0):
 LX=LY=4;site=lambda x,y:(x%LX)+LX*(y%LY);b=[]
 for y in range(LY):
  for x in range(LX):
   i=site(x,y)
   b += [(i,site(x+1,y),1.0),(i,site(x,y+1),1.0),(i,site(x+1,y+1),J2),(i,site(x+1,y-1),J2)]
 sq.append(audit_model(LX,LY,b,"square",J2))
res=dict(triangular=tri,square=sq)
Path(OUT/"symmetry_sector_audit.json").write_text(json.dumps(res,indent=2))
for block in [tri]+sq:
 print("MODEL",block["model"],block["J2"],"amp",block["amp_Tx"],block["amp_Ty"],flush=True)
 for r in block["rows"]: print(r,flush=True)
