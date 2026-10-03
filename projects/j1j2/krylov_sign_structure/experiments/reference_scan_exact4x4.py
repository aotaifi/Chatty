import itertools,json
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
L=4;N=16
J2S=[0.4,0.5,0.6,0.65,0.7,0.8,0.9,1.0]
q=lambda x,y:(x%L)+L*(y%L)
NN=[];NNN=[]
for y in range(L):
  for x in range(L):
    i=q(x,y); NN += [(i,q(x+1,y)),(i,q(x,y+1))]
    NNN += [(i,q(x+1,y+1)),(i,q(x+1,y-1))]
states=[]
for c in itertools.combinations(range(N),N//2):
  s=0
  for i in c:s|=1<<i
  states.append(s)
states=np.asarray(states,np.uint32);dim=len(states);idx={int(s):i for i,s in enumerate(states)}
def build(bonds):
  rr=[];cc=[];vv=[];d=np.zeros(dim)
  for a,s0 in enumerate(states):
    s=int(s0)
    for i,j in bonds:
      same=((s>>i)&1)==((s>>j)&1);d[a]+=.25 if same else -.25
      if not same:rr.append(a);cc.append(idx[s^(1<<i)^(1<<j)]);vv.append(.5)
  rr.extend(range(dim));cc.extend(range(dim));vv.extend(d.tolist())
  return sp.coo_matrix((vv,(rr,cc)),shape=(dim,dim)).tocsr()
H1=build(NN);H2=build(NNN)
masks={
 'neel':sum(1<<(x+L*y) for y in range(L) for x in range(L) if (x+y)%2==0),
 'stripe_x':sum(1<<(x+L*y) for y in range(L) for x in range(L) if x%2==0),
 'stripe_y':sum(1<<(x+L*y) for y in range(L) for x in range(L) if y%2==0)}
refs={n:np.asarray([1. if ((int(s)&m).bit_count()%2)==0 else -1. for s in states]) for n,m in masks.items()}
def eopt(H,a,r,sref):
  W=H.copy().tolil();W.setdiag(0);W=W.tocsr();W=sp.diags(a)@W@sp.diags(a)
  dE=float(np.sum(H.diagonal()*a*a))
  s=-sref.copy();E=dE+float(s@(W@s));h=np.asarray(W@s).ravel();order=np.argsort(r)
  bE=float(E);bs=s.copy();bk=-1
  for k,ii in enumerate(order):
    old=s[ii];E += -4*old*h[ii];s[ii]=-old
    st,en=W.indptr[ii],W.indptr[ii+1];js=W.indices[st:en];ws=W.data[st:en]
    h[js] += -2*old*ws
    if E<bE-1e-13:bE=float(E);bs=s.copy();bk=k
  return bE,bs,bk
v0=None;out=[]
for J2 in J2S:
  H=H1+J2*H2
  ev,g=sla.eigsh(H,k=1,which='SA',tol=2e-11,maxiter=200000,v0=v0)
  g=np.asarray(g[:,0]);v0=g.copy()
  # choose a stable global orientation using Neel reference
  if np.dot(g,refs['neel'])<0:g=-g
  a=np.abs(g);p=a*a;p/=p.sum();st=np.where(g>=0,1.,-1.)
  row={'J2':J2,'E0':float(ev[0])}
  for name,sref in refs.items():
    # global orientation irrelevant in overlap
    ob=abs(float(np.sum(p*sref*st))); wb=(1-ob)/2
    psi=a*sref
    safe=np.where(np.abs(psi)>1e-300,psi,1e-300)
    r=np.asarray((H@psi)/safe,float)
    Eb,sb,k=eopt(H,a,r,sref)
    oo=abs(float(np.sum(p*sb*st)));wo=(1-oo)/2
    flip=min(float(np.sum(p[sb!=sref])),float(np.sum(p[sb!=-sref])))
    row[name]={'O0':ob,'wrong0':wb,'E0ref':float(psi@(H@psi)),
               'Eopt':Eb,'O1':oo,'wrong1':wo,
               'removed':((wb-wo)/wb if wb>1e-15 else None),'flipmass':flip}
  out.append(row);print('ROW',json.dumps(row,sort_keys=True),flush=True)
with open('reference_scan_exact4x4.json','w') as f:json.dump(out,f,indent=2)
