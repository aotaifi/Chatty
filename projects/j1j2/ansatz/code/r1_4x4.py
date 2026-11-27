# R1 known case on 4x4 J1-J2 (J2=0.5, PBC, Sz=0): ten Haaf FN energy before/after R1 flips; broken = flip where B<0.
import numpy as np, itertools, scipy.sparse as sp, scipy.sparse.linalg as sl
L=4;N=16;J2=0.5
def idx(i,j):return (i%L)*L+(j%L)
b1=set();b2=set()
for i in range(L):
  for j in range(L):
    for d in [(0,1),(1,0)]: b1.add(tuple(sorted((idx(i,j),idx(i+d[0],j+d[1])))))
    for d in [(1,1),(1,-1)]: b2.add(tuple(sorted((idx(i,j),idx(i+d[0],j+d[1])))))
bonds=[(a,b,1.0) for a,b in b1]+[(a,b,J2) for a,b in b2]
st=[s for s in range(1<<N) if bin(s).count('1')==8];D=len(st);pos={s:k for k,s in enumerate(st)}
r=[];c=[];v=[];diag=np.zeros(D)
for k,s in enumerate(st):
  for a,b,J in bonds:
    sa=(s>>a)&1;sb=(s>>b)&1
    if sa==sb: diag[k]+=J/4
    else:
      diag[k]-=J/4; t=s^(1<<a)^(1<<b); r.append(k);c.append(pos[t]);v.append(J/2)
H=sp.csr_matrix((v,(r,c)),shape=(D,D))+sp.diags(diag)
Hoff=sp.csr_matrix((v,(r,c)),shape=(D,D))
E0,psi=sl.eigsh(H,k=1,which='SA');E0=E0[0];psi=psi[:,0]
# Marshall sign (sublattice A spins up count parity)
A=[idx(i,j) for i in range(L) for j in range(L) if (i+j)%2==0]
mars=np.array([(-1)**sum((s>>a)&1 for a in A) for s in st],float)
S0=np.sign(psi); S0[S0==0]=1
if (S0*mars).sum()<0: mars=-mars
def efn(S,a):
  Ho=Hoff.tocoo(); m=Ho.data*S[Ho.row]*S[Ho.col]
  keep=m<=0; viol=~keep
  K=sp.csr_matrix((m[keep],(Ho.row[keep],Ho.col[keep])),shape=(D,D))
  V=np.zeros(D); np.add.at(V,Ho.row[viol],m[viol]*a[Ho.col[viol]]/a[Ho.row[viol]])
  Hf=K+sp.diags(diag+V); e=sl.eigsh(Hf,k=1,which='SA')[0][0]; return e
def evar(S,a):
  p=S*a; return p@(H@p)/(p@p)
def B(S,a):
  return S*(Hoff@(S*a))/a
rng=np.random.default_rng(0)
print('D',D,'E0/N',E0/N,'wrong-weight Marshall',(psi**2)[mars!=S0].sum())
for sig in [0.0,0.1,0.3]:
  a=np.abs(psi)*np.exp(sig*rng.standard_normal(D))
  S=mars.copy(); Bx=B(S,a); F=Bx>0; Fb=(Bx<0)&(rng.random(D)<F.mean()/max((Bx<0).mean(),1e-12))
  S1=S.copy();S1[F]*=-1; Sb=S.copy();Sb[Fb]*=-1
  w=lambda T:(psi**2)[T!=S0].sum()
  print(f'sigma {sig}: E_FN-E0 per site  Marshall {(efn(S,a)-E0)/N:.3e} R1 {(efn(S1,a)-E0)/N:.3e} broken {(efn(Sb,a)-E0)/N:.3e} | Evar M {(evar(S,a)-E0)/N:.3e} R1 {(evar(S1,a)-E0)/N:.3e} | w M {w(S):.2e} R1 {w(S1):.2e} | flips {F.sum()} P {(S[F]!=S0[F]).mean() if F.sum() else 0:.3f}')
# fixed point: exact amplitude & exact sign -> R1 flips nothing
print('flags at exact psi0 with exact sign:', (B(S0,np.abs(psi))>0).sum())
