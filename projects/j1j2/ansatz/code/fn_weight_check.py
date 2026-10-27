# Check dE_FN ~ sum_{violating pairs} |H_xy psi(x) psi(y)| 2(cosh(eps_y-eps_x)-1) for psi_T = psi0 e^eps (signs kept), 4x4 J1-J2 0.5
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sla
L=4; N=16; J2=0.5
def s(x,y): return (x%L)+L*(y%L)
NN=[(s(x,y),s(x+1,y)) for y in range(L) for x in range(L)]+[(s(x,y),s(x,y+1)) for y in range(L) for x in range(L)]
NNN=[(s(x,y),s(x+1,y+1)) for y in range(L) for x in range(L)]+[(s(x,y),s(x+1,y-1)) for y in range(L) for x in range(L)]
allx=np.arange(1<<N,dtype=np.int64); pc=np.zeros(allx.size,np.int8)
for i in range(N): pc+=((allx>>i)&1).astype(np.int8)
B=allx[pc==N//2]; D=B.size
rows=[];cols=[];vals=[];diag=np.zeros(D)
for bonds,J in ((NN,1.0),(NNN,J2)):
    for i,j in bonds:
        anti=((B>>i)&1)!=((B>>j)&1); diag+=J*np.where(anti,-0.25,0.25)
        r=np.nonzero(anti)[0]; c=np.searchsorted(B,B[r]^((1<<i)|(1<<j)))
        rows+=list(r); cols+=list(c); vals+=[0.5*J]*r.size
Hoff=sp.csr_matrix((vals,(rows,cols)),shape=(D,D)); H=Hoff+sp.diags(diag)
E,V=sla.eigsh(H,k=1,which='SA',tol=1e-13); E0=E[0]; p0=V[:,0]
def EFN(pt):
    C=Hoff.tocoo(); m=pt[C.row]*C.data*pt[C.col]>0
    keep=sp.csr_matrix((C.data[~m],(C.row[~m],C.col[~m])),shape=(D,D))
    Vd=np.zeros(D); np.add.at(Vd,C.row[m],C.data[m]*pt[C.col[m]]/pt[C.row[m]])
    return sla.eigsh(keep+sp.diags(diag+Vd),k=1,which='SA',tol=1e-13)[0][0]
print('E0',E0,'E_FN(psi0)-E0',EFN(p0)-E0)
C=Hoff.tocoo(); m=p0[C.row]*C.data*p0[C.col]>0
print('violating ordered pairs',int(m.sum()),'of',C.nnz,' weight sum|Hpp| over unordered',0.5*np.sum(np.abs(C.data[m]*p0[C.row[m]]*p0[C.col[m]])))
rng=np.random.default_rng(0)
for sig in (0.01,0.03,0.1,0.3):
    eps=rng.normal(0,sig,D); pt=p0*np.exp(eps)
    dE=EFN(pt)-E0
    pred=0.5*np.sum(np.abs(C.data[m]*p0[C.row[m]]*p0[C.col[m]])*2*(np.cosh(eps[C.col[m]]-eps[C.row[m]])-1))
    print(f'sigma {sig}: dE_FN {dE:.4e}  first-order formula {pred:.4e}  ratio {dE/pred:.3f}')
print('--- same copy loss (sum psi^2 eps^2 = 1e-6) placed in one |psi0|^2 decade at a time ---')
p2=p0**2; dec=np.floor(np.log10(p2)).astype(int)
for d in sorted(set(dec.tolist()),reverse=True):
    msk=dec==d
    if msk.sum()<20: continue
    out=[]
    for seed in range(3):
        r=np.random.default_rng(seed); eps=np.zeros(D); eps[msk]=r.normal(0,1,msk.sum())
        eps*=np.sqrt(1e-6/np.sum(p2*eps**2)); pt=p0*np.exp(eps)
        out.append(EFN(pt)-E0)
    print(f'decade {d:4d}: n {int(msk.sum()):6d} weight {p2[msk].sum():.2e}  rms eps {np.sqrt(1e-6/p2[msk].sum()):.3e}  dE_FN {np.mean(out):.3e} (+- {np.std(out):.1e})  dE_FN per unit copy loss {np.mean(out)/1e-6:.3g}')
