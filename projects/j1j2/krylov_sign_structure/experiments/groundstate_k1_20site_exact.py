#!/usr/bin/env python3
import json,time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

LX,LY=4,5
N=LX*LY
J2=.5
ROOT=Path('/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure')
OUT=ROOT/'results/groundstate_k1_20site_exact_J2p5.json'

def red(x,y):
    q=y//LY; y-=q*LY; x-=q
    return (x%LX)+LX*y

NN=[]; NNN=[]
for y in range(LY):
    for x in range(LX):
        i=red(x,y)
        NN += [(i,red(x+1,y)),(i,red(x,y+1))]
        NNN += [(i,red(x+1,y+1)),(i,red(x+1,y-1))]

basis=np.array([s for s in range(1<<N) if s.bit_count()==N//2],dtype=np.uint32)
D=len(basis)
state_to_idx=np.full(1<<N,-1,dtype=np.int32)
state_to_idx[basis]=np.arange(D,dtype=np.int32)

def build_H():
    diag=np.zeros(D,float); rr=[];cc=[];vv=[]
    rows=np.arange(D,dtype=np.int32)
    for bonds,J in ((NN,1.0),(NNN,J2)):
        for u,v in bonds:
            bu=(basis>>u)&1; bv=(basis>>v)&1
            same=bu==bv
            diag += J*np.where(same,.25,-.25)
            sel=~same; r=rows[sel]
            t=basis[sel]^(np.uint32(1<<u)|np.uint32(1<<v))
            rr.append(r); cc.append(state_to_idx[t]); vv.append(np.full(r.size,.5*J))
    rr.append(rows);cc.append(rows);vv.append(diag.copy())
    H=sp.coo_matrix((np.concatenate(vv),(np.concatenate(rr),np.concatenate(cc))),shape=(D,D)).tocsr()
    return H,diag

def marshall_signs():
    mask=0
    for y in range(LY):
        for x in range(LX):
            if (x+y)%2==0: mask |= 1<<red(x,y)
    return np.array([1 if (int(s)&mask).bit_count()%2==0 else -1 for s in basis],dtype=np.int8)

def energy_opt_threshold(H,diag,a,s0,r):
    W=H.copy().tolil(); W.setdiag(0); W=W.tocsr()
    W=sp.diags(a)@W@sp.diags(a)
    diagE=float(np.sum(diag*a*a))
    s=(-s0).astype(np.int8)
    E=diagE+float(s@(W@s))
    h=np.asarray(W@s).ravel()
    order=np.argsort(r)
    bestE=E; best_t=float(r[order[0]]-max(1.0,abs(r[order[0]])*.01)); best_s=s.copy()
    for n,ii in enumerate(order):
        old=int(s[ii]); E += -4.0*old*h[ii]; s[ii]=-old
        st,en=W.indptr[ii],W.indptr[ii+1]
        js=W.indices[st:en]; ws=W.data[st:en]
        h[js] += -2.0*old*ws
        if E < bestE-1e-13:
            bestE=float(E); best_t=float(r[ii]); best_s=s.copy()
        if n and n%50000==0: print('scan',n,'/',D,'bestE',bestE,flush=True)
    return best_t,best_s,float(bestE)

def overlap_and_wrong(s,truth,p):
    z=float(np.sum(p*s*truth)); O=abs(z)
    return O,(1-O)/2

t0=time.time(); print('D',D,'building H',flush=True)
H,diag=build_H(); print('H nnz',H.nnz,'build sec',time.time()-t0,flush=True)
print('eigsh',flush=True)
ev,V=sla.eigsh(H,k=2,which='SA',tol=1e-11,maxiter=100000)
o=np.argsort(ev);ev=ev[o];V=V[:,o]
vec=np.asarray(V[:,0],float)
if vec[np.argmax(np.abs(vec))]<0: vec=-vec
E0=float(ev[0]);gap=float(ev[1]-ev[0]);a=np.abs(vec);p=a*a;p/=p.sum()
truth=np.where(vec>=0,1,-1).astype(np.int8);s0=marshall_signs()
psi0=a*s0
r=(H@psi0)/np.where(a>1e-300,psi0,1e-300)
Ebase=float(psi0@(H@psi0))
O0,w0=overlap_and_wrong(s0,truth,p)
print('E0',E0,'gap',gap,'baseline E',Ebase,'O0',O0,'w0',w0,'sec',time.time()-t0,flush=True)
t,s1,E1=energy_opt_threshold(H,diag,a,s0,r)
O1,w1=overlap_and_wrong(s1,truth,p)
eps0=(Ebase-E0)/abs(E0); eps1=(E1-E0)/abs(E0)
out=dict(geometry='20-site bipartite skew torus T=(4,0),(1,5)',N=N,D=D,J2=J2,E0=E0,gap=gap,
         threshold_energyopt=t,E_fixed_baseline=Ebase,E_fixed_step1=E1,
         O_S_0=O0,O_S_1=O1,wrong_weight_0=w0,wrong_weight_1=w1,
         epsilon_rel_0=eps0,epsilon_rel_1=eps1,
         wrong_reduction_factor=w0/w1 if w1>0 else float('inf'),
         energy_reduction_factor=eps0/eps1 if eps1>0 else float('inf'),
         elapsed_sec=time.time()-t0)
OUT.write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2),flush=True)
print('WROTE',OUT,flush=True)
