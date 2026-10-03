#!/usr/bin/env python3
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

LX,LY=4,5; N=LX*LY
ROOT=Path('/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure')

def red(x,y):
    q=y//LY; y-=q*LY; x-=q
    return (x%LX)+LX*y
NN=[];NNN=[]
for y in range(LY):
    for x in range(LX):
        i=red(x,y)
        NN += [(i,red(x+1,y)),(i,red(x,y+1))]
        NNN += [(i,red(x+1,y+1)),(i,red(x+1,y-1))]
basis=np.array([s for s in range(1<<N) if s.bit_count()==N//2],dtype=np.uint32)
D=len(basis)
state_to_idx=np.full(1<<N,-1,dtype=np.int32);state_to_idx[basis]=np.arange(D,dtype=np.int32)

def build_H(J2):
    diag=np.zeros(D,float);rr=[];cc=[];vv=[];rows=np.arange(D,dtype=np.int32)
    for bonds,J in ((NN,1.0),(NNN,J2)):
        for u,v in bonds:
            bu=(basis>>u)&1;bv=(basis>>v)&1;same=bu==bv
            diag += J*np.where(same,.25,-.25)
            sel=~same;r=rows[sel];t=basis[sel]^(np.uint32(1<<u)|np.uint32(1<<v))
            rr.append(r);cc.append(state_to_idx[t]);vv.append(np.full(r.size,.5*J))
    rr=np.concatenate(rr);cc=np.concatenate(cc);vv=np.concatenate(vv)
    H=sp.coo_matrix((np.r_[vv,diag],(np.r_[rr,rows],np.r_[cc,rows])),shape=(D,D)).tocsr()
    # one copy of each undirected off-diagonal edge
    up=sp.triu(H-sp.diags(diag),k=1).tocoo()
    return H,diag,up.row.astype(np.int32),up.col.astype(np.int32),up.data.astype(float)

def ground(A,v0=None,tol=2e-10):
    ew,ev=sla.eigsh(A,k=1,which='SA',v0=v0,tol=tol,maxiter=300000)
    v=np.asarray(ev[:,0],float);v/=np.linalg.norm(v)
    return float(ew[0]),v

def marshall_signs():
    mask=0
    for y in range(LY):
        for x in range(LX):
            if (x+y)%2==0: mask |= 1<<red(x,y)
    return np.array([1 if (int(s)&mask).bit_count()%2==0 else -1 for s in basis],dtype=np.int8)

def canonical(s):
    s=np.asarray(s,dtype=np.int8).copy()
    if s[0]<0:s=-s
    return s

def physical_energy(H,a,s):
    psi=a*s;return float(psi@(H@psi)/(psi@psi))

def sign_hash(s):
    return hashlib.sha1(np.packbits((s>0).astype(np.uint8)).tobytes()).hexdigest()[:12]

def group_ids(r,atol=1e-11,rtol=1e-11):
    order=np.argsort(r,kind='mergesort');rs=r[order]
    scale=np.maximum(np.maximum(np.abs(rs[1:]),np.abs(rs[:-1])),1.0)
    br=np.empty(D,dtype=bool);br[0]=True;br[1:]=np.abs(rs[1:]-rs[:-1])>(atol+rtol*scale)
    gs=np.cumsum(br,dtype=np.int32)-1;gid=np.empty(D,dtype=np.int32);gid[order]=gs
    return gid,int(gs[-1])+1

def projected_krylov_update(H,diag,ei,ej,hij,a,s):
    psi=a*s;r=np.asarray((H@psi)/np.where(np.abs(psi)>1e-300,psi,1.0),float)
    gid,G=group_ids(r)
    wij=hij*a[ei]*a[ej]
    c0=-s.copy();val=2.0*wij*c0[ei]*c0[ej]
    diagE=float(np.sum(diag*a*a));e0=diagE+float(np.sum(val))
    gi=gid[ei];gj=gid[ej];lo=np.minimum(gi,gj);hi=np.maximum(gi,gj);mask=lo<hi
    delta=(np.bincount(lo[mask],weights=-2.0*val[mask],minlength=G)+
           np.bincount(hi[mask],weights=+2.0*val[mask],minlength=G))
    candidates=np.r_[e0,e0+np.cumsum(delta)];kbest=int(np.argmin(candidates))-1
    sn=c0.copy()
    if kbest>=0:sn[gid<=kbest]*=-1
    sn=canonical(sn)
    if kbest<0:t=float(np.min(r)-1)
    elif kbest==G-1:t=float(np.max(r)+1)
    else:t=float(.5*(np.max(r[gid==kbest])+np.min(r[gid==kbest+1])))
    return sn,t,float(np.min(candidates)),G

def fixed_node_solve(H,diag,ei,ej,hij,a,s):
    af=np.maximum(a,1e-15)
    # symmetric directed arrays from unique edges
    rr=np.r_[ei,ej];cc=np.r_[ej,ei];hh=np.r_[hij,hij]
    kij=s[rr].astype(float)*hh*s[cc].astype(float)
    keep=kij<0;bad=~keep
    corr=np.bincount(rr[bad],weights=kij[bad]*af[cc[bad]]/af[rr[bad]],minlength=D)
    d=diag+corr
    rows=np.r_[rr[keep],np.arange(D,dtype=np.int32)]
    cols=np.r_[cc[keep],np.arange(D,dtype=np.int32)]
    vals=np.r_[kij[keep],d]
    F=sp.coo_matrix((vals,(rows,cols)),shape=(D,D)).tocsr()
    e,v=ground(F,v0=af)
    v=np.abs(v);v/=np.linalg.norm(v)
    return e,v

def run(maxiter,outpath):
    t0=time.time();print('D',D,'build target',flush=True)
    H,diag,ei,ej,hij=build_H(.5);print('nnz',H.nnz,'edges',len(hij),'sec',time.time()-t0,flush=True)
    Hinit,_,_,_,_=build_H(0.0)
    print('target ED',flush=True);E0,psi0=ground(H)
    print('init ED',flush=True);_,psiinit=ground(Hinit)
    sM=canonical(marshall_signs())
    if np.dot(psi0,sM)<0:psi0=-psi0
    if np.dot(psiinit,sM)<0:psiinit=-psiinit
    atrue=np.abs(psi0);atrue/=np.linalg.norm(atrue);strue=canonical(np.where(psi0>=0,1,-1).astype(np.int8));ptrue=atrue**2
    a=np.abs(psiinit);a/=np.linalg.norm(a);s=sM.copy()
    hist=[]
    def record(it,stage,fn=None,t=None,amp_delta=None):
        E=physical_energy(H,a,s);O=abs(float(np.sum(ptrue*s*strue)))
        rec=dict(it=it,stage=stage,E_guide=E,epsilon_rel=(E-E0)/abs(E0),O_sign=O,wrong_weight=(1-O)/2,
                 F_amp=float(abs(np.dot(a,atrue))**2),sign_hash=sign_hash(s))
        if fn is not None:rec['E_FN']=fn;rec['epsilon_rel_FN']=(fn-E0)/abs(E0)
        if t is not None:rec['threshold_t']=t
        if amp_delta is not None:rec['amp_delta_l2']=amp_delta
        return rec
    hist.append(record(0,'initial'));print('ITER 0',json.dumps(hist[-1]),flush=True)
    terminal='maxiter'
    for it in range(1,maxiter+1):
        sold=s.copy();aold=a.copy();tt=time.time()
        efn,afn=fixed_node_solve(H,diag,ei,ej,hij,aold,sold)
        snew,t,eth,G=projected_krylov_update(H,diag,ei,ej,hij,afn,sold)
        a=afn;s=snew;rec=record(it,'after_fn_krylov',efn,t,float(np.linalg.norm(a-aold)));rec['r_groups']=G;rec['threshold_energy']=eth
        hist.append(rec);print('ITER',it,json.dumps(rec),'sec',time.time()-tt,flush=True)
        if np.array_equal(s,sold) and rec['amp_delta_l2']<1e-9:terminal='fixed_pair';break
        if rec['wrong_weight']<1e-12 and 1-rec['F_amp']<1e-10:terminal='exact_ground_state';break
    out=dict(method={'geometry':'20-site bipartite skew torus T=(4,0),(1,5)','target_J2':.5,'init_amplitude_source_J2':0.0,
                     'initial_signs':'Marshall','FN':'exact standard lattice fixed-node solve','sign_update':'energy-optimal one-step Krylov threshold','ED':'diagnostics only'},
             N=N,D=D,E0=E0,terminal=terminal,elapsed_sec=time.time()-t0,history=hist)
    Path(outpath).write_text(json.dumps(out,indent=2));print('TERMINAL',terminal,'WROTE',outpath,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--maxiter',type=int,default=20);ap.add_argument('--out',default=str(ROOT/'results/closed_fn_krylov_20site_J2p5_J2zero_init.json'));args=ap.parse_args();run(args.maxiter,args.out)
