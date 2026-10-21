#!/usr/bin/env python3
"""Exact ED & optimized Krylov sign chamber + self-consistent FN, 20-site torus.
ED labels used only for scoring. No target amplitude/sign used in FN/Krylov.
"""
import os, time, json
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
from itertools import combinations
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh, LinearOperator
from scipy.stats import rankdata

N=20; J2=.5; Lx,Ly=4,5
T0=time.time()
def log(*s): print(f'[{time.time()-T0:7.1f}s]', *s,flush=True)
def site(x,y):
    q, r=divmod(y,Ly)
    return ((x-q)%Lx)+Lx*r
bonds1=[]; bonds2=[]
for y in range(Ly):
 for x in range(Lx):
  i=site(x,y)
  for dx,dy in [(1,0),(0,1)]: bonds1.append((i,site(x+dx,y+dy)))
  for dx,dy in [(1,1),(-1,1)]: bonds2.append((i,site(x+dx,y+dy)))
assert len(set(tuple(sorted(b)) for b in bonds1)) == 40
assert len(set(tuple(sorted(b)) for b in bonds2)) == 40
states=np.fromiter((sum(1<<i for i in comb) for comb in combinations(range(N),N//2)),dtype=np.uint32,count=184756)
D=states.size
lookup=np.full(1<<N,-1,dtype=np.int32);lookup[states]=np.arange(D,dtype=np.int32)
diag1=np.zeros(D);diag2=np.zeros(D)

def build(bonds,coef,label):
    rows=[];cols=[]
    dd=np.zeros(D)
    for i,j in bonds:
        flip=((states>>i)^(states>>j))&1
        dd += coef*(0.25-0.5*flip)
        rr=np.flatnonzero(flip).astype(np.int32)
        cc=lookup[states[rr] ^ np.uint32((1<<i)|(1<<j))]
        assert (cc>=0).all()
        rows.append(rr); cols.append(cc)
    rows=np.concatenate(rows);cols=np.concatenate(cols)
    from scipy.sparse import coo_matrix
    X=coo_matrix((np.full(len(rows),0.5*coef), (rows,cols)),shape=(D,D)).tocsr()
    log(label,'nnz',X.nnz,'memory MB',round((X.data.nbytes+X.indices.nbytes+X.indptr.nbytes)/1e6,1))
    return X,dd
H1,d1=build(bonds1,1.,'J1'); H2,d2=build(bonds2,J2,'J2')
diag=d1+d2
maskA=sum(1<<site(x,y) for y in range(Ly) for x in range(Lx) if (x+y)%2==0)
bits=np.zeros(D,dtype=np.int8)
a=states & np.uint32(maskA)
for k in range(N):bits^=((a>>np.uint32(k))&1).astype(np.int8)
sM=np.where(bits==0,1.,-1.)

def energy_solve(M, label, tol=2e-11, v0=None):
    ev, vv=eigsh(M,k=1,which='SA',tol=tol,v0=v0,maxiter=1500,ncv=32)
    v=vv[:,0]
    log(label, 'E=',float(ev[0]))
    return float(ev[0]),v
E20,p20=energy_solve(diags(d1)+H1,'J2=0 ED')
# choose Marshall-positive amplitudes
if np.dot(sM,p20)<0:p20=-p20
bad_J20=np.sum(sM*p20<0);assert bad_J20==0,(bad_J20,np.min(sM*p20))
E0,psi0=energy_solve(diags(diag)+H1+H2,'J2=0.5 ED',v0=p20)
if np.dot(sM,psi0)<0:psi0=-psi0
wrong=(sM*psi0<0)
wt=psi0**2
log('physical wrong states',int(wrong.sum()),'fraction',wrong.mean(),'weight',wt[wrong].sum())
a0=np.abs(p20);a0/=np.linalg.norm(a0)


from scipy.sparse import triu
from scipy import linalg as spla
from pathlib import Path
Hoff = (H1 + H2).tocsr()
Htot = diags(diag) + Hoff
U=triu(Hoff,k=1).tocoo()
edge_i=U.row.astype(np.int32);edge_j=U.col.astype(np.int32);edge_h=U.data
log('EDGE DATA',len(edge_i),'upper edges')

def krylov_step(a,s,label):
    # t>=0: s'_i = s_i sign(T-r_i), T=1/t. For each threshold, get its exact energy.
    v=s*a
    local = diag + (Hoff @ v) / np.maximum(np.abs(v),1e-290) * s  # (Hv)/(s a) = diag + s(Hoff@(s a))/a
    order=np.argsort(local,kind='stable')
    rv=local[order]
    sc=np.maximum(np.maximum(abs(rv[1:]),abs(rv[:-1])),1.)
    distinct=np.r_[True,np.abs(np.diff(rv))>1e-10*sc]
    gs=np.cumsum(distinct).astype(np.int32)-1
    group=np.empty(D,dtype=np.int32);group[order]=gs
    G=int(gs[-1])+1
    mask=group[edge_i]!=group[edge_j]
    i=edge_i[mask];j=edge_j[mask]
    u=group[i];vgrp=group[j]
    lo=np.minimum(u,vgrp);hi=np.maximum(u,vgrp)
    weights=-4*edge_h[mask]*a[i]*a[j]*s[i]*s[j]
    diff=np.bincount(lo,weights=weights,minlength=G)-np.bincount(hi,weights=weights,minlength=G)
    Eb=float((s*a)@(Htot@(s*a)))
    allE=Eb+np.cumsum(diff)
    k=np.argmin(np.r_[Eb, allE])-1
    sn=np.where(group<=k,s,-s)
    if np.dot(sn*a,sM*a)<0: sn=-sn  # fix the irrelevant global phase against Marshall
    En=float((sn*a)@(Htot@(sn*a)))
    log('KRYLOV',label,'groups',G,'k',int(k),'flip-orbits/states',int(np.sum(sn!=s)),
        'E_before',Eb,'E_est',float(np.min(np.r_[Eb,allE])),'E_actual',En,'dE',En-Eb)
    assert abs(En-np.min(np.r_[Eb,allE]))<2e-8,(En,np.min(np.r_[Eb,allE]))
    return sn,dict(E_before=Eb,E_after=En,threshold=float(rv[np.searchsorted(gs,k,side='right')-1]) if k>=0 else None,
                   n_flip=int(np.sum(sn!=s)),G=G)

def fn_step(a,s,label,tol=3e-10):
    sw=s*a
    positive=.5*(Hoff@a + s*(Hoff@sw))
    # robust cancellation: compute forbidden as rows in sign gauge
    Dfn=diag+positive/np.maximum(a,1e-250)
    def Fmul(x):
        return Dfn*x-.5*(Hoff@x-s*(Hoff@(s*x)))
    # For sign-improved chambers tiny amplitudes produce very large FN diagonal.
    # Diagonally preconditioned 3-dimensional Rayleigh-Ritz as in the project.
    x=a/np.linalg.norm(a)
    Ax=Fmul(x)
    th=float(x@Ax)
    P=AP=None
    nr=np.inf
    for it in range(1,1001):
        r=Ax-th*x
        nr=float(np.linalg.norm(r))
        if nr <= max(tol*abs(th),1e-9):break
        z=r/np.maximum(Dfn-th,1.0)
        z-=float(x@z)*x
        if P is not None:
            z-=float(P@z)*P
        nz=np.linalg.norm(z)
        if nz<1e-14:break
        z/=nz
        Az=Fmul(z)
        basis=[x,z]+([P] if P is not None else [])
        abasis=[Ax,Az]+([AP] if P is not None else [])
        G=np.array([[float(basis[ii]@abasis[jj]) for jj in range(len(basis))] for ii in range(len(basis))]);G=(G+G.T)*.5
        B=np.array([[float(basis[ii]@basis[jj]) for jj in range(len(basis))] for ii in range(len(basis))]);B=(B+B.T)*.5
        try:
            ew,ev=spla.eigh(G,B)
        except spla.LinAlgError:
            P=AP=None;continue
        c=ev[:,0]
        xn=sum(c[j]*basis[j] for j in range(len(c)))
        Axn=sum(c[j]*abasis[j] for j in range(len(c)))
        pn=sum(c[j]*basis[j] for j in range(1,len(c)))
        apn=sum(c[j]*abasis[j] for j in range(1,len(c)))
        x=xn/np.linalg.norm(xn);Ax=Axn/np.linalg.norm(xn)
        pp=np.linalg.norm(pn)
        P=pn/pp if pp>1e-14 else None
        AP=apn/pp if P is not None else None
        th=float(x@Ax)
        if it%100==0:log(label,'precond iter',it,'E',th,'resid',nr)
    if np.dot(x,a)<0:x=-x
    negmass=float(np.sum(x[x<0]**2))
    v=np.maximum(x,1e-220);v/=np.linalg.norm(v)
    eig=float(v@Fmul(v))
    resid=float(np.linalg.norm(Fmul(v)-eig*v))
    log(label,'E',eig,'precond iter',it,'resid',resid,'negmass',negmass)
    return eig,v,dict(eig=eig,neg_mass=negmass,resid=resid,max_diag=float(np.max(Dfn)),iterations=it)


def auc(scores, labels):
    rank=rankdata(scores,method='average')
    n1=labels.sum();n0=len(labels)-n1
    return float((rank[labels].sum()-n1*(n1+1)/2)/(n1*n0))

def conditioned_auc(scores,amp,labels,nbins=25):
    order=np.argsort(amp,kind='stable')
    num=0.;den=0.;
    for inds in np.array_split(order,nbins):
        n1=labels[inds].sum(); n0=len(inds)-n1
        if n1==0 or n0==0:continue
        num+=auc(scores[inds],labels[inds])*n1*n0; den+=n1*n0
    return float(num/den) if den else None

def metrics(a_before,a_after,s,label,step,info):
    wrong=s*psi0<0;wrong_weight=float(wt[wrong].sum())
    cr=np.log(np.maximum(a_before,1e-220))-np.log(np.maximum(a_after,1e-220))
    prioramp=-np.log(np.maximum(a_before,1e-220))
    curramp=-np.log(np.maximum(a_after,1e-220))
    g=s*(Hoff@(s*a_before)); total=Hoff@a_before
    q=g/np.maximum(total,1e-220)
    row=dict(chamber=label,step=step,wrong_fraction=float(wrong.mean()),wrong_weight=wrong_weight,
             auc_initial_amplitude=auc(prioramp,wrong),auc_current_amplitude=auc(curramp,wrong),
             auc_fn_collapse=auc(cr,wrong),auc_fn_collapse_amplitude_matched=conditioned_auc(cr,a_before,wrong),
             auc_fn_collapse_currentamp_matched=conditioned_auc(cr,a_after,wrong),
             auc_q=auc(q,wrong),auc_q_amplitude_matched=conditioned_auc(q,a_before,wrong),
             median_cr_wrong=float(np.median(cr[wrong])),median_cr_right=float(np.median(cr[~wrong])),
             fraction_below_epsilon=float(np.mean(a_after<1e-16)),**info)
    for frac in (0.01,0.05):
        n=int(frac*D)
        sel=np.argpartition(cr,-n)[-n:]
        row[f'top{frac:g}_frac_wrong']=float(wrong[sel].mean())
        row[f'top{frac:g}_weight_recall']=float(wt[sel][wrong[sel]].sum()/wrong_weight)
    log('METRICS',json.dumps(row))
    return row

import sys
NITER=int(os.environ.get('NITER','6'))
ALL={"model":"20-site tilted torus J1-J2 = 0.5","D":int(D),"E0":E0,"E20":E20,
 "init":"exact J2=0 ground state amplitudes; ED J2=0.5 used only as hidden labels",
 "chambers":{},"records":[]}
# Try from both raw J2=0 amplitudes and from Marshall fixed-node amplitudes
for case in ('raw_J20','Marshall_FN_1'):
    a=a0.copy()
    if case == 'Marshall_FN_1':
        _,a,_=fn_step(a,sM,'Marshall preparatory FN')
    s=sM.copy()
    s,kry=krylov_step(a,s,case)
    wrong=s*psi0<0
    ALL['chambers'][case]=dict(**kry,wrong_count=int(wrong.sum()),wrong_prob=float(wt[wrong].sum()))
    for iteration in range(1,NITER+1):
        old=a
        en,a,info=fn_step(a,s,f'{case} FN {iteration}')
        row=metrics(old,a,s,case,iteration,info)
        ALL['records'].append(row)
        Path('/tmp/avsl05-20site-structural-data/fn_collapse_exact20_krylov_results.json').write_text(json.dumps(ALL,indent=2))
    if case=='raw_J20':
        # Krylov again after FN amplitude improves, then freeze further.
        s2,kry2=krylov_step(a,s,'K2_after_FN')
        wrong=s2*psi0<0
        ALL['chambers']['K2_after_FN']=dict(**kry2,wrong_count=int(wrong.sum()),wrong_prob=float(wt[wrong].sum()))
        log('K2 SIGN QUALITY',int(wrong.sum()),float(wt[wrong].sum()))
        Path('/tmp/avsl05-20site-structural-data/fn_collapse_exact20_krylov_checkpoint.npz').parent.mkdir(exist_ok=True)
        np.savez_compressed('/tmp/avsl05-20site-structural-data/fn_collapse_exact20_krylov_checkpoint.npz',a=a,s2=s2,s1=sM,psi0=psi0,diag=diag)
        Path('/tmp/avsl05-20site-structural-data/fn_collapse_exact20_krylov_results.json').write_text(json.dumps(ALL,indent=2))
        for iteration in range(1,NITER+1):
            old=a
            en,a,info=fn_step(a,s2,f'K2_after_FN FN {iteration}')
            ALL['records'].append(metrics(old,a,s2,'K2_after_FN',iteration,info))
            Path('/tmp/avsl05-20site-structural-data/fn_collapse_exact20_krylov_results.json').write_text(json.dumps(ALL,indent=2))
log('DONE')
