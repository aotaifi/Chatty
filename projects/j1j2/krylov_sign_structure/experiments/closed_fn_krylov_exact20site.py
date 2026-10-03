#!/usr/bin/env python3
import argparse, json, hashlib, time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from finite_tau_matching_20site_exact import build_H, marshall_signs, D, N

ROOT=Path('/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure')

def ground(A,v0=None,tol=2e-10):
    ew,ev=sla.eigsh(A,k=1,which='SA',v0=v0,tol=tol,maxiter=100000)
    v=np.asarray(ev[:,0],float); v/=np.linalg.norm(v)
    return float(ew[0]),v

def canonical(s):
    s=np.asarray(s,dtype=np.int8).copy()
    if s[0]<0: s=-s
    return s

def sign_hash(s):
    return hashlib.sha1(np.packbits((s>0).astype(np.uint8)).tobytes()).hexdigest()[:12]

def run(maxiter,outpath):
    H,diag=build_H(.5); Hinit,_=build_H(0.0)
    co=sp.triu(H-sp.diags(diag),k=1).tocoo()
    ei=co.row.astype(np.int32); ej=co.col.astype(np.int32); hij=np.asarray(co.data,float)
    full=H.tocoo(); off=full.row!=full.col
    fr=full.row[off].astype(np.int32); fc=full.col[off].astype(np.int32); fh=np.asarray(full.data[off],float)

    E0,psi0=ground(H); _,psi_init=ground(Hinit)
    sM=canonical(marshall_signs())
    if np.dot(psi0,sM)<0: psi0=-psi0
    if np.dot(psi_init,sM)<0: psi_init=-psi_init
    atrue=np.abs(psi0); atrue/=np.linalg.norm(atrue)
    strue=canonical(np.where(psi0>=0,1,-1).astype(np.int8)); ptrue=atrue**2
    a=np.abs(psi_init); a/=np.linalg.norm(a); s=sM.copy()

    def physical_energy(aa,ss):
        p=aa*ss; return float(p@(H@p))

    def fixed_node(aa,ss):
        af=np.maximum(aa,1e-15)
        kij=fh*ss[fr]*ss[fc]
        bad=kij>0
        d=diag + np.bincount(fr[bad],weights=kij[bad]*af[fc[bad]]/af[fr[bad]],minlength=D)
        keep=~bad
        rr=np.concatenate([fr[keep],np.arange(D,dtype=np.int32)])
        cc=np.concatenate([fc[keep],np.arange(D,dtype=np.int32)])
        vv=np.concatenate([kij[keep],d])
        F=sp.coo_matrix((vv,(rr,cc)),shape=(D,D)).tocsr()
        e,v=ground(F,v0=aa); v=np.abs(v); v/=np.linalg.norm(v)
        return e,v

    def k1_energyopt(aa,ss):
        psi=aa*ss; r=np.asarray((H@psi)/psi,float)
        order=np.argsort(r,kind='mergesort'); rs=r[order]
        scale=np.maximum(np.maximum(np.abs(rs[1:]),np.abs(rs[:-1])),1.0)
        br=np.empty(D,bool); br[0]=True
        br[1:]=np.abs(rs[1:]-rs[:-1])>(1e-11+1e-11*scale)
        gs=np.cumsum(br,dtype=np.int32)-1
        gid=np.empty(D,np.int32); gid[order]=gs; G=int(gs[-1])+1
        c0=-ss.copy(); wij=hij*aa[ei]*aa[ej]; val=2*wij*c0[ei]*c0[ej]
        diagE=float(np.sum(diag*aa*aa)); ebase=diagE+float(val.sum())
        gi=gid[ei]; gj=gid[ej]; lo=np.minimum(gi,gj); hi=np.maximum(gi,gj); m=lo<hi
        delta=(np.bincount(lo[m],weights=-2*val[m],minlength=G)
               +np.bincount(hi[m],weights=2*val[m],minlength=G))
        cand=np.r_[ebase,ebase+np.cumsum(delta)]; kbest=int(np.argmin(cand))-1
        sn=c0.copy()
        if kbest>=0: sn[gid<=kbest]*=-1
        sn=canonical(sn)
        if kbest<0: threshold=float(np.min(r)-max(1.0,abs(np.min(r))*.1))
        elif kbest==G-1: threshold=float(np.max(r)+max(1.0,abs(np.max(r))*.1))
        else: threshold=float(.5*(np.max(r[gid==kbest])+np.min(r[gid==kbest+1])))
        return sn,threshold,float(np.min(cand)),G

    def record(it,efn=None,threshold=None,groups=None,t_fn=None,t_k1=None):
        E=physical_energy(a,s); O=float(abs(np.sum(ptrue*s*strue)))
        rec=dict(it=int(it),sign_hash=sign_hash(s),E_guide=E,E_error=float(E-E0),
                 E_relative_error=float((E-E0)/abs(E0)),O_sign=O,
                 wrong_sign_weight=float((1-O)/2),F_amp=float(np.dot(a,atrue)**2),
                 min_amp=float(np.min(a)),max_amp=float(np.max(a)))
        if efn is not None:
            rec['E_FN']=float(efn); rec['E_FN_relative_error']=float((efn-E0)/abs(E0))
        if threshold is not None: rec['threshold_t']=float(threshold)
        if groups is not None: rec['r_groups']=int(groups)
        if t_fn is not None: rec['time_fn_s']=float(t_fn)
        if t_k1 is not None: rec['time_k1_s']=float(t_k1)
        return rec

    hist=[record(0)]; print('ITER 0',json.dumps(hist[-1],sort_keys=True),flush=True)
    terminal='maxiter'
    for it in range(1,maxiter+1):
        aold=a.copy(); sold=s.copy()
        t=time.time(); efn,afn=fixed_node(aold,sold); tfn=time.time()-t
        t=time.time(); snew,T,eth,G=k1_energyopt(afn,sold); tk=time.time()-t
        a=afn; s=snew
        rec=record(it,efn,T,G,tfn,tk); rec['threshold_energy']=eth
        hist.append(rec); print('ITER',it,json.dumps(rec,sort_keys=True),flush=True)
        if rec['wrong_sign_weight']<1e-12 and 1-rec['F_amp']<1e-10:
            terminal='exact_ground_state'; break

    out=dict(method=dict(system='20-site skew torus T=(4,0),(1,5), Sz=0',D=D,target_J2=.5,
                         init_amplitude_source_J2=0.0,initial_signs='Marshall',
                         FN='standard lattice fixed node from current amplitude and signs',
                         sign_update='one projected Krylov step with exact fixed-amplitude energy-optimal threshold',
                         oracle_policy='target ED ground state used only for diagnostics; never in FN or sign update'),
             E0=E0,terminal=terminal,history=hist)
    Path(outpath).write_text(json.dumps(out,indent=2))
    print('WROTE',outpath,'terminal',terminal,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--maxiter',type=int,default=40)
    ap.add_argument('--out',default=str(ROOT/'results/closed_fn_krylov_20site_J2p5_J2zero_init.json'))
    a=ap.parse_args(); run(a.maxiter,a.out)
