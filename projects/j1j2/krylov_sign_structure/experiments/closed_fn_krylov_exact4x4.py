#!/usr/bin/env python3
import argparse, json, hashlib
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

L=4; N=L*L
ROOT=Path("/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure")
site=lambda x,y:(x%L)+L*(y%L)

NN=[]; NNN=[]
for y in range(L):
    for x in range(L):
        i=site(x,y)
        NN += [(i,site(x+1,y)),(i,site(x,y+1))]
        NNN += [(i,site(x+1,y+1)),(i,site(x+1,y-1))]

basis=np.array([s for s in range(1<<N) if s.bit_count()==N//2],dtype=np.uint32)
D=len(basis); idx={int(s):i for i,s in enumerate(basis)}

def build_H(J2):
    rr=[];cc=[];vv=[];diag=np.zeros(D)
    for bi,s0 in enumerate(basis):
        s=int(s0); e=0.0
        for bonds,J in ((NN,1.0),(NNN,J2)):
            for u,v in bonds:
                same=((s>>u)&1)==((s>>v)&1)
                e += J*(0.25 if same else -0.25)
                if not same:
                    t=s^(1<<u)^(1<<v)
                    rr.append(bi);cc.append(idx[t]);vv.append(0.5*J)
        diag[bi]=e
    rr.extend(range(D));cc.extend(range(D));vv.extend(diag.tolist())
    return sp.coo_matrix((vv,(rr,cc)),shape=(D,D)).tocsr(),diag

def ground(A,v0=None,tol=1e-11):
    ew,ev=sla.eigsh(A,k=1,which="SA",v0=v0,tol=tol,maxiter=200000)
    v=np.asarray(ev[:,0],float)
    return float(ew[0]),v/np.linalg.norm(v)

def marshall_signs():
    mask=sum(1<<site(x,y) for y in range(L) for x in range(L) if (x+y)%2==0)
    return np.array([1 if ((int(s)&mask).bit_count()%2)==0 else -1 for s in basis],dtype=np.int8)

def canonical(s):
    s=np.asarray(s,dtype=np.int8).copy()
    if s[0] < 0: s=-s
    return s

def physical_energy(H,a,s):
    psi=a*s
    return float(psi@(H@psi)/(psi@psi))

def group_ids(r,atol=1e-11,rtol=1e-11):
    order=np.argsort(r,kind="mergesort")
    rs=r[order]
    scale=np.maximum(np.maximum(np.abs(rs[1:]),np.abs(rs[:-1])),1.0)
    br=np.empty(D,dtype=bool); br[0]=True
    br[1:]=np.abs(rs[1:]-rs[:-1]) > (atol+rtol*scale)
    gs=np.cumsum(br,dtype=np.int32)-1
    gid=np.empty(D,dtype=np.int32); gid[order]=gs
    return order,gid,int(gs[-1])+1

def prep_edges(H,a,diag):
    co=sp.triu(H-sp.diags(diag),k=1).tocoo()
    ei=co.row.astype(np.int32); ej=co.col.astype(np.int32)
    wij=(co.data*a[ei]*a[ej]).astype(float)
    return ei,ej,wij

def projected_krylov_update(H,diag,a,s):
    # IMPORTANT: no target ground-state data enters this function.
    psi=a*s
    r=np.asarray((H@psi)/np.where(np.abs(psi)>1e-300,psi,1.0),float)
    order,gid,G=group_ids(r)
    ei,ej,wij=prep_edges(H,a,diag)

    # t below min(r): candidate is -s, globally equivalent to s.
    c0=-s.copy()
    val=2.0*wij*c0[ei]*c0[ej]
    diagE=float(np.sum(diag*a*a))
    e0=diagE+float(np.sum(val))

    gi=gid[ei]; gj=gid[ej]
    lo=np.minimum(gi,gj); hi=np.maximum(gi,gj)
    mask=lo<hi
    delta=(np.bincount(lo[mask],weights=-2.0*val[mask],minlength=G)
           +np.bincount(hi[mask],weights=+2.0*val[mask],minlength=G))
    energies=e0+np.cumsum(delta)
    candidates=np.r_[e0,energies]
    kbest=int(np.argmin(candidates))-1

    sn=c0.copy()
    if kbest>=0:
        sn[gid<=kbest]*=-1
    sn=canonical(sn)

    if kbest<0:
        t=float(np.min(r)-max(1.0,abs(np.min(r))*0.1))
    elif kbest==G-1:
        t=float(np.max(r)+max(1.0,abs(np.max(r))*0.1))
    else:
        left=np.max(r[gid==kbest]); right=np.min(r[gid==kbest+1])
        t=float(0.5*(left+right))
    return sn,t,float(np.min(candidates)),r,G

def build_fixed_node(H,diag,a,s):
    # Standard lattice fixed node in the gauge defined by s.
    # K_ij = s_i H_ij s_j. Keep K_ij<0; move forbidden K_ij>0 to diagonal.
    af=np.maximum(np.asarray(a,float),1e-15)
    d=diag.astype(float).copy()
    rr=[];cc=[];vv=[]
    for i in range(D):
        st,en=H.indptr[i],H.indptr[i+1]
        for p in range(st,en):
            j=int(H.indices[p]); hij=float(H.data[p])
            if j==i: continue
            kij=float(s[i])*hij*float(s[j])
            if kij < 0:
                rr.append(i);cc.append(j);vv.append(kij)
            else:
                d[i] += kij*af[j]/af[i]
    rr.extend(range(D));cc.extend(range(D));vv.extend(d.tolist())
    return sp.coo_matrix((vv,(rr,cc)),shape=(D,D)).tocsr()

def fixed_node_solve(H,diag,a,s):
    # IMPORTANT: no target ground-state data enters this function.
    F=build_fixed_node(H,diag,a,s)
    e,v=ground(F,v0=np.asarray(a,float),tol=2e-11)
    # Stoquastic Perron ground state can be chosen nonnegative.
    v=np.abs(v)
    v/=np.linalg.norm(v)
    return e,v

def sign_hash(s):
    return hashlib.sha1(np.packbits((s>0).astype(np.uint8)).tobytes()).hexdigest()[:12]

def run(target_J2,init_source_J2,maxiter,outpath):
    H,diag=build_H(target_J2)
    Hinit,_=build_H(init_source_J2)

    # Target ED is diagnostics ONLY.
    E0,psi0=ground(H)
    sM=canonical(marshall_signs())
    if np.dot(psi0,sM)<0: psi0=-psi0
    atrue=np.abs(psi0); atrue/=np.linalg.norm(atrue)
    strue=canonical(np.where(psi0>=0,1,-1).astype(np.int8))
    ptrue=atrue*atrue

    # Oracle-free initialization: exact modulus of unfrustrated/reference J2 source.
    _,psi_init=ground(Hinit)
    if np.dot(psi_init,sM)<0: psi_init=-psi_init
    a=np.abs(psi_init); a/=np.linalg.norm(a)
    s=sM.copy()

    def diag_record(it,stage,fn_energy=None,t=None,groups=None,amp_delta=None,sign_change_trial=None):
        psi=a*s
        rec=dict(
            it=int(it),stage=stage,
            sign_hash=sign_hash(s),
            E_guide=physical_energy(H,a,s),
            E_error=float(physical_energy(H,a,s)-E0),
            O_sign=float(abs(np.sum(ptrue*s*strue))),
            F_amp=float(abs(np.dot(a,atrue))**2),
            min_amp=float(np.min(a)),
            max_amp=float(np.max(a)),
        )
        if fn_energy is not None: rec["E_FN"]=float(fn_energy)
        if t is not None: rec["threshold_t"]=float(t)
        if groups is not None: rec["r_groups"]=int(groups)
        if amp_delta is not None: rec["amp_delta_l2"]=float(amp_delta)
        if sign_change_trial is not None: rec["sign_change_trial_weight"]=float(sign_change_trial)
        return rec

    hist=[diag_record(0,"initial")]
    seen={}
    terminal="maxiter"
    for it in range(maxiter):
        pairkey=(sign_hash(s),tuple(np.round(a,12)))
        if pairkey in seen:
            terminal=f"pair_cycle:{seen[pairkey]}->{it}"
            break
        seen[pairkey]=it

        aold=a.copy(); sold=s.copy()
        efn,afn=fixed_node_solve(H,diag,aold,sold)

        snew,t,eth,r,G=projected_krylov_update(H,diag,afn,sold)
        amp_delta=float(np.linalg.norm(afn-aold))
        sign_change_trial=float(np.sum((afn*afn)*(snew!=sold)))

        a=afn; s=snew
        rec=diag_record(it+1,"after_fn_krylov",fn_energy=efn,t=t,groups=G,
                        amp_delta=amp_delta,sign_change_trial=sign_change_trial)
        rec["threshold_energy"]=float(eth)
        hist.append(rec)
        print("ITER",it+1,json.dumps(rec,sort_keys=True),flush=True)

        if np.array_equal(s,sold) and amp_delta<1e-10:
            terminal="fixed_pair"
            break
        if rec["O_sign"]>1-1e-12 and rec["F_amp"]>1-1e-12:
            terminal="exact_ground_state"
            break

    out=dict(
        method={
            "target_J2":target_J2,
            "init_amplitude_source_J2":init_source_J2,
            "initial_signs":"Marshall",
            "FN":"standard lattice fixed node; current a,s define H_FN",
            "sign_update":"current-sign projected Krylov r=(H a s)/(a s); exact fixed-amplitude energy-optimal grouped threshold",
            "oracle_policy":"target ED ground state used only for diagnostics O_sign,F_amp,E0; never in FN or sign update",
            "maxiter":maxiter,
        },
        E0=E0,terminal=terminal,history=hist
    )
    Path(outpath).write_text(json.dumps(out,indent=2))
    print("TERMINAL",terminal,"LAST",json.dumps(hist[-1],sort_keys=True),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--target-j2",type=float,default=0.5)
    ap.add_argument("--init-source-j2",type=float,default=0.0)
    ap.add_argument("--maxiter",type=int,default=20)
    ap.add_argument("--out",type=str,default=str(ROOT/"results/closed_fn_krylov_exact4x4.json"))
    args=ap.parse_args()
    run(args.target_j2,args.init_source_j2,args.maxiter,args.out)
