#!/usr/bin/env python3
import argparse, json, math
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import scipy.sparse.csgraph as cs
from finite_tau_matching_20site_exact import build_H,D,special_columns

ROOT=Path("/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure")
H,diag=build_H(.5)
HO=(H-sp.diags(diag)).tocsr()
co=H.tocoo(); off=co.row!=co.col
R=co.row[off].astype(np.int32)
C=co.col[off].astype(np.int32)
HV=co.data[off]
IDX=np.arange(D,dtype=np.int32)

def distance_sign(y):
    A=HO.copy(); A.data=np.ones_like(A.data)
    d=np.asarray(cs.shortest_path(A,directed=False,unweighted=True,indices=[int(y)]))[0]
    return np.where((d.astype(np.int64)%2)==0,1.0,-1.0)

def uniform_fn(s):
    prod=s[R]*s[C]; good=prod<0; bad=~good
    lam=np.bincount(R[good],weights=HV[good],minlength=D)
    shift=np.bincount(R[bad],weights=HV[bad],minlength=D)
    F=sp.coo_matrix((
        np.concatenate([-HV[good],diag+shift]),
        (np.concatenate([R[good],IDX]),np.concatenate([C[good],IDX]))
    ),shape=H.shape).tocsr()
    J=sp.coo_matrix((HV[good],(R[good],C[good])),shape=H.shape).tocsr()
    pot=lam-(diag+shift)
    return F,J,lam,pot,bad

def guide_fn(s,g):
    prod=s[R]*s[C]; good=prod<0; bad=~good
    gg=np.maximum(np.asarray(g,float),1e-10/np.sqrt(D))
    shift=np.bincount(R[bad],weights=HV[bad]*gg[C[bad]]/gg[R[bad]],minlength=D)
    return sp.coo_matrix((
        np.concatenate([-HV[good],diag+shift]),
        (np.concatenate([R[good],IDX]),np.concatenate([C[good],IDX]))
    ),shape=H.shape).tocsr()

def sample_paths(y,tau,M,J,lam,pot,seed):
    rng=np.random.default_rng(seed)
    end=np.empty(M,np.int32); logw=np.empty(M,float)
    for q in range(M):
        x=int(y); t=0.0; lw=0.0
        while True:
            rate=float(lam[x])
            if rate<=0:
                lw+=float(pot[x])*(tau-t); break

            wait=float(rng.exponential(1.0/rate))
            if t+wait>=tau:
                lw+=float(pot[x])*(tau-t); break
            lw+=float(pot[x])*wait; t+=wait
            a,b=J.indptr[x],J.indptr[x+1]
            rates=J.data[a:b]
            u=rng.random()*rate
            j=int(np.searchsorted(np.cumsum(rates),u,side="right"))
            x=int(J.indices[a+min(j,len(rates)-1)])
        end[q]=x; logw[q]=lw
    logw-=np.max(logw)
    w=np.exp(logw)
    hist=np.bincount(end,weights=w,minlength=D).astype(float)
    ess=float(w.sum()**2/np.dot(w,w))
    return hist,ess,int(np.unique(end).size)

def normalized(v):
    v=np.asarray(v,float); n=np.linalg.norm(v)
    return v/n if n else v

def residual(s,g,a,bad):
    gg=np.maximum(np.asarray(g,float),1e-12/np.sqrt(D))
    term=np.bincount(R[bad],weights=HV[bad]*a[C[bad]],minlength=D)
    comp=np.bincount(R[bad],weights=HV[bad]*gg[C[bad]]/gg[R[bad]],minlength=D)
    ca=term-a*comp
    p=a*a; p/=p.sum()

    mask=a>1e-14*np.max(a)
    eps=np.zeros(D); eps[mask]=ca[mask]/a[mask]
    return {"l2":float(np.linalg.norm(ca)),
            "wabs":float(np.sum(p*np.abs(eps))),
            "wrms":float(np.sqrt(np.sum(p*eps*eps)))}

def ratio_metrics(h1,h2,a,bad,minmass):
    total1=h1.sum(); total2=h2.sum()
    p1=h1/total1; p2=h2/total2
    rr=R[bad]; cc=C[bad]
    keep=(p1[rr]>minmass)&(p1[cc]>minmass)&(p2[rr]>minmass)&(p2[cc]>minmass)
    rr=rr[keep]; cc=cc[keep]
    if len(rr)==0: return {"n_edges":0}
    lr1=np.log(p1[cc]/p1[rr]); lr2=np.log(p2[cc]/p2[rr])
    lrt=np.log(np.maximum(a[cc],1e-300)/np.maximum(a[rr],1e-300))
    wt=HV[bad][keep]*a[rr]*a[cc]; wt/=wt.sum()
    return {"n_edges":int(len(rr)),
            "replica_logratio_rmse":float(np.sqrt(np.sum(wt*(lr1-lr2)**2))),
            "oracle_logratio_rmse_rep1":float(np.sqrt(np.sum(wt*(lr1-lrt)**2))),
            "oracle_logratio_rmse_rep2":float(np.sqrt(np.sum(wt*(lr2-lrt)**2)))}

def mismatch(s,psi):
    p=psi*psi; p/=p.sum()
    st=np.where(psi>=0,1.0,-1.0)
    return float(np.sum(p[s!=st]))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--M",type=int,default=20000)
    ap.add_argument("--tau",type=float,default=.25)
    ap.add_argument("--dt-k1",type=float,default=.05)
    ap.add_argument("--prior-frac",type=float,default=.01)
    ap.add_argument("--out",default=str(ROOT/"results/finite_tau_ct_local_ratio_20site.json"))
    args=ap.parse_args()
    ys=special_columns(np.random.default_rng(20260930),2)
    y=int(ys[1]); s=distance_sign(y)
    F,J,lam,pot,bad=uniform_fn(s)
    e=np.zeros(D); e[y]=1.0
    afn=normalized(np.maximum(sla.expm_multiply(-args.tau*F,e),0.0))
    psi=normalized(sla.expm_multiply(-args.tau*H,e))
    aphys=np.abs(psi)
    print("START",{"y":y,"M":args.M,"tau":args.tau},flush=True)

    h1,ess1,u1=sample_paths(y,args.tau,args.M,J,lam,pot,71001)
    print("REP1",{"ess":ess1,"unique":u1},flush=True)
    h2,ess2,u2=sample_paths(y,args.tau,args.M,J,lam,pot,72001)
    print("REP2",{"ess":ess2,"unique":u2},flush=True)
    q1=normalized(h1); q2=normalized(h2)
    fid1=float(np.dot(q1,afn)**2); fid2=float(np.dot(q2,afn)**2)
    cross=float(np.dot(q1,q2)**2)
    rms=[]
    for mult in (1,2,5):
        rms.append({"threshold_mult":mult,
                    **ratio_metrics(h1,h2,afn,bad,mult/args.M)})
    pmean=.5*(h1/h1.sum()+h2/h2.sum())
    ghat=pmean+args.prior_frac/D
    ghat=normalized(ghat)
    gunif=np.ones(D)/np.sqrt(D)
    res_u=residual(s,gunif,aphys,bad)
    res_h=residual(s,ghat,aphys,bad)
    res_x=residual(s,aphys,aphys,bad)
    base=np.maximum(pmean+args.prior_frac/D,1e-300)
    eta_scan=[]
    for eta in (0.0,0.1,0.2,0.3,0.5,0.7,1.0):
        ge=normalized(base**eta)
        eta_scan.append({"eta":eta,**residual(s,ge,aphys,bad)})
    p1=h1/h1.sum(); p2=h2/h2.sum()
    b1=np.maximum(p1+args.prior_frac/D,1e-300)
    b2=np.maximum(p2+args.prior_frac/D,1e-300)
    cv=[]
    for eta in np.linspace(0.05,1.0,20):
        q1=b1**eta; q1/=q1.sum()
        q2=b2**eta; q2/=q2.sum()
        ce=-0.5*(np.sum(p2*np.log(q1))+np.sum(p1*np.log(q2)))
        cv.append((float(eta),float(ce)))
    eta_cv=min(cv,key=lambda z:z[1])[0]
    gref=normalized(base**eta_cv)
    Fref=guide_fn(s,gref)
    aref=normalized(np.maximum(sla.expm_multiply(-args.tau*Fref,e),0.0))
    refresh={"eta_cv":eta_cv,"cv_curve":cv,
             "uniform_amp_fidelity":float(np.dot(afn,aphys)**2),
             "refreshed_amp_fidelity":float(np.dot(aref,aphys)**2),
             "uniform_full_fidelity":float(np.dot(s*afn,psi)**2),
             "refreshed_full_fidelity":float(np.dot(s*aref,psi)**2)}

    exact_next=normalized(sla.expm_multiply(-args.dt_k1*H,psi))
    sfixed=mismatch(s,exact_next)
    pred_hat=s*ghat-args.dt_k1*(H@(s*ghat))
    shat=np.where(pred_hat>=0,1.0,-1.0)
    skhat=mismatch(shat,exact_next)
    pred_fn=s*afn-args.dt_k1*(H@(s*afn))
    sfn=np.where(pred_fn>=0,1.0,-1.0)
    skfn=mismatch(sfn,exact_next)
    pred_ref=s*aref-args.dt_k1*(H@(s*aref))
    sref=np.where(pred_ref>=0,1.0,-1.0)
    skref=mismatch(sref,exact_next)
    refresh["k1_next_sign_mismatch"]=skref
    pred_ex=psi-args.dt_k1*(H@psi)
    sex=np.where(pred_ex>=0,1.0,-1.0)
    skex=mismatch(sex,exact_next)
    out={"y":y,"M":args.M,"tau":args.tau,
         "replicas":{"ess":[ess1,ess2],"unique":[u1,u2],
                     "fn_amp_fidelity":[fid1,fid2],"cross_fidelity":cross},
         "ratio_metrics":rms,
         "physical_residual":{"uniform":res_u,"walker_guide":res_h,"exact":res_x,
                              "eta_scan":eta_scan},
         "one_refresh":refresh,
         "k1_next_sign":{"fixed":sfixed,"walker_amp":skhat,
                         "exact_fn_amp":skfn,"exact_physical_amp":skex}}
    print("RESULT",json.dumps(out,sort_keys=True),flush=True)
    Path(args.out).write_text(json.dumps(out,indent=2))
    print("WROTE",args.out,flush=True)

if __name__=="__main__":
    main()
