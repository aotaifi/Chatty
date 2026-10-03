#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from finite_tau_matching_20site_exact import D,special_columns
from finite_tau_ct_local_ratio_20site import H,diag,R,C,HV,IDX,distance_sign,normalized,mismatch
from finite_tau_ct_smc_fn20 import ct_block
from finite_tau_localratio_mlp_oracle20 import MLP,feat,make_wd
from finite_tau_ct_smc_edgeratio_longbeta20 import ratio_params,k1_ratio,reconstruct,exact_sequence,full_score

DT=.05
HO=H.copy();HO.setdiag(0);HO.eliminate_zeros()

def exact_F_from_ratios(s,logg):
    prod=s[R]*s[C]; good=prod<0; bad=~good
    rat=np.exp(np.clip(logg[C]-logg[R],-20,20))
    shift=np.bincount(R[bad],weights=HV[bad]*rat[bad],minlength=D)
    return sp.coo_matrix((
        np.concatenate([-HV[good],diag+shift]),
        (np.concatenate([R[good],IDX]),np.concatenate([C[good],IDX]))
    ),shape=H.shape).tocsr()

def make_pairs(states,y,wd,tau,seed,nfit=20000,K=4):
    rg=np.random.default_rng(seed)
    if len(states)>nfit:
        states=states[rg.choice(len(states),size=nfit,replace=False)]
    src=[];dst=[];ww=[]
    for x0 in states:
        x=int(x0);a,b=HO.indptr[x],HO.indptr[x+1]
        nb=HO.indices[a:b]; hv=np.abs(HO.data[a:b])
        if not len(nb): continue
        kk=min(K,len(nb));pick=rg.choice(len(nb),size=kk,replace=False)
        fac=len(nb)/kk
        src.extend([x]*kk);dst.extend(nb[pick].tolist());ww.extend((fac*hv[pick]).tolist())
    src=np.asarray(src,np.int32);dst=np.asarray(dst,np.int32);ww=np.asarray(ww,float)
    return feat(src,y,wd,tau),feat(dst,y,wd,tau),src,dst,ww

def apply(net,grads,lr):
    net.t+=1
    for p,g,m,v in zip(net.par(),grads,net.m,net.v):
        m*=.9;m+=.1*g;v*=.999;v+=.001*g*g
        p-=lr*(m/(1-.9**net.t))/(np.sqrt(v/(1-.999**net.t))+1e-8)

def pair_loss(net,A,B,src,dst,w,logg,eta=1.0):
    df=net.pred(B)-net.pred(A)
    z=np.clip((logg[dst]-logg[src])+eta*df,-40,40)
    return float(np.average(np.logaddexp(0,z),weights=w))

def pair_step(net,A,B,src,dst,w,logg,lr=8e-4):
    fi,ci=net.fwd(A);fj,cj=net.fwd(B)
    z=np.clip((logg[dst]-logg[src])+(fj-fi),-40,40)
    ww=w/max(w.mean(),1e-12); dz=ww/(1+np.exp(-z))/len(z)
    gi=net.grads(ci,-dz);gj=net.grads(cj,dz)
    apply(net,[a+b for a,b in zip(gi,gj)],lr)
    return float(np.average(np.logaddexp(0,z),weights=w))

def fit_pair(st1,st2,logg,y,wd,tau,seed,epochs=6,nfit=20000,K=4):
    A,B,src,dst,w=make_pairs(st1,y,wd,tau,seed+1,nfit,K)
    VA,VB,vsrc,vdst,vw=make_pairs(st2,y,wd,tau,seed+2,nfit,K)
    X=np.vstack([A,B]);mu=X.mean(0);sd=X.std(0);sd[sd<1e-6]=1
    A=(A-mu)/sd;B=(B-mu)/sd;VA=(VA-mu)/sd;VB=(VB-mu)/sd
    net=MLP(A.shape[1],64,seed)
    net.W3[:]=0;net.b3[:]=0
    rg=np.random.default_rng(seed)
    base=pair_loss(net,VA,VB,vsrc,vdst,vw,logg,0.0)
    best=base;bestp=[x.copy() for x in net.par()]
    for ep in range(epochs):
        order=rg.permutation(len(src))
        for q in range(0,len(order),2048):
            z=order[q:q+2048]
            pair_step(net,A[z],B[z],src[z],dst[z],w[z],logg)
        vl=pair_loss(net,VA,VB,vsrc,vdst,vw,logg,1.0)
        if vl<best:
            best=vl;bestp=[x.copy() for x in net.par()]
    for a,b in zip(net.par(),bestp):a[:]=b
    df=net.pred(VB)-net.pred(VA)
    grid=np.linspace(0,1.5,31)
    losses=[]
    for eta in grid:
        z=np.clip((logg[vdst]-logg[vsrc])+eta*df,-40,40)
        losses.append(np.average(np.logaddexp(0,z),weights=vw))
    j=int(np.argmin(losses));eta=float(grid[j]);gain=float((base-losses[j])/max(abs(base),1e-12))
    if gain<0.002:eta=0.0
    return net,mu,sd,eta,{"pair_base":base,"pair_val":float(losses[j]),"pair_gain":gain,
                          "eta":eta,"train_pairs":int(len(src)),"val_pairs":int(len(vsrc))}

def exact_ratio_diag(net,mu,sd,eta,states,logg,ex,y,wd,tau,seed,nfit=5000,K=4,prefix="oracle"):
    A,B,src,dst,w=make_pairs(states,y,wd,tau,seed,nfit,K)
    m=(np.abs(ex[src])>1e-14)&(np.abs(ex[dst])>1e-14)
    if not np.any(m):
        return {f"{prefix}_edge_n":0}
    A=A[m];B=B[m];src=src[m];dst=dst[m];w=w[m]
    base=logg[dst]-logg[src]
    df=net.pred((B-mu)/sd)-net.pred((A-mu)/sd)
    pred=base+eta*df
    targ=np.log(np.abs(ex[dst]))-np.log(np.abs(ex[src]))
    def wrmse(z):
        return float(np.sqrt(np.average((z-targ)**2,weights=w)))
    def wcorr(a,b):
        ww=w/w.sum(); am=np.sum(ww*a); bm=np.sum(ww*b)
        av=a-am;bv=b-bm
        den=np.sqrt(np.sum(ww*av*av)*np.sum(ww*bv*bv))
        return float(np.sum(ww*av*bv)/den) if den>0 else 0.0
    return {f"{prefix}_edge_n":int(len(targ)),
            f"{prefix}_ratio_rmse_base":wrmse(base),
            f"{prefix}_ratio_rmse_update":wrmse(pred),
            f"{prefix}_ratio_corr_base":wcorr(base,targ),
            f"{prefix}_ratio_corr_update":wcorr(pred,targ)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--M",type=int,default=25000)
    ap.add_argument("--beta",type=float,default=2.5)
    ap.add_argument("--epochs",type=int,default=6)
    ap.add_argument("--nfit",type=int,default=20000)
    ap.add_argument("--K",type=int,default=4)
    ap.add_argument("--seed-offset",type=int,default=0)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()

    y=int(special_columns(np.random.default_rng(20260930),2)[1])
    s=distance_sign(y);wd=make_wd(H,[y])[0]
    exact=exact_sequence(y,a.beta)
    st1,st2,logg=reconstruct(y,s,a.M)
    rg1=np.random.default_rng(1010001+a.seed_offset)
    rg2=np.random.default_rng(1020001+a.seed_offset)
    rows=[];t=.5

    diag_times={.55,1.0,1.5,2.0,2.5,3.0,4.0,5.0,10.0}
    while t<a.beta-1e-12:
        tnext=round(t+.05,10)
        J,lam,pot=ratio_params(s,logg)
        fn_target2=None
        if tnext in diag_times:
            pre2=normalized(np.bincount(st2,minlength=D).astype(float))
            F=exact_F_from_ratios(s,logg)
            fn_target2=normalized(np.maximum(sla.expm_multiply(-DT*F,pre2),0.0))
        st1,e1,u1=ct_block(st1,.05,J,lam,pot,rg1)
        st2,e2,u2=ct_block(st2,.05,J,lam,pot,rg2)
        t=tnext
        h1=np.bincount(st1,minlength=D).astype(float)
        h2=np.bincount(st2,minlength=D).astype(float)
        q1=normalized(h1);q2=normalized(h2);q=normalized(h1+h2)

        net,mu,sd,eta,pd=fit_pair(st1,st2,logg,y,wd,t,
                                   1030001+a.seed_offset+int(round(100*t)),
                                   a.epochs,a.nfit,a.K)
        od=exact_ratio_diag(net,mu,sd,eta,st2,logg,exact[t],y,wd,t,
                            1040001+a.seed_offset+int(round(100*t)),
                            min(5000,a.nfit),a.K,prefix="oracle")
        fd={}
        if fn_target2 is not None:
            fd=exact_ratio_diag(net,mu,sd,eta,st2,logg,fn_target2,y,wd,t,
                                1050001+a.seed_offset+int(round(100*t)),
                                min(5000,a.nfit),a.K,prefix="fn")
            fd["walker_fn_fidelity"]=float(np.dot(q2,fn_target2)**2)
            fd["fn_physical_fidelity"]=float(np.dot(fn_target2,np.abs(exact[t]))**2)
        if eta>0:
            fs=[]
            for q0 in range(0,D,4096):
                ix=np.arange(q0,min(q0+4096,D))
                fs.append(net.pred((feat(ix,y,wd,t)-mu)/sd))
            f=np.concatenate(fs)
            loggn=logg+eta*f;loggn-=loggn.max()
        else:
            loggn=logg.copy()

        ex=exact[t]
        row={"tau":t,"full":float(np.dot(s*q,ex)**2),
             "amp":float(np.dot(q,np.abs(ex))**2),
             "sign":mismatch(s,ex),
             "cross":float(np.dot(q1,q2)**2),
             "ess":[e1,e2],"unique":[u1,u2],**pd,**od,**fd}
        if t<a.beta-1e-12:
            sn=k1_ratio(s,loggn)
            row["next_sign"]=mismatch(sn,exact[round(t+.05,10)])
            row["flip_weight"]=float(np.sum((ex*ex)[sn!=s]))
            s=sn
        rows.append(row);logg=loggn
        if abs((t*20)%5)<1e-9 or t in (.55,a.beta):
            print("PAIRRATIO",json.dumps(row,sort_keys=True),flush=True)

    result={"y":y,"M":a.M,"beta":a.beta,"dt":DT,
            "epochs":a.epochs,"nfit":a.nfit,"K":a.K,
            "seed_offset":a.seed_offset,"rows":rows}
    Path(a.out).write_text(json.dumps(result,indent=2))
    print("SUMMARY",json.dumps(rows[-1],sort_keys=True),flush=True)
    print("WROTE",a.out,flush=True)

if __name__=="__main__":main()
