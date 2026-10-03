#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np

from finite_tau_matching_20site_exact import D,special_columns
from finite_tau_ct_local_ratio_20site import H,distance_sign,normalized,mismatch
from finite_tau_ct_smc_fn20 import systematic
from finite_tau_localratio_mlp_oracle20 import MLP,make_wd
from finite_tau_ct_smc_edgeratio_longbeta20 import (
    ratio_params,k1_ratio,edge_data,wstep,reconstruct,exact_sequence,full_score,HO
)

def ct_block_stats(states,dt,J,lam,pot,rng):
    M=len(states); out=np.empty(M,np.int32);logw=np.empty(M)
    paths=[]
    for q,x0 in enumerate(states):
        x=int(x0);t=0.;lw=0.; seg=[]
        while True:
            rate=float(lam[x])
            if rate<=0:
                dur=dt-t;seg.append((x,dur));lw+=float(pot[x])*dur;break
            wait=float(rng.exponential(1./rate))
            if t+wait>=dt:
                dur=dt-t;seg.append((x,dur));lw+=float(pot[x])*dur;break
            seg.append((x,wait));lw+=float(pot[x])*wait;t+=wait
            a,b=J.indptr[x],J.indptr[x+1];rates=J.data[a:b]
            u=rng.random()*rate;j=int(np.searchsorted(np.cumsum(rates),u,side="right"))
            x=int(J.indices[a+min(j,len(rates)-1)])
        out[q]=x;logw[q]=lw;paths.append(seg)
    logw-=logw.max();w=np.exp(logw)
    ess=float(w.sum()**2/np.dot(w,w))
    end=np.bincount(out,weights=w,minlength=D).astype(float)
    dwell=np.zeros(D,float)
    for q,seg in enumerate(paths):
        z=float(w[q])
        for x,dur in seg:dwell[x]+=z*dur/dt
    end*=M/max(end.sum(),1e-300);dwell*=M/max(dwell.sum(),1e-300)
    sel=systematic(w,rng,M);new=out[sel]
    return new,ess,int(np.unique(new).size),end,dwell

def replica_ratio_rmse(a,b,minm=.25,maxedges=50000,seed=1):
    rg=np.random.default_rng(seed);src=[];dst=[]
    obs=np.flatnonzero((a>=minm)&(b>=minm))
    for x in obs:
        lo,hi=HO.indptr[x],HO.indptr[x+1];nb=HO.indices[lo:hi]
        nb=nb[(nb>x)&(a[nb]>=minm)&(b[nb]>=minm)]
        if len(nb):src.extend([int(x)]*len(nb));dst.extend(nb.tolist())
    if not src:return np.inf,0
    src=np.asarray(src,np.int32);dst=np.asarray(dst,np.int32)
    if len(src)>maxedges:
        z=rg.choice(len(src),maxedges,replace=False);src,dst=src[z],dst[z]
    r1=np.log(a[dst])-np.log(a[src]);r2=np.log(b[dst])-np.log(b[src])
    mass=np.sqrt(np.maximum(a[src]*a[dst]*b[src]*b[dst],1e-30))
    return float(np.sqrt(np.average((r1-r2)**2,weights=mass))),int(len(src))

def choose_stats(e1,d1,e2,d2,seed):
    rows=[]
    for alpha in (0.,.25,.5):
        a=(1-alpha)*e1+alpha*d1;b=(1-alpha)*e2+alpha*d2
        rm,n=replica_ratio_rmse(a,b,.25,50000,seed+int(100*alpha))
        rows.append((rm,alpha,n))
    rm,alpha,n=min(rows,key=lambda z:z[0])
    return (1-alpha)*e1+alpha*d1,(1-alpha)*e2+alpha*d2,float(alpha),float(rm),int(n)

def fit_stats(A,V,seed,epochs=8):
    if A is None or V is None:return None,0.,{}
    XA,XB,t,w,_,_=A;VA,VB,vt,vw,_,_=V
    net=MLP(XA.shape[1],64,seed);rg=np.random.default_rng(seed)
    for ep in range(epochs):
        order=rg.permutation(len(t))
        for k in range(0,len(order),2048):
            z=order[k:k+2048];wstep(net,XA[z],XB[z],t[z],w[z],5e-4)
    pd=net.pred(VB)-net.pred(VA)
    grid=np.linspace(0,1,21)
    loss=np.array([np.average((e*pd-vt)**2,weights=vw) for e in grid])
    j=int(np.argmin(loss));eta=float(grid[j])
    rm0=float(np.sqrt(np.average(vt*vt,weights=vw)));rm=float(np.sqrt(loss[j]))
    gain=(rm0-rm)/rm0 if rm0>0 else 0.
    if gain<.005:eta=0.
    return net,eta,{"eta":eta,"eta_raw":float(grid[j]),"edge_gain":float(gain),
                    "edge_rmse0":rm0,"edge_rmse":rm,
                    "train_edges":int(len(t)),"val_edges":int(len(vt))}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--M",type=int,default=25000)
    ap.add_argument("--beta",type=float,default=2.5)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    y=int(special_columns(np.random.default_rng(20260930),2)[1])
    s=distance_sign(y);wd=make_wd(H,[y])[0]
    exact=exact_sequence(y,a.beta)
    st1,st2,logg=reconstruct(y,s,a.M)
    rg1=np.random.default_rng(1110001);rg2=np.random.default_rng(1120001)
    rows=[];t=.5

    while t<a.beta-1e-12:
        J,lam,pot=ratio_params(s,logg)
        st1,e1,u1,end1,dw1=ct_block_stats(st1,.05,J,lam,pot,rg1)
        st2,e2,u2,end2,dw2=ct_block_stats(st2,.05,J,lam,pot,rg2)
        t=round(t+.05,10)
        stat1,stat2,alpha,rep_rm,rep_n=choose_stats(end1,dw1,end2,dw2,1200000+int(100*t))
        A=edge_data(stat1,logg,y,wd,t,.25,50000,1210000+int(100*t))
        V=edge_data(stat2,logg,y,wd,t,.25,50000,1220000+int(100*t))
        net,eta,fd=fit_stats(A,V,1230000+int(100*t))
        if net is not None and eta>0:
            f=full_score(net,y,wd,t);loggn=logg+eta*f;loggn-=loggn.max()
        else:loggn=logg.copy()

        h1=np.bincount(st1,minlength=D).astype(float);h2=np.bincount(st2,minlength=D).astype(float)
        q1=normalized(h1);q2=normalized(h2);q=normalized(h1+h2);ex=exact[t]
        row={"tau":t,"full":float(np.dot(s*q,ex)**2),"amp":float(np.dot(q,np.abs(ex))**2),
             "sign":mismatch(s,ex),"cross":float(np.dot(q1,q2)**2),
             "ess":[e1,e2],"unique":[u1,u2],"dwell_alpha":alpha,
             "replica_ratio_rmse":rep_rm,"replica_ratio_edges":rep_n,**fd}
        if t<a.beta-1e-12:
            sn=k1_ratio(s,loggn);row["next_sign"]=mismatch(sn,exact[round(t+.05,10)]);s=sn
        logg=loggn;rows.append(row)
        if abs((t*20)%5)<1e-9 or t in (.55,a.beta):
            print("PATHRATIO",json.dumps(row,sort_keys=True),flush=True)
    Path(a.out).write_text(json.dumps({"y":y,"M":a.M,"beta":a.beta,"rows":rows},indent=2))
    print("SUMMARY",json.dumps(rows[-1],sort_keys=True),flush=True)
if __name__=="__main__":main()
