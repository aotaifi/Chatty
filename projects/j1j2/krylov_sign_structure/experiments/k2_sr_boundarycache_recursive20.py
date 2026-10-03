#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np
import scipy.sparse.linalg as sla

from finite_tau_matching_20site_exact import D
from finite_tau_ct_local_ratio_20site import H,normalized,mismatch
from finite_tau_localratio_mlp_oracle20 import feat,make_wd
from k2_sr_tangent_oneblock20 import TinyMLP

ROOT=Path(__file__).resolve().parents[1]
HO=H.copy();HO.setdiag(0);HO.eliminate_zeros()

def shell(xs):
    nb=[]
    for x in xs:
        lo,hi=HO.indptr[int(x)],HO.indptr[int(x)+1]
        nb.extend(HO.indices[lo:hi].tolist())
    return np.unique(np.asarray(nb,np.int32))

def neighbors(xs):
    nb=[]
    for x in xs:
        lo,hi=H.indptr[int(x)],H.indptr[int(x)+1]
        nb.extend(H.indices[lo:hi].tolist())
    return np.unique(np.asarray(nb,np.int32)) if nb else np.asarray([],np.int32)

def sr_project(g,targ,y,wd,tau,ns,h,seed):
    p=g*g;p/=p.sum();rg=np.random.default_rng(seed)
    tr=rg.choice(D,ns,p=p);va=rg.choice(D,ns,p=p)
    X=feat(tr,y,wd,tau);V=feat(va,y,wd,tau);mu=X.mean(0);sd=X.std(0);sd[sd<1e-6]=1
    X=(X-mu)/sd;V=(V-mu)/sd
    net=TinyMLP(X.shape[1],h,seed+991)
    J=net.jac(X);JV=net.jac(V)
    Jc=J-J.mean(0,keepdims=True);tc=targ[tr]-targ[tr].mean()
    JVc=JV-JV.mean(0,keepdims=True);tv=targ[va]-targ[va].mean()
    S=(Jc.T@Jc)/len(Jc);b=(Jc.T@tc)/len(Jc)
    scale=max(float(np.trace(S)/len(S)),1e-14)
    ev,U=np.linalg.eigh(S);ub=U.T@b
    lams=scale*np.array([1e-5,1e-4,1e-3,1e-2,1e-1,1.,10.])
    vals=[];sols=[]
    for lam in lams:
        sol=U@(ub/(ev+lam))
        vals.append(float(np.sqrt(np.mean((JVc@sol-tv)**2))));sols.append(sol)
    k=int(np.argmin(vals));sol=sols[k]
    out=[]
    for q in range(0,D,4096):
        ix=np.arange(q,min(D,q+4096));out.append(net.jac((feat(ix,y,wd,tau)-mu)/sd)@sol)
    f=np.concatenate(out);f-=np.sum(p*f)
    return f,tr,{"lambda":float(lams[k]),"val_rmse":vals[k],
                 "f_rms_p":float(np.sqrt(np.sum(p*f*f)))}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--beta",type=float,default=3.0)
    ap.add_argument("--dt",type=float,default=.0125)
    ap.add_argument("--ns",type=int,default=2000)
    ap.add_argument("--h",type=int,default=8)
    ap.add_argument("--margin",type=float,default=.2)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()

    c=np.load(ROOT/"results/reconstruct_M100000_y59279.npz")
    y=int(c["y"]);s=c["s"].astype(float);logg=c["logg"].copy()
    wd=make_wd(H,[y])[0]

    # exact only for diagnostic scoring
    e=np.zeros(D);e[y]=1.;psi_ex=normalized(sla.expm_multiply(-.5*H,e))
    exact={.5:psi_ex.copy()};tt=.5
    while tt<a.beta-1e-12:
        psi_ex=normalized(sla.expm_multiply(-a.dt*H,psi_ex))
        tt=round(tt+a.dt,10);exact[tt]=psi_ex.copy()

    rows=[];t=.5
    while t<a.beta-1e-12:
        g=normalized(np.exp(np.clip(logg-logg.max(),-700,0)));p=g*g;p/=p.sum();psi=s*g
        h1=H@psi
        raw2=psi-a.dt*h1+.5*a.dt*a.dt*(H@h1)
        psi2=normalized(raw2)
        Rraw=raw2/np.where(np.abs(psi)>1e-300,psi,np.where(psi>=0,1.,-1.)*1e-300)
        tnext=round(t+a.dt,10)
        targ=np.log(np.maximum(np.abs(psi2),1e-300))-np.log(np.maximum(g,1e-300))

        f,tr,diag=sr_project(g,targ,y,wd,tnext,a.ns,a.h,3100001+int(round(10000*tnext)))

        sh=shell(tr)
        B=sh[(Rraw[sh]<0)|(np.abs(Rraw[sh])<a.margin)]
        BN=neighbors(B)
        if len(B):
            BN=np.unique(np.concatenate([B,BN]))
            # align teacher correction to the SR gauge using current physical measure
            gauge=float(np.sum(p*(f-targ)))
            f[BN]=targ[BN]+gauge

        logg=logg+f;logg-=logg.max()
        s=np.where(raw2>=0,1.0,-1.0)
        gn=normalized(np.exp(np.clip(logg-logg.max(),-700,0)))
        ex=exact[tnext]
        row={"tau":tnext,"full":float(np.dot(s*gn,ex)**2),
             "amp":float(np.dot(gn,np.abs(ex))**2),"sign":mismatch(s,ex),
             "k2_target_full":float(np.dot(psi2,ex)**2),
             "projection_target_fid":float(np.dot(gn,np.abs(psi2))**2),
             "shell_n":int(len(sh)),"boundary_n":int(len(B)),"cache_n":int(len(BN)),
             "cache_frac_D":float(len(BN)/D),**diag}
        rows.append(row);t=tnext
        if abs((t*4)-round(t*4))<1e-9 or t in (.5125,.55,1.,1.5,2.,2.5,3.,a.beta):
            print("K2SRCACHE",json.dumps(row,sort_keys=True),flush=True)

    out={"beta":a.beta,"dt":a.dt,"ns":a.ns,"h":a.h,"margin":a.margin,"rows":rows}
    Path(a.out).write_text(json.dumps(out,indent=2))
    print("FINAL",json.dumps(rows[-1],sort_keys=True),flush=True)
if __name__=="__main__":main()
