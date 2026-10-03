#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np

from finite_tau_matching_20site_exact import D,special_columns
from finite_tau_ct_local_ratio_20site import H,distance_sign,normalized
from finite_tau_ct_smc_edgeratio_longbeta20 import reconstruct
from finite_tau_localratio_mlp_oracle20 import feat,make_wd
from k2_sr_tangent_oneblock20 import TinyMLP

def edge_set(xs):
    src=[]; dst=[]; w=[]
    for x in xs:
        lo,hi=H.indptr[x],H.indptr[x+1]
        js=H.indices[lo:hi]; hs=np.abs(H.data[lo:hi]); m=js!=x
        js=js[m]; hs=hs[m]
        src.extend([int(x)]*len(js)); dst.extend(js.tolist()); w.extend(hs.tolist())
    return np.asarray(src,dtype=np.int32),np.asarray(dst,dtype=np.int32),np.asarray(w,float)

def jac_for(net,ids,y,wd,tau,mu,sd,batch=2048):
    out=[]
    for q in range(0,len(ids),batch):
        z=ids[q:q+batch]
        X=(feat(z,y,wd,tau)-mu)/sd
        out.append(net.jac(X))
    return np.vstack(out)

def wrmse(e,w):
    return float(np.sqrt(np.sum(w*e*e)/np.sum(w)))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--M",type=int,default=100000)
    ap.add_argument("--dt",type=float,default=.0125)
    ap.add_argument("--ns",type=int,default=2000)
    ap.add_argument("--h",type=int,default=8)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    y=int(special_columns(np.random.default_rng(20260930),2)[1]); wd=make_wd(H,[y])[0]
    s=distance_sign(y); _,_,logg=reconstruct(y,s,a.M)
    g=normalized(np.exp(np.clip(logg-logg.max(),-700,0))); psi=s*g
    h1=H@psi; psi2=normalized(psi-a.dt*h1+.5*a.dt*a.dt*(H@h1)); amp2=np.abs(psi2)
    targ=np.log(np.maximum(amp2,1e-300))-np.log(np.maximum(g,1e-300))
    tau=.5+a.dt; p=g*g; p/=p.sum(); rg=np.random.default_rng(1900001+a.M+int(10000*a.dt))
    trx=rg.choice(D,a.ns,p=p); vax=rg.choice(D,a.ns,p=p)
    trsrc,trdst,trw=edge_set(trx); vasrc,vadst,vaw=edge_set(vax)
    # normalize features from sampled x plus their train neighbors
    ids=np.concatenate([trx,trdst])
    X=feat(ids,y,wd,tau); mu=X.mean(0); sd=X.std(0); sd[sd<1e-6]=1
    net=TinyMLP(X.shape[1],a.h,77)
    Jx=jac_for(net,trsrc,y,wd,tau,mu,sd); Jy=jac_for(net,trdst,y,wd,tau,mu,sd)
    A=Jy-Jx; tt=targ[trdst]-targ[trsrc]
    ww=trw/max(float(np.mean(trw)),1e-12)
    sw=np.sqrt(ww)[:,None]; Aw=A*sw; tw=tt*np.sqrt(ww)
    S=(Aw.T@Aw)/len(Aw); b=(Aw.T@tw)/len(Aw)
    scale=max(float(np.trace(S)/len(S)),1e-14)
    ev,U=np.linalg.eigh(S); ub=U.T@b
    Vx=jac_for(net,vasrc,y,wd,tau,mu,sd); Vy=jac_for(net,vadst,y,wd,tau,mu,sd)
    VA=Vy-Vx; vt=targ[vadst]-targ[vasrc]
    lams=scale*np.array([1e-5,1e-4,1e-3,1e-2,1e-1,1.,10.])
    vals=[]; sols=[]
    for lam in lams:
        sol=U@(ub/(ev+lam)); pred=VA@sol
        vals.append(wrmse(pred-vt,vaw)); sols.append(sol)
    k=int(np.argmin(vals)); sol=sols[k]
    allf=[]
    for q in range(0,D,4096):
        ix=np.arange(q,min(D,q+4096))
        allf.append(net.jac((feat(ix,y,wd,tau)-mu)/sd)@sol)
    f=np.concatenate(allf); f-=np.sum(p*f)
    gp=normalized(g*np.exp(np.clip(f,-20,20)))
    # independent validation-edge diagnostics; classify by whether y entered train expanded shell
    train_exp=np.zeros(D,dtype=bool); train_exp[np.unique(np.concatenate([trsrc,trdst]))]=True
    pred=VA@sol; base=vt; err=pred-vt; seen=train_exp[vadst]
    out={"M":a.M,"dt":a.dt,"ns":a.ns,"h":a.h,
         "train_edges":int(len(trsrc)),"val_edges":int(len(vasrc)),
         "train_expanded_unique":int(train_exp.sum()),
         "lambda":float(lams[k]),"val_edge_rmse":vals[k],
         "val_edge_base_rmse":wrmse(base,vaw),
         "val_edge_seenY_frac":float(seen.mean()),
         "val_edge_proj_rmse_seenY":wrmse(err[seen],vaw[seen]) if np.any(seen) else None,
         "val_edge_base_rmse_seenY":wrmse(base[seen],vaw[seen]) if np.any(seen) else None,
         "val_edge_proj_rmse_unseenY":wrmse(err[~seen],vaw[~seen]) if np.any(~seen) else None,
         "val_edge_base_rmse_unseenY":wrmse(base[~seen],vaw[~seen]) if np.any(~seen) else None,
         "target_fid":float((gp@amp2)**2)}
    print(json.dumps(out,sort_keys=True),flush=True)
    Path(a.out).write_text(json.dumps(out,indent=2))
if __name__=="__main__":main()
