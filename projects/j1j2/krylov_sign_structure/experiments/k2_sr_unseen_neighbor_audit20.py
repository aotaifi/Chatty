#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np

from finite_tau_matching_20site_exact import D,special_columns
from finite_tau_ct_local_ratio_20site import H,distance_sign,normalized
from finite_tau_ct_smc_edgeratio_longbeta20 import reconstruct
from finite_tau_localratio_mlp_oracle20 import feat,make_wd
from k2_sr_tangent_oneblock20 import TinyMLP

def project_with_sample(g,targ,y,wd,tau,ns,h,seed):
    p=g*g; p/=p.sum(); rg=np.random.default_rng(seed)
    tr=rg.choice(D,ns,p=p); va=rg.choice(D,ns,p=p)
    X=feat(tr,y,wd,tau); V=feat(va,y,wd,tau)
    mu=X.mean(0); sd=X.std(0); sd[sd<1e-6]=1
    X=(X-mu)/sd; V=(V-mu)/sd
    net=TinyMLP(X.shape[1],h,seed+991)
    J=net.jac(X); JV=net.jac(V)
    Jc=J-J.mean(0,keepdims=True); tc=targ[tr]-targ[tr].mean()
    JVc=JV-J.mean(0,keepdims=True); tv=targ[va]-targ[va].mean()
    S=(Jc.T@Jc)/len(Jc); b=(Jc.T@tc)/len(Jc)
    scale=max(float(np.trace(S)/len(S)),1e-14)
    ev,U=np.linalg.eigh(S); ub=U.T@b
    lams=scale*np.array([1e-5,1e-4,1e-3,1e-2,1e-1,1.,10.])
    vals=[]; sols=[]
    for lam in lams:
        sol=U@(ub/(ev+lam))
        vals.append(float(np.sqrt(np.mean((JVc@sol-tv)**2)))); sols.append(sol)
    k=int(np.argmin(vals)); sol=sols[k]
    allf=[]
    for q in range(0,D,4096):
        ix=np.arange(q,min(D,q+4096))
        Z=(feat(ix,y,wd,tau)-mu)/sd
        allf.append(net.jac(Z)@sol)
    f=np.concatenate(allf); f-=np.sum(p*f)
    return f,tr,va,{"lambda":float(lams[k]),"val_rmse":vals[k]}

def wrms(e,w):
    return float(np.sqrt(np.sum(w*e*e)/np.sum(w))) if np.sum(w)>0 else float("nan")

def cond_fid(a,b,mask):
    aa=a[mask]; bb=b[mask]
    na=np.linalg.norm(aa); nb=np.linalg.norm(bb)
    return float((aa@bb/(na*nb))**2) if na>0 and nb>0 else float("nan")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--M",type=int,default=100000)
    ap.add_argument("--dt",type=float,default=.0125)
    ap.add_argument("--ns",type=int,default=2000)
    ap.add_argument("--h",type=int,default=8)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    y=int(special_columns(np.random.default_rng(20260930),2)[1])
    wd=make_wd(H,[y])[0]; s=distance_sign(y)
    _,_,logg=reconstruct(y,s,a.M)
    g=normalized(np.exp(np.clip(logg-logg.max(),-700,0))); psi=s*g
    h1=H@psi; psi2=normalized(psi-a.dt*h1+.5*a.dt*a.dt*(H@h1))
    amp2=np.abs(psi2)
    targ=np.log(np.maximum(amp2,1e-300))-np.log(np.maximum(g,1e-300))
    tau=.5+a.dt
    seed=1500001+a.M+int(round(100*tau))
    f,tr,va,pd=project_with_sample(g,targ,y,wd,tau,a.ns,a.h,seed)
    gp=normalized(g*np.exp(np.clip(f,-20,20)))

    train_seen=np.zeros(D,dtype=bool); train_seen[np.unique(tr)]=True
    shell=np.zeros(D,dtype=bool); shell[train_seen]=True
    # directed offdiagonal edges from training sample with multiplicity
    errs=[]; bases=[]; weights=[]; seen_y=[]
    for x in tr:
        lo,hi=H.indptr[x],H.indptr[x+1]
        js=H.indices[lo:hi]; hs=H.data[lo:hi]
        m=js!=x; js=js[m]; hs=np.abs(hs[m])
        shell[js]=True
        exact=targ[js]-targ[x]
        pred=f[js]-f[x]
        errs.extend((pred-exact).tolist())
        bases.extend(exact.tolist())
        weights.extend(hs.tolist())
        seen_y.extend(train_seen[js].tolist())
    errs=np.asarray(errs); bases=np.asarray(bases); weights=np.asarray(weights); seen_y=np.asarray(seen_y,dtype=bool)

    masks={"train":train_seen,"shell_only":shell & ~train_seen,"far":~shell}
    sectors={}
    p2=amp2*amp2; pp=gp*gp; p0=g*g
    for name,m in masks.items():
        sectors[name]={
            "n":int(m.sum()),
            "target_mass":float(p2[m].sum()),
            "proj_mass":float(pp[m].sum()),
            "base_mass":float(p0[m].sum()),
            "cond_fid_base":cond_fid(g,amp2,m),
            "cond_fid_proj":cond_fid(gp,amp2,m)
        }
    out={
      "M":a.M,"dt":a.dt,"ns":a.ns,"h":a.h,**pd,
      "train_unique":int(train_seen.sum()),"shell_total":int(shell.sum()),
      "edge_n":int(len(errs)),
      "edge_seen_frac":float(seen_y.mean()),
      "edge_rmse_base_all":wrms(bases,weights),
      "edge_rmse_proj_all":wrms(errs,weights),
      "edge_rmse_base_seenY":wrms(bases[seen_y],weights[seen_y]),
      "edge_rmse_proj_seenY":wrms(errs[seen_y],weights[seen_y]),
      "edge_rmse_base_unseenY":wrms(bases[~seen_y],weights[~seen_y]),
      "edge_rmse_proj_unseenY":wrms(errs[~seen_y],weights[~seen_y]),
      "target_fid_global":float((gp@amp2)**2),
      "sectors":sectors
    }
    print(json.dumps(out,sort_keys=True),flush=True)
    Path(a.out).write_text(json.dumps(out,indent=2))
if __name__=="__main__":main()
