#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import scipy.sparse.linalg as sla
import torch

from finite_tau_matching_20site_exact import basis,D,special_columns
from finite_tau_ct_local_ratio_20site import H,distance_sign,normalized
from finite_tau_ct_smc_edgeratio_longbeta20 import reconstruct
from k2_gnn_tempered_reconstructed20 import PairGNN,bits_idx,make_wd,globals_from_wd

torch.set_num_threads(4)
HO=H.copy(); HO.setdiag(0); HO.eliminate_zeros()

def edge_audit(rg, dist, targ, f, seen, n=4000):
    xs=rg.choice(D,n,p=dist)
    src=[]; dst=[]; ww=[]
    for x in xs:
        lo,hi=HO.indptr[x],HO.indptr[x+1]
        js=HO.indices[lo:hi]; hs=np.abs(HO.data[lo:hi])
        src.extend([int(x)]*len(js)); dst.extend(js.tolist()); ww.extend(hs.tolist())
    src=np.asarray(src,np.int32); dst=np.asarray(dst,np.int32); ww=np.asarray(ww,float)
    tar=targ[dst]-targ[src]; err=(f[dst]-f[src])-tar
    both=seen[src]&seen[dst]
    def wrms(z,w):
        return float(np.sqrt(np.sum(w*z*z)/np.sum(w))) if np.sum(w)>0 else None
    return {
        "nedge":int(len(src)),
        "both_seen_weight_frac":float(np.sum(ww[both])/np.sum(ww)),
        "base_rmse":wrms(tar,ww),
        "proj_rmse":wrms(err,ww),
        "seen_rmse":wrms(err[both],ww[both]) if np.any(both) else None,
        "unseen_rmse":wrms(err[~both],ww[~both]) if np.any(~both) else None,
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode",choices=["exact","reconstructed"],required=True)
    ap.add_argument("--M",type=int,default=100000)
    ap.add_argument("--dt",type=float,default=.0125)
    ap.add_argument("--alpha",type=float,default=.8)
    ap.add_argument("--centers",type=int,default=2000)
    ap.add_argument("--epochs",type=int,default=4)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()

    y=int(special_columns(np.random.default_rng(20260930),2)[1])
    wd=make_wd(y); s=distance_sign(y)
    e=np.zeros(D); e[y]=1.0
    exact0=normalized(sla.expm_multiply(-.5*H,e))
    if a.mode=="exact":
        g=np.abs(exact0).copy()
        s=np.where(exact0>=0,1.0,-1.0)
    else:
        _,_,logg=reconstruct(y,s,a.M)
        g=normalized(np.exp(np.clip(logg-logg.max(),-700,0)))

    psi=s*g; h1=H@psi
    psi2=normalized(psi-a.dt*h1+.5*a.dt*a.dt*(H@h1))
    amp2=np.abs(psi2)
    targ=np.log(np.maximum(amp2,1e-300))-np.log(np.maximum(g,1e-300))

    ptr=g**(2*a.alpha); ptr/=ptr.sum()
    rg=np.random.default_rng(740001 if a.mode=="exact" else 740002)
    torch.manual_seed(740001 if a.mode=="exact" else 740002)
    cent=rg.choice(D,a.centers,p=ptr)
    counts=np.bincount(cent,minlength=D).astype(float)
    for x0 in np.unique(cent):
        lo,hi=HO.indptr[x0],HO.indptr[x0+1]
        ns=HO.indices[lo:hi]; hs=np.abs(HO.data[lo:hi])
        counts[ns]+=hs/max(float(hs.mean()),1e-12)
    train=np.flatnonzero(counts>0).astype(np.int32)
    seen=counts>0; w=counts[train]; w/=w.mean()
    yb=torch.tensor(bits_idx([y])[0])
    def data(ids,t):
        ids=np.asarray(ids,np.int64)
        x=torch.tensor(bits_idx(ids))
        yy=yb[None,:].repeat(len(ids),1)
        gg=torch.tensor(globals_from_wd(wd,t,ids))
        return x,yy,gg

    model=PairGNN()
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-6)
    bs=2048
    for ep in range(a.epochs):
        order=rg.permutation(len(train)); ls=[]
        model.train()
        for q in range(0,len(order),bs):
            z=order[q:q+bs]; ids=train[z]
            tt=torch.tensor(targ[ids].astype(np.float32))
            ww=torch.tensor(w[z].astype(np.float32))
            opt.zero_grad(); pred=model(*data(ids,.5+a.dt))
            loss=(ww*(pred-tt)**2).sum()/ww.sum()
            loss.backward(); opt.step(); ls.append(float(loss.detach()))
        print("TRAIN",a.mode,ep+1,float(np.mean(ls)),flush=True)

    outv=[]; model.eval()
    with torch.no_grad():
        for q in range(0,D,4096):
            ids=np.arange(q,min(q+4096,D))
            outv.append(model(*data(ids,.5+a.dt)).numpy())
    f=np.concatenate(outv)
    f-=np.sum(g*g*f)
    gp=normalized(g*np.exp(np.clip(f,-5,5)))

    p_exact0=exact0*exact0; p_exact0/=p_exact0.sum()
    p_guide=g*g; p_guide/=p_guide.sum()
    res={
      "mode":a.mode,"M":a.M,"dt":a.dt,"alpha":a.alpha,
      "centers":a.centers,"epochs":a.epochs,
      "train_unique":int(seen.sum()),"train_frac_D":float(seen.mean()),
      "physical_state_mass_seen":float(np.sum(p_exact0[seen])),
      "guide_state_mass_seen":float(np.sum(p_guide[seen])),
      "target_fid":float((gp@amp2)**2),
      "physical_edge_audit":edge_audit(rg,p_exact0,targ,f,seen),
      "guide_edge_audit":edge_audit(rg,p_guide,targ,f,seen),
    }
    Path(a.out).write_text(json.dumps(res,indent=2))
    print("RESULT",json.dumps(res,sort_keys=True),flush=True)

if __name__=="__main__":
    main()
