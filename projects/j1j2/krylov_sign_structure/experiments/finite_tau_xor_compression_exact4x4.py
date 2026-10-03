#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import scipy.sparse.linalg as sla
from closed_fn_krylov_exact4x4 import build_H, basis, idx, site, canonical, marshall_signs, NN, NNN

ROOT=Path("/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure")

def exact_columns(H,ys,tau):
    E=np.zeros((H.shape[0],len(ys)))
    E[ys,np.arange(len(ys))]=1.0
    G=np.asarray(sla.expm_multiply(-tau*H,E),float)
    G/=np.linalg.norm(G,axis=0,keepdims=True)
    return G

def bit(mask,i): return (int(mask)>>i)&1

def feat_basic(mask):
    k=int(mask).bit_count()
    nn=sum(bit(mask,u)!=bit(mask,v) for u,v in NN)
    nnn=sum(bit(mask,u)!=bit(mask,v) for u,v in NNN)
    return (k,nn,nnn)

def feat_plaquette(mask):
    h=[0]*16
    for y in range(4):
        for x in range(4):
            ids=[site(x,y),site(x+1,y),site(x,y+1),site(x+1,y+1)]
            pat=sum(bit(mask,i)<<j for j,i in enumerate(ids))
            h[pat]+=1
    return tuple(h)

def feat_combo(mask):
    return feat_basic(mask)+feat_plaquette(mask)
def loo_feature_error(keys,corr,p,feature_of):
    D,K=keys.shape
    ids={}; lut=np.empty(1<<16,dtype=np.int32)
    for m in range(1<<16):
        f=feature_of(m)
        if f not in ids: ids[f]=len(ids)
        lut[m]=ids[f]
    fid=lut[keys]
    F=len(ids)
    total=np.bincount(fid.ravel(),weights=(p*corr).ravel(),minlength=F)
    errs=[]
    for k in range(K):
        held=np.bincount(fid[:,k],weights=p[:,k]*corr[:,k],minlength=F)
        vote=total-held
        pred=np.where(vote[fid[:,k]]>=0,1,-1)
        errs.append(float(np.sum(p[:,k]*(pred!=corr[:,k]))))
    return errs,len(ids)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--tau",type=float,default=0.5)
    ap.add_argument("--ncols",type=int,default=64)
    ap.add_argument("--seed",type=int,default=1234)
    ap.add_argument("--out",default=str(ROOT/"results/finite_tau_xor_compression_exact4x4.json"))
    args=ap.parse_args()
    H,_=build_H(0.5)
    rng=np.random.default_rng(args.seed)
    ys=[int(x) for x in rng.choice(np.arange(H.shape[0]),size=args.ncols,replace=False)]
    G=exact_columns(H,ys,args.tau)
    sm=canonical(marshall_signs())
    base=sm[:,None]*sm[np.asarray(ys)][None,:]
    true=np.where(G>=0,1,-1).astype(np.int8)
    corr=true*base
    p=G*G
    bvals=np.asarray(basis,dtype=np.int64)
    keys=np.empty((H.shape[0],args.ncols),dtype=np.int32)
    for k,y in enumerate(ys):
        keys[:,k]=np.bitwise_xor(bvals,int(basis[y])).astype(np.int32)

    baseline=np.sum(p*(corr<0),axis=0)
    maps={
        "hamming":lambda m:(int(m).bit_count(),),
        "basic":feat_basic,
        "plaquette":feat_plaquette,
        "combo":feat_combo,
    }
    out={"tau":args.tau,"ncols":args.ncols,"seed":args.seed,
         "baseline_mean":float(np.mean(baseline)),"models":{}}
    for name,f in maps.items():
        errs,nbins=loo_feature_error(keys,corr,p,f)
        out["models"][name]={"mean_error":float(np.mean(errs)),
                             "max_error":float(np.max(errs)),
                             "nbins":int(nbins),
                             "errors":[float(x) for x in errs]}
        print(name,"mean",np.mean(errs),"max",np.max(errs),"bins",nbins,flush=True)

    Path(args.out).write_text(json.dumps(out,indent=2))
    print("baseline",out["baseline_mean"],flush=True)
    print("WROTE",args.out,flush=True)

if __name__=="__main__":
    main()
