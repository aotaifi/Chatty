#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import scipy.sparse.linalg as sla

from closed_fn_krylov_exact4x4 import build_H, basis, idx, site, canonical, marshall_signs

ROOT=Path("/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure")

def named_indices():
    neel=sum(1<<site(x,y) for y in range(4) for x in range(4) if (x+y)%2==0)
    stripe=sum(1<<site(x,y) for y in range(4) for x in range(4) if x%2==0)
    return idx[neel],idx[stripe]

def exact_columns(H,ys,tau):
    E=np.zeros((H.shape[0],len(ys)))
    E[ys,np.arange(len(ys))]=1.0
    G=np.asarray(sla.expm_multiply(-tau*H,E),float)
    G/=np.linalg.norm(G,axis=0,keepdims=True)
    return G
def analyze(G,ys):
    D,K=G.shape
    sm=canonical(marshall_signs())
    base=sm[:,None]*sm[np.asarray(ys)][None,:]
    true=np.where(G>=0,1,-1).astype(np.int8)
    corr=true*base
    p=G*G
    baseline=np.sum(p*(corr<0),axis=0)

    # Transfer one column's correction vector to another target column.
    transfer=np.empty((K,K))
    for j in range(K):
        pred=base*corr[:,j:j+1]
        transfer[j]=np.sum(p*(pred!=true),axis=0)

    # Best single source by mean target error, excluding self.
    mean_off=[]
    for j in range(K):
        mask=np.arange(K)!=j
        mean_off.append(float(np.mean(transfer[j,mask])))
    best=int(np.argmin(mean_off))

    # Shared y-independent weighted-majority correction, leave-one-out.
    loo=[]
    for k in range(K):
        keep=np.arange(K)!=k
        vote=np.sum(p[:,keep]*corr[:,keep],axis=1)
        cshared=np.where(vote>=0,1,-1)
        pred=base[:,k]*cshared
        loo.append(float(np.sum(p[:,k]*(pred!=true[:,k]))))
    # Joint (x,y) compression test: correction as a function only of XOR=x xor y.
    keys=np.empty((D,K),dtype=np.int32)
    bvals=np.asarray(basis,dtype=np.int64)
    for k,y in enumerate(ys):
        keys[:,k]=np.bitwise_xor(bvals,int(basis[y])).astype(np.int32)
    xor_loo=[]
    ham_loo=[]
    pop=np.array([i.bit_count() for i in range(1<<16)],dtype=np.int8)
    for k in range(K):
        keep=np.arange(K)!=k
        trk=keys[:,keep].ravel()
        trv=(p[:,keep]*corr[:,keep]).ravel()
        vote=np.bincount(trk,weights=trv,minlength=1<<16)
        seen=np.bincount(trk,weights=p[:,keep].ravel(),minlength=1<<16)
        ck=np.ones(D,dtype=np.int8)
        tk=keys[:,k]
        have=seen[tk]>0
        ck[have]=np.where(vote[tk[have]]>=0,1,-1)
        pred=base[:,k]*ck
        xor_loo.append(float(np.sum(p[:,k]*(pred!=true[:,k]))))

        # Extremely compact control: use only Hamming distance |x xor y|.
        hk=pop[trk]
        hvote=np.bincount(hk,weights=trv,minlength=17)
        hseen=np.bincount(hk,weights=p[:,keep].ravel(),minlength=17)
        th=pop[tk]
        ch=np.ones(D,dtype=np.int8)
        hh=hseen[th]>0
        ch[hh]=np.where(hvote[th[hh]]>=0,1,-1)
        predh=base[:,k]*ch
        ham_loo.append(float(np.sum(p[:,k]*(predh!=true[:,k]))))

    # Weighted low-rank diagnostic of the correction family.
    wrow=np.mean(p,axis=1)
    mask=wrow>1e-12
    X=np.sqrt(wrow[mask,None])*corr[mask].astype(float)
    s=np.linalg.svd(X,compute_uv=False)
    frac=np.cumsum(s*s)/np.sum(s*s)

    return {
        "baseline_mismatch":baseline.tolist(),
        "transfer_matrix":transfer.tolist(),
        "best_single_source":best,
        "best_single_source_mean_offdiag_error":mean_off[best],
        "mean_single_source_offdiag_error":float(np.mean(mean_off)),
        "loo_shared_majority_errors":loo,
        "loo_shared_majority_mean_error":float(np.mean(loo)),
        "xor_loo_errors":xor_loo,
        "xor_loo_mean_error":float(np.mean(xor_loo)),
        "hamming_loo_errors":ham_loo,
        "hamming_loo_mean_error":float(np.mean(ham_loo)),
        "sv_energy_frac":[float(frac[min(r-1,len(frac)-1)]) for r in (1,2,4,8,16)],
        "effective_rows":int(np.sum(mask))
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--tau",type=float,default=0.5)
    ap.add_argument("--ncols",type=int,default=16)
    ap.add_argument("--seed",type=int,default=1234)
    ap.add_argument("--out",default=str(ROOT/"results/finite_tau_cross_column_exact4x4.json"))
    args=ap.parse_args()
    H,_=build_H(0.5)
    neel,stripe=named_indices()
    rng=np.random.default_rng(args.seed)
    pool=np.setdiff1d(np.arange(H.shape[0]),np.array([neel,stripe]))
    ys=[int(neel),int(stripe)]+[int(x) for x in rng.choice(pool,size=args.ncols-2,replace=False)]
    G=exact_columns(H,ys,args.tau)
    res=analyze(G,ys)
    out={"tau":args.tau,"ncols":args.ncols,"seed":args.seed,"ys":ys,**res}
    Path(args.out).write_text(json.dumps(out,indent=2))
    print("BASELINE mean",float(np.mean(res["baseline_mismatch"])))
    print("SHARED majority LOO mean",res["loo_shared_majority_mean_error"])
    print("BEST source mean offdiag",res["best_single_source_mean_offdiag_error"])
    print("XOR LOO mean",res["xor_loo_mean_error"])
    print("HAMMING LOO mean",res["hamming_loo_mean_error"])
    print("SVD fractions",res["sv_energy_frac"])
    print("WROTE",args.out)

if __name__=="__main__":
    main()
