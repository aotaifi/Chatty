#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import scipy.sparse.csgraph as cs
from scipy.optimize import linear_sum_assignment
from finite_tau_matching_20site_exact import build_H, basis, D, N, NN, NNN, special_columns

ROOT=Path("/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure")

def site_graph_dist():
    rr=[];cc=[]
    for u,v in NN+NNN:
        rr += [u,v]; cc += [v,u]
    A=sp.coo_matrix((np.ones(len(rr)),(rr,cc)),shape=(N,N)).tocsr()
    return cs.shortest_path(A,directed=False,unweighted=True)
SD=site_graph_dist()

def matching_parity(si,sj):
    a=int(basis[si]); b=int(basis[sj])
    src=[i for i in range(N) if ((b>>i)&1) and not ((a>>i)&1)]
    dst=[i for i in range(N) if ((a>>i)&1) and not ((b>>i)&1)]
    if not src: return 1
    C=SD[np.ix_(src,dst)]
    r,c=linear_sum_assignment(C)
    d=int(round(float(C[r,c].sum())))
    return 1 if d%2==0 else -1
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ncols",type=int,default=4)
    ap.add_argument("--nsamp",type=int,default=20000)
    ap.add_argument("--seed",type=int,default=20260930)
    ap.add_argument("--out",default=str(ROOT/"results/finite_tau_residual_sign_20site.json"))
    args=ap.parse_args()
    rng=np.random.default_rng(args.seed)
    H,diag=build_H(.5)
    Habs=2*sp.diags(diag)-H
    ys=special_columns(rng,args.ncols)
    E=np.zeros((D,args.ncols)); E[ys,np.arange(args.ncols)]=1.0
    out={"D":D,"ys":ys,"nsamp":args.nsamp,"times":{}}

    for tau in (0.0125,0.025,0.05,0.1,0.15):
        print("tau",tau,flush=True)
        G=np.asarray(sla.expm_multiply(-tau*H,E),float)
        A=np.asarray(sla.expm_multiply(-tau*Habs,E),float)
        rows=[]
        for k,y in enumerate(ys):
            den=float(np.sum(A[:,k]))
            raw=float(np.sum(G[:,k])/den)
            prob=A[:,k]/den
            xs=rng.choice(np.arange(D),size=args.nsamp,replace=True,p=prob)
            vals=np.empty(args.nsamp,float)
            for q,x in enumerate(xs):
                a=A[x,k]
                vals[q]=matching_parity(int(x),int(y))*G[x,k]/a if a>0 else 0.0
            rows.append({"y":int(y),"raw_average_sign":raw,
                         "guided_average_sign":float(np.mean(vals)),
                         "guided_se":float(np.std(vals,ddof=1)/np.sqrt(args.nsamp)),
                         "guided_negative_sample_fraction":float(np.mean(vals<0))})
        out["times"][str(tau)]={"rows":rows,
            "raw_mean":float(np.mean([r["raw_average_sign"] for r in rows])),
            "guided_mean":float(np.mean([r["guided_average_sign"] for r in rows]))}
        print(out["times"][str(tau)],flush=True)
    Path(args.out).write_text(json.dumps(out,indent=2))
    print("WROTE",args.out,flush=True)

if __name__=="__main__":
    main()
