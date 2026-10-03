#!/usr/bin/env python3
import argparse, json, math
from pathlib import Path
from collections import defaultdict
import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as cs
from scipy.optimize import linear_sum_assignment

L=6; N=L*L
ROOT=Path("/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure")
site=lambda x,y:(x%L)+L*(y%L)

NN=[]; NNN=[]
for y in range(L):
    for x in range(L):
        i=site(x,y)
        NN += [(i,site(x+1,y)),(i,site(x,y+1))]
        NNN += [(i,site(x+1,y+1)),(i,site(x+1,y-1))]
B1=len(NN); B2=len(NNN)
RMAX=.5*B1+.25*B2
INC=[[] for _ in range(N)]
for typ,bonds in enumerate((NN,NNN)):
    for k,(u,v) in enumerate(bonds):
        INC[u].append((typ,k,u,v)); INC[v].append((typ,k,u,v))

rr=[];cc=[]
for u,v in NN+NNN:
    rr += [u,v]; cc += [v,u]
A=sp.coo_matrix((np.ones(len(rr)),(rr,cc)),shape=(N,N)).tocsr()
SITE_DIST=cs.shortest_path(A,directed=False,unweighted=True)

def unlike(s,u,v):
    return ((s>>u)&1)!=((s>>v)&1)

def counts(s):
    return (sum(unlike(s,u,v) for u,v in NN),
            sum(unlike(s,u,v) for u,v in NNN))

def Vpot(u1,u2):
    # r-H_diag = J1(U1-B1/4)+J2(U2-B2/4), J2=1/2.
    return (u1-B1/4)+.5*(u2-B2/4)

def flip_update(s,u,v,u1,u2):
    aff={(typ,k,a,b) for typ,k,a,b in INC[u]+INC[v]}
    old1=old2=new1=new2=0
    sn=s^(1<<u)^(1<<v)
    for typ,k,a,b in aff:
        old=unlike(s,a,b); new=unlike(sn,a,b)
        if typ==0: old1+=old; new1+=new
        else: old2+=old; new2+=new
    return sn,u1+(new1-old1),u2+(new2-old2)
def run_paths(y,tau,npaths,seed,cref=27.0):
    rng=np.random.default_rng(seed)
    agg=defaultdict(lambda:[0.0,0.0,0.0,0])
    for _ in range(npaths):
        s=int(y); u1,u2=counts(s); t=0.0; logw=0.0; sg=1
        while True:
            dt=float(rng.exponential(1.0/RMAX))
            if t+dt>=tau:
                logw+=(Vpot(u1,u2)-cref)*(tau-t)
                break
            logw+=(Vpot(u1,u2)-cref)*dt
            t+=dt
            if rng.random() < (0.5*B1)/RMAX:
                u,v=NN[int(rng.integers(B1))]
            else:
                u,v=NNN[int(rng.integers(B2))]
            if unlike(s,u,v):
                s,u1,u2=flip_update(s,u,v,u1,u2)
                sg=-sg
        w=math.exp(logw)
        z=sg*w
        a=agg[s]; a[0]+=z; a[1]+=w; a[2]+=z*z; a[3]+=1
    return agg

def matching_distance(x,y):
    src=[i for i in range(N) if ((y>>i)&1) and not ((x>>i)&1)]
    dst=[i for i in range(N) if ((x>>i)&1) and not ((y>>i)&1)]
    if not src:return 0
    C=SITE_DIST[np.ix_(src,dst)]
    r,c=linear_sum_assignment(C)
    return int(round(float(C[r,c].sum())))
def marshall_pair(x,y):
    mask=sum(1<<site(a,b) for b in range(L) for a in range(L) if (a+b)%2==0)
    return 1 if (((x&mask).bit_count()-(y&mask).bit_count())%2)==0 else -1

def score(y,a,b,npaths,topk):
    keys=set(a)|set(b)
    rows=[]
    for x in keys:
        z1=a.get(x,[0,0,0,0])[0]/npaths
        z2=b.get(x,[0,0,0,0])[0]/npaths
        z=.5*(z1+z2)
        amp2=z*z
        if amp2==0: continue
        s1=1 if z1>=0 else -1; s2=1 if z2>=0 else -1
        rows.append((amp2,x,s1,s2,z1,z2))
    rows.sort(reverse=True)
    rows=rows[:topk]
    denom=sum(r[0] for r in rows)
    match_bad=marshall_bad=rep_bad=0.0
    details=[]
    cache={}
    for amp2,x,s1,s2,z1,z2 in rows:
        ds=cache.setdefault(x,matching_distance(x,y))
        smatch=1 if ds%2==0 else -1
        semp=1 if (z1+z2)>=0 else -1
        sm=marshall_pair(x,y)
        match_bad+=amp2*(smatch!=semp)
        marshall_bad+=amp2*(sm!=semp)
        rep_bad+=amp2*(s1!=s2)
        details.append((x,ds,semp,smatch,s1,s2,amp2))
    return {
      "topk":len(rows),"topk_amp2_sum":denom,
      "matching_weighted_error":match_bad/denom if denom else None,
      "marshall_weighted_error":marshall_bad/denom if denom else None,
      "replicate_sign_disagreement":rep_bad/denom if denom else None,
      "n_unique_union":len(keys),
      "details_top20":[list(r) for r in details[:20]]
    }
def initial_states(seed):
    neel=sum(1<<site(x,y) for y in range(L) for x in range(L) if (x+y)%2==0)
    stripe=sum(1<<site(x,y) for y in range(L) for x in range(L) if x%2==0)
    rng=np.random.default_rng(seed)
    occ=rng.choice(np.arange(N),size=N//2,replace=False)
    rnd=sum(1<<int(i) for i in occ)
    return {"neel":neel,"stripe":stripe,"random":rnd}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--tau",type=float,default=.5)
    ap.add_argument("--npaths",type=int,default=100000)
    ap.add_argument("--topk",type=int,default=2000)
    ap.add_argument("--columns",default="stripe,random")
    ap.add_argument("--seed",type=int,default=20260930)
    ap.add_argument("--out",default=str(ROOT/"results/finite_tau_ctmc_6x6_matching.json"))
    args=ap.parse_args()
    states=initial_states(args.seed)
    out={"tau":args.tau,"npaths_per_replica":args.npaths,"Rmax":RMAX,"columns":{}}
    for q,name in enumerate(args.columns.split(",")):
        name=name.strip(); y=states[name]
        print("COLUMN",name,"start",flush=True)
        a=run_paths(y,args.tau,args.npaths,args.seed+100*q)
        b=run_paths(y,args.tau,args.npaths,args.seed+100*q+1)
        sc=score(y,a,b,args.npaths,args.topk)
        out["columns"][name]={"y":int(y),**sc}
        print(name,json.dumps(sc,sort_keys=True),flush=True)
    Path(args.out).write_text(json.dumps(out,indent=2))
    print("WROTE",args.out,flush=True)

if __name__=="__main__":
    main()
