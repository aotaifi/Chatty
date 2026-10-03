#!/usr/bin/env python3
import json,time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as cs
from finite_tau_matching_20site_exact import build_H,D,special_columns
ROOT=Path("/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure")
H,diag=build_H(.5); HO=(H-sp.diags(diag)).tocsr()

def sign_for_y(y):
    A=HO.copy();A.data=np.ones_like(A.data)
    d=np.asarray(cs.shortest_path(A,directed=False,unweighted=True,indices=[y]))[0]
    return np.where((d.astype(np.int64)%2)==0,1,-1).astype(np.int8)

def local_cache(sign):
    cache={}
    def loc(x):
        x=int(x)
        if x in cache:return cache[x]
        a,b=HO.indptr[x],HO.indptr[x+1]; ys=HO.indices[a:b]; hs=HO.data[a:b]
        opp=sign[ys]!=sign[x]
        rates=hs[opp].astype(float)
        dfn=float(diag[x]+np.sum(hs[~opp]))
        z=(dfn,ys[opp].astype(np.int32),rates);cache[x]=z;return z
    return loc

def systematic(w,rng,M):
    w=np.asarray(w,float);w/=w.sum();c=np.cumsum(w);c[-1]=1
    return np.searchsorted(c,rng.random()/M+np.arange(M)/M,side='right')

def run(y,M,beta,tau_max,seed):
    rng=np.random.default_rng(seed);sign=sign_for_y(y);loc=local_cache(sign)
    walkers=np.full(M,int(y),np.int32);t=0.;steps=0
    while t<beta-1e-14:
        dat=[loc(x) for x in walkers]
        dfn=np.array([z[0] for z in dat]);el=np.array([z[0]-z[2].sum() for z in dat])
        Eref=float(el.mean());mx=max(0.,float(np.max(dfn-Eref)))
        dt=min(tau_max,beta-t,0.8/mx if mx>0 else tau_max)
        nxt=np.empty(M,np.int32);bw=np.empty(M)
        for k,(x,(dv,ys,rates)) in enumerate(zip(walkers,dat)):
            stay=1-dt*(dv-Eref);move=dt*rates;tot=stay+move.sum()
            u=rng.random()*tot
            if u<stay:nxt[k]=x
            else:
                j=np.searchsorted(np.cumsum(move),u-stay,side='right');nxt[k]=int(ys[min(j,len(ys)-1)])
            bw[k]=tot
        walkers=nxt[systematic(bw,rng,M)];t+=dt;steps+=1
    return walkers,steps
def main():
    rng=np.random.default_rng(20260930);ys=special_columns(rng,6);train_y=ys[:4]
    M=8192;beta=.1;reps={}
    for rep,base in [(1,31000),(2,41000)]:
        arr=[]
        for k,y in enumerate(train_y):
            w,steps=run(int(y),M,beta,.005,base+k)
            arr.append(w);print("REP",rep,"y",y,"unique",len(np.unique(w)),"steps",steps,flush=True)
        reps[rep]=np.stack(arr)
    path=ROOT/'results/finite_tau_gfmc_uniform_block20.npz'
    np.savez_compressed(path,ys=np.asarray(ys,np.int32),train_y=np.asarray(train_y,np.int32),
                        rep1=reps[1],rep2=reps[2],M=M,beta=beta)
    print("WROTE",path,flush=True)
if __name__=='__main__':main()
