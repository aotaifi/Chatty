#!/usr/bin/env python3
import argparse,json,math,time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as cs
import scipy.sparse.linalg as sla
from finite_tau_matching_20site_exact import build_H,D,special_columns

ROOT=Path("/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure")
H,diag=build_H(.5)
HO=(H-sp.diags(diag)).tocsr()

def labels_for_y(y):
    co=HO.tocoo(); v=np.abs(co.data)
    cost=np.where(v>0.375,1000.0,1001.0)
    W=sp.coo_matrix((cost,(co.row,co.col)),shape=H.shape).tocsr()
    wd=np.asarray(cs.shortest_path(W,directed=False,indices=[y]))[0]
    d=(wd//1000).astype(np.int16)
    n2=(wd-1000*d).round().astype(np.int16)
    ec=np.rint(8.0*diag).astype(np.int16)+100
    raw=d.astype(np.int64)*100000+n2.astype(np.int64)*1000+ec.astype(np.int64)
    vals,lab=np.unique(raw,return_inverse=True)
    return d,lab.astype(np.int32),np.bincount(lab,minlength=len(vals)).astype(float)
def systematic(w,rng,M):
    w=np.asarray(w,float); w/=w.sum()
    c=np.cumsum(w); c[-1]=1.0
    return np.searchsorted(c,rng.random()/M+np.arange(M)/M,side="right")

def make_local(theta,lab,sgn):
    cache={}
    def local(x):
        x=int(x)
        if x in cache:return cache[x]
        gx=theta[lab[x]]
        a,b=HO.indptr[x],HO.indptr[x+1]
        ys=HO.indices[a:b]; hs=HO.data[a:b]
        lr=theta[lab[ys]]-gx
        rat=np.exp(np.clip(lr,-30,30))
        opp=sgn[ys]!=sgn[x]
        rates=hs[opp]*rat[opp]
        dfn=float(diag[x]+np.sum(hs[~opp]*rat[~opp]))
        cache[x]=(dfn,ys[opp],rates)
        return cache[x]
    return local

def propagate(walkers,theta,lab,sgn,block_dt,tau_max,rng):
    t=0.0; steps=0
    local=make_local(theta,lab,sgn)
    M=len(walkers)
    while t<block_dt-1e-14:
        dat=[local(int(x)) for x in walkers]
        dfn=np.array([z[0] for z in dat])
        el=np.array([z[0]-np.sum(z[2]) for z in dat])
        Eref=float(el.mean())
        mx=max(0.0,float(np.max(dfn-Eref)))
        dt=min(tau_max,block_dt-t,0.8/mx if mx>0 else tau_max)
        nxt=np.empty(M,dtype=np.int32); bw=np.empty(M)
        for k,(x,(dv,ys,rates)) in enumerate(zip(walkers,dat)):
            stay=1-dt*(dv-Eref); move=dt*rates; tot=stay+move.sum()
            if stay<0 or tot<=0: raise RuntimeError(("bad",stay,tot,dt,dv,Eref))
            u=rng.random()*tot
            if u<stay:nxt[k]=x
            else:
                j=np.searchsorted(np.cumsum(move),u-stay,side="right")
                nxt[k]=int(ys[min(j,len(ys)-1)])
            bw[k]=tot
        walkers=nxt[systematic(bw,rng,M)]
        t+=dt; steps+=1
    return walkers,steps
def guide_fidelity(theta,lab,target):
    g=np.exp(np.clip(theta[lab]-np.max(theta),-60,0))
    g/=np.linalg.norm(g)
    a=np.abs(target); a/=np.linalg.norm(a)
    return float(np.dot(g,a)**2)

def update_theta(theta,lab,bin_size,walkers,eta,prior,pc):
    B=len(theta); M=len(walkers)
    cnt=np.bincount(lab[walkers],minlength=B).astype(float)
    z=2*(theta-np.max(theta))
    q=bin_size*np.exp(np.clip(z,-60,0)); q/=q.sum()
    if pc>=0:
        m=(cnt+pc)/(M+pc*B)
    else:
        m=(cnt+prior*q)/(M+prior)
    delta=np.log(np.maximum(m,1e-300)/np.maximum(q,1e-300))
    delta-=np.sum(q*delta)
    new=np.clip(theta+eta*delta,-30,30)
    # reweight mixed walkers from old g*a to new g*a
    rw=np.exp(np.clip(new[lab[walkers]]-theta[lab[walkers]],-30,30))
    return new,rw,cnt,q,m

def run_column(y,M,block_dt,beta,tau_max,eta,prior,pc,seed):
    rng=np.random.default_rng(seed)
    d,lab,bin_size=labels_for_y(y); sgn=np.where(d%2==0,1,-1).astype(np.int8)
    theta=np.zeros(len(bin_size),float)
    walkers=np.full(M,int(y),dtype=np.int32)
    exact=np.zeros(D); exact[y]=1.0
    rec=[]; t0=time.time()
    for n in range(1,int(round(beta/block_dt))+1):
        walkers,nsub=propagate(walkers,theta,lab,sgn,block_dt,tau_max,rng)
        exact=np.asarray(sla.expm_multiply(-block_dt*H,exact),float)
        exact/=np.linalg.norm(exact)
        old=theta.copy()
        theta,rw,cnt,q,m=update_theta(theta,lab,bin_size,walkers,eta,prior,pc)
        walkers=walkers[systematic(rw,rng,M)]
        true_s=np.where(exact>=0,1,-1)
        p=exact*exact
        row={"tau":n*block_dt,"guide_fidelity":guide_fidelity(theta,lab,exact),
             "sign_mismatch_mass":float(np.sum(p[sgn!=true_s])),
             "occupied_bins":int(np.sum(cnt>0)),"n_bins":int(len(theta)),
             "unique_walkers":int(len(np.unique(walkers))),"nsub":int(nsub),
             "theta_rms_step":float(np.sqrt(np.mean((theta-old)**2)))}
        rec.append(row); print("M",M,"y",y,row,flush=True)
    return rec
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--M",type=int,default=2048)
    ap.add_argument("--block-dt",type=float,default=.05)
    ap.add_argument("--beta",type=float,default=.5)
    ap.add_argument("--tau-max",type=float,default=.005)
    ap.add_argument("--eta",type=float,default=.5)
    ap.add_argument("--prior",type=float,default=32.)
    ap.add_argument("--pc",type=float,default=-1.)
    ap.add_argument("--ncols",type=int,default=2)
    ap.add_argument("--seed",type=int,default=20260930)
    ap.add_argument("--out",default=str(ROOT/"results/finite_tau_table_gfmc_20site.json"))
    args=ap.parse_args()
    rg=np.random.default_rng(args.seed); ys=special_columns(rg,max(2,args.ncols))[:args.ncols]
    out={"M":args.M,"block_dt":args.block_dt,"beta":args.beta,"tau_max":args.tau_max,
         "eta":args.eta,"prior":args.prior,"pc":args.pc,"ys":ys,"columns":{}}
    for k,y in enumerate(ys):
        out["columns"][str(y)]=run_column(int(y),args.M,args.block_dt,args.beta,args.tau_max,
                                          args.eta,args.prior,args.pc,args.seed+100+k)
    Path(args.out).write_text(json.dumps(out,indent=2))
    print("WROTE",args.out,flush=True)
if __name__=="__main__":main()
