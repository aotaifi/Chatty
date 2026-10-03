#!/usr/bin/env python3
import json,math
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as cs
import scipy.sparse.linalg as sla
from finite_tau_matching_20site_exact import build_H,D,special_columns

ROOT=Path("/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure")
H,diag=build_H(.5); HO=(H-sp.diags(diag)).tocsr()

def labels_for_y(y):
    co=HO.tocoo(); cost=np.where(np.abs(co.data)>0.375,1000.0,1001.0)
    W=sp.coo_matrix((cost,(co.row,co.col)),shape=H.shape).tocsr()
    wd=np.asarray(cs.shortest_path(W,directed=False,indices=[y]))[0]
    d=(wd//1000).astype(np.int16); n2=(wd-1000*d).round().astype(np.int16)
    ec=np.rint(8.0*diag).astype(np.int32)+100
    raw=d.astype(np.int64)*100000+n2.astype(np.int64)*1000+ec.astype(np.int64)
    _,lab=np.unique(raw,return_inverse=True)
    return d,lab.astype(np.int32),np.bincount(lab).astype(float)

def systematic(w,rng,M):
    p=np.asarray(w,float); p/=p.sum(); c=np.cumsum(p); c[-1]=1
    return np.searchsorted(c,rng.random()/M+np.arange(M)/M,side="right")

def ess(w):
    w=np.asarray(w,float); return float(w.sum()**2/np.dot(w,w))

def make_local(theta,lab,sgn):
    cache={}
    def local(x):
        x=int(x)
        if x in cache:return cache[x]
        gx=theta[lab[x]]; a,b=HO.indptr[x],HO.indptr[x+1]
        ys=HO.indices[a:b]; hs=HO.data[a:b]
        rat=np.exp(np.clip(theta[lab[ys]]-gx,-30,30))
        opp=sgn[ys]!=sgn[x]
        rates=hs[opp]*rat[opp]
        dfn=float(diag[x]+np.sum(hs[~opp]*rat[~opp]))
        cache[x]=(dfn,ys[opp],rates)
        return cache[x]
    return local
def propagate_weighted(states,w,theta,lab,sgn,block_dt,tau_max,rng,ess_frac=.35):
    M=len(states); t=0.; nr=0; local=make_local(theta,lab,sgn)
    while t<block_dt-1e-14:
        dat=[local(int(x)) for x in states]
        dfn=np.array([z[0] for z in dat])
        el=np.array([z[0]-np.sum(z[2]) for z in dat])
        Eref=float(np.average(el,weights=w))
        mx=max(0.,float(np.max(dfn-Eref)))
        dt=min(tau_max,block_dt-t,0.8/mx if mx>0 else tau_max)
        nxt=np.empty(M,dtype=np.int32); mult=np.empty(M)
        for k,(x,(dv,ys,rates)) in enumerate(zip(states,dat)):
            stay=1-dt*(dv-Eref); move=dt*rates; tot=stay+move.sum()
            if stay<0 or tot<=0: raise RuntimeError(("bad",stay,tot,dt,dv,Eref))
            u=rng.random()*tot
            if u<stay: nxt[k]=x
            else:
                j=np.searchsorted(np.cumsum(move),u-stay,side="right")
                nxt[k]=int(ys[min(j,len(ys)-1)])
            mult[k]=tot
        states=nxt; w=w*mult
        w*=M/w.sum()
        if ess(w)<ess_frac*M:
            states=states[systematic(w,rng,M)]; w=np.ones(M); nr+=1
        t+=dt
    return states,w,nr

def update_theta_weighted(theta,lab,bin_size,states,w,eta=.5,prior=32.):
    B=len(theta); cnt=np.bincount(lab[states],weights=w,minlength=B).astype(float)
    cnt*=len(states)/cnt.sum()
    z=2*(theta-np.max(theta)); q=bin_size*np.exp(np.clip(z,-60,0)); q/=q.sum()
    m=(cnt+prior*q)/(cnt.sum()+prior)
    delta=np.log(np.maximum(m,1e-300)/np.maximum(q,1e-300)); delta-=np.sum(q*delta)
    new=np.clip(theta+eta*delta,-30,30)
    return new,cnt

def guide_fidelity(theta,lab,exact):
    g=np.exp(np.clip(theta[lab]-np.max(theta),-60,0)); g/=np.linalg.norm(g)
    a=np.abs(exact); a/=np.linalg.norm(a)
    return float(np.dot(g,a)**2)

def amplitude_measure(states,w,theta,lab):
    aw=w*np.exp(np.clip(-theta[lab[states]],-40,40))
    c=np.bincount(states,weights=aw,minlength=D)
    return c
def run_column(y,M=8192,block_dt=.05,beta=.5,tau_max=.005,eta=.5,ess_frac=.35,seed=1):
    rng=np.random.default_rng(seed); d,lab,bs=labels_for_y(y)
    sgn=np.where(d%2==0,1,-1).astype(np.int8)
    theta=np.zeros(len(bs)); states=np.full(M,int(y),dtype=np.int32); w=np.ones(M)
    exact=np.zeros(D); exact[y]=1.; rec=[]
    for n in range(1,int(round(beta/block_dt))+1):
        states,w,nr=propagate_weighted(states,w,theta,lab,sgn,block_dt,tau_max,rng,ess_frac)
        exact=np.asarray(sla.expm_multiply(-block_dt*H,exact),float); exact/=np.linalg.norm(exact)
        c=amplitude_measure(states,w,theta,lab); pc=c/c.sum(); pe=np.abs(exact); pe/=pe.sum()
        bc=float(np.sum(np.sqrt(pc*pe))); tv=float(.5*np.sum(np.abs(pc-pe)))
        old=theta.copy(); new,cnt=update_theta_weighted(theta,lab,bs,states,w,eta)
        # Change importance function: f_new/f_old = g_new/g_old.
        w*=np.exp(np.clip(new[lab[states]]-old[lab[states]],-30,30)); w*=M/w.sum()
        theta=new
        if ess(w)<ess_frac*M:
            states=states[systematic(w,rng,M)]; w=np.ones(M); nr+=1
        row={"tau":n*block_dt,"guide_fidelity":guide_fidelity(theta,lab,exact),
             "amplitude_BC":bc,"amplitude_TV":tv,"ESS":ess(w),
             "unique_states":int(len(np.unique(states))),"resamples":int(nr)}
        rec.append(row); print("SMC",y,row,flush=True)
    return rec

def main():
    rg=np.random.default_rng(20260930); ys=special_columns(rg,2)
    out={"M":8192,"ys":ys,"columns":{}}
    for k,y in enumerate(ys): out["columns"][str(y)]=run_column(int(y),seed=51001+k)
    path=ROOT/"results/finite_tau_weighted_smc_20site.json"; path.write_text(json.dumps(out,indent=2))
    print("WROTE",path,flush=True)
if __name__=="__main__":main()
