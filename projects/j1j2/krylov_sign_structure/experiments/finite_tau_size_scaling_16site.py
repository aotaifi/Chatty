#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as cs
import scipy.sparse.linalg as sla
from closed_fn_krylov_exact4x4 import build_H,basis,D,NN,NNN
from finite_tau_amplitude_regression_20site import RegMLP,fit,grouped_rmse

ROOT=Path("/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure")
H,diag=build_H(.5); HO=(H-sp.diags(diag)).tocsr()
f1=np.zeros(D); f2=np.zeros(D)
for u,v in NN: f1 += (((basis>>u)^(basis>>v))&1)
for u,v in NNN: f2 += (((basis>>u)^(basis>>v))&1)

def systematic(w,rng,M):
    w=np.asarray(w,float); w/=w.sum(); c=np.cumsum(w); c[-1]=1
    return np.searchsorted(c,rng.random()/M+np.arange(M)/M,side="right")

def endpoint(y):
    co=HO.tocoo(); cost=np.where(np.abs(co.data)>.375,1000.,1001.)
    W=sp.coo_matrix((cost,(co.row,co.col)),shape=H.shape).tocsr()
    wd=np.asarray(cs.shortest_path(W,directed=False,indices=[y]))[0]
    d=(wd//1000).astype(int); n2=np.rint(wd-1000*d).astype(int)
    ec=np.rint(8*diag).astype(int)+100
    raw=d.astype(np.int64)*100000+n2.astype(np.int64)*1000+ec
    _,lab=np.unique(raw,return_inverse=True); bs=np.bincount(lab).astype(float)
    rich=np.column_stack([d,n2,diag,f1,f2])
    vals,rlab=np.unique(rich,axis=0,return_inverse=True); rbs=np.bincount(rlab).astype(float)
    return d,n2,lab.astype(np.int32),bs,rich,vals,rlab.astype(np.int32),rbs

def xrows(vals,y,tau):
    return np.column_stack([vals,np.tile([diag[y],f1[y],f2[y]],(len(vals),1)),np.full(len(vals),tau)])
def propagate_table(walkers,theta,lab,s,block,rng,tau_max=.005):
    M=len(walkers); t=0.; cache={}
    def local(x):
        x=int(x)
        if x in cache:return cache[x]
        a,b=HO.indptr[x],HO.indptr[x+1]; ys=HO.indices[a:b]; hs=HO.data[a:b]
        rat=np.exp(np.clip(theta[lab[ys]]-theta[lab[x]],-30,30)); opp=s[ys]!=s[x]
        rate=hs[opp]*rat[opp]; d0=float(diag[x]+np.sum(hs[~opp]*rat[~opp]))
        cache[x]=(d0,ys[opp],rate); return cache[x]
    while t<block-1e-14:
        dat=[local(x) for x in walkers]; dv=np.array([z[0] for z in dat]); el=np.array([z[0]-z[2].sum() for z in dat])
        Eref=float(el.mean()); mx=max(0.,float(np.max(dv-Eref))); dt=min(tau_max,block-t,0.8/mx if mx>0 else tau_max)
        nxt=np.empty(M,np.int32); bw=np.empty(M)
        for k,(x,(d0,ys,rate)) in enumerate(zip(walkers,dat)):
            stay=1-dt*(d0-Eref); mv=dt*rate; tot=stay+mv.sum(); u=rng.random()*tot
            if stay<0 or tot<=0: raise RuntimeError("bad")
            if u<stay:nxt[k]=x
            else:
                j=np.searchsorted(np.cumsum(mv),u-stay,side="right"); nxt[k]=int(ys[min(j,len(ys)-1)])
            bw[k]=tot
        walkers=nxt[systematic(bw,rng,M)]; t+=dt
    return walkers

def update(theta,lab,bs,walkers,eta=.5,prior=32.):
    M=len(walkers); B=len(theta); cnt=np.bincount(lab[walkers],minlength=B).astype(float)
    q=bs*np.exp(np.clip(2*(theta-theta.max()),-60,0)); q/=q.sum()
    m=(cnt+prior*q)/(M+prior); delta=np.log(np.maximum(m,1e-300)/np.maximum(q,1e-300)); delta-=np.sum(q*delta)
    new=np.clip(theta+eta*delta,-30,30)
    rw=np.exp(np.clip(new[lab[walkers]]-theta[lab[walkers]],-30,30))
    return new,rw

def collect(ys,M,seed,block=.05,beta=.5):
    rg=np.random.default_rng(seed); Xs=[];Ts=[];Ws=[];Gs=[];oracle={}
    for y in ys:
        arr=endpoint(y); d,n2,lab,bs,rich,vals,rlab,rbs=arr; s=np.where(d%2==0,1,-1).astype(np.int8)
        theta=np.zeros(len(bs)); walkers=np.full(M,y,np.int32); exact=np.zeros(D); exact[y]=1
        for n in range(1,int(round(beta/block))+1):
            tau=n*block; old=theta.copy(); walkers=propagate_table(walkers,old,lab,s,block,rg)
            lw=-old[lab[walkers]]; lw-=lw.max(); iw=np.exp(lw); iw*=M/iw.sum()
            cnt=np.bincount(rlab[walkers],minlength=len(vals)).astype(float)
            mass=np.bincount(rlab[walkers],weights=iw,minlength=len(vals)).astype(float)
            keep=cnt>=2; tar=np.log(np.maximum(mass[keep],1e-30)/rbs[keep]); ww=cnt[keep]; tar-=np.average(tar,weights=ww)
            Xs.append(xrows(vals[keep],y,tau)); Ts.append(tar); Ws.append(ww); Gs.extend([(y,round(tau,8))]*int(np.sum(keep)))
            theta,rw=update(old,lab,bs,walkers); walkers=walkers[systematic(rw,rg,M)]
            exact=np.asarray(sla.expm_multiply(-block*H,exact)); exact/=np.linalg.norm(exact); oracle[(y,round(tau,8))]=(arr,exact.copy())
    return np.concatenate(Xs),np.concatenate(Ts),np.concatenate(Ws),np.asarray(Gs,dtype=object),oracle
def guide(net,mu,sd,y,tau,arr):
    rich=arr[4]; X=np.column_stack([rich,np.tile([diag[y],f1[y],f2[y]],(D,1)),np.full(D,tau)])
    z=net.pred((X-mu)/sd); z-=z.max(); g=np.exp(np.clip(z,-40,0)); g/=np.linalg.norm(g); return g

def propagate_vec(walkers,g,s,block,rng,tau_max=.005):
    M=len(walkers); t=0.; cache={}
    def local(x):
        x=int(x)
        if x in cache:return cache[x]
        a,b=HO.indptr[x],HO.indptr[x+1]; ys=HO.indices[a:b]; hs=HO.data[a:b]
        rat=g[ys]/max(g[x],1e-300); opp=s[ys]!=s[x]; rate=hs[opp]*rat[opp]; d0=float(diag[x]+np.sum(hs[~opp]*rat[~opp]))
        cache[x]=(d0,ys[opp],rate); return cache[x]
    while t<block-1e-14:
        dat=[local(x) for x in walkers]; dv=np.array([z[0] for z in dat]); el=np.array([z[0]-z[2].sum() for z in dat])
        Eref=float(el.mean()); mx=max(0.,float(np.max(dv-Eref))); dt=min(tau_max,block-t,0.8/mx if mx>0 else tau_max)
        nxt=np.empty(M,np.int32); bw=np.empty(M)
        for k,(x,(d0,ys,rate)) in enumerate(zip(walkers,dat)):
            stay=1-dt*(d0-Eref); mv=dt*rate; tot=stay+mv.sum(); u=rng.random()*tot
            if stay<0 or tot<=0: raise RuntimeError("bad")
            if u<stay:nxt[k]=x
            else:
                j=np.searchsorted(np.cumsum(mv),u-stay,side="right"); nxt[k]=int(ys[min(j,len(ys)-1)])
            bw[k]=tot
        walkers=nxt[systematic(bw,rng,M)]; t+=dt
    return walkers

def run(net,mu,sd,y,M,seed,switch=.15,dt=.05,beta=.5):
    arr=endpoint(y); d,_,_,_,rich,vals,rlab,rbs=arr; s=np.where(d%2==0,1,-1).astype(np.int8)
    rg=np.random.default_rng(seed); walkers=np.full(M,y,np.int32); exact=np.zeros(D); exact[y]=1.; oldg=np.full(D,1/np.sqrt(D)); rec=[]
    for n in range(1,int(round(beta/dt))+1):
        tp=(n-1)*dt; g=guide(net,mu,sd,y,tp,arr) if tp>=switch-1e-12 else np.full(D,1/np.sqrt(D))
        if n>1 and np.max(np.abs(g-oldg))>1e-15: walkers=walkers[systematic(g[walkers]/np.maximum(oldg[walkers],1e-300),rg,M)]
        walkers=propagate_vec(walkers,g,s,dt,rg); exact=np.asarray(sla.expm_multiply(-dt*H,exact)); exact/=np.linalg.norm(exact)
        lw=-np.log(np.maximum(g[walkers],1e-300)); lw-=lw.max(); iw=np.exp(lw); iw*=M/iw.sum()
        mass=np.bincount(rlab[walkers],weights=iw,minlength=len(vals)); aest=(mass/rbs)[rlab]; aest/=np.linalg.norm(aest)
        at=np.abs(exact); at/=np.linalg.norm(at); F=float(np.dot(aest,at)**2)
        rec.append((n*dt,F,len(np.unique(walkers)))); oldg=g
    return rec
def main():
    rg=np.random.default_rng(160930); ys=[int(x) for x in rg.choice(np.arange(D),size=2,replace=False)]
    Xa,ta,wa,ga,oa=collect(ys,4096,81001); Xb,tb,wb,gb,ob=collect(ys,4096,82001)
    net,mu,sd=fit(Xa,ta,wa,epochs=300,seed=17)
    tr=grouped_rmse(net.pred((Xa-mu)/sd),ta,wa,ga); va=grouped_rmse(net.pred((Xb-mu)/sd),tb,wb,gb)
    print("YS",ys,"RMSE",tr,va,flush=True)
    # choose harder column by M=2048 final fidelity
    pilot=[]
    for k,y in enumerate(ys):
        r=run(net,mu,sd,y,2048,83001+k); pilot.append((r[-1][1],y)); print("PILOT",y,r[-1],flush=True)
    hard=min(pilot)[1]; rows=[]
    for M in (2048,8192,32768):
        r=run(net,mu,sd,hard,M,84000+M); row={"M":M,"F":r[-1][1],"unique":r[-1][2]}; rows.append(row); print("SCALE",row,flush=True)
    out={"N":16,"D":D,"ys":ys,"hard_y":hard,"train_rmse":tr,"val_rmse":va,"rows":rows}
    path=ROOT/"results/finite_tau_size_scaling_16site.json"; path.write_text(json.dumps(out,indent=2)); print("WROTE",path,flush=True)
if __name__=="__main__": main()
