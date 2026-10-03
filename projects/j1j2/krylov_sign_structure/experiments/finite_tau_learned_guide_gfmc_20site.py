#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import scipy.sparse.linalg as sla
from finite_tau_matching_20site_exact import build_H,D,special_columns
from finite_tau_amplitude_regression_20site import RegMLP,endpoint,diag,f1,f2,H,HO
from finite_tau_table_gfmc_20site import systematic

ROOT=Path("/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure")

def load_model():
    z=np.load(ROOT/"results/finite_tau_amplitude_regression_20site_weights.npz")
    net=RegMLP(len(z["mu"]),48,1)
    net.W1[:]=z["W1"]; net.b1[:]=z["b1"]; net.W2[:]=z["W2"]; net.b2[:]=z["b2"]
    net.W3[:]=z["W3"]; net.b3[:]=z["b3"]
    return net,z["mu"],z["sd"]

def guide_vec(net,mu,sd,y,tau,arr):
    rich=arr[4]
    X=np.column_stack([rich,np.tile([diag[y],f1[y],f2[y]],(D,1)),np.full(D,tau)])
    z=net.pred((X-mu)/sd); z-=z.max()
    g=np.exp(np.clip(z,-40,0)); g/=np.linalg.norm(g)
    return g

def propagate(walkers,g,s,block=.05,tau_max=.005,seed=None,rng=None):
    if rng is None: rng=np.random.default_rng(seed)
    cache={}; M=len(walkers); t=0
    def local(x):
        x=int(x)
        if x in cache:return cache[x]
        a,b=HO.indptr[x],HO.indptr[x+1]; ys=HO.indices[a:b]; hs=HO.data[a:b]
        rat=g[ys]/max(g[x],1e-30); opp=s[ys]!=s[x]
        rates=hs[opp]*rat[opp]
        dfn=float(diag[x]+np.sum(hs[~opp]*rat[~opp]))
        cache[x]=(dfn,ys[opp],rates); return cache[x]
    while t<block-1e-14:
        dat=[local(x) for x in walkers]
        dv=np.array([z[0] for z in dat]); el=np.array([z[0]-z[2].sum() for z in dat])
        Eref=float(el.mean()); mx=max(0.,float(np.max(dv-Eref)))
        dt=min(tau_max,block-t,0.8/mx if mx>0 else tau_max)
        nxt=np.empty(M,np.int32); bw=np.empty(M)
        for k,(x,(d0,ys,rate)) in enumerate(zip(walkers,dat)):
            stay=1-dt*(d0-Eref); mv=dt*rate; tot=stay+mv.sum()
            if stay<0 or tot<=0: raise RuntimeError(("bad",stay,tot,dt))
            u=rng.random()*tot
            if u<stay:nxt[k]=x
            else:
                j=np.searchsorted(np.cumsum(mv),u-stay,side="right")
                nxt[k]=int(ys[min(j,len(ys)-1)])
            bw[k]=tot
        walkers=nxt[systematic(bw,rng,M)]; t+=dt
    return walkers
def run(y,M=8192,dt=.05,beta=.5,seed=1,switch_tau=.05):
    net,mu,sd=load_model(); arr=endpoint(y)
    d,_,_,_,rich,vals,rlab,rbs=arr; s=np.where(d%2==0,1,-1).astype(np.int8)
    rg=np.random.default_rng(seed); walkers=np.full(M,y,np.int32)
    exact=np.zeros(D); exact[y]=1.; rec=[]
    oldg=np.full(D,1/np.sqrt(D))
    for n in range(1,int(round(beta/dt))+1):
        tp=(n-1)*dt
        use_learned=(tp>=switch_tau-1e-12)
        g=guide_vec(net,mu,sd,y,tp,arr) if use_learned else np.full(D,1/np.sqrt(D))
        if n>1 and np.max(np.abs(g-oldg))>1e-15:
            handoff=g[walkers]/np.maximum(oldg[walkers],1e-300)
            walkers=walkers[systematic(handoff,rg,M)]
        walkers=propagate(walkers,g,s,block=dt,tau_max=.005,rng=rg)
        exact=np.asarray(sla.expm_multiply(-dt*H,exact),float); exact/=np.linalg.norm(exact)
        # walkers sample g*a; undo g and estimate amplitude per rich feature bin.
        lw=-np.log(np.maximum(g[walkers],1e-300)); lw-=lw.max()
        iw=np.exp(lw); iw*=M/iw.sum()
        mass=np.bincount(rlab[walkers],weights=iw,minlength=len(vals))
        aest=(mass/rbs)[rlab]; aest/=np.linalg.norm(aest)
        at=np.abs(exact); at/=np.linalg.norm(at)
        Fest=float(np.dot(aest,at)**2)
        gn=guide_vec(net,mu,sd,y,n*dt,arr); Fmodel=float(np.dot(gn,at)**2)
        row={"tau":n*dt,"walker_amp_fidelity":Fest,"model_amp_fidelity":Fmodel,
             "unique_walkers":int(len(np.unique(walkers))),"occupied_rich_bins":int(np.sum(mass>0))}
        rec.append(row); print("Y",y,row,flush=True)
        oldg=g
    return rec

def main():
    rg=np.random.default_rng(20260930); ys=special_columns(rg,2)
    out={"M":8192,"ys":ys,"columns":{}}
    for k,y in enumerate(ys): out["columns"][str(y)]=run(int(y),8192,seed=51001+k)
    path=ROOT/"results/finite_tau_learned_guide_gfmc_20site.json"; path.write_text(json.dumps(out,indent=2))
    print("WROTE",path,flush=True)
if __name__=="__main__": main()

def run_online(y,M=8192,dt=.05,beta=.5,seed=1,prior_strength=512.):
    net,mu,sd=load_model(); arr=endpoint(y)
    d,_,_,_,rich,vals,rlab,rbs=arr; s=np.where(d%2==0,1,-1).astype(np.int8)
    rg=np.random.default_rng(seed); walkers=np.full(M,y,np.int32)
    exact=np.zeros(D); exact[y]=1.; rec=[]; g=np.full(D,1/np.sqrt(D))
    for n in range(1,int(round(beta/dt))+1):
        walkers=propagate(walkers,g,s,block=dt,tau_max=.005,rng=rg)
        exact=np.asarray(sla.expm_multiply(-dt*H,exact),float); exact/=np.linalg.norm(exact)
        lw=-np.log(np.maximum(g[walkers],1e-300)); lw-=lw.max()
        iw=np.exp(lw); iw*=M/iw.sum()
        mass=np.bincount(rlab[walkers],weights=iw,minlength=len(vals))
        # Learned callable model is a smooth prior over all rich bins.
        gm=guide_vec(net,mu,sd,y,n*dt,arr)
        pmass=np.bincount(rlab,weights=gm,minlength=len(vals))
        pmass/=pmass.sum()
        post=mass+prior_strength*pmass
        gnew=(post/rbs)[rlab]; gnew/=np.linalg.norm(gnew)
        aest=(mass/rbs)[rlab]; aest/=np.linalg.norm(aest)
        at=np.abs(exact); at/=np.linalg.norm(at)
        Fest=float(np.dot(aest,at)**2); Fguide=float(np.dot(gnew,at)**2)
        # Change importance representation before next block.
        hand=gnew[walkers]/np.maximum(g[walkers],1e-300)
        ess=float((hand.sum()**2)/(np.sum(hand*hand)*M))
        walkers=walkers[systematic(hand,rg,M)]
        row={"tau":n*dt,"walker_amp_fidelity":Fest,"next_guide_fidelity":Fguide,
             "handoff_ess_fraction":ess,"unique_walkers":int(len(np.unique(walkers))),
             "occupied_rich_bins":int(np.sum(mass>0))}
        rec.append(row); print("ONLINE",y,row,flush=True)
        g=gnew
    return rec

def _ess(w):
    w=np.asarray(w,float); return float((w.sum()**2)/max(np.sum(w*w),1e-300))

def propagate_weighted(walkers,pw,g,s,block,rng,tau_max=.005,ess_frac=.5):
    cache={}; M=len(walkers); t=0.; nres=0
    def local(x):
        x=int(x)
        if x in cache:return cache[x]
        a,b=HO.indptr[x],HO.indptr[x+1]; ys=HO.indices[a:b]; hs=HO.data[a:b]
        rat=g[ys]/max(g[x],1e-300); opp=s[ys]!=s[x]
        rate=hs[opp]*rat[opp]; d0=float(diag[x]+np.sum(hs[~opp]*rat[~opp]))
        cache[x]=(d0,ys[opp],rate); return cache[x]
    while t<block-1e-14:
        dat=[local(x) for x in walkers]
        dv=np.array([z[0] for z in dat]); el=np.array([z[0]-z[2].sum() for z in dat])
        Eref=float(np.average(el,weights=pw)); mx=max(0.,float(np.max(dv-Eref)))
        dt=min(tau_max,block-t,0.8/mx if mx>0 else tau_max)
        nxt=np.empty(M,np.int32); mult=np.empty(M)
        for k,(x,(d0,ys,rate)) in enumerate(zip(walkers,dat)):
            stay=1-dt*(d0-Eref); mv=dt*rate; tot=stay+mv.sum()
            if stay<0 or tot<=0: raise RuntimeError(("bad",stay,tot,dt))
            u=rng.random()*tot
            if u<stay:nxt[k]=x
            else:
                j=np.searchsorted(np.cumsum(mv),u-stay,side="right"); nxt[k]=int(ys[min(j,len(ys)-1)])
            mult[k]=tot
        walkers=nxt; pw*=mult; pw*=M/pw.sum(); t+=dt
        if _ess(pw)<ess_frac*M:
            walkers=walkers[systematic(pw,rng,M)]; pw=np.ones(M); nres+=1
    return walkers,pw,nres

def run_online_weighted(y,M=8192,dt=.05,beta=.5,seed=1,prior_strength=512.,ess_frac=.5):
    net,mu,sd=load_model(); arr=endpoint(y)
    d,_,_,_,rich,vals,rlab,rbs=arr; s=np.where(d%2==0,1,-1).astype(np.int8)
    rg=np.random.default_rng(seed); walkers=np.full(M,y,np.int32); pw=np.ones(M)
    exact=np.zeros(D); exact[y]=1.; rec=[]; g=np.full(D,1/np.sqrt(D))
    for n in range(1,int(round(beta/dt))+1):
        walkers,pw,nres=propagate_weighted(walkers,pw,g,s,dt,rg,ess_frac=ess_frac)
        exact=np.asarray(sla.expm_multiply(-dt*H,exact),float); exact/=np.linalg.norm(exact)
        aw=pw/np.maximum(g[walkers],1e-300); aw*=M/aw.sum()
        mass=np.bincount(rlab[walkers],weights=aw,minlength=len(vals))
        gm=guide_vec(net,mu,sd,y,n*dt,arr)
        pmass=np.bincount(rlab,weights=gm,minlength=len(vals)); pmass/=pmass.sum()
        post=mass+prior_strength*pmass
        gnew=(post/rbs)[rlab]; gnew/=np.linalg.norm(gnew)
        aest=(mass/rbs)[rlab]; aest/=np.linalg.norm(aest)
        at=np.abs(exact); at/=np.linalg.norm(at)
        Fest=float(np.dot(aest,at)**2); Fguide=float(np.dot(gnew,at)**2)
        # Exact change of importance representation; keep weights unless ESS forces resampling.
        pw*=gnew[walkers]/np.maximum(g[walkers],1e-300); pw*=M/pw.sum()
        handess=_ess(pw)/M; handres=0
        if _ess(pw)<ess_frac*M:
            walkers=walkers[systematic(pw,rg,M)]; pw=np.ones(M); handres=1
        row={"tau":n*dt,"walker_amp_fidelity":Fest,"next_guide_fidelity":Fguide,
             "particle_ess_fraction":float(_ess(pw)/M),"pre_resample_handoff_ess":float(handess),
             "block_resamples":int(nres),"handoff_resample":int(handres),
             "unique_walkers":int(len(np.unique(walkers))),"occupied_rich_bins":int(np.sum(mass>0))}
        rec.append(row); print("WEIGHTED",y,row,flush=True); g=gnew
    return rec

def propagate_fn_uniform_importance(walkers,aguide,s,block,rng,tau_max=.005):
    cache={}; M=len(walkers); t=0.
    def local(x):
        x=int(x)
        if x in cache:return cache[x]
        a,b=HO.indptr[x],HO.indptr[x+1]; ys=HO.indices[a:b]; hs=HO.data[a:b]
        opp=s[ys]!=s[x]
        rat=aguide[ys]/max(aguide[x],1e-300)
        dfn=float(diag[x]+np.sum(hs[~opp]*rat[~opp]))
        rate=hs[opp]  # q(x)=1 importance function: no amplitude ratio on allowed hops.
        cache[x]=(dfn,ys[opp],rate); return cache[x]
    while t<block-1e-14:
        dat=[local(x) for x in walkers]
        dv=np.array([z[0] for z in dat]); el=np.array([z[0]-z[2].sum() for z in dat])
        Eref=float(el.mean()); mx=max(0.,float(np.max(dv-Eref)))
        dt=min(tau_max,block-t,0.8/mx if mx>0 else tau_max)
        nxt=np.empty(M,np.int32); bw=np.empty(M)
        for k,(x,(d0,ys,rate)) in enumerate(zip(walkers,dat)):
            stay=1-dt*(d0-Eref); mv=dt*rate; tot=stay+mv.sum()
            if stay<0 or tot<=0: raise RuntimeError(("bad",stay,tot,dt,d0,Eref))
            u=rng.random()*tot
            if u<stay:nxt[k]=x
            else:
                j=np.searchsorted(np.cumsum(mv),u-stay,side="right"); nxt[k]=int(ys[min(j,len(ys)-1)])
            bw[k]=tot
        walkers=nxt[systematic(bw,rng,M)]; t+=dt
    return walkers

def run_separate_fn(y,M=8192,dt=.05,beta=.5,seed=1,use_model=True):
    net,mu,sd=load_model(); arr=endpoint(y)
    d,_,_,_,rich,vals,rlab,rbs=arr; s=np.where(d%2==0,1,-1).astype(np.int8)
    rg=np.random.default_rng(seed); walkers=np.full(M,y,np.int32)
    exact=np.zeros(D); exact[y]=1.; rec=[]
    for n in range(1,int(round(beta/dt))+1):
        tp=(n-1)*dt
        aguide=guide_vec(net,mu,sd,y,tp,arr) if (use_model and n>1) else np.full(D,1/np.sqrt(D))
        walkers=propagate_fn_uniform_importance(walkers,aguide,s,dt,rg)
        exact=np.asarray(sla.expm_multiply(-dt*H,exact),float); exact/=np.linalg.norm(exact)
        mass=np.bincount(rlab[walkers],minlength=len(vals)).astype(float)
        aest=(mass/rbs)[rlab]; aest/=np.linalg.norm(aest)
        at=np.abs(exact); at/=np.linalg.norm(at)
        Fest=float(np.dot(aest,at)**2)
        row={"tau":n*dt,"walker_amp_fidelity":Fest,"unique_walkers":int(len(np.unique(walkers))),
             "occupied_rich_bins":int(np.sum(mass>0))}
        rec.append(row); print("SEPARATE_FN",y,row,flush=True)
    return rec
