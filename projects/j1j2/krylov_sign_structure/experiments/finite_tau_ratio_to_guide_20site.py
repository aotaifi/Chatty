#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from finite_tau_matching_20site_exact import build_H,basis,D,NN,NNN,special_columns
from finite_tau_table_gfmc_20site import labels_for_y,propagate,update_theta,systematic
from finite_tau_learned_amplitude_20site import MLP,weighted_auc

ROOT=Path("/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure")
H,diag=build_H(.5)
f1=np.zeros(D,float); f2=np.zeros(D,float)
for u,v in NN: f1+=(((basis>>u)^(basis>>v))&1)
for u,v in NNN: f2+=(((basis>>u)^(basis>>v))&1)

def feats(d,n2,idx,tau):
    ii=np.asarray(idx,np.int64)
    return np.column_stack([d[ii],n2[ii],diag[ii],f1[ii],f2[ii],np.full(len(ii),tau)])

def d_n2_from_labels(y):
    # recompute lexicographic endpoint distances for features
    import scipy.sparse.csgraph as cs
    HO=(H-sp.diags(diag)).tocoo()
    cost=np.where(np.abs(HO.data)>0.375,1000.0,1001.0)
    W=sp.coo_matrix((cost,(HO.row,HO.col)),shape=H.shape).tocsr()
    wd=np.asarray(cs.shortest_path(W,directed=False,indices=[y]))[0]
    d=(wd//1000).astype(float); n2=(wd-1000*d).round().astype(float)
    return d,n2

def collect(seed,M=4096,beta=.5,block=.05,tau_max=.005,eta=.5,prior=32.):
    rg=np.random.default_rng(seed); ys=special_columns(np.random.default_rng(20260930),2)
    rows=[]; oracle={}
    for y in ys:
        d,lab,bs=labels_for_y(int(y)); d=d.astype(float)
        _,n2=d_n2_from_labels(int(y))
        sgn=np.where((d.astype(np.int64)%2)==0,1,-1).astype(np.int8)
        theta=np.zeros(len(bs)); walkers=np.full(M,int(y),np.int32)
        exact=np.zeros(D); exact[int(y)]=1.
        for n in range(1,int(round(beta/block))+1):
            tau=n*block
            old=theta.copy()
            walkers,_=propagate(walkers,old,lab,sgn,block,tau_max,rg)
            # Reference q proportional to g_old^2; exact enumeration is ONLY a sampler convenience on 20 sites.
            lq=2*old[lab]; lq-=lq.max(); q=np.exp(np.clip(lq,-60,0)); q/=q.sum()
            ref=rg.choice(D,size=M,replace=True,p=q)
            Xp=feats(d,n2,walkers,tau); Xq=feats(d,n2,ref,tau)
            basep=old[lab[walkers]]; baseq=old[lab[ref]]
            rows.append((Xp,np.ones(M),basep))
            rows.append((Xq,np.zeros(M),baseq))
            theta,rw,_,_,_=update_theta(old,lab,bs,walkers,eta,prior,-1.)
            walkers=walkers[systematic(rw,rg,M)]
            exact=np.asarray(sla.expm_multiply(-block*H,exact),float); exact/=np.linalg.norm(exact)
            oracle[(int(y),round(tau,8))]=(d,n2,lab,old.copy(),exact.copy())
    X=np.concatenate([r[0] for r in rows]); Y=np.concatenate([r[1] for r in rows]); B=np.concatenate([r[2] for r in rows])
    return ys,X,Y,B,oracle
def train(X,y,seed=7,epochs=25,batch=4096):
    mu=X.mean(0); sd=X.std(0); sd[sd<1e-8]=1
    Z=(X-mu)/sd; net=MLP(Z.shape[1],32,seed); rg=np.random.default_rng(seed); w=np.ones(len(y))
    for ep in range(epochs):
        o=rg.permutation(len(Z)); ls=[]
        for i in range(0,len(Z),batch):
            j=o[i:i+batch]; ls.append(net.step(Z[j],y[j],w[j]))
        if ep in (0,4,9,14,19,24):
            print("TRAIN",ep+1,np.mean(ls),weighted_auc(net.pred(Z),y,w),flush=True)
    return net,mu,sd

def evaluate(net,mu,sd,ys,oracle,eta_res=1.0):
    out=[]
    for y in ys:
        for tau in np.arange(.05,.5001,.05):
            d,n2,lab,old,exact=oracle[(int(y),round(float(tau),8))]
            F=feats(d,n2,np.arange(D),float(tau))
            eta_here=eta_res.get(round(float(tau),8),1.0) if isinstance(eta_res,dict) else eta_res
            resid=eta_here*net.pred((F-mu)/sd)
            la=old[lab]+resid; la-=la.max()
            a=np.exp(np.clip(la,-60,0)); a/=np.linalg.norm(a)
            at=np.abs(exact); at/=np.linalg.norm(at)
            out.append({"y":int(y),"tau":float(tau),"fidelity":float(np.dot(a,at)**2)})
    return out

def ce(score,y,eta):
    z=np.clip(eta*score,-40,40)
    return float(np.mean(np.logaddexp(0,z)-y*z))

def main():
    ys,Xa,ya,Ba,oa=collect(51001)
    _,Xb,yb,Bb,ob=collect(52001)
    net,mu,sd=train(Xa,ya)
    sa=net.pred((Xa-mu)/sd); sb=net.pred((Xb-mu)/sd)
    aucA=weighted_auc(sa,ya,np.ones(len(ya))); aucB=weighted_auc(sb,yb,np.ones(len(yb)))
    grid=np.linspace(.05,2,196); losses=np.array([ce(sb,yb,e) for e in grid]); eta=float(grid[np.argmin(losses)])
    eta_tau={}
    for tau in np.arange(.05,.5001,.05):
        m=np.isclose(Xb[:,5],tau)
        lt=np.array([ce(sb[m],yb[m],e) for e in grid])
        eta_tau[round(float(tau),8)]=float(grid[np.argmin(lt)])
    print("TRANSFER",aucA,aucB,"eta",eta,"eta_tau",eta_tau,"rawCE",ce(sb,yb,1),"calCE",losses.min(),flush=True)
    sc=evaluate(net,mu,sd,ys,ob,eta_tau)
    for r in sc:
        if abs(r["tau"]-.25)<1e-8 or abs(r["tau"]-.5)<1e-8: print("ORACLE",r,flush=True)
    out={"ys":[int(x) for x in ys],"train_auc":aucA,"val_auc":aucB,"eta":eta,"eta_tau":eta_tau,"oracle":sc}
    path=ROOT/"results/finite_tau_ratio_to_guide_20site.json"; path.write_text(json.dumps(out,indent=2)); print("WROTE",path)
if __name__=="__main__":main()
