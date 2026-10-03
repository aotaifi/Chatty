#!/usr/bin/env python3
import json,math
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as cs
import scipy.sparse.linalg as sla
from finite_tau_matching_20site_exact import build_H,basis,D,N,NN,NNN,special_columns
from finite_tau_table_gfmc_20site import propagate,update_theta,systematic

ROOT=Path("/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure")
H,diag=build_H(.5); HO=(H-sp.diags(diag)).tocsr()
f1=np.zeros(D,dtype=np.float64); f2=np.zeros(D,dtype=np.float64)
for u,v in NN: f1 += (((basis>>u)^(basis>>v))&1)
for u,v in NNN: f2 += (((basis>>u)^(basis>>v))&1)

def endpoint_arrays(y):
    co=HO.tocoo(); cost=np.where(np.abs(co.data)>0.375,1000.0,1001.0)
    W=sp.coo_matrix((cost,(co.row,co.col)),shape=H.shape).tocsr()
    wd=np.asarray(cs.shortest_path(W,directed=False,indices=[y]))[0]
    d=(wd//1000).astype(np.float64)
    n2=(wd-1000*d).round().astype(np.float64)
    ec=np.rint(8.0*diag).astype(np.int32)+100
    raw=d.astype(np.int64)*100000+n2.astype(np.int64)*1000+ec.astype(np.int64)
    vals,lab=np.unique(raw,return_inverse=True)
    return d,n2,lab.astype(np.int32),np.bincount(lab,minlength=len(vals)).astype(float)

def feat(arrs,idx,tau):
    d,n2,_,_=arrs
    ii=np.asarray(idx,dtype=np.int64)
    return np.column_stack([d[ii],n2[ii],diag[ii],f1[ii],f2[ii],np.full(len(ii),tau)])

def collect_replica(ys,M,seed,block_dt=.05,beta=.5,tau_max=.005,eta=.5,prior=32.):
    rng=np.random.default_rng(seed); rows=[]; oracle={}
    for ky,y in enumerate(ys):
        arrs=endpoint_arrays(int(y)); d,n2,lab,bs=arrs
        sgn=np.where((d.astype(np.int64)%2)==0,1,-1).astype(np.int8)
        theta=np.zeros(len(bs)); walkers=np.full(M,int(y),dtype=np.int32)
        exact=np.zeros(D); exact[int(y)]=1.
        for n in range(1,int(round(beta/block_dt))+1):
            tau=n*block_dt; theta_old=theta.copy()
            walkers,_=propagate(walkers,theta_old,lab,sgn,block_dt,tau_max,rng)
            # importance undo: walkers ~ g_old*a_new, so weights ~1/g_old.
            lw=-theta_old[lab[walkers]]
            lw-=np.max(lw); wp=np.exp(lw); wp*=M/wp.sum()
            neg=rng.integers(0,D,size=M,dtype=np.int64)
            rows.append((feat(arrs,walkers,tau),np.ones(M),wp,int(y),tau))
            rows.append((feat(arrs,neg,tau),np.zeros(M),np.ones(M),int(y),tau))
            theta,rw,_,_,_=update_theta(theta_old,lab,bs,walkers,eta,prior,-1.)
            walkers=walkers[systematic(rw,rng,M)]
            exact=np.asarray(sla.expm_multiply(-block_dt*H,exact),float); exact/=np.linalg.norm(exact)
            oracle[(int(y),round(tau,8))]=(arrs,exact.copy())
    X=np.concatenate([r[0] for r in rows]); Y=np.concatenate([r[1] for r in rows])
    W=np.concatenate([r[2] for r in rows])
    return X,Y,W,oracle
class MLP:
    def __init__(self,nin,h=32,seed=1):
        rg=np.random.default_rng(seed)
        self.W1=rg.normal(0,.25,(nin,h)); self.b1=np.zeros(h)
        self.W2=rg.normal(0,.25,(h,h)); self.b2=np.zeros(h)
        self.W3=rg.normal(0,.25,(h,1)); self.b3=np.zeros(1)
        self.m=[np.zeros_like(x) for x in self.params()]
        self.v=[np.zeros_like(x) for x in self.params()]
        self.t=0
    def params(self): return [self.W1,self.b1,self.W2,self.b2,self.W3,self.b3]
    def forward(self,X):
        z1=X@self.W1+self.b1; h1=np.tanh(z1)
        z2=h1@self.W2+self.b2; h2=np.tanh(z2)
        z3=(h2@self.W3+self.b3).ravel()
        return z3,(X,h1,h2)
    def step(self,X,y,w,lr=2e-3):
        z,(X,h1,h2)=self.forward(X)
        p=1/(1+np.exp(-np.clip(z,-30,30)))
        ww=w/w.mean(); dz=(p-y)*ww/len(y)
        gW3=h2.T@dz[:,None]; gb3=np.array([dz.sum()])
        dh2=dz[:,None]@self.W3.T; dz2=dh2*(1-h2*h2)
        gW2=h1.T@dz2; gb2=dz2.sum(0)
        dh1=dz2@self.W2.T; dz1=dh1*(1-h1*h1)
        gW1=X.T@dz1; gb1=dz1.sum(0)
        grads=[gW1,gb1,gW2,gb2,gW3,gb3]
        self.t+=1
        for p0,g,m,v in zip(self.params(),grads,self.m,self.v):
            m*=.9; m+=.1*g; v*=.999; v+=.001*g*g
            mh=m/(1-.9**self.t); vh=v/(1-.999**self.t)
            p0-=lr*mh/(np.sqrt(vh)+1e-8)
        eps=1e-12
        loss=-np.sum(ww*(y*np.log(p+eps)+(1-y)*np.log(1-p+eps)))/np.sum(ww)
        return float(loss)
    def pred(self,X): return self.forward(X)[0]

def weighted_auc(score,y,w):
    o=np.argsort(score); y=y[o]; w=w[o]
    wp=w*y; wn=w*(1-y)
    cneg=np.cumsum(wn)-wn
    ties=0.5*wn
    num=np.sum(wp*(cneg+ties)); den=wp.sum()*wn.sum()
    return float(num/den) if den>0 else np.nan

def train(X,y,w,seed=3,epochs=25,batch=4096):
    mu=X.mean(0); sd=X.std(0); sd[sd<1e-8]=1
    Z=(X-mu)/sd; net=MLP(Z.shape[1],32,seed)
    rg=np.random.default_rng(seed)
    for ep in range(epochs):
        o=rg.permutation(len(Z)); losses=[]
        for i in range(0,len(Z),batch):
            j=o[i:i+batch]; losses.append(net.step(Z[j],y[j],w[j]))
        if ep in (0,4,9,14,19,24):
            print("TRAIN",ep+1,"loss",np.mean(losses),"auc",weighted_auc(net.pred(Z),y,w),flush=True)
    return net,mu,sd
def calib_loss(score,y,w,eta):
    z=np.clip(eta*score,-40,40)
    return float(np.sum(w*(np.logaddexp(0,z)-y*z))/np.sum(w))

def oracle_scores(net,mu,sd,ys,oracle,eta=1.0):
    out=[]
    for y in ys:
        for tau in np.arange(.05,.5001,.05):
            arrs,exact=oracle[(int(y),round(float(tau),8))]
            F=feat(arrs,np.arange(D),float(tau))
            loga=eta*net.pred((F-mu)/sd); loga-=loga.max()
            a=np.exp(np.clip(loga,-60,0)); a/=np.linalg.norm(a)
            at=np.abs(exact); at/=np.linalg.norm(at)
            out.append({"y":int(y),"tau":float(tau),"fidelity":float(np.dot(a,at)**2)})
    return out

def main():
    rg=np.random.default_rng(20260930); ys=special_columns(rg,2)
    print("YS",ys,flush=True)
    Xa,ya,wa,oa=collect_replica(ys,4096,31001)
    Xb,yb,wb,ob=collect_replica(ys,4096,32001)
    net,mu,sd=train(Xa,ya,wa)
    sa=net.pred((Xa-mu)/sd); sb=net.pred((Xb-mu)/sd)
    aucA=weighted_auc(sa,ya,wa); aucB=weighted_auc(sb,yb,wb)
    grid=np.linspace(.05,1.25,121)
    losses=np.array([calib_loss(sb,yb,wb,e) for e in grid])
    eta=float(grid[np.argmin(losses)])
    print("CROSS_REPLICA","train_auc",aucA,"val_auc",aucB,"eta",eta,
          "val_loss_raw",calib_loss(sb,yb,wb,1.0),"val_loss_cal",losses.min(),flush=True)
    cals={}
    for tau in np.arange(.05,.5001,.05):
        m=np.isclose(Xb[:,5],tau)
        cals[round(float(tau),8)]=fit_ratio_calibrator(sb[m],yb[m],wb[m],nb=48)
    sc=oracle_scores_cal(net,mu,sd,ys,ob,cals)
    for r in sc:
        if abs(r["tau"]-.5)<1e-8 or abs(r["tau"]-.25)<1e-8:
            print("ORACLE",r,flush=True)
    out={"ys":[int(x) for x in ys],"train_auc":aucA,"val_auc":aucB,"eta":eta,"oracle":sc}
    path=ROOT/"results/finite_tau_learned_amplitude_20site.json"
    path.write_text(json.dumps(out,indent=2)); print("WROTE",path,flush=True)
    np.savez_compressed(ROOT/"results/finite_tau_learned_amplitude_20site_weights.npz",
        mu=mu,sd=sd,W1=net.W1,b1=net.b1,W2=net.W2,b2=net.b2,W3=net.W3,b3=net.b3)

def pava_monotone(v,w):
    vals=[]; wg=[]; n=[]
    for a,b in zip(v,w):
        vals.append(float(a)); wg.append(float(b)); n.append(1)
        while len(vals)>=2 and vals[-2]>vals[-1]:
            ww=wg[-2]+wg[-1]
            vv=(vals[-2]*wg[-2]+vals[-1]*wg[-1])/max(ww,1e-30)
            nn=n[-2]+n[-1]
            vals[-2:]=[vv]; wg[-2:]=[ww]; n[-2:]=[nn]
    out=[]
    for a,nn in zip(vals,n): out.extend([a]*nn)
    return np.asarray(out)

def fit_ratio_calibrator(score,y,w,nb=64):
    q=np.unique(np.quantile(score,np.linspace(0,1,nb+1)))
    if len(q)<4: raise RuntimeError("too few score bins")
    b=np.clip(np.searchsorted(q,score,side="right")-1,0,len(q)-2)
    K=len(q)-1
    pos=np.bincount(b,weights=w*y,minlength=K)
    neg=np.bincount(b,weights=w*(1-y),minlength=K)
    cen=np.zeros(K); bw=pos+neg
    for k in range(K):
        m=b==k
        cen[k]=np.average(score[m],weights=w[m]) if np.any(m) else .5*(q[k]+q[k+1])
    # normalized density ratio; Jeffreys pseudocount stabilizes tails.
    ratio=((pos+.5)/(pos.sum()+.5*K))/((neg+.5)/(neg.sum()+.5*K))
    lr=np.log(ratio)
    lr=pava_monotone(lr,bw+1.)
    return cen,lr

def apply_calibrator(score,cal):
    x,z=cal
    return np.interp(score,x,z,left=z[0],right=z[-1])

def oracle_scores_cal(net,mu,sd,ys,oracle,cals):
    out=[]
    for y in ys:
        for tau in np.arange(.05,.5001,.05):
            arrs,exact=oracle[(int(y),round(float(tau),8))]
            F=feat(arrs,np.arange(D),float(tau))
            score=net.pred((F-mu)/sd)
            loga=apply_calibrator(score,cals[round(float(tau),8)])
            loga-=loga.max()
            a=np.exp(np.clip(loga,-60,0)); a/=np.linalg.norm(a)
            at=np.abs(exact); at/=np.linalg.norm(at)
            out.append({"y":int(y),"tau":float(tau),"fidelity":float(np.dot(a,at)**2)})
    return out

if __name__=="__main__": main()
