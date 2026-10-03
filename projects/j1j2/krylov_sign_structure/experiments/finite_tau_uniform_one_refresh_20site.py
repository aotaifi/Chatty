#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import scipy.sparse.csgraph as cs
from finite_tau_matching_20site_exact import build_H,basis,D,NN,NNN,special_columns
from finite_tau_table_gfmc_20site import labels_for_y,propagate
from finite_tau_learned_amplitude_20site import MLP,weighted_auc

ROOT=Path("/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure")
H,diag=build_H(.5); HO=(H-sp.diags(diag)).tocsr()
f1=np.zeros(D,float); f2=np.zeros(D,float)
for u,v in NN: f1+=(((basis>>u)^(basis>>v))&1)
for u,v in NNN: f2+=(((basis>>u)^(basis>>v))&1)

def arrays(y):
    co=HO.tocoo(); cost=np.where(np.abs(co.data)>0.375,1000.,1001.)
    W=sp.coo_matrix((cost,(co.row,co.col)),shape=H.shape).tocsr()
    wd=np.asarray(cs.shortest_path(W,directed=False,indices=[int(y)]))[0]
    d=(wd//1000).astype(float); n2=(wd-1000*d).round().astype(float)
    _,lab,bs=labels_for_y(int(y))
    return d,n2,lab,bs

def feat(d,n2,idx):
    ii=np.asarray(idx,np.int64)
    return np.column_stack([d[ii],n2[ii],diag[ii],f1[ii],f2[ii]])

def walkers_uniform(y,M,seed):
    rg=np.random.default_rng(seed); d,n2,lab,bs=arrays(y)
    s=np.where((d.astype(np.int64)%2)==0,1,-1).astype(np.int8)
    th=np.zeros(len(bs)); w=np.full(M,int(y),np.int32)
    w,_=propagate(w,th,lab,s,.5,.005,rg)
    ref=rg.integers(0,D,size=M,dtype=np.int64)
    X=np.concatenate([feat(d,n2,w),feat(d,n2,ref)])
    Y=np.concatenate([np.ones(M),np.zeros(M)])
    return X,Y,(d,n2,s)

def train(X,y,seed=1,epochs=30):
    mu=X.mean(0); sd=X.std(0); sd[sd<1e-8]=1
    Z=(X-mu)/sd; net=MLP(Z.shape[1],32,seed); rg=np.random.default_rng(seed); ww=np.ones(len(y))
    for ep in range(epochs):
        o=rg.permutation(len(Z))
        for i in range(0,len(Z),4096):
            j=o[i:i+4096]; net.step(Z[j],y[j],ww[j])
    return net,mu,sd
def build_fn_from_guide(s,g):
    co=H.tocoo(); off=co.row!=co.col
    r=co.row[off]; c=co.col[off]; h=co.data[off]
    prod=s[r]*s[c]; keep=prod<0
    gg=np.maximum(g,1e-14)
    shift=np.bincount(r[~keep],weights=h[~keep]*gg[c[~keep]]/gg[r[~keep]],minlength=D)
    rr=np.concatenate([r[keep],np.arange(D)])
    cc=np.concatenate([c[keep],np.arange(D)])
    vv=np.concatenate([h[keep]*prod[keep],diag+shift])
    return sp.coo_matrix((vv,(rr,cc)),shape=H.shape).tocsr()

def ce(score,y,e):
    z=np.clip(e*score,-40,40)
    return float(np.mean(np.logaddexp(0,z)-y*z))

def main():
    ys=special_columns(np.random.default_rng(20260930),2)
    XA=[];YA=[]; XB=[];YB=[]; meta={}
    for k,y in enumerate(ys):
        x,yv,m=walkers_uniform(int(y),8192,61001+k); XA.append(x);YA.append(yv);meta[int(y)]=m
        x,yv,_=walkers_uniform(int(y),8192,62001+k); XB.append(x);YB.append(yv)
    XA=np.concatenate(XA);YA=np.concatenate(YA);XB=np.concatenate(XB);YB=np.concatenate(YB)
    net,mu,sd=train(XA,YA,7)
    sa=net.pred((XA-mu)/sd); sb=net.pred((XB-mu)/sd)
    aucA=weighted_auc(sa,YA,np.ones(len(YA))); aucB=weighted_auc(sb,YB,np.ones(len(YB)))
    grid=np.linspace(.05,2,196); loss=np.array([ce(sb,YB,e) for e in grid]); eta=float(grid[np.argmin(loss)])
    print("TRANSFER",aucA,aucB,"eta",eta,flush=True)
    rows=[]
    for y in ys:
        d,n2,s=meta[int(y)]; F=feat(d,n2,np.arange(D))
        lg=eta*net.pred((F-mu)/sd); lg-=lg.max(); g=np.exp(np.clip(lg,-60,0))
        e=np.zeros(D); e[int(y)]=1.
        FF=build_fn_from_guide(s,g)
        a=np.asarray(sla.expm_multiply(-.5*FF,e),float); a=np.maximum(a,0);a/=np.linalg.norm(a)
        ex=np.asarray(sla.expm_multiply(-.5*H,e),float); ex/=np.linalg.norm(ex)
        at=np.abs(ex); p=ex*ex; ts=np.where(ex>=0,1,-1)
        model_fid=float(np.dot(g/np.linalg.norm(g),at)**2)
        row={"y":int(y),"model_amplitude_fidelity":model_fid,
             "one_refresh_amplitude_fidelity":float(np.dot(a,at)**2),
             "one_refresh_full_fidelity":float(np.dot(a*s,ex)**2),
             "sign_mismatch_mass":float(np.sum(p[s!=ts]))}
        rows.append(row); print(row,flush=True)
    path=ROOT/"results/finite_tau_uniform_one_refresh_20site.json"
    path.write_text(json.dumps({"train_auc":aucA,"val_auc":aucB,"eta":eta,"rows":rows},indent=2))
    print("WROTE",path)
if __name__=="__main__":main()
