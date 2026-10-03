#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import scipy.sparse.csgraph as cs
import torch
import torch.nn as nn
import torch.nn.functional as F

from finite_tau_matching_20site_exact import basis,D,N,NN,NNN,special_columns
from finite_tau_ct_local_ratio_20site import H,diag,distance_sign,normalized,mismatch
from finite_tau_ct_smc_edgeratio_longbeta20 import reconstruct

torch.set_num_threads(4)

def adj(bonds):
    A=np.zeros((N,N),np.float32)
    for u,v in bonds:
        A[u,v]=A[v,u]=1.0
    A/=np.maximum(A.sum(1,keepdims=True),1.0)
    return torch.tensor(A)
A1=adj(NN); A2=adj(NNN)

def bits_idx(idx):
    ss=basis[np.asarray(idx,np.int64)]
    b=((ss[:,None]>>np.arange(N,dtype=np.uint32))&1).astype(np.float32)
    return 2*b-1

def make_wd(y):
    HO=H.copy(); HO.setdiag(0); HO.eliminate_zeros()
    co=HO.tocoo()
    cost=np.where(np.abs(co.data)>0.375,1000.,1001.)
    W=sp.coo_matrix((cost,(co.row,co.col)),shape=H.shape).tocsr()
    return np.asarray(cs.shortest_path(W,directed=False,indices=[int(y)]))[0]

class PairGNN(nn.Module):
    def __init__(self,h=48,layers=4):
        super().__init__()
        self.layers=layers
        self.inp=nn.Linear(3,h)
        self.d1=nn.ModuleList([nn.Linear(3*h,2*h) for _ in range(layers)])
        self.d2=nn.ModuleList([nn.Linear(2*h,h) for _ in range(layers)])
        self.p1=nn.Linear(2*h+3,64)
        self.p2=nn.Linear(64,32)
        self.out=nn.Linear(32,1)
    def forward(self,x,y,g):
        h=self.inp(torch.stack([x,y,x*y],dim=-1))
        for l in range(self.layers):
            m1=torch.einsum('ij,bjh->bih',A1,h)
            m2=torch.einsum('ij,bjh->bih',A2,h)
            z=torch.cat([h,m1,m2],dim=-1)
            h=F.gelu(h+self.d2[l](F.gelu(self.d1[l](z))))
        p=torch.cat([h.mean(1),(h*h).mean(1),g],dim=-1)
        z=F.gelu(self.p1(p)); z=F.gelu(self.p2(z))
        return self.out(z).squeeze(-1)

def globals_from_wd(wd,tau,idx):
    idx=np.asarray(idx,np.int64)
    d=(wd[idx]//1000).astype(np.float32)
    n2=np.rint(wd[idx]-1000*(wd[idx]//1000)).astype(np.float32)
    return np.column_stack([d/10,n2/10,np.full(len(idx),tau/.5,np.float32)])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--M",type=int,default=100000)
    ap.add_argument("--steps",type=int,default=5)
    ap.add_argument("--dt",type=float,default=.0125)
    ap.add_argument("--alpha",type=float,default=.8)
    ap.add_argument("--centers",type=int,default=2000)
    ap.add_argument("--epochs",type=int,default=4)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()

    y=int(special_columns(np.random.default_rng(20260930),2)[1])
    wd=make_wd(y)
    HO=H.copy(); HO.setdiag(0); HO.eliminate_zeros()
    s=distance_sign(y)
    st1,st2,logg=reconstruct(y,s,a.M)
    g=normalized(np.exp(np.clip(logg-logg.max(),-700,0)))

    e=np.zeros(D); e[y]=1.0
    exact=normalized(sla.expm_multiply(-.5*H,e))
    tau=.5
    yb=torch.tensor(bits_idx([y])[0])
    rows=[]

    def data(ids,t):
        ids=np.asarray(ids,np.int64)
        x=torch.tensor(bits_idx(ids))
        yy=yb[None,:].repeat(len(ids),1)
        gg=torch.tensor(globals_from_wd(wd,t,ids))
        return x,yy,gg

    def full_pred(model,t):
        out=[]; model.eval()
        with torch.no_grad():
            for q in range(0,D,4096):
                ids=np.arange(q,min(q+4096,D))
                out.append(model(*data(ids,t)).numpy())
        return np.concatenate(out)

    for step in range(1,a.steps+1):
        psi=s*g
        h1=H@psi
        psi2=normalized(psi-a.dt*h1+.5*a.dt*a.dt*(H@h1))
        amp2=np.abs(psi2)
        sn=np.where(psi2>=0,1.0,-1.0)
        target=np.log(np.maximum(amp2,1e-300))-np.log(np.maximum(g,1e-300))

        ptr=g**(2*a.alpha); ptr/=ptr.sum()
        rg=np.random.default_rng(500000+step)
        torch.manual_seed(500000+step)
        cent=rg.choice(D,a.centers,p=ptr)
        counts=np.bincount(cent,minlength=D).astype(float)
        for x0 in np.unique(cent):
            lo,hi=HO.indptr[x0],HO.indptr[x0+1]
            ns=HO.indices[lo:hi]; hs=np.abs(HO.data[lo:hi])
            counts[ns]+=hs/max(float(hs.mean()),1e-12)
        train=np.flatnonzero(counts>0).astype(np.int32)
        w=counts[train]; w/=w.mean()

        model=PairGNN()
        opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-6)
        bs=2048
        for ep in range(a.epochs):
            order=rg.permutation(len(train))
            ls=[]
            model.train()
            for q in range(0,len(order),bs):
                z=order[q:q+bs]; ids=train[z]
                tt=torch.tensor(target[ids].astype(np.float32))
                ww=torch.tensor(w[z].astype(np.float32))
                opt.zero_grad()
                pred=model(*data(ids,tau+a.dt))
                err=pred-tt
                loss=(ww*err*err).sum()/ww.sum()
                loss.backward(); opt.step()
                ls.append(float(loss.detach()))
            print("TRAIN",step,ep+1,float(np.mean(ls)),flush=True)

        f=full_pred(model,tau+a.dt)
        g=normalized(g*np.exp(np.clip(f,-5,5)))
        s=sn
        tau=round(tau+a.dt,10)
        exact=normalized(sla.expm_multiply(-a.dt*H,exact))

        row={"step":step,"tau":tau,
             "full":float(np.dot(s*g,exact)**2),
             "amp":float(np.dot(g,np.abs(exact))**2),
             "target_fid":float(np.dot(g,amp2)**2),
             "sign":mismatch(s,exact),
             "ntrain":int(len(train))}
        rows.append(row)
        print("ROW",json.dumps(row,sort_keys=True),flush=True)

    out={"M":a.M,"dt":a.dt,"alpha":a.alpha,"centers":a.centers,
         "epochs":a.epochs,"rows":rows}
    Path(a.out).write_text(json.dumps(out,indent=2))
    print("FINAL",json.dumps(rows[-1],sort_keys=True),flush=True)

if __name__=="__main__":
    main()
