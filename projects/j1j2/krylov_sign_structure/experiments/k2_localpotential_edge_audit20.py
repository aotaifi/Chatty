#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import scipy.sparse.linalg as sla
import torch
import torch.nn as nn
import torch.nn.functional as F

from finite_tau_matching_20site_exact import basis,D,N,NN,NNN,special_columns
from finite_tau_ct_local_ratio_20site import H,distance_sign,normalized
from finite_tau_ct_smc_edgeratio_longbeta20 import reconstruct

torch.set_num_threads(4)
HO=H.copy(); HO.setdiag(0); HO.eliminate_zeros()

def adj(bonds):
    A=np.zeros((N,N),np.float32)
    for u,v in bonds: A[u,v]=A[v,u]=1.
    A/=np.maximum(A.sum(1,keepdims=True),1.)
    return torch.tensor(A)
A1,A2=adj(NN),adj(NNN)

def bits_idx(idx):
    ss=basis[np.asarray(idx,np.int64)]
    b=((ss[:,None]>>np.arange(N,dtype=np.uint32))&1).astype(np.float32)
    return 2*b-1

class LocalPotential(nn.Module):
    def __init__(self,h=48,layers=4):
        super().__init__()
        self.layers=layers
        self.inp=nn.Linear(3,h)
        self.d1=nn.ModuleList([nn.Linear(3*h,2*h) for _ in range(layers)])
        self.d2=nn.ModuleList([nn.Linear(2*h,h) for _ in range(layers)])
        self.site1=nn.Linear(h,48); self.site2=nn.Linear(48,1)
    def forward(self,x,y):
        h=self.inp(torch.stack([x,y,x*y],dim=-1))
        for l in range(self.layers):
            m1=torch.einsum('ij,bjh->bih',A1,h)
            m2=torch.einsum('ij,bjh->bih',A2,h)
            z=torch.cat([h,m1,m2],dim=-1)
            h=F.gelu(h+self.d2[l](F.gelu(self.d1[l](z))))
        e=self.site2(F.gelu(self.site1(h))).squeeze(-1)
        return e.sum(dim=1)/np.sqrt(N)
def draw_pairs(rg,p,q,n,mix):
    nm=int(round(n*mix))
    src=np.concatenate([rg.choice(D,n-nm,p=p),rg.choice(D,nm,p=q)]).astype(np.int32)
    dst=np.empty(n,np.int32)
    for k,x in enumerate(src):
        lo,hi=HO.indptr[x],HO.indptr[x+1]
        js=HO.indices[lo:hi]; hs=np.abs(HO.data[lo:hi])
        dst[k]=int(rg.choice(js,p=hs/hs.sum()))
    return src,dst

def edge_audit(rg,dist,targ,f,n=4000):
    xs=rg.choice(D,n,p=dist)
    src=[]; dst=[]; ww=[]
    for x in xs:
        lo,hi=HO.indptr[x],HO.indptr[x+1]
        js=HO.indices[lo:hi]; hs=np.abs(HO.data[lo:hi])
        src.extend([int(x)]*len(js)); dst.extend(js.tolist()); ww.extend(hs.tolist())
    src=np.asarray(src,np.int32); dst=np.asarray(dst,np.int32); ww=np.asarray(ww,float)
    tar=targ[dst]-targ[src]; err=(f[dst]-f[src])-tar
    wrms=lambda z: float(np.sqrt(np.sum(ww*z*z)/np.sum(ww)))
    return {"nedge":int(len(src)),"base_rmse":wrms(tar),"proj_rmse":wrms(err)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode",choices=["exact","reconstructed"],required=True)
    ap.add_argument("--M",type=int,default=100000)
    ap.add_argument("--dt",type=float,default=.0125)
    ap.add_argument("--alpha",type=float,default=.8)
    ap.add_argument("--mix",type=float,default=.5)
    ap.add_argument("--ntrain",type=int,default=30000)
    ap.add_argument("--nval",type=int,default=10000)
    ap.add_argument("--epochs",type=int,default=15)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    y=int(special_columns(np.random.default_rng(20260930),2)[1])
    s=distance_sign(y)
    e=np.zeros(D); e[y]=1.
    exact0=normalized(sla.expm_multiply(-.5*H,e))
    if a.mode=="exact":
        g=np.abs(exact0).copy(); s=np.where(exact0>=0,1.,-1.)
    else:
        _,_,logg=reconstruct(y,s,a.M)
        g=normalized(np.exp(np.clip(logg-logg.max(),-700,0)))

    p=g*g; p/=p.sum()
    q=np.power(np.maximum(p,1e-300),a.alpha); q/=q.sum()
    psi=s*g; h1=H@psi
    psi2=normalized(psi-a.dt*h1+.5*a.dt*a.dt*(H@h1))
    amp2=np.abs(psi2)
    targ=np.log(np.maximum(amp2,1e-300))-np.log(np.maximum(g,1e-300))

    rg=np.random.default_rng(990001 if a.mode=="exact" else 990002)
    torch.manual_seed(990001 if a.mode=="exact" else 990002)
    tr_i,tr_j=draw_pairs(rg,p,q,a.ntrain,a.mix)
    va_i,va_j=draw_pairs(rg,p,q,a.nval,a.mix)
    yb=torch.tensor(bits_idx([y])[0])

    def data(ids):
        ids=np.asarray(ids,np.int64)
        x=torch.tensor(bits_idx(ids))
        yy=yb[None,:].repeat(len(ids),1)
        return x,yy

    tr_t=torch.tensor((targ[tr_j]-targ[tr_i]).astype(np.float32))
    va_t=torch.tensor((targ[va_j]-targ[va_i]).astype(np.float32))
    model=LocalPotential()
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-6)
    bs=1024; best=None; bestv=1e99
    for ep in range(a.epochs):
        order=rg.permutation(a.ntrain); ls=[]; model.train()
        for q0 in range(0,a.ntrain,bs):
            z=order[q0:q0+bs]
            opt.zero_grad()
            fi=model(*data(tr_i[z])); fj=model(*data(tr_j[z]))
            loss=((fj-fi-tr_t[z])**2).mean()
            loss.backward(); opt.step(); ls.append(float(loss.detach()))
        model.eval(); vals=[]
        with torch.no_grad():
            for q0 in range(0,a.nval,bs):
                z=slice(q0,min(q0+bs,a.nval))
                fi=model(*data(va_i[z])); fj=model(*data(va_j[z]))
                vals.append(float(((fj-fi-va_t[z])**2).mean()))
        vr=float(np.sqrt(np.mean(vals)))
        if vr<bestv:
            bestv=vr; best={k:v.detach().clone() for k,v in model.state_dict().items()}
        if ep in (0,1,2,4,9,a.epochs-1):
            print("EPOCH",a.mode,ep+1,"train_rmse",float(np.sqrt(np.mean(ls))),"val_rmse",vr,flush=True)

    model.load_state_dict(best); model.eval()
    fv=[]
    with torch.no_grad():
        for q0 in range(0,D,4096):
            ids=np.arange(q0,min(q0+4096,D))
            fv.append(model(*data(ids)).numpy())
    f=np.concatenate(fv); f-=np.sum(p*f)
    gp=normalized(g*np.exp(np.clip(f,-5,5)))
    pphys=exact0*exact0; pphys/=pphys.sum()
    res={"mode":a.mode,"M":a.M,"dt":a.dt,"alpha":a.alpha,"mix":a.mix,
         "ntrain":a.ntrain,"nval":a.nval,"epochs":a.epochs,
         "val_edge_rmse":bestv,"raw_f_rms_guide":float(np.sqrt(np.sum(p*f*f))),
         "target_fid":float((gp@amp2)**2),
         "physical_edge_audit":edge_audit(rg,pphys,targ,f),
         "guide_edge_audit":edge_audit(rg,p,targ,f)}
    Path(a.out).write_text(json.dumps(res,indent=2))
    print("RESULT",json.dumps(res,sort_keys=True),flush=True)

if __name__=="__main__": main()
