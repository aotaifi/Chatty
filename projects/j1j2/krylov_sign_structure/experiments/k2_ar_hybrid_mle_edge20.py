#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np
import scipy.sparse.linalg as sla
import torch
import torch.nn as nn
import torch.nn.functional as F

from finite_tau_matching_20site_exact import basis,D,N
from finite_tau_ct_local_ratio_20site import H,normalized

torch.set_num_threads(4)
HO=H.copy();HO.setdiag(0);HO.eliminate_zeros()

def bits_idx(idx):
    ss=basis[np.asarray(idx,np.int64)]
    return ((ss[:,None]>>np.arange(N,dtype=np.uint32))&1).astype(np.int64)

class SectorAR(nn.Module):
    def __init__(self,h=96,e=16):
        super().__init__(); self.h=h
        self.spin=nn.Embedding(2,e); self.end=nn.Embedding(2,8); self.pos=nn.Embedding(N,e)
        self.ctx=nn.Linear(N,h); self.cell=nn.GRUCell(e+8+e,h); self.out=nn.Linear(h+e,1)
    def logits(self,bits,ybits):
        B=bits.shape[0]; yv=(2*ybits.float()-1).unsqueeze(0).repeat(B,1)
        h=torch.tanh(self.ctx(yv)); outs=[]
        for i in range(N):
            pi=self.pos.weight[i].unsqueeze(0).repeat(B,1)
            outs.append(self.out(torch.cat([h,pi],1)).squeeze(1))
            yi=self.end(ybits[i]).unsqueeze(0).repeat(B,1)
            h=self.cell(torch.cat([self.spin(bits[:,i]),yi,pi],1),h)
        return torch.stack(outs,1)
    def logprob(self,bits,ybits):
        lg=self.logits(bits,ybits)
        ones_before=torch.cumsum(bits,1)-bits
        rem=N-torch.arange(N,device=bits.device).unsqueeze(0)
        need=N//2-ones_before
        forced=(need==0)|(need==rem)
        lp=-F.binary_cross_entropy_with_logits(lg,bits.float(),reduction='none')
        return torch.where(forced,torch.zeros_like(lp),lp).sum(1)
def draw_edges(rg,q,n):
    src=rg.choice(D,n,p=q).astype(np.int32); dst=np.empty(n,np.int32)
    for k,x in enumerate(src):
        lo,hi=HO.indptr[x],HO.indptr[x+1]; js=HO.indices[lo:hi]; hs=np.abs(HO.data[lo:hi])
        dst[k]=int(rg.choice(js,p=hs/hs.sum()))
    return src,dst

def edge_rmse(rg,dist,target_log,pred_log,n=6000):
    xs=rg.choice(D,n,p=dist); src=[];dst=[];ww=[]
    for x in xs:
        lo,hi=HO.indptr[x],HO.indptr[x+1]; js=HO.indices[lo:hi];hs=np.abs(HO.data[lo:hi])
        src.extend([int(x)]*len(js));dst.extend(js.tolist());ww.extend(hs.tolist())
    src=np.asarray(src);dst=np.asarray(dst);ww=np.asarray(ww,float)
    e=(pred_log[dst]-pred_log[src])-(target_log[dst]-target_log[src])
    return float(np.sqrt(np.sum(ww*e*e)/np.sum(ww)))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--M",type=int,default=25000); ap.add_argument("--nedges",type=int,default=30000)
    ap.add_argument("--epochs",type=int,default=30); ap.add_argument("--hidden",type=int,default=96)
    ap.add_argument("--alpha",type=float,default=.8); ap.add_argument("--out",required=True)
    a=ap.parse_args()
    z=np.load("../results/reconstruct_M100000_y59279.npz")
    y=int(z["y"]);s=z["s"].astype(float);logg=z["logg"].astype(float)
    g=normalized(np.exp(np.clip(logg-logg.max(),-700,0))); pg=g*g;pg/=pg.sum()
    psi=s*g;dt=.0125;h1=H@psi
    psi2=normalized(psi-dt*h1+.5*dt*dt*(H@h1));amp2=np.abs(psi2);pt=amp2*amp2;pt/=pt.sum()
    target_log=np.log(np.maximum(amp2,1e-300))
    qt=np.power(np.maximum(pg,1e-300),a.alpha);qt/=qt.sum()
    rg=np.random.default_rng(1448001)
    train_idx=rg.choice(D,a.M,p=pt); val_idx=rg.choice(D,10000,p=pt)
    ei,ej=draw_edges(rg,qt,a.nedges)
    et=(target_log[ej]-target_log[ei]).astype(np.float32); escale=max(float(np.sqrt(np.mean(et*et))),1e-6)
    X=torch.tensor(bits_idx(train_idx),dtype=torch.long); VX=torch.tensor(bits_idx(val_idx),dtype=torch.long)
    EI=torch.tensor(bits_idx(ei),dtype=torch.long); EJ=torch.tensor(bits_idx(ej),dtype=torch.long)
    ET=torch.tensor(et); yb=torch.tensor(bits_idx([y])[0],dtype=torch.long)
    torch.manual_seed(1448001);model=SectorAR(a.hidden)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-6)
    bs=512;best=None;bestv=1e99;hist=[]
    for ep in range(a.epochs):
        order=rg.permutation(a.M); eorder=rg.permutation(a.nedges); ls=[]; es=[]
        model.train()
        for q0 in range(0,a.M,bs):
            z0=order[q0:q0+bs]
            ez=eorder[(q0%a.nedges):min((q0%a.nedges)+len(z0),a.nedges)]
            if len(ez)<len(z0): ez=np.concatenate([ez,eorder[:len(z0)-len(ez)]])
            opt.zero_grad()
            nll=-model.logprob(X[z0],yb).mean()
            li=.5*model.logprob(EI[ez],yb); lj=.5*model.logprob(EJ[ez],yb)
            edge=((lj-li-ET[ez])/escale).pow(2).mean()
            loss=nll+edge
            loss.backward();opt.step();ls.append(float(nll.detach()));es.append(float(torch.sqrt(edge).detach()))
        model.eval();vls=[]
        with torch.no_grad():
            for q0 in range(0,len(VX),bs):
                vls.append(float((-model.logprob(VX[q0:q0+bs],yb).mean()).detach()))
        v=float(np.mean(vls));tr=float(np.mean(ls));er=float(np.mean(es));hist.append([ep+1,tr,v,er])
        if v<bestv:
            bestv=v;best={k:t.detach().clone() for k,t in model.state_dict().items()}
        if ep in (0,1,2,4,9,19,a.epochs-1):
            print("EPOCH",ep+1,"train_nll",tr,"val_nll",v,"edge_scaled_rmse",er,flush=True)
    model.load_state_dict(best);model.eval();lp=[]
    with torch.no_grad():
        for q0 in range(0,D,4096):
            b=torch.tensor(bits_idx(np.arange(q0,min(q0+4096,D))),dtype=torch.long)
            lp.append(model.logprob(b,yb).numpy())
    lp=np.concatenate(lp);m=lp.max();pm=np.exp(lp-m);zsum=float(pm.sum());pm/=zsum;gm=np.sqrt(pm)
    e=np.zeros(D);e[y]=1.;exact0=normalized(sla.expm_multiply(-.5*H,e));exact1=normalized(sla.expm_multiply(-dt*H,exact0))
    pp=exact1*exact1;pp/=pp.sum();current_log=np.log(np.maximum(g,1e-300));model_log=.5*np.log(np.maximum(pm,1e-300))
    out={"M":a.M,"nedges":a.nedges,"alpha":a.alpha,"epochs":a.epochs,"hidden":a.hidden,
         "train_unique":int(np.unique(train_idx).size),"best_val_nll":bestv,"edge_scale":escale,
         "sector_prob_sum_raw":zsum*np.exp(m),"target_fid":float((gm@amp2)**2),
         "physical_exact_fid":float((gm@np.abs(exact1))**2),"current_physical_fid":float((g@np.abs(exact0))**2),
         "k2_target_physical_fid":float((amp2@np.abs(exact1))**2),
         "current_target_edge_rmse":edge_rmse(rg,pp,target_log,current_log),
         "ar_target_edge_rmse":edge_rmse(rg,pp,target_log,model_log),"history":hist}
    Path(a.out).write_text(json.dumps(out,indent=2))
    print("RESULT",json.dumps({k:v for k,v in out.items() if k!="history"},sort_keys=True),flush=True)

if __name__=="__main__":main()
