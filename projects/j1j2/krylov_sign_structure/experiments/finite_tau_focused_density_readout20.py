#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from finite_tau_matching_20site_exact import special_columns,D
from finite_tau_uniform_one_refresh_20site import H
from finite_tau_two_learned_refresh_20site import propagate_vec
from finite_tau_localratio_mlp_oracle20 import feat,make_wd,mm
from finite_tau_learned_amplitude_20site import MLP,weighted_auc,calib_loss

ROOT=Path("/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure")

def norm(v):
    v=np.asarray(v,float);return v/np.linalg.norm(v)

def train_clf(X,y,seed=111,epochs=30):
    mu=X.mean(0);sd=X.std(0);sd[sd<1e-8]=1;Z=(X-mu)/sd
    net=MLP(Z.shape[1],64,seed);rg=np.random.default_rng(seed);w=np.ones(len(y))
    for ep in range(epochs):
        o=rg.permutation(len(Z))
        for q in range(0,len(o),2048):
            j=o[q:q+2048];net.step(Z[j],y[j],w[j],lr=1e-3)
    return net,mu,sd

def score_full(net,mu,sd,y,wd,tau=.5):
    out=[]
    for q in range(0,D,4096):
        ix=np.arange(q,min(q+4096,D))
        out.append(net.pred((feat(ix,y,wd,tau)-mu)/sd))
    return np.concatenate(out)

def main():
    y=int(special_columns(np.random.default_rng(20260930),2)[1]);M=8192;dt=.05
    z=np.load(ROOT/"results/finite_tau_densityratio_k1_eval20_long_guide.npz")
    lg=z["logg"].astype(float);s=z["s"].astype(np.int8)
    g=norm(np.exp(np.clip(lg-lg.max(),-60,0)));wd=make_wd(H,[y])[0]
    margin=1-dt*s*(H@(s*g))/np.maximum(g,1e-300)
    c=1+200*np.exp(-(np.abs(margin)/.5)**2)
    q=np.exp(np.clip(2*(lg-lg.max()),-60,0))*c;q/=q.sum()

    wa=propagate_vec(y,lg,s,M,111001);wb=propagate_vec(y,lg,s,M,112001)
    ra=np.random.default_rng(113001);rb=np.random.default_rng(114001)
    na=ra.choice(D,size=M,p=q);nb=rb.choice(D,size=M,p=q)
    XA=np.vstack([feat(wa,y,wd,.5),feat(na,y,wd,.5)])
    YA=np.concatenate([np.ones(M),np.zeros(M)])
    XB=np.vstack([feat(wb,y,wd,.5),feat(nb,y,wd,.5)])
    YB=np.concatenate([np.ones(M),np.zeros(M)])
    net,mu,sd=train_clf(XA,YA)
    sa=net.pred((XA-mu)/sd);sb=net.pred((XB-mu)/sd)

    grid=np.linspace(.1,2.0,191)
    losses=np.array([calib_loss(sb,YB,np.ones(len(YB)),e) for e in grid])
    eta=float(grid[np.argmin(losses)])
    sf=score_full(net,mu,sd,y,wd)
    corr=eta*sf+np.log(c)
    e=np.zeros(D);e[y]=1.;ex=norm(sla.expm_multiply(-.5*H,e));exn=norm(sla.expm_multiply(-dt*H,ex))
    rows=[]
    Hoff=(H-sp.diags(H.diagonal())).tocsr()
    for th in (.2,.5,1.,1.5):
        cand=np.abs(margin)<th
        neigh=np.asarray((np.abs(Hoff)@cand.astype(float))>0).ravel()
        for spread,mask in (("self",cand),("self+1hop",cand|neigh)):
            cc=np.zeros(D);cc[mask]=np.clip(corr[mask],-3,3)
            la=lg+cc;la-=la.max()
            ah=norm(np.exp(np.clip(la,-60,0)))
            sn=np.where((s*ah-dt*(H@(s*ah)))>=0,1,-1)
            row={"threshold":th,"spread":spread,"mask_n":int(mask.sum()),
                 "amp_fid":float(np.dot(ah,np.abs(ex))**2),"k1_next_err":mm(sn,exn)}
            rows.append(row);print("LOCAL",row,flush=True)
    out={"M":M,"auc_train":weighted_auc(sa,YA,np.ones(len(YA))),
         "auc_val":weighted_auc(sb,YB,np.ones(len(YB))),"eta":eta,
         "guide_amp_fid":float(np.dot(g,np.abs(ex))**2),
         "carry_next_err":mm(s,exn),"rows":rows}
    path=ROOT/"results/finite_tau_focused_density_readout20.json"
    path.write_text(json.dumps(out,indent=2));print("WROTE",path,flush=True)

if __name__=="__main__":main()
