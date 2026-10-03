#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np

from finite_tau_matching_20site_exact import D,special_columns
from finite_tau_ct_local_ratio_20site import H,distance_sign,normalized
from finite_tau_ct_smc_edgeratio_longbeta20 import reconstruct
from finite_tau_localratio_mlp_oracle20 import feat,make_wd
from k2_sr_tangent_oneblock20 import TinyMLP
from k2_sr_edge_tangent_oneblock20 import edge_set,jac_for,wrmse

def sample_neighbors(xs,K,rg):
    out=[]
    for x in xs:
        lo,hi=H.indptr[x],H.indptr[x+1]
        js=H.indices[lo:hi]; hs=np.abs(H.data[lo:hi]); m=js!=x
        js=js[m]; hs=hs[m]; pr=hs/hs.sum()
        out.extend(rg.choice(js,size=K,replace=True,p=pr).tolist())
    return np.asarray(out,np.int32)

def weighted_center(J,t,w):
    w=w/w.sum()
    return J-(w@J)[None,:], t-float(w@t)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--M",type=int,default=100000)
    ap.add_argument("--dt",type=float,default=.0125)
    ap.add_argument("--ns",type=int,default=2000)
    ap.add_argument("--K",type=int,default=4)
    ap.add_argument("--h",type=int,default=8)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()

    y=int(special_columns(np.random.default_rng(20260930),2)[1]); wd=make_wd(H,[y])[0]
    s=distance_sign(y); _,_,logg=reconstruct(y,s,a.M)
    g=normalized(np.exp(np.clip(logg-logg.max(),-700,0))); p=g*g; p/=p.sum()
    psi=s*g; h1=H@psi
    psi2=normalized(psi-a.dt*h1+.5*a.dt*a.dt*(H@h1)); amp2=np.abs(psi2)
    targ=np.log(np.maximum(amp2,1e-300))-np.log(np.maximum(g,1e-300))
    tau=.5+a.dt; rg=np.random.default_rng(2300001+a.M+a.K)

    trx=rg.choice(D,a.ns,p=p); vax=rg.choice(D,a.ns,p=p)
    trn=sample_neighbors(trx,a.K,rg); van=sample_neighbors(vax,a.K,rg)
    tri=np.concatenate([trx,trn]); vai=np.concatenate([vax,van])
    # Give centers and sampled-neighbor cloud equal total weight.
    tw=np.concatenate([np.full(len(trx),.5/len(trx)),np.full(len(trn),.5/len(trn))])
    vw=np.concatenate([np.full(len(vax),.5/len(vax)),np.full(len(van),.5/len(van))])

    X0=feat(tri,y,wd,tau); mu=X0.mean(0); sd=X0.std(0); sd[sd<1e-6]=1
    net=TinyMLP(X0.shape[1],a.h,456)
    J=jac_for(net,tri,y,wd,tau,mu,sd); tt=targ[tri]
    Jc,tc=weighted_center(J,tt,tw)
    sw=np.sqrt(tw*len(tw)); A=Jc*sw[:,None]; z=tc*sw
    S=(A.T@A)/len(A); b=(A.T@z)/len(A)

    JV=jac_for(net,vai,y,wd,tau,mu,sd); vt=targ[vai]
    JVc,vtc=weighted_center(JV,vt,vw)
    scale=max(float(np.trace(S)/len(S)),1e-14)
    ev,U=np.linalg.eigh(S); ub=U.T@b
    lams=scale*np.array([1e-5,1e-4,1e-3,1e-2,1e-1,1.,10.])
    vals=[]; sols=[]
    for lam in lams:
        sol=U@(ub/(ev+lam))
        e=JVc@sol-vtc
        vals.append(float(np.sqrt(np.sum(vw*e*e))))
        sols.append(sol)
    k=int(np.argmin(vals)); sol=sols[k]

    allf=[]
    for q in range(0,D,4096):
        ix=np.arange(q,min(D,q+4096))
        allf.append(net.jac((feat(ix,y,wd,tau)-mu)/sd)@sol)
    f=np.concatenate(allf); f-=np.sum(p*f)
    raw_rms=float(np.sqrt(np.sum(p*f*f)))
    eta=min(1.0,0.05/max(raw_rms,1e-14))
    f*=eta
    gp=normalized(g*np.exp(np.clip(f,-20,20)))

    # Blind all-edge audit from independent validation centers.
    vasrc,vadst,vaw=edge_set(vax)
    prededge=f[vadst]-f[vasrc]
    tedge=targ[vadst]-targ[vasrc]
    train_seen=np.zeros(D,dtype=bool); train_seen[np.unique(tri)]=True
    seen=train_seen[vadst]
    out={"M":a.M,"dt":a.dt,"ns":a.ns,"K":a.K,"h":a.h,
         "train_unique":int(train_seen.sum()),"train_frac_D":float(train_seen.mean()),
         "lambda":float(lams[k]),"val_mix_rmse":vals[k],
         "raw_f_rms_p":raw_rms,"eta":float(eta),
         "target_fid":float((gp@amp2)**2),
         "edge_seenY_frac":float(seen.mean()),
         "edge_base_rmse":wrmse(tedge,vaw),
         "edge_proj_rmse":wrmse(prededge-tedge,vaw),
         "edge_base_seenY":wrmse(tedge[seen],vaw[seen]) if np.any(seen) else None,
         "edge_proj_seenY":wrmse((prededge-tedge)[seen],vaw[seen]) if np.any(seen) else None,
         "edge_base_unseenY":wrmse(tedge[~seen],vaw[~seen]) if np.any(~seen) else None,
         "edge_proj_unseenY":wrmse((prededge-tedge)[~seen],vaw[~seen]) if np.any(~seen) else None,
         "f_rms_p":float(np.sqrt(np.sum(p*f*f))),"f_absmax":float(np.max(np.abs(f)))}
    print(json.dumps(out,sort_keys=True),flush=True)
    Path(a.out).write_text(json.dumps(out,indent=2))
if __name__=="__main__": main()
