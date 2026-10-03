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

def flat_params(net):
    arr=[net.W1,net.b1,net.W2,net.b2,net.W3,net.b3]
    shapes=[x.shape for x in arr]; sizes=[x.size for x in arr]
    return np.concatenate([x.ravel() for x in arr]),shapes,sizes

def set_flat(net,v,shapes,sizes):
    arr=[]; q=0
    for sh,n in zip(shapes,sizes):
        arr.append(v[q:q+n].reshape(sh));q+=n
    net.W1[:]=arr[0];net.b1[:]=arr[1];net.W2[:]=arr[2]
    net.b2[:]=arr[3];net.W3[:]=arr[4];net.b3[:]=arr[5]

def full_pred(net,y,wd,tau,mu,sd,batch=4096):
    out=[]
    for q in range(0,D,batch):
        ix=np.arange(q,min(D,q+batch))
        out.append(net.fwd((feat(ix,y,wd,tau)-mu)/sd)[0])
    return np.concatenate(out)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--M",type=int,default=100000)
    ap.add_argument("--dt",type=float,default=.0125)
    ap.add_argument("--ns",type=int,default=2000)
    ap.add_argument("--h",type=int,default=8)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()

    y=int(special_columns(np.random.default_rng(20260930),2)[1]); wd=make_wd(H,[y])[0]
    s=distance_sign(y); _,_,logg0=reconstruct(y,s,a.M)
    g0=normalized(np.exp(np.clip(logg0-logg0.max(),-700,0))); p0=g0*g0; p0/=p0.sum()
    rg=np.random.default_rng(2200001+a.M+int(10000*a.dt))
    tr=rg.choice(D,a.ns,p=p0); va=rg.choice(D,a.ns,p=p0)

    # Persistent network starts with a random full tangent. Absorb its initial output
    # into a fixed base so total log amplitude exactly equals reconstructed logg0.
    X0=feat(tr,y,wd,.5); mu=X0.mean(0); sd=X0.std(0); sd[sd<1e-6]=1
    net=TinyMLP(X0.shape[1],a.h,321)
    f0=full_pred(net,y,wd,.5,mu,sd)
    base=logg0-f0

    psi=s*g0; h1=H@psi; h2=H@h1
    psi2=normalized(psi-a.dt*h1+.5*a.dt*a.dt*h2); amp2=np.abs(psi2)
    targ=np.log(np.maximum(amp2,1e-300))-np.log(np.maximum(g0,1e-300))

    E=h1/np.where(np.abs(psi)>1e-300,psi,np.sign(psi)*1e-300)
    Etr=E[tr]; Eva=E[va]
    J=jac_for(net,tr,y,wd,.5,mu,sd); JV=jac_for(net,va,y,wd,.5,mu,sd)
    Jc=J-J.mean(0,keepdims=True); JVc=JV-JV.mean(0,keepdims=True)
    et=Etr-Etr.mean(); ev=Eva-Eva.mean()
    S=(Jc.T@Jc)/len(Jc); b=(Jc.T@et)/len(Jc)
    scale=max(float(np.trace(S)/len(S)),1e-14)
    ew,U=np.linalg.eigh(S); ub=U.T@b
    lams=scale*np.array([1e-5,1e-4,1e-3,1e-2,1e-1,1.,10.])
    vals=[]; sols=[]
    for lam in lams:
        sol=U@(ub/(ew+lam))
        pred=-a.dt*(JVc@sol); tv=-a.dt*ev
        vals.append(float(np.sqrt(np.mean((pred-tv)**2))))
        sols.append(sol)
    k=int(np.argmin(vals)); sol=sols[k]

    pflat,shapes,sizes=flat_params(net)
    delta=-a.dt*sol
    # Safety cap is geometric only; normally inactive at this dt.
    dlog=Jc@delta; rms=float(np.sqrt(np.mean(dlog*dlog)))
    cap=.10
    scale_eta=min(1.0,cap/max(rms,1e-14))
    delta*=scale_eta
    set_flat(net,pflat+delta,shapes,sizes)

    f1=full_pred(net,y,wd,.5,mu,sd)
    # tau feature changed; keep base fixed and compare actual total update.
    logg1=base+f1; g1=normalized(np.exp(np.clip(logg1-logg1.max(),-700,0)))
    sn=np.where(psi2>=0,1.0,-1.0)

    # Edge audit against the exact K2 amplitude correction.
    vasrc,vadst,vaw=edge_set(va)
    corr=(logg1-logg0)
    prededge=corr[vadst]-corr[vasrc]
    tedge=targ[vadst]-targ[vasrc]
    trainmask=np.zeros(D,dtype=bool);trainmask[np.unique(tr)]=True
    seen=trainmask[vadst]
    out={"M":a.M,"dt":a.dt,"ns":a.ns,"h":a.h,
         "lambda":float(lams[k]),"tdvp_val_rmse":vals[k],
         "train_dlog_rms":rms,"eta":float(scale_eta),
         "target_fid":float((g1@amp2)**2),
         "physical_full":float((sn*g1)@psi2)**2,
         "edge_seenY_frac":float(seen.mean()),
         "edge_base_rmse":wrmse(tedge,vaw),
         "edge_proj_rmse":wrmse(prededge-tedge,vaw),
         "edge_base_seenY":wrmse(tedge[seen],vaw[seen]) if np.any(seen) else None,
         "edge_proj_seenY":wrmse((prededge-tedge)[seen],vaw[seen]) if np.any(seen) else None,
         "edge_base_unseenY":wrmse(tedge[~seen],vaw[~seen]) if np.any(~seen) else None,
         "edge_proj_unseenY":wrmse((prededge-tedge)[~seen],vaw[~seen]) if np.any(~seen) else None}
    print(json.dumps(out,sort_keys=True),flush=True)
    Path(a.out).write_text(json.dumps(out,indent=2))

if __name__=="__main__":main()
