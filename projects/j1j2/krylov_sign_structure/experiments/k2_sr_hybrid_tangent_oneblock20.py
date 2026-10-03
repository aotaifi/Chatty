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

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--M",type=int,default=100000)
    ap.add_argument("--dt",type=float,default=.0125)
    ap.add_argument("--ns",type=int,default=2000)
    ap.add_argument("--h",type=int,default=8)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()

    y=int(special_columns(np.random.default_rng(20260930),2)[1]); wd=make_wd(H,[y])[0]
    s=distance_sign(y); _,_,logg=reconstruct(y,s,a.M)
    g=normalized(np.exp(np.clip(logg-logg.max(),-700,0))); psi=s*g
    h1=H@psi; psi2=normalized(psi-a.dt*h1+.5*a.dt*a.dt*(H@h1)); amp2=np.abs(psi2)
    targ=np.log(np.maximum(amp2,1e-300))-np.log(np.maximum(g,1e-300))
    tau=.5+a.dt; p=g*g; p/=p.sum(); rg=np.random.default_rng(2000001+a.M+int(10000*a.dt))
    trx=rg.choice(D,a.ns,p=p); vax=rg.choice(D,a.ns,p=p)
    trsrc,trdst,trw=edge_set(trx); vasrc,vadst,vaw=edge_set(vax)

    ids=np.concatenate([trx,trdst])
    X0=feat(ids,y,wd,tau); mu=X0.mean(0); sd=X0.std(0); sd[sd<1e-6]=1
    net=TinyMLP(X0.shape[1],a.h,91)

    # state tangent equations, centered to remove normalization gauge
    J=jac_for(net,trx,y,wd,tau,mu,sd); JV=jac_for(net,vax,y,wd,tau,mu,sd)
    Jc=J-J.mean(0,keepdims=True); tc=targ[trx]-targ[trx].mean()
    JVc=JV-J.mean(0,keepdims=True); tv=targ[vax]-targ[vax].mean()
    state_scale=max(float(np.mean(tc*tc)),1e-12)

    # edge-ratio tangent equations
    Jx=jac_for(net,trsrc,y,wd,tau,mu,sd); Jy=jac_for(net,trdst,y,wd,tau,mu,sd)
    A=Jy-Jx; et=targ[trdst]-targ[trsrc]
    ww=trw/max(float(np.mean(trw)),1e-12); sw=np.sqrt(ww)
    edge_scale=max(float(np.sum(trw*et*et)/np.sum(trw)),1e-12)

    # Equal relative weight: each branch is normalized by its own target variance.
    S=(Jc.T@Jc)/(len(Jc)*state_scale)
    b=(Jc.T@tc)/(len(Jc)*state_scale)
    Aw=A*sw[:,None]; ew=et*sw
    S += (Aw.T@Aw)/(len(Aw)*edge_scale)
    b += (Aw.T@ew)/(len(Aw)*edge_scale)

    scale=max(float(np.trace(S)/len(S)),1e-14)
    ev,U=np.linalg.eigh(S); ub=U.T@b
    Vx=jac_for(net,vasrc,y,wd,tau,mu,sd); Vy=jac_for(net,vadst,y,wd,tau,mu,sd)
    VA=Vy-Vx; vet=targ[vadst]-targ[vasrc]
    state_base=float(np.mean(tv*tv)); edge_base=float(np.sum(vaw*vet*vet)/np.sum(vaw))
    lams=scale*np.array([1e-5,1e-4,1e-3,1e-2,1e-1,1.,10.])
    scores=[]; sols=[]; parts=[]
    for lam in lams:
        sol=U@(ub/(ev+lam))
        sr=float(np.mean((JVc@sol-tv)**2))
        er=float(np.sum(vaw*(VA@sol-vet)**2)/np.sum(vaw))
        score=sr/max(state_base,1e-12)+er/max(edge_base,1e-12)
        scores.append(score); sols.append(sol); parts.append((sr,er))
    k=int(np.argmin(scores)); sol=sols[k]; sr,er=parts[k]

    allf=[]
    for q in range(0,D,4096):
        ix=np.arange(q,min(D,q+4096))
        allf.append(net.jac((feat(ix,y,wd,tau)-mu)/sd)@sol)
    f=np.concatenate(allf); f-=np.sum(p*f)
    gp=normalized(g*np.exp(np.clip(f,-20,20)))

    train_states=np.zeros(D,dtype=bool); train_states[np.unique(trx)]=True
    pred=VA@sol; err=pred-vet; seen=train_states[vadst]
    out={"M":a.M,"dt":a.dt,"ns":a.ns,"h":a.h,
         "lambda":float(lams[k]),"validation_score":float(scores[k]),
         "state_rmse_base":float(np.sqrt(state_base)),
         "state_rmse_proj":float(np.sqrt(sr)),
         "edge_rmse_base":float(np.sqrt(edge_base)),
         "edge_rmse_proj":float(np.sqrt(er)),
         "edge_seenY_frac":float(seen.mean()),
         "edge_base_seenY":wrmse(vet[seen],vaw[seen]) if np.any(seen) else None,
         "edge_proj_seenY":wrmse(err[seen],vaw[seen]) if np.any(seen) else None,
         "edge_base_unseenY":wrmse(vet[~seen],vaw[~seen]) if np.any(~seen) else None,
         "edge_proj_unseenY":wrmse(err[~seen],vaw[~seen]) if np.any(~seen) else None,
         "target_fid":float((gp@amp2)**2),
         "f_rms_p":float(np.sqrt(np.sum(p*f*f))),
         "f_absmax":float(np.max(np.abs(f)))}
    print(json.dumps(out,sort_keys=True),flush=True)
    Path(a.out).write_text(json.dumps(out,indent=2))
if __name__=="__main__": main()
