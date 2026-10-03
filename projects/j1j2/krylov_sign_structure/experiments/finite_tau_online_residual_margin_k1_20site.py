#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import scipy.sparse.linalg as sla

from finite_tau_matching_20site_exact import special_columns,D
from finite_tau_amplitude_regression_20site import endpoint,H
from finite_tau_learned_guide_gfmc_20site import load_model,guide_vec,propagate_fn_uniform_importance
from finite_tau_localratio_mlp_oracle20 import MLP,make_wd,mm
from finite_tau_online_residual_k1_20site import pairs,fit_block,total_guide,norm

ROOT=Path("/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure")

def focus_weights(g,s,src,dst,w,dt=.05,lam=200.,sigma=.5):
    q=1-dt*s*(H@(s*g))/np.maximum(g,1e-300)
    edge=np.minimum(np.abs(q[src]),np.abs(q[dst]))
    return w*(1+lam*np.exp(-(edge/sigma)**2))

def run(y,M=8192,dt=.05,beta=.5,seed=921001):
    arr=endpoint(y);d0=arr[0];s=np.where(d0%2==0,1,-1).astype(np.int8);wd=make_wd(H,[y])[0]
    bnet,bmu,bsd=load_model();ra=np.random.default_rng(seed);rb=np.random.default_rng(seed+1);rt=np.random.default_rng(seed+2)
    wa=np.full(M,y,np.int32);wb=wa.copy();g=np.full(D,1/np.sqrt(D))
    net=MLP(63,64,93);net.W3*=0;net.b3[:]=0
    exact=np.zeros(D);exact[y]=1.;rows=[]

    for n in range(1,int(round(beta/dt))+1):
        wa=propagate_fn_uniform_importance(wa,g,s,dt,ra);wb=propagate_fn_uniform_importance(wb,g,s,dt,rb)
        tau=n*dt;A,B,W,src,dst=pairs(wa,y,wd,tau,4,ra);VA,VB,VW,vsrc,vdst=pairs(wb,y,wd,tau,4,rb)
        g0=guide_vec(bnet,bmu,bsd,y,tau,arr);lg=np.log(np.maximum(g0,1e-300))
        O=lg[dst]-lg[src];VO=lg[vdst]-lg[vsrc]
        basevl=fit_block(net,A,B,O,W,VA,VB,VO,VW,8 if n==1 else 4,rt)
        g=total_guide(net,bnet,bmu,bsd,y,arr,wd,tau)
        FW=focus_weights(g,s,src,dst,W);FVW=focus_weights(g,s,vsrc,vdst,VW)
        focusvl=fit_block(net,A,B,O,FW,VA,VB,VO,FVW,2,rt)
        g=total_guide(net,bnet,bmu,bsd,y,arr,wd,tau)

        exact=norm(sla.expm_multiply(-dt*H,exact));exnext=norm(sla.expm_multiply(-dt*H,exact))
        sn=np.where((s*g-dt*(H@(s*g)))>=0,1,-1).astype(np.int8)
        row={"tau":tau,"base_val":basevl,"focus_val":focusvl,
             "ampfid":float(np.dot(g,np.abs(exact))**2),
             "sign_err_current":mm(s,exact),"carry_next":mm(s,exnext),"k1_next":mm(sn,exnext)}
        rows.append(row);print("ONLINE_FOCUS",row,flush=True);s=sn
    return rows

def main():
    y=int(special_columns(np.random.default_rng(20260930),2)[1]);rows=run(y)
    path=ROOT/"results/finite_tau_online_residual_margin_k1_20site.json"
    path.write_text(json.dumps({"y":y,"rows":rows},indent=2));print("WROTE",path,flush=True)

if __name__=="__main__":main()
