#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import scipy.sparse.linalg as sla
from finite_tau_matching_20site_exact import special_columns,D
from finite_tau_amplitude_regression_20site import endpoint,H
from finite_tau_learned_guide_gfmc_20site import load_model,guide_vec,propagate_fn_uniform_importance

ROOT=Path("/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure")
def norm(v): v=np.asarray(v,float);return v/np.linalg.norm(v)
def mm(s,v):
    p=v*v;p/=p.sum();return float(np.sum(p[s!=np.where(v>=0,1,-1)]))

def run(y,kfac,M=8192,seed=1,dt=.05,prior=512.):
    net,mu,sd=load_model();arr=endpoint(y);d,_,_,_,_,vals,rlab,rbs=arr
    s=np.where(d%2==0,1,-1).astype(np.int8);rg=np.random.default_rng(seed)
    w=np.full(M,y,np.int32);g=np.full(D,1/np.sqrt(D));ex=np.zeros(D);ex[y]=1.;rows=[]
    for n in range(1,11):
        w=propagate_fn_uniform_importance(w,g,s,dt,rg);ex=norm(sla.expm_multiply(-dt*H,ex))
        mass=np.bincount(rlab[w],minlength=len(vals)).astype(float)
        gm=guide_vec(net,mu,sd,y,n*dt,arr);pm=np.bincount(rlab,weights=gm,minlength=len(vals));pm/=pm.sum()
        g=norm(np.maximum(((mass+prior*pm)/rbs)[rlab],1e-14))
        psi=s*g;sn=np.where((psi-kfac*dt*(H@psi))>=0,1,-1).astype(np.int8)
        rows.append({"tau":n*dt,"sign_err":mm(s,ex),"next_err":mm(sn,norm(sla.expm_multiply(-dt*H,ex)))})
        s=sn
    return rows

def main():
    y=int(special_columns(np.random.default_rng(20260930),2)[1]);out={}
    for q,k in enumerate((.5,.75,1.,1.25,1.5,2.)):
        rows=run(y,k,8192,98000+q)
        out[str(k)]=rows
        print("KFAC",k,"tau05",rows[-1]["sign_err"],"next055",rows[-1]["next_err"],flush=True)
    path=ROOT/"results/finite_tau_k1_stepsize_sweep20.json"
    path.write_text(json.dumps({"y":y,"runs":out},indent=2));print("WROTE",path,flush=True)
if __name__=="__main__":main()
