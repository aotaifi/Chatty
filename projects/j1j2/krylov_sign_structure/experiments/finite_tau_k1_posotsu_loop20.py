#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import scipy.sparse.linalg as sla
from finite_tau_matching_20site_exact import special_columns,D
from finite_tau_amplitude_regression_20site import endpoint,H
from finite_tau_learned_guide_gfmc_20site import load_model,guide_vec,propagate_fn_uniform_importance

ROOT=Path("/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure")
def norm(v):v=np.asarray(v,float);return v/np.linalg.norm(v)
def mm(s,v):
    p=v*v;p/=p.sum();return float(np.sum(p[s!=np.where(v>=0,1,-1)]))
def otsu_pos(r,w,use_log=False):
    m=(r>0)&(w>0)
    if m.sum()<4:return np.inf,0.,0.
    x=np.log(r[m]) if use_log else r[m]; ww=w[m].astype(float); ww/=ww.sum()
    o=np.argsort(x);x=x[o];ww=ww[o];cw=np.cumsum(ww);cx=np.cumsum(ww*x);mu=cx[-1]
    good=(cw>1e-5)&(cw<1-1e-5);sep=np.full(len(x),-np.inf)
    sep[good]=cw[good]*(1-cw[good])*(cx[good]/cw[good]-(mu-cx[good])/(1-cw[good]))**2
    k=int(np.argmax(sep));xt=.5*(x[k]+x[min(k+1,len(x)-1)])
    t=float(np.exp(xt) if use_log else xt)
    return t,float(sep[k]),float(np.sum(w[r>t]))

def run(y,M,seed,mode,dt=.05,prior=512.):
    net,mu,sd=load_model();arr=endpoint(y);d,_,_,_,_,vals,rlab,rbs=arr
    s=np.where(d%2==0,1,-1).astype(np.int8);rg=np.random.default_rng(seed)
    walkers=np.full(M,y,np.int32);g=np.full(D,1/np.sqrt(D));ex=np.zeros(D);ex[y]=1.;rows=[]
    for n in range(1,11):
        walkers=propagate_fn_uniform_importance(walkers,g,s,dt,rg);ex=norm(sla.expm_multiply(-dt*H,ex))
        mass=np.bincount(rlab[walkers],minlength=len(vals)).astype(float)
        gm=guide_vec(net,mu,sd,y,n*dt,arr);pm=np.bincount(rlab,weights=gm,minlength=len(vals));pm/=pm.sum()
        g=norm(np.maximum(((mass+prior*pm)/rbs)[rlab],1e-14))
        psi=s*g;safe=np.where(np.abs(psi)>1e-14,psi,np.where(psi>=0,1e-14,-1e-14))
        r=np.asarray(H@psi)/safe;wg=g*g;wg/=wg.sum()
        if mode=="fixed":
            th=1/(1.25*dt);sep=0.;flip=float(np.sum(wg[r>th]))
        else:
            th,sep,flip=otsu_pos(r,wg,use_log=(mode=="log"))
        sn=s.copy();sn[r>th]*=-1
        exn=norm(sla.expm_multiply(-dt*H,ex))
        row={"tau":n*dt,"threshold":th,"kfac":1/(th*dt) if np.isfinite(th) and th>0 else 0.,
             "sep":sep,"flip_mass":flip,"sign_err":mm(s,ex),"next_err":mm(sn,exn)}
        rows.append(row);print(mode,M,seed,row,flush=True);s=sn
    return rows

def main():
    y=int(special_columns(np.random.default_rng(20260930),2)[1]);out={}
    for mode in ("fixed","linear","log"):
        for rep in (0,1):
            key=f"{mode}_r{rep}";out[key]=run(y,8192,120100+100*("fixed","linear","log").index(mode)+rep,mode)
    path=ROOT/"results/finite_tau_k1_posotsu_loop20.json"
    path.write_text(json.dumps({"y":y,"runs":out},indent=2));print("WROTE",path,flush=True)
if __name__=="__main__":main()
