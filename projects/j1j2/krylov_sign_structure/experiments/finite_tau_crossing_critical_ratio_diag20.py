#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import scipy.sparse.linalg as sla

from finite_tau_matching_20site_exact import special_columns,D
from finite_tau_amplitude_regression_20site import H
from finite_tau_localratio_mlp_oracle20 import MLP,feat,make_wd

ROOT=Path("/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure")
HO=H.copy(); HO.setdiag(0); HO.eliminate_zeros()

def norm(v):
    v=np.asarray(v,float); return v/np.linalg.norm(v)

def load_shared():
    z=np.load(ROOT/"results/finite_tau_shared_ratio_walkers20_weights.npz")
    net=MLP(len(z["mu"]),64,17)
    net.W1[:]=z["W1"]; net.b1[:]=z["b1"]
    net.W2[:]=z["W2"]; net.b2[:]=z["b2"]
    net.W3[:]=z["W3"]; net.b3[:]=z["b3"]
    return net,z["mu"],z["sd"]

def full_logamp(net,mu,sd,y,wd,tau):
    out=[]
    for q in range(0,D,4096):
        ix=np.arange(q,min(q+4096,D))
        out.append(net.pred((feat(ix,y,wd,tau)-mu)/sd))
    f=np.concatenate(out)
    return f-f.max()

def recursive_signs(y,tau,dt=.05):
    wd=make_wd(H,[y])[0]
    d=(wd//1000).astype(np.int64)
    s=np.where(d%2==0,1,-1)
    psi=np.zeros(D); psi[y]=1.
    n=int(round(tau/dt))
    for k in range(1,n+1):
        psi=norm(sla.expm_multiply(-dt*H,psi))
        if k<n:
            a=np.abs(psi)
            pred=s*a-dt*(H@(s*a))
            s=np.where(pred>=0,1,-1)
    return psi,s,wd

def one_tau(net,mu,sd,y,tau,dt=.05):
    psi,s,wd=recursive_signs(y,tau,dt)
    a=np.abs(psi); lp=full_logamp(net,mu,sd,y,wd,tau)

    phi=s*a
    margin=phi-dt*(H@phi)
    scale=np.abs(phi)+dt*np.asarray(np.abs(H)@a)
    rel=np.abs(margin)/np.maximum(scale,1e-30)

    p=a*a; p/=p.sum()
    cuts=np.quantile(rel[p>1e-12],[.01,.05,.10,.50])
    rows=[]

    # directed physical edges
    src=[]; dst=[]
    for i in range(D):
        lo,hi=HO.indptr[i],HO.indptr[i+1]
        if hi>lo:
            src.extend([i]*(hi-lo)); dst.extend(HO.indices[lo:hi].tolist())
    src=np.asarray(src,np.int32); dst=np.asarray(dst,np.int32)

    exact_lr=np.log(np.maximum(a[dst],1e-300))-np.log(np.maximum(a[src],1e-300))
    pred_lr=lp[dst]-lp[src]
    e2=(pred_lr-exact_lr)**2
    ew=p[src]

    for label,mask_state in [
        ("critical_1pct", rel<=cuts[0]),
        ("critical_5pct", rel<=cuts[1]),
        ("critical_10pct", rel<=cuts[2]),
        ("bulk_50plus", rel>=cuts[3]),
    ]:
        m=mask_state[src] & (a[src]>1e-15) & (a[dst]>1e-15)
        ww=ew[m]
        rows.append({
            "group":label,
            "nstates":int(mask_state.sum()),
            "nedges":int(m.sum()),
            "state_weight":float(p[mask_state].sum()),
            "ratio_rmse":float(np.sqrt(np.average(e2[m],weights=ww))) if m.any() else None,
            "ratio_mae":float(np.average(np.abs(pred_lr[m]-exact_lr[m]),weights=ww)) if m.any() else None,
            "median_rel_margin":float(np.median(rel[mask_state])) if mask_state.any() else None,
        })

    true_next=norm(sla.expm_multiply(-dt*H,psi))
    true_sign=np.where(true_next>=0,1,-1)
    pred_model=s*np.exp(lp)
    k1_model=np.where((pred_model-dt*(H@pred_model))>=0,1,-1)
    err=(k1_model!=true_sign)
    rows.append({
        "group":"wrong_k1_states",
        "nstates":int(err.sum()),
        "state_weight":float((true_next*true_next)[err].sum()),
        "median_rel_margin":float(np.median(rel[err])) if err.any() else None,
        "p90_rel_margin":float(np.quantile(rel[err],.9)) if err.any() else None,
    })
    return {"tau":tau,"cuts":cuts.tolist(),"rows":rows}

def main():
    y=int(special_columns(np.random.default_rng(20260930),2)[1])
    net,mu,sd=load_shared()
    out={"y":y,"diagnostics":[]}
    for tau in (.25,.40,.50):
        r=one_tau(net,mu,sd,y,tau)
        out["diagnostics"].append(r)
        print("TAU",tau,"cuts",r["cuts"],flush=True)
        for z in r["rows"]: print("ROW",z,flush=True)
    path=ROOT/"results/finite_tau_crossing_critical_ratio_diag20.json"
    path.write_text(json.dumps(out,indent=2))
    print("WROTE",path,flush=True)

if __name__=="__main__": main()
