#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from finite_tau_matching_20site_exact import build_H,D,special_columns
from finite_tau_amplitude_regression_20site import RegMLP,endpoint,xrows,diag,f1,f2

ROOT=Path("/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure")
H,diagH=build_H(.5)

def load_model():
    z=np.load(ROOT/"results/finite_tau_amplitude_regression_20site_weights.npz")
    net=RegMLP(len(z["mu"]),48,1)
    net.W1[:]=z["W1"]; net.b1[:]=z["b1"]; net.W2[:]=z["W2"]; net.b2[:]=z["b2"]
    net.W3[:]=z["W3"]; net.b3[:]=z["b3"]
    return net,z["mu"],z["sd"]

def build_fn(H,diag,guide,s):
    co=H.tocoo(); off=co.row!=co.col
    r=co.row[off]; c=co.col[off]; h=co.data[off]
    prod=s[r]*s[c]; keep=prod<0
    g=np.maximum(guide,1e-14)
    shift=np.bincount(r[~keep],weights=h[~keep]*g[c[~keep]]/g[r[~keep]],minlength=H.shape[0])
    rr=np.concatenate([r[keep],np.arange(H.shape[0])])
    cc=np.concatenate([c[keep],np.arange(H.shape[0])])
    vv=np.concatenate([h[keep]*prod[keep],diag+shift])
    return sp.coo_matrix((vv,(rr,cc)),shape=H.shape).tocsr()

def guide_vec(net,mu,sd,y,tau):
    arr=endpoint(y); rich=arr[4]
    X=np.column_stack([rich,np.tile([diag[y],f1[y],f2[y]],(D,1)),np.full(D,tau)])
    z=net.pred((X-mu)/sd); z-=z.max()
    g=np.exp(np.clip(z,-60,0)); g/=np.linalg.norm(g)
    return g,arr

def run(y,dt=.05,beta=.5):
    net,mu,sd=load_model()
    g0,arr=guide_vec(net,mu,sd,y,.05)
    d=arr[0]; s=np.where(d%2==0,1,-1).astype(np.int8)
    a=np.zeros(D); a[y]=1.; exact=a.copy(); rec=[]
    for n in range(1,int(round(beta/dt))+1):
        tau_prev=(n-1)*dt
        if n==1: guide=np.full(D,1/np.sqrt(D))
        else: guide,_=guide_vec(net,mu,sd,y,tau_prev)
        F=build_fn(H,diagH,guide,s)
        a=np.asarray(sla.expm_multiply(-dt*F,a)); a=np.maximum(a,0); a/=np.linalg.norm(a)
        exact=np.asarray(sla.expm_multiply(-dt*H,exact)); exact/=np.linalg.norm(exact)
        full=float(np.dot(a*s,exact)**2); amp=float(np.dot(a,np.abs(exact))**2)
        row={"tau":n*dt,"full_fidelity":full,"amplitude_fidelity":amp}
        rec.append(row); print("Y",y,row,flush=True)
    return rec

def main():
    rg=np.random.default_rng(20260930); ys=special_columns(rg,2)
    out={"ys":ys,"columns":{}}
    for y in ys: out["columns"][str(y)]=run(int(y))
    path=ROOT/"results/finite_tau_learned_fn_exact20.json"; path.write_text(json.dumps(out,indent=2))
    print("WROTE",path,flush=True)
if __name__=="__main__": main()
