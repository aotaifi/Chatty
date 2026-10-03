#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import scipy.sparse.linalg as sla
from finite_tau_matching_20site_exact import D,special_columns
from finite_tau_amplitude_regression_20site import RegMLP,fit,grouped_rmse,endpoint,xrows,H,diag,f1,f2
from finite_tau_learned_guide_gfmc_20site import propagate_fn_uniform_importance

ROOT=Path("/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure")

def load_stage1():
    z=np.load(ROOT/"results/finite_tau_amplitude_regression_20site_weights.npz")
    net=RegMLP(len(z["mu"]),48,1)
    net.W1[:]=z["W1"]; net.b1[:]=z["b1"]; net.W2[:]=z["W2"]; net.b2[:]=z["b2"]; net.W3[:]=z["W3"]; net.b3[:]=z["b3"]
    return net,z["mu"],z["sd"]

def guide(net,mu,sd,y,tau,arr):
    rich=arr[4]
    X=np.column_stack([rich,np.tile([diag[y],f1[y],f2[y]],(D,1)),np.full(D,tau)])
    z=net.pred((X-mu)/sd); z-=z.max(); g=np.exp(np.clip(z,-40,0)); g/=np.linalg.norm(g)
    return g

def collect(ys,M,seed,beta=.75,dt=.05):
    base,mu0,sd0=load_stage1(); rg=np.random.default_rng(seed)
    Xs=[];Ts=[];Ws=[];Gs=[]
    for y in ys:
        arr=endpoint(y); d,_,_,_,rich,vals,rlab,rbs=arr; s=np.where(d%2==0,1,-1).astype(np.int8)
        walkers=np.full(M,int(y),np.int32)
        for n in range(1,int(round(beta/dt))+1):
            tp=(n-1)*dt
            ag=np.full(D,1/np.sqrt(D)) if n==1 else guide(base,mu0,sd0,int(y),tp,arr)
            walkers=propagate_fn_uniform_importance(walkers,ag,s,dt,rg)
            cnt=np.bincount(rlab[walkers],minlength=len(vals)).astype(float)
            keep=cnt>=2
            tar=np.log(np.maximum(cnt[keep],1e-30)/rbs[keep]); ww=cnt[keep]
            tar-=np.average(tar,weights=ww)
            Xs.append(xrows(vals[keep],int(y),n*dt)); Ts.append(tar); Ws.append(ww)
            Gs.extend([(int(y),round(n*dt,8))]*int(np.sum(keep)))
    return np.concatenate(Xs),np.concatenate(Ts),np.concatenate(Ws),np.asarray(Gs,dtype=object)
def eval_model(net,mu,sd,ys,beta=1.0,M=32768,seed=170001,dt=.05):
    rg=np.random.default_rng(seed); rows=[]
    for ky,y in enumerate(ys):
        arr=endpoint(int(y)); d,_,_,_,rich,vals,rlab,rbs=arr; s=np.where(d%2==0,1,-1).astype(np.int8)
        walkers=np.full(M,int(y),np.int32); exact=np.zeros(D); exact[int(y)]=1.
        for n in range(1,int(round(beta/dt))+1):
            tp=(n-1)*dt
            ag=np.full(D,1/np.sqrt(D)) if n==1 else guide(net,mu,sd,int(y),tp,arr)
            walkers=propagate_fn_uniform_importance(walkers,ag,s,dt,rg)
            exact=np.asarray(sla.expm_multiply(-dt*H,exact),float); exact/=np.linalg.norm(exact)
            cnt=np.bincount(rlab[walkers],minlength=len(vals)).astype(float)
            aest=(cnt/rbs)[rlab]; aest/=np.linalg.norm(aest)
            at=np.abs(exact); at/=np.linalg.norm(at)
            F=float(np.dot(aest,at)**2)
            if n in (10,15,20):
                print("EVAL",y,n*dt,F,len(np.unique(walkers)),flush=True)
            rows.append({"y":int(y),"tau":n*dt,"fidelity":F,"unique":int(len(np.unique(walkers)))})
    return rows

def direct_oracle(net,mu,sd,ys):
    out=[]
    for y in ys:
        arr=endpoint(int(y)); e=np.zeros(D); e[int(y)]=1.
        for tau in (.5,.75,1.0):
            ex=np.asarray(sla.expm_multiply(-tau*H,e),float); ex/=np.linalg.norm(ex)
            g=guide(net,mu,sd,int(y),tau,arr); at=np.abs(ex); at/=np.linalg.norm(at)
            out.append({"y":int(y),"tau":tau,"fidelity":float(np.dot(g,at)**2)})
            print("DIRECT",out[-1],flush=True)
    return out
def main():
    rg=np.random.default_rng(20260930); ys=special_columns(rg,2)
    Xa,ta,wa,ga=collect(ys,16384,160001); Xb,tb,wb,gb=collect(ys,16384,160002)
    print("DATA",len(Xa),len(Xb),flush=True)
    net,mu,sd=fit(Xa,ta,wa,epochs=300,seed=23)
    tr=grouped_rmse(net.pred((Xa-mu)/sd),ta,wa,ga); va=grouped_rmse(net.pred((Xb-mu)/sd),tb,wb,gb)
    print("STAGE2_RMSE",tr,va,flush=True)
    direct=direct_oracle(net,mu,sd,ys)
    rows=eval_model(net,mu,sd,ys,beta=1.0,M=32768)
    out={"ys":[int(x) for x in ys],"train_rmse":tr,"val_rmse":va,"direct":direct,"propagation":rows}
    path=ROOT/"results/finite_tau_stage2_amplitude_20site.json"; path.write_text(json.dumps(out,indent=2))
    np.savez_compressed(ROOT/"results/finite_tau_stage2_amplitude_20site_weights.npz",mu=mu,sd=sd,
        W1=net.W1,b1=net.b1,W2=net.W2,b2=net.b2,W3=net.W3,b3=net.b3)
    print("WROTE",path,flush=True)
if __name__=="__main__": main()
