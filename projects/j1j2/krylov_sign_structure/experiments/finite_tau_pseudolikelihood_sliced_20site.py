#!/usr/bin/env python3
import json
import numpy as np
from pathlib import Path
from finite_tau_pseudolikelihood_20site import (
    ROOT,D,special_columns,collect_pairs,train,ploss,exact_ratio_rmse,feat
)

def one_tau(tau,A1,B1,w1,m1,A2,B2,w2,m2,orc,amap,ys):
    mt=np.array([abs(z[1]-tau)<1e-9 for z in m1])
    mv=np.array([abs(z[1]-tau)<1e-9 for z in m2])
    net,mu,sd=train(A1[mt],B1[mt],w1[mt],epochs=35,batch=2048,seed=int(7000+100*tau))
    tr=ploss(net,(A1[mt]-mu)/sd,(B1[mt]-mu)/sd,w1[mt])
    va=ploss(net,(A2[mv]-mu)/sd,(B2[mv]-mu)/sd,w2[mv])
    er=exact_ratio_rmse(net,mu,sd,A2[mv],B2[mv],w2[mv],[m2[i] for i in np.flatnonzero(mv)],orc)
    fs=[]
    for y0 in ys:
        y=int(y0); arrs=amap[y]
        X=feat(arrs,np.arange(D),tau,y); z=net.pred((X-mu)/sd); z-=z.max()
        a=np.exp(np.clip(z,-60,0)); a/=np.linalg.norm(a)
        ex=np.abs(orc[(y,round(tau,8))]); ex/=np.linalg.norm(ex)
        fs.append(float(np.dot(a,ex)**2))
    row={"tau":tau,"ntrain":int(mt.sum()),"nval":int(mv.sum()),"train_ploss":tr,"val_ploss":va,
         "exact_ratio_rmse":er[0],"exact_ratio_corr":er[1],"fidelity":fs}
    print("SLICED",row,flush=True); return row

def main():
    rg=np.random.default_rng(20260930); ys=special_columns(rg,2)
    A1,B1,w1,m1,_,_=collect_pairs(ys,4096,71001,K=6,oracle=False)
    A2,B2,w2,m2,orc,amap=collect_pairs(ys,4096,72001,K=6,oracle=True)
    rows=[one_tau(t,A1,B1,w1,m1,A2,B2,w2,m2,orc,amap,ys) for t in (.25,.5)]
    out={"ys":[int(x) for x in ys],"rows":rows}
    path=ROOT/"results/finite_tau_pseudolikelihood_sliced_20site.json"
    path.write_text(json.dumps(out,indent=2)); print("WROTE",path,flush=True)
if __name__=="__main__": main()
