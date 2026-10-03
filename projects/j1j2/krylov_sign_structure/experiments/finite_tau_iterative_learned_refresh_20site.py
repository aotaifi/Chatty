#!/usr/bin/env python3
import json
import numpy as np
import scipy.sparse.linalg as sla
from finite_tau_two_learned_refresh_20site import (
    ROOT,H,D,diag,arrays,feat,build_fn_from_guide,propagate_vec,fit_classifier
)
from finite_tau_matching_20site_exact import special_columns

def collect_from_guides(ys,guides,M,seed0):
    XA=[];YA=[];XB=[];YB=[]
    for k,y in enumerate(ys):
        lg,s,d,n2=guides[int(y)]
        for seed,XX,YY in [(seed0+100+k,XA,YA),(seed0+200+k,XB,YB)]:
            w=propagate_vec(int(y),lg,s,M,seed)
            rg=np.random.default_rng(seed+9000)
            q=np.exp(np.clip(2*lg,-60,0)); q/=q.sum()
            ref=rg.choice(D,size=M,replace=True,p=q)
            XX.append(np.concatenate([feat(d,n2,w),feat(d,n2,ref)]))
            YY.append(np.concatenate([np.ones(M),np.zeros(M)]))
    return np.concatenate(XA),np.concatenate(YA),np.concatenate(XB),np.concatenate(YB)

def exact_scores(ys,guides):
    rows=[]
    for y in ys:
        lg,s,d,n2=guides[int(y)]
        g=np.exp(np.clip(lg-lg.max(),-60,0))
        e=np.zeros(D); e[int(y)]=1.
        F=build_fn_from_guide(s,g)
        a=np.asarray(sla.expm_multiply(-.5*F,e),float); a=np.maximum(a,0); a/=np.linalg.norm(a)
        ex=np.asarray(sla.expm_multiply(-.5*H,e),float); ex/=np.linalg.norm(ex)
        rows.append({"y":int(y),
                     "full_fidelity":float(np.dot(a*s,ex)**2),
                     "amplitude_fidelity":float(np.dot(a,np.abs(ex))**2)})
    return rows

def main():
    ys=special_columns(np.random.default_rng(20260930),2); M=8192
    guides={}
    for y in ys:
        d,n2,lab,bs=arrays(int(y))
        s=np.where((d.astype(np.int64)%2)==0,1,-1).astype(np.int8)
        guides[int(y)]=(np.zeros(D,float),s,d,n2)
    history=[{"iter":0,"rows":exact_scores(ys,guides)}]
    print("ITER0",history[-1]["rows"],flush=True)
    for it in range(1,4):
        XA,YA,XB,YB=collect_from_guides(ys,guides,M,80000+1000*it)
        net,mu,sd,eta,aucA,aucB=fit_classifier(XA,YA,XB,YB,10*it+7)
        new={}
        for y in ys:
            lg,s,d,n2=guides[int(y)]
            F=feat(d,n2,np.arange(D))
            resid=eta*net.pred((F-mu)/sd)
            lg2=lg+resid; lg2-=lg2.max()
            new[int(y)]=(lg2,s,d,n2)
        guides=new
        rows=exact_scores(ys,guides)
        history.append({"iter":it,"train_auc":aucA,"val_auc":aucB,"eta":eta,"rows":rows})
        print("ITER",it,"AUC",aucA,aucB,"eta",eta,"rows",rows,flush=True)
    path=ROOT/"results/finite_tau_iterative_learned_refresh_20site.json"
    path.write_text(json.dumps({"history":history},indent=2)); print("WROTE",path)
if __name__=="__main__":main()
