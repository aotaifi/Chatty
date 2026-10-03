#!/usr/bin/env python3
import json, numpy as np, scipy.sparse.linalg as sla
from finite_tau_matching_20site_exact import D,special_columns
from finite_tau_ct_local_ratio_20site import H,normalized
from finite_tau_localratio_mlp_oracle20 import make_wd
from finite_tau_ct_smc_pairratio_longbeta20 import fit_pair,exact_ratio_diag

def main():
    y=int(special_columns(np.random.default_rng(20260930),2)[1])
    wd=make_wd(H,[y])[0]
    e=np.zeros(D);e[y]=1.0
    out=[]
    for tau in (1.0,2.0,3.0):
        ex=normalized(sla.expm_multiply(-tau*H,e))
        a=np.abs(ex); p=a/a.sum()
        rg=np.random.default_rng(881000+int(100*tau))
        M=100000
        st1=rg.choice(D,size=M,p=p).astype(np.int32)
        st2=rg.choice(D,size=M,p=p).astype(np.int32)
        logg=np.zeros(D)
        net,mu,sd,eta,pd=fit_pair(st1,st2,logg,y,wd,tau,990000+int(100*tau),
                                  epochs=4,nfit=M,K=4)
        # evaluate full learned update eta=1 as well as selected eta
        od_sel=exact_ratio_diag(net,mu,sd,eta,st2,logg,ex,y,wd,tau,
                                991000+int(100*tau),nfit=10000,K=4,prefix="sel")
        od_one=exact_ratio_diag(net,mu,sd,1.0,st2,logg,ex,y,wd,tau,
                                992000+int(100*tau),nfit=10000,K=4,prefix="one")
        row={"tau":tau,"eta":eta,**pd,**od_sel,**od_one}
        out.append(row); print(json.dumps(row,sort_keys=True),flush=True)
    open("../results/pairratio_oracle_capacity20.json","w").write(json.dumps(out,indent=2))
if __name__=="__main__": main()
