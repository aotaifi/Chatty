#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
from finite_tau_matching_20site_exact import special_columns
from finite_tau_margin_ratio_walkers20 import (
    ROOT,collect_meta,load_shared,margin_weights,refine
)
from finite_tau_shared_ratio_walkers20 import oracle_diagnostics,sampled_replay

def main():
    y=int(special_columns(np.random.default_rng(20260930),2)[1]);M=8192
    net,mu,sd=load_shared()
    A,B,W,T,S,Z,wd,s0=collect_meta(y,M,701001)
    VA,VB,VW,VT,VS,VZ,_,_=collect_meta(y,M,702001)
    MW=margin_weights(net,mu,sd,y,wd,s0,T,S,Z,W,lam=200.,sigma=.5)
    MVW=margin_weights(net,mu,sd,y,wd,s0,VT,VS,VZ,VW,lam=200.,sigma=.5)
    bestv,hist=refine(net,mu,sd,A,B,MW,VA,VB,MVW,epochs=12,seed=51)
    diag=oracle_diagnostics(net,mu,sd,y,wd,s0)
    for r in diag:
        if r["tau"] in (.1,.25,.5):print("ORACLE",r,flush=True)
    replay=sampled_replay(net,mu,sd,y,M,703001)
    out={"y":y,"M":M,"lambda":200.,"sigma":.5,"best_val":bestv,
         "history":hist,"oracle_diagnostics":diag,"sampled_replay":replay}
    path=ROOT/"results/finite_tau_strong_margin_ratio20.json"
    path.write_text(json.dumps(out,indent=2))
    print("WROTE",path,flush=True)

if __name__=="__main__":main()
