#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import scipy.sparse.linalg as sla

from finite_tau_matching_20site_exact import special_columns,D
from finite_tau_uniform_one_refresh_20site import H,arrays,feat,build_fn_from_guide
from finite_tau_iterative_learned_refresh_20site import collect_from_guides
from finite_tau_two_learned_refresh_20site import fit_classifier

ROOT=Path("/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure")

def norm(v):
    v=np.asarray(v,float);return v/np.linalg.norm(v)

def mm(s,psi):
    p=psi*psi;p/=p.sum();return float(np.sum(p[s!=np.where(psi>=0,1,-1)]))

def score(y,lg,s,dt=.05):
    g=norm(np.exp(np.clip(lg-lg.max(),-60,0)))
    e=np.zeros(D);e[y]=1.
    ex=norm(sla.expm_multiply(-.5*H,e));exn=norm(sla.expm_multiply(-dt*H,ex))
    F=build_fn_from_guide(s,g)
    afn=np.asarray(sla.expm_multiply(-.5*F,e),float);afn=norm(np.maximum(afn,0))
    sg=np.where((s*g-dt*(H@(s*g)))>=0,1,-1)
    sf=np.where((s*afn-dt*(H@(s*afn)))>=0,1,-1)
    return {"guide_fid":float(np.dot(g,np.abs(ex))**2),
            "fn_amp_fid":float(np.dot(afn,np.abs(ex))**2),
            "guide_k1_next_err":mm(sg,exn),"fn_k1_next_err":mm(sf,exn),
            "carry_next_err":mm(s,exn)}

def main():
    y=int(special_columns(np.random.default_rng(20260930),2)[1]);ys=np.array([y]);M=8192
    d,n2,lab,bs=arrays(y);s=np.where((d.astype(np.int64)%2)==0,1,-1).astype(np.int8)
    guides={y:(np.zeros(D,float),s,d,n2)};hist=[]
    base=score(y,*guides[y][:2]);base["iter"]=0;hist.append(base);print("ITER",base,flush=True)
    for it in range(1,4):
        XA,YA,XB,YB=collect_from_guides(ys,guides,M,95000+1000*it)
        net,mu,sd,eta,aucA,aucB=fit_classifier(XA,YA,XB,YB,100+it)
        lg,ss,dd,nn=guides[y];F=feat(dd,nn,np.arange(D))
        resid=eta*net.pred((F-mu)/sd);lg2=lg+resid;lg2-=lg2.max()
        guides={y:(lg2,ss,dd,nn)}
        row=score(y,lg2,ss);row.update({"iter":it,"auc_train":aucA,"auc_val":aucB,"eta":eta})
        hist.append(row);print("ITER",row,flush=True)
    path=ROOT/"results/finite_tau_densityratio_k1_eval20.json"
    path.write_text(json.dumps({"y":y,"history":hist},indent=2));print("WROTE",path,flush=True)

if __name__=="__main__":main()
