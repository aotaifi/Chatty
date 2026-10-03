#!/usr/bin/env python3
import numpy as np, json
from pathlib import Path
from finite_tau_learned_amplitude_20site import collect_replica,endpoint_arrays,feat,H,D,special_columns,ROOT
import scipy.sparse.linalg as sla

def keyrows(X):
    # exact discrete physical feature key except tau
    A=np.rint(np.column_stack([X[:,0],X[:,1],8*X[:,2],X[:,3],X[:,4]])).astype(np.int64)
    return [tuple(r) for r in A]

def fit_table(X,y,w,tau,pc=.5):
    m=np.isclose(X[:,5],tau); X=X[m]; y=y[m]; w=w[m]
    kp=keyrows(X[y==1]); wp=w[y==1]
    kn=keyrows(X[y==0]); wn=w[y==0]
    pos={}; neg={}
    for k,z in zip(kp,wp): pos[k]=pos.get(k,0.)+float(z)
    for k,z in zip(kn,wn): neg[k]=neg.get(k,0.)+float(z)
    keys=set(pos)|set(neg); K=max(1,len(keys))
    sp=sum(pos.values()); sn=sum(neg.values())
    out={}
    for k in keys:
        pp=(pos.get(k,0.)+pc)/(sp+pc*K)
        pn=(neg.get(k,0.)+pc)/(sn+pc*K)
        out[k]=np.log(pp/pn)
    default=np.log((pc/(sp+pc*K))/(pc/(sn+pc*K)))
    return out,default

def score_table(tab,default,F):
    return np.array([tab.get(k,default) for k in keyrows(F)],float)

def main():
    rg=np.random.default_rng(20260930); ys=special_columns(rg,2)
    Xa,ya,wa,oa=collect_replica(ys,8192,41001)
    rows=[]
    for tau in np.arange(.05,.5001,.05):
        tab,default=fit_table(Xa,ya,wa,float(tau),pc=.5)
        for y in ys:
            arrs,exact=oa[(int(y),round(float(tau),8))]
            F=feat(arrs,np.arange(D),float(tau))
            la=score_table(tab,default,F); la-=la.max()
            a=np.exp(np.clip(la,-60,0)); a/=np.linalg.norm(a)
            at=np.abs(exact); at/=np.linalg.norm(at)
            fid=float(np.dot(a,at)**2)
            rows.append({"tau":float(tau),"y":int(y),"fidelity":fid,"nkeys":len(tab)})
            if abs(tau-.25)<1e-8 or abs(tau-.5)<1e-8: print(rows[-1],flush=True)
    out={"ys":[int(x) for x in ys],"rows":rows}
    path=ROOT/"results/finite_tau_direct_ratio_20site.json"
    path.write_text(json.dumps(out,indent=2)); print("WROTE",path)
if __name__=="__main__":main()
