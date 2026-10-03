#!/usr/bin/env python3
import argparse,json,numpy as np
from finite_tau_matching_20site_exact import D,special_columns
from finite_tau_ct_local_ratio_20site import H,distance_sign,normalized
from finite_tau_ct_smc_fn20 import ct_block
from finite_tau_ct_smc_edgeratio_longbeta20 import ratio_params,reconstruct,exact_sequence

DT=.05
HO=H.copy();HO.setdiag(0);HO.eliminate_zeros()

def expand(S):
    out=set(int(x) for x in S)
    for x in list(out):
        a,b=HO.indptr[x],HO.indptr[x+1]
        out.update(int(z) for z in HO.indices[a:b])
    return np.fromiter(out,dtype=np.int32)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--M",type=int,default=25000);a=ap.parse_args()
    y=int(special_columns(np.random.default_rng(20260930),2)[1]); s=distance_sign(y)
    exact=exact_sequence(y,.55); st1,st2,logg=reconstruct(y,s,a.M)
    g=np.exp(np.clip(logg-logg.max(),-700,0)); psi=s*g
    h1=H@psi; psi2=psi-DT*h1+.5*DT*DT*(H@h1); sn=np.where(psi2>=0,1.,-1.)
    J,lam,pot=ratio_params(s,logg)
    st1,_,_=ct_block(st1,DT,J,lam,pot,np.random.default_rng(1310001))
    st2,_,_=ct_block(st2,DT,J,lam,pot,np.random.default_rng(1320001))
    ex=exact[.55]; k2=normalized(psi2); target=np.abs(psi2)
    S=np.unique(np.concatenate([st1,st2]))
    rows=[]
    for hops in (0,1,2):
        if hops>0:S=expand(S)
        gn=g.copy();gn[S]=target[S];gn=normalized(gn)
        rows.append({"hops":hops,"covered":int(len(S)),
                     "physical_mass":float(np.sum(ex[S]**2)),
                     "k2_mass":float(np.sum(k2[S]**2)),
                     "full":float(np.dot(sn*gn,ex)**2),
                     "amp":float(np.dot(gn,np.abs(ex))**2),
                     "to_k2":float(np.dot(gn,np.abs(k2))**2)})
    print(json.dumps({"M":a.M,"k2_full":float(np.dot(k2,ex)**2),"rows":rows},indent=2),flush=True)
if __name__=="__main__":main()
