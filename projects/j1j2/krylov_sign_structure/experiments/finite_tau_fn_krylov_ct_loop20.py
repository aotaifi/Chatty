#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import scipy.sparse.csgraph as cs

from finite_tau_matching_20site_exact import build_H,D,special_columns

ROOT=Path("/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure")
H,diag=build_H(.5)
HO=(H-sp.diags(diag)).tocsr()
CO=H.tocoo()
OFF=CO.row!=CO.col
R=CO.row[OFF].astype(np.int32)
C=CO.col[OFF].astype(np.int32)
HV=CO.data[OFF]
IDX=np.arange(D,dtype=np.int32)

def distance_sign(y):
    A=HO.copy(); A.data=np.ones_like(A.data)
    d=np.asarray(cs.shortest_path(A,directed=False,unweighted=True,indices=[int(y)]))[0]
    return np.where((d.astype(np.int64)%2)==0,1.0,-1.0)

def build_fn(s,g):
    prod=s[R]*s[C]
    good=prod<0
    floor=1e-8/np.sqrt(D)
    gg=np.maximum(np.asarray(g,float),floor)
    shift=np.bincount(R[~good],weights=HV[~good]*gg[C[~good]]/gg[R[~good]],minlength=D)
    rr=np.concatenate([R[good],IDX])
    cc=np.concatenate([C[good],IDX])
    vv=np.concatenate([HV[good]*prod[good],diag+shift])
    F=sp.coo_matrix((vv,(rr,cc)),shape=H.shape).tocsr()
    return F,~good

def residual_metrics(s,g,a,bad):
    gg=np.maximum(np.asarray(g,float),1e-14)
    term=np.bincount(R[bad],weights=HV[bad]*a[C[bad]],minlength=D)
    comp=np.bincount(R[bad],weights=HV[bad]*gg[C[bad]]/gg[R[bad]],minlength=D)
    ca=term-a*comp
    mask=a>1e-14*np.max(a)
    eps=np.zeros(D); eps[mask]=ca[mask]/a[mask]
    p=a*a; p/=p.sum()
    return {
        "residual_l2":float(np.linalg.norm(ca)),
        "weighted_abs_epsilon":float(np.sum(p*np.abs(eps))),
        "weighted_rms_epsilon":float(np.sqrt(np.sum(p*eps*eps))),
    }

def normalized(v):
    v=np.asarray(v,float)
    n=np.linalg.norm(v)
    return v/n if n else v

def mismatch_mass(s,psi):
    p=psi*psi; p/=p.sum()
    st=np.where(psi>=0,1.0,-1.0)
    return float(np.sum(p[s!=st]))

def epsilon_scan(y,dt=.05):
    e=np.zeros(D); e[int(y)]=1.0
    rows=[]
    prev=None
    for tau in (0.10,0.25,0.50):
        psi=normalized(sla.expm_multiply(-tau*H,e))
        a=np.abs(psi); s=np.where(psi>=0,1.0,-1.0)
        guides={"uniform":np.ones(D)/np.sqrt(D),"exact":a.copy()}
        if tau>=dt:
            prev=normalized(np.abs(sla.expm_multiply(-(tau-dt)*H,e)))
            guides["lag_dt"]=prev
        exact_next=normalized(sla.expm_multiply(-dt*H,psi))
        htilde_a=s*(H@(s*a))
        prod=s[R]*s[C]; bad=prod>0
        for name,g in guides.items():
            met=residual_metrics(s,g,a,bad)
            gg=np.maximum(np.asarray(g,float),1e-14)
            term=np.bincount(R[bad],weights=HV[bad]*a[C[bad]],minlength=D)
            comp=np.bincount(R[bad],weights=HV[bad]*gg[C[bad]]/gg[R[bad]],minlength=D)
            ca=term-a*comp
            ae=normalized(a+dt*(-htilde_a+ca))
            approx=s*ae
            met.update({"tau":tau,"guide":name,
                        "euler_next_full_fidelity":float(np.dot(approx,exact_next)**2)})
            rows.append(met)
    return rows

def k1_sign_track(y,dt=.05,beta=1.0):
    e=np.zeros(D); e[int(y)]=1.0
    psi=normalized(sla.expm_multiply(-dt*H,e))
    sfix=distance_sign(y)
    sad=sfix.copy()
    rows=[{"tau":dt,"fixed":mismatch_mass(sfix,psi),
           "k1":mismatch_mass(sad,psi)}]
    n=int(round(beta/dt))
    for k in range(1,n):
        a=np.abs(psi)
        phi=sad*a
        pred=phi-dt*(H@phi)
        sn=np.where(pred>=0,1.0,-1.0)
        psi=normalized(sla.expm_multiply(-dt*H,psi))
        rows.append({"tau":(k+1)*dt,
                     "fixed":mismatch_mass(sfix,psi),
                     "k1":mismatch_mass(sn,psi),
                     "flip_mass":float(np.sum((a*a)[sn!=sad]))})
        sad=sn
    return rows

def closed_loop(y,dt=.05,beta=1.0,use_k1=True):
    e=np.zeros(D); e[int(y)]=1.0
    exact=e.copy(); a=e.copy()
    s=distance_sign(y); g=np.ones(D)/np.sqrt(D)
    rows=[]
    for k in range(int(round(beta/dt))):
        F,bad=build_fn(s,g)
        a=normalized(np.maximum(sla.expm_multiply(-dt*F,a),0.0))
        exact=normalized(sla.expm_multiply(-dt*H,exact))
        approx=s*a

        row={"tau":(k+1)*dt,
             "amplitude_fidelity":float(np.dot(a,np.abs(exact))**2),
             "full_fidelity":float(np.dot(approx,exact)**2),
             "sign_mismatch_mass":mismatch_mass(s,exact)}
        rows.append(row)
        g=np.maximum(a,1e-14); g=normalized(g)
        if use_k1:
            pred=approx-dt*(H@approx)
            sn=np.where(pred>=0,1.0,-1.0)
            row["k1_flip_mass"]=float(np.sum((a*a)[sn!=s]))
            s=sn
    return rows

def main():
    ys=special_columns(np.random.default_rng(20260930),2)
    y=int(ys[1])
    print("HARD_COLUMN",y,flush=True)
    eps=epsilon_scan(y)
    for r in eps: print("EPS",r,flush=True)
    kt=k1_sign_track(y)
    for r in kt: print("K1",r,flush=True)
    fixed=closed_loop(y,use_k1=False)
    adap=closed_loop(y,use_k1=True)
    for a,b in zip(fixed,adap):
        print("LOOP",{"tau":a["tau"],
              "fixed_full":a["full_fidelity"],
              "k1_full":b["full_fidelity"],
              "fixed_sign":a["sign_mismatch_mass"],
              "k1_sign":b["sign_mismatch_mass"],
              "k1_flip_mass":b.get("k1_flip_mass",0.0)},flush=True)
    out={"y":y,"epsilon_scan":eps,"k1_sign_track":kt,
         "fixed_loop":fixed,"k1_loop":adap}
    path=ROOT/"results/finite_tau_fn_krylov_ct_loop20.json"
    path.write_text(json.dumps(out,indent=2))
    print("WROTE",path,flush=True)

if __name__=="__main__":
    main()
