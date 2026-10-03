import json,csv,time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

L=4; N=16; J1=1.0
J2_VALUES=[0.60,0.62,0.64,0.66,0.68,0.70,0.72,0.74,0.76,0.78,0.80]
OUT=Path("/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure/results")
site=lambda x,y:(x%L)+L*(y%L)
NN=[]; NNN=[]
for y in range(L):
    for x in range(L):
        i=site(x,y)
        NN += [(i,site(x+1,y)),(i,site(x,y+1))]
        NNN += [(i,site(x+1,y+1)),(i,site(x+1,y-1))]
basis=np.array([s for s in range(1<<N) if s.bit_count()==N//2],np.uint32)
D=len(basis); idx={int(s):i for i,s in enumerate(basis)}

def build_H(J2):
    rr=[];cc=[];vv=[];diag=np.zeros(D)
    for bi,s0 in enumerate(basis):
        s=int(s0); e=0.
        for bonds,J in ((NN,1.0),(NNN,J2)):
            if J==0: continue
            for u,v in bonds:
                if ((s>>u)&1)==((s>>v)&1): e+=.25*J
                else:
                    e-=.25*J
                    t=s^(1<<u)^(1<<v)
                    rr.append(bi);cc.append(idx[t]);vv.append(.5*J)
        diag[bi]=e
    rr.extend(range(D));cc.extend(range(D));vv.extend(diag)
    return sp.coo_matrix((vv,(rr,cc)),shape=(D,D)).tocsr(),diag

def mask_for(kind):
    if kind=="marshall":
        return sum(1<<site(x,y) for y in range(L) for x in range(L) if (x+y)%2==0)
    if kind=="stripe_x":
        return sum(1<<site(x,y) for y in range(L) for x in range(L) if x%2==0)
    if kind=="stripe_y":
        return sum(1<<site(x,y) for y in range(L) for x in range(L) if y%2==0)
def gauge(kind):
    m=mask_for(kind)
    return np.array([1 if ((int(s)&m).bit_count()%2)==0 else -1 for s in basis],np.int8)

BASE={k:gauge(k) for k in ("marshall","stripe_x","stripe_y")}

def metrics(H,diag,E0,a,truth,s0,s1):
    p=a*a; p/=p.sum()
    O0=float(abs(np.sum(p*s0*truth)))
    O1=float(abs(np.sum(p*s1*truth)))
    psi0=a*s0; psi1=a*s1
    Ebase=float(psi0@(H@psi0)); E1=float(psi1@(H@psi1))
    h=(H@psi1)-diag*psi1
    stable=(s1*h)<=1e-12
    return dict(O_S_0=O0,O_S_1=O1,
                fixed_amp_energy_error0_per_site=(Ebase-E0)/N,
                fixed_amp_energy_error1_per_site=(E1-E0)/N,
                E_fixed_baseline=Ebase,E_fixed_step1=E1,
                P_stable_1=float(np.sum(p*stable)),
                changed_weight=float(np.sum(p*(s1!=s0))),
                wrong_weight_1=float(np.sum(p*(s1!=truth))))

def energy_opt_threshold(H,diag,a,s0,r):
    W=H.copy().tolil();W.setdiag(0);W=W.tocsr()
    W=sp.diags(a)@W@sp.diags(a)
    diagE=float(np.sum(diag*a*a))
    s=(-s0).astype(np.int8)  # t below all r: global baseline flip
    E=diagE+float(s@(W@s))
    h=np.asarray(W@s).ravel()
    order=np.argsort(r)
    bestE=E; best_t=float(r[order[0]]-max(1.0,abs(r[order[0]])*.01)); best_s=s.copy()
    for ii in order:
        old=int(s[ii])
        E += -4.0*old*h[ii]
        s[ii]=-old
        st,en=W.indptr[ii],W.indptr[ii+1]
        js=W.indices[st:en]; ws=W.data[st:en]
        h[js] += -2.0*old*ws
        if E < bestE-1e-13:
            bestE=float(E);best_t=float(r[ii]);best_s=s.copy()
    return best_t,best_s,float(bestE)

def oracle_overlap_threshold(a,truth,s0,r):
    p=a*a;p/=p.sum()
    order=np.argsort(r)
    s=(-s0).astype(np.int8)
    z=float(np.sum(p*s*truth))
    bestO=abs(z);best_t=float(r[order[0]]-1);best_s=s.copy()
    for ii in order:
        old=int(s[ii]);new=-old;s[ii]=new
        z += p[ii]*(new-old)*truth[ii]
        if abs(z)>bestO+1e-15:
            bestO=abs(z);best_t=float(r[ii]);best_s=s.copy()
    return best_t,best_s,float(bestO)

rows=[];prev=None
for J2 in J2_VALUES:
    t0=time.time();H,diag=build_H(J2)
    kw=dict(k=2,which="SA",tol=1e-11,maxiter=50000)
    if prev is not None:kw["v0"]=prev
    ev,V=sla.eigsh(H,**kw);o=np.argsort(ev);ev=ev[o];V=V[:,o]
    vec=np.asarray(V[:,0],float);prev=vec
    if vec[np.argmax(np.abs(vec))]<0:vec=-vec
    E0=float(ev[0]);gap=float(ev[1]-ev[0]);a=np.abs(vec)
    truth=np.where(vec>=0,1,-1).astype(np.int8)
    baselines=["marshall","stripe_x","stripe_y"]
    for b in baselines:
        s0=BASE[b]
        psi0=a*s0
        r=(H@psi0)/np.where(np.abs(psi0)>1e-300,psi0,1e-300)
        te,se,Ee=energy_opt_threshold(H,diag,a,s0,r)
        to,so,Oo=oracle_overlap_threshold(a,truth,s0,r)
        m=metrics(H,diag,E0,a,truth,s0,se)
        mo=metrics(H,diag,E0,a,truth,s0,so)
        row=dict(J2=J2,baseline=b,gap=gap,threshold_energyopt=te,
                 threshold_oracle_overlap=to,oracle_O_S_1=Oo,
                 oracle_fixed_amp_error1_per_site=mo["fixed_amp_energy_error1_per_site"],
                 **m)
        rows.append(row)
        print("ENERGYOPT",J2,b,json.dumps({k:row[k] for k in
          ["O_S_0","O_S_1","oracle_O_S_1","fixed_amp_energy_error0_per_site",
           "fixed_amp_energy_error1_per_site","P_stable_1","changed_weight",
           "threshold_energyopt"]}),"gap",gap,"sec",time.time()-t0,flush=True)

with open(OUT/"square_exact_allgauges_dense.json","w") as f:
    json.dump(dict(method={
      "system":"4x4 periodic square J1-J2, Sz=0, exact ground-state amplitudes",
      "one_step":"s1=s0 sign(t-r0), r0=(H a s0)/(a s0)",
      "threshold_primary":"t chosen by exact minimization of fixed-amplitude variational energy over all r0 thresholds; label-free",
      "oracle":"best hidden-sign-overlap threshold recorded only to diagnose coordinate capacity",
      "P_stable_1":"|psi|^2-weighted single-sign-flip stability after update"
    },results=rows),f,indent=2)
fields=list(rows[0].keys())
with open(OUT/"square_exact_allgauges_dense.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
print("DONE",flush=True)
