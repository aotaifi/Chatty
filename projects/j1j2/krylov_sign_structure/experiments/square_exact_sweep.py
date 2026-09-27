import json, math, csv, time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

L=4
N=L*L
J1=1.0
ALPHA_THRESH=0.8
J2_VALUES=[0.0,0.2,0.4,0.5,0.6,0.8,1.0]
OUT=Path("/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure/results")
OUT.mkdir(parents=True,exist_ok=True)

def site(x,y): return (x%L)+L*(y%L)

NN=[]
NNN=[]
for y in range(L):
    for x in range(L):
        i=site(x,y)
        NN += [(i,site(x+1,y)),(i,site(x,y+1))]
        NNN += [(i,site(x+1,y+1)),(i,site(x+1,y-1))]

basis=np.array([s for s in range(1<<N) if s.bit_count()==N//2],dtype=np.uint32)
D=len(basis)
index={int(s):i for i,s in enumerate(basis)}
print("BASIS",D,"NN",len(NN),"NNN",len(NNN),flush=True)

def build_H(J2):
    rows=[]; cols=[]; vals=[]
    diag=np.zeros(D,float)
    for bi,s0 in enumerate(basis):
        s=int(s0)
        e=0.0
        for bonds,J in ((NN,J1),(NNN,J2)):
            if J==0: continue
            for u,v in bonds:
                bu=(s>>u)&1; bv=(s>>v)&1
                if bu==bv:
                    e += 0.25*J
                else:
                    e -= 0.25*J
                    t=s^(1<<u)^(1<<v)
                    rows.append(bi); cols.append(index[t]); vals.append(0.5*J)
        diag[bi]=e
    rows.extend(range(D)); cols.extend(range(D)); vals.extend(diag.tolist())
    H=sp.coo_matrix((vals,(rows,cols)),shape=(D,D)).tocsr()
    return H,diag

A_checker=sum(1<<site(x,y) for y in range(L) for x in range(L) if (x+y)%2==0)
A_x=sum(1<<site(x,y) for y in range(L) for x in range(L) if x%2==0)
A_y=sum(1<<site(x,y) for y in range(L) for x in range(L) if y%2==0)

def gauge_sign(mask):
    return np.array([1 if ((int(s)&mask).bit_count()%2)==0 else -1 for s in basis],dtype=np.int8)

BASELINES={
    "marshall":gauge_sign(A_checker),
    "stripe_x":gauge_sign(A_x),
    "stripe_y":gauge_sign(A_y),
}

def weighted_quantile(x,w,q):
    idx=np.argsort(x)
    xs=np.asarray(x)[idx]; ws=np.asarray(w)[idx]
    c=np.cumsum(ws)
    if c[-1]<=0: return float(np.quantile(xs,q))
    c=c/c[-1]
    return float(np.interp(q,c,xs))

def choose_threshold(r,a,alpha=ALPHA_THRESH):
    # Exact analog of sampling q(x) ∝ |psi|^alpha, robustly trimming 10%-90%.
    w=np.power(np.maximum(a,1e-300),alpha)
    w=w/w.sum()
    lo=weighted_quantile(r,w,0.10)
    hi=weighted_quantile(r,w,0.90)
    keep=(r>=lo)&(r<=hi)
    rr=r[keep]; ww=w[keep]
    ww=ww/ww.sum()
    if len(rr)<2 or np.sqrt(np.sum(ww*(rr-np.sum(ww*rr))**2))<1e-10:
        return float(np.sum(ww*rr)), (float(np.sum(ww*rr)),float(np.sum(ww*rr))), "degenerate"
    c1=weighted_quantile(rr,ww,0.30)
    c2=weighted_quantile(rr,ww,0.80)
    if c1>c2: c1,c2=c2,c1
    for _ in range(200):
        cut=0.5*(c1+c2)
        left=rr<=cut
        if not left.any() or left.all(): break
        n1=float(np.sum(ww[left]*rr[left])/np.sum(ww[left]))
        n2=float(np.sum(ww[~left]*rr[~left])/np.sum(ww[~left]))
        if abs(n1-c1)+abs(n2-c2)<1e-12:
            c1,c2=n1,n2
            break
        c1,c2=n1,n2
    return 0.5*(c1+c2),(float(c1),float(c2)),"weighted_2means_trim10_90"

def overlap(sign,truth,p):
    return float(abs(np.sum(p*sign*truth)))

def analyze(H,diag,E0,vec,baseline_name):
    a=np.abs(vec)
    p=a*a
    p=p/p.sum()
    truth=np.where(vec>=0,1,-1).astype(np.int8)
    s0=BASELINES[baseline_name].copy()
    psi0=a*s0
    Hv=H@psi0
    support=a>1e-14
    r=np.empty(D,float)
    r[support]=Hv[support]/psi0[support]
    # values off support have zero physical weight; set to weighted median later if needed
    med=float(np.median(r[support]))
    r[~support]=med
    t,centers,rule=choose_threshold(r,a)
    if rule=="degenerate":
        q1=np.ones(D,dtype=np.int8)
    else:
        q1=np.where(r<=t,1,-1).astype(np.int8)
    s1=(s0*q1).astype(np.int8)

    O0=overlap(s0,truth,p)
    O1=overlap(s1,truth,p)

    psi1=a*s1
    Ebase=float(psi0@(H@psi0))
    E1=float(psi1@(H@psi1))
    err0=(Ebase-E0)/N
    err1=(E1-E0)/N

    # Single-sign-flip stability for the fixed-amplitude sign-Ising objective.
    h1=(H@psi1)-diag*psi1
    margin=s1*h1
    stable=margin<=1e-12
    Pstable=float(np.sum(p*stable))
    Pstable_unw=float(np.mean(stable[support]))
    # Hidden exact signs as a sanity check.
    psit=a*truth
    ht=(H@psit)-diag*psit
    Ptruth=float(np.sum(p*((truth*ht)<=1e-12)))

    changed=float(np.sum(p*(s1!=s0)))
    wrong=float(np.sum(p*(s1!=truth)))
    rmean=float(np.sum(p*r))
    rstd=float(np.sqrt(np.sum(p*(r-rmean)**2)))
    return dict(
        baseline=baseline_name, threshold=float(t), centers=list(centers), threshold_rule=rule,
        O_S_0=O0,O_S_1=O1,wrong_weight_1=wrong,changed_weight=changed,
        E0=float(E0),E_fixed_baseline=Ebase,E_fixed_step1=E1,
        fixed_amp_energy_error0_per_site=float(err0),
        fixed_amp_energy_error1_per_site=float(err1),
        P_stable_1=Pstable,P_stable_1_unweighted=Pstable_unw,
        P_stable_truth=Ptruth,r0_phys_mean=rmean,r0_phys_std=rstd,
        support=int(np.sum(support))
    )

all_results=[]
prev=None
for J2 in J2_VALUES:
    t0=time.time()
    H,diag=build_H(J2)
    # two levels expose accidental/near degeneracy; v0 reuse accelerates adjacent couplings
    kwargs=dict(k=2,which="SA",tol=1e-11,maxiter=50000)
    if prev is not None: kwargs["v0"]=prev
    ev,vecs=sla.eigsh(H,**kwargs)
    order=np.argsort(ev); ev=ev[order]; vecs=vecs[:,order]
    E0=float(ev[0]); gap=float(ev[1]-ev[0]); vec=np.asarray(vecs[:,0],float)
    prev=vec
    # deterministic global phase
    j=np.argmax(np.abs(vec))
    if vec[j]<0: vec=-vec
    main=analyze(H,diag,E0,vec,"marshall")
    main.update(J2=J2,gap=gap,size="4x4",model="square_J1J2_exact")
    all_results.append(main)
    print("SWEEP",J2,json.dumps({k:main[k] for k in
          ["O_S_0","O_S_1","fixed_amp_energy_error0_per_site",
           "fixed_amp_energy_error1_per_site","P_stable_1","threshold","changed_weight"]}),
          "gap",gap,"sec",time.time()-t0,flush=True)

    if J2 in (0.8,1.0):
        for b in ("stripe_x","stripe_y"):
            z=analyze(H,diag,E0,vec,b)
            z.update(J2=J2,gap=gap,size="4x4",model="square_J1J2_exact")
            all_results.append(z)
            print("BASELINE_COMPARE",J2,b,json.dumps({k:z[k] for k in
                  ["O_S_0","O_S_1","fixed_amp_energy_error1_per_site",
                   "P_stable_1","threshold","changed_weight"]}),flush=True)

with open(OUT/"square_exact_sweep.json","w") as f:
    json.dump(dict(method={
        "system":"4x4 periodic square lattice, Sz=0 exact diagonalization",
        "threshold":"weighted two-means of r0 under |psi|^0.8, robust 10%-90% trim; no hidden signs",
        "O_S":"absolute |psi|^2-weighted sign overlap with exact ground-state signs",
        "fixed_amp_energy_error":"(E[a_exact,s]-E0)/N",
        "P_stable_1":"|psi|^2-weighted fraction satisfying s1(x)*sum_{y!=x}Hxy a(y)s1(y) <= 0"
    },results=all_results),f,indent=2)

fields=["J2","baseline","gap","O_S_0","O_S_1","fixed_amp_energy_error0_per_site",
        "fixed_amp_energy_error1_per_site","P_stable_1","P_stable_1_unweighted",
        "changed_weight","wrong_weight_1","threshold"]
with open(OUT/"square_exact_sweep.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
    for r in all_results: w.writerow({k:r[k] for k in fields})
print("WROTE",OUT/"square_exact_sweep.json",OUT/"square_exact_sweep.csv",flush=True)
