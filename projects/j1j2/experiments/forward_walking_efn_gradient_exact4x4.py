import json, sys, numpy as np
from pathlib import Path

KDIR=Path("/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure/experiments")
sys.path.insert(0,str(KDIR))
from closed_fn_krylov_exact4x4 import (
    build_H, ground, marshall_signs, canonical, basis,
    fixed_node_solve,
)

H,diag0=build_H(0.5)
H0,_=build_H(0.0)
D=len(basis)
sM=canonical(marshall_signs())
_,psi_init=ground(H0)
if np.dot(psi_init,sM)<0: psi_init=-psi_init
a0=np.abs(psi_init); a0/=np.linalg.norm(a0)
Efn,afn=fixed_node_solve(H,diag0,a0,sM)
f_exact=a0*afn; f_exact/=f_exact.sum()
pure_exact=afn*afn; pure_exact/=pure_exact.sum()

diag=diag0.astype(float).copy()
allow=[[] for _ in range(D)]
for i in range(D):
    st,en=H.indptr[i],H.indptr[i+1]
    for q in range(st,en):
        j=int(H.indices[q]); hij=float(H.data[q])
        if j==i: continue
        kij=float(sM[i])*hij*float(sM[j])
        if kij<0:
            allow[i].append((j,abs(kij)))
        elif kij>0:
            diag[i]+=kij*a0[j]/a0[i]
maxd=max(map(len,allow))
nei=np.zeros((D,maxd),np.int32)
rate=np.zeros((D,maxd),float)
for i,ls in enumerate(allow):
    for k,(j,h) in enumerate(ls):
        nei[i,k]=j
        rate[i,k]=h*a0[j]/a0[i]
Eloc=diag-rate.sum(axis=1)
print("EXACT",Efn,"mixedE",float(np.sum(f_exact*Eloc)),"maxd",maxd,flush=True)

def sysidx(w,rng,M):
    c=np.cumsum(w); c[-1]=1.0
    return np.searchsorted(c,rng.random()/M+np.arange(M)/M,"right")

def one_step(walkers,labels,rng,tau=0.001):
    Eref=float(np.mean(Eloc[walkers]))
    st=1-tau*(diag[walkers]-Eref)
    if np.any(st<0): raise RuntimeError(("negative stay",float(st.min())))
    mw=tau*rate[walkers]
    total=st+mw.sum(axis=1)
    u=rng.random(len(walkers))*total
    c=np.cumsum(np.c_[st,mw],axis=1)
    k=(u[:,None]>c).sum(axis=1)
    nxt=walkers.copy()
    mv=k>0
    nxt[mv]=nei[walkers[mv],k[mv]-1]
    take=sysidx(total/total.sum(),rng,len(walkers))
    return nxt[take],labels[take]

def grad_from_pure(pure):
    g=np.zeros(D,float)
    for i in range(D):
        st,en=H.indptr[i],H.indptr[i+1]
        for q in range(st,en):
            j=int(H.indices[q]); hij=float(H.data[q])
            if j==i: continue
            kij=float(sM[i])*hij*float(sM[j])
            if kij>0:
                c=float(pure[i])*kij*a0[j]/a0[i]
                g[j]+=c; g[i]-=c
    return g

g_exact=grad_from_pure(pure_exact)
ng=np.linalg.norm(g_exact)

M=20000
BURN=3000
CHECKS={0,50,100,200,500,1000}
rng=np.random.default_rng(20261001)
walkers=rng.choice(D,M,p=a0*a0)
dummy=np.arange(M,dtype=np.int32)
for it in range(BURN):
    walkers,dummy=one_step(walkers,dummy,rng)
    if it in (0,999,1999,2999):
        print("BURN",it+1,"E",float(np.mean(Eloc[walkers])),flush=True)

ancestor=walkers.copy()
labels=np.arange(M,dtype=np.int32)
mixed_hist=np.bincount(ancestor,minlength=D).astype(float); mixed_hist/=mixed_hist.sum()
rows=[]
def report(step,labels):
    cnt=np.bincount(labels,minlength=M).astype(float)
    hist=np.bincount(ancestor,weights=cnt,minlength=D).astype(float)
    hist/=hist.sum()
    g=grad_from_pure(hist)
    cos=float((g@g_exact)/(max(np.linalg.norm(g),1e-300)*ng))
    rel=float(np.linalg.norm(g-g_exact)/ng)
    ess=float(cnt.sum()**2/(cnt@cnt))
    row=dict(
        fw_steps=step,descendant_ESS=ess,
        pure_TV=.5*float(np.abs(hist-pure_exact).sum()),
        pure_BC=float(np.sqrt(hist*pure_exact).sum()),
        grad_cos=cos,grad_relerr=rel,
        mixed_BC=float(np.sqrt(mixed_hist*f_exact).sum()),
        mixed_to_pure_BC=float(np.sqrt(mixed_hist*pure_exact).sum()),
    )
    rows.append(row)
    print("FW",json.dumps(row,sort_keys=True),flush=True)

report(0,labels)
for it in range(1,1001):
    walkers,labels=one_step(walkers,labels,rng)
    if it in CHECKS:
        report(it,labels)

out=dict(M=M,burn_steps=BURN,tau=0.001,rows=rows,
         exact_grad_norm=float(ng),
         mixed_TV=.5*float(np.abs(mixed_hist-f_exact).sum()))
print("FORWARD_WALK_FINAL",json.dumps(out,sort_keys=True),flush=True)
Path("/Users/aliotaifi/Chatty/projects/j1j2/results/forward_walking_efn_gradient_exact4x4.json").write_text(json.dumps(out,indent=2))
