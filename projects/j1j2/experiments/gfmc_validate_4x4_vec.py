import numpy as np
exec(open("fn_krylov_loop_4x4_kmeans_a12.py").read().split('report_direct("exact_abs",a0)')[0])
Efn,afn,_=fn_solve(a0,sM)
f_exact=a0*afn; f_exact/=f_exact.sum()

diag=H.diagonal().astype(float).copy()
allow=[[] for _ in range(D)]
for ia,s0 in enumerate(states):
    x=int(s0)
    for i,j,J in BONDS:
        if not (((x>>i)^(x>>j))&1): continue
        ib=idx[x^(1<<i)^(1<<j)]
        if sM[ia]*sM[ib] < 0: allow[ia].append((ib,0.5*J))
        else: diag[ia]+=0.5*J*a0[ib]/a0[ia]
maxd=max(map(len,allow))
nei=np.zeros((D,maxd),np.int32)
rate=np.zeros((D,maxd),float)
mask=np.zeros((D,maxd),bool)
for x,ls in enumerate(allow):
    for j,(y,h) in enumerate(ls):
        nei[x,j]=y; rate[x,j]=h*a0[y]/a0[x]; mask[x,j]=1
Eloc=diag-rate.sum(axis=1)
print("EXACT",Efn,"mixed",np.sum(f_exact*Eloc),
      "diagRange",diag.min(),diag.max(),"ElocRange",Eloc.min(),Eloc.max(),"maxd",maxd,flush=True)

def sysidx(w,rng,M):
    c=np.cumsum(w); c[-1]=1
    u=rng.random()/M+np.arange(M)/M
    return np.searchsorted(c,u,'right')

def run(M,tau,steps,burn,seed):
    rng=np.random.default_rng(seed)
    walkers=rng.choice(D,M,p=a0*a0)
    hist=np.zeros(D); ens=[]; minStay=1e9
    for it in range(steps):
        Eref=float(np.mean(Eloc[walkers]))
        st=1-tau*(diag[walkers]-Eref)
        minStay=min(minStay,float(st.min()))
        if np.any(st<0): raise RuntimeError(("negative stay",st.min()))
        mw=tau*rate[walkers]
        total=st+mw.sum(axis=1)
        u=rng.random(M)*total
        c=np.cumsum(np.c_[st,mw],axis=1)
        k=(u[:,None]>c).sum(axis=1)
        nxt=walkers.copy()
        mv=k>0
        nxt[mv]=nei[walkers[mv],k[mv]-1]
        ww=total/total.sum()
        walkers=nxt[sysidx(ww,rng,M)]
        if it>=burn and (it-burn)%10==0:
            hist+=np.bincount(walkers,minlength=D); ens.append(np.mean(Eloc[walkers]))
        if it in (0,99,499,999,1999,3999):
            print("STEP",M,tau,it,"E",np.mean(Eloc[walkers]),flush=True)
    ph=hist/hist.sum()
    print("RESULT",M,tau,"E",np.mean(ens),"SEsnap",np.std(ens)/np.sqrt(len(ens)),
          "TV",.5*np.abs(ph-f_exact).sum(),"BC",np.sqrt(ph*f_exact).sum(),
          "minStay",minStay,flush=True)

run(10000,.001,4000,1000,20260928)
