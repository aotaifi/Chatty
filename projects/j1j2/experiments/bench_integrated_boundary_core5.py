import numpy as np
import scipy.sparse as sp

L=6
D=np.load("boundary_cache_core5.npz")
arr=D["arr"]; la=D["la"]; tru=D["tru"].astype(np.int8)
u=D["uk"].astype(np.int32); v=D["vk"].astype(np.int32); w=D["wk"].astype(float)
cii=D["cii"].astype(np.int32)
outside=D["outside"]; xs_i=D["xs_i"].astype(np.int32); inv=D["inv"].astype(np.int32); cw=D["cw"].astype(float)
n=len(arr); no=len(outside)

def overlap(p):
    ww=np.exp(2*(la[cii]-np.max(la[cii]))); ww/=ww.sum()
    return abs(np.sum(ww*p[cii]*tru[cii]))

def energy_eff(s):
    q=np.bincount(inv,weights=cw*s[xs_i],minlength=no)
    return float(2*np.sum(w*s[u]*s[v])-2*np.sum(np.abs(q)))

def solve_wgreedy():
    adj=[[] for _ in range(n)]
    for a,b,c in zip(u,v,w):
        a=int(a); b=int(b); c=float(c)
        adj[a].append((b,c)); adj[b].append((a,c))
    sign=np.ones(n,dtype=np.int8)
    cid=np.full(n,-1,dtype=np.int32); members={}; nextcid=0
    for q0 in np.argsort(w)[::-1]:
        a=int(u[q0]); b=int(v[q0]); c=float(w[q0])
        ca=int(cid[a]); cb=int(cid[b])
        if ca>=0 and cb>=0:
            if ca==cb: continue
            frustrated=(int(sign[a])*int(sign[b])*c)>0
            if len(members[ca])<len(members[cb]): ca,cb=cb,ca
            if frustrated:
                for x in members[cb]: sign[x]*=-1
            for x in members[cb]: cid[x]=ca
            members[ca].extend(members[cb]); del members[cb]
        elif ca>=0 or cb>=0:
            if ca>=0: cc=ca; x=b
            else: cc=cb; x=a
            m=sum(cj*int(sign[j]) for j,cj in adj[x] if int(cid[j])==cc)
            if m>0: sign[x]=-1
            cid[x]=cc; members[cc].append(x)
        else:
            cc=nextcid; nextcid+=1
            cid[a]=cid[b]=cc; members[cc]=[a,b]
            sign[a]=1; sign[b]=-1 if c>0 else 1
    A=sp.coo_matrix((np.r_[w,w],(np.r_[u,v],np.r_[v,u])),shape=(n,n)).tocsr()
    h=A@sign.astype(float)
    while True:
        changed=0
        for i in range(n):
            if -4.0*float(sign[i])*float(h[i]) < -1e-15:
                old=int(sign[i]); sign[i]=-old; changed+=1
                a,b=A.indptr[i],A.indptr[i+1]
                h[A.indices[a:b]] += -2*old*A.data[a:b]
        if not changed: break
    return sign

A=sp.coo_matrix((np.r_[w,w],(np.r_[u,v],np.r_[v,u])),shape=(n,n)).tocsr()
order=np.argsort(xs_i,kind="stable")
si=xs_i[order]; sy=inv[order]; sw=cw[order]
ptr=np.searchsorted(si,np.arange(n+1))

def integrated_descent(s,max_sweeps=50):
    s=np.array(s,dtype=np.int8,copy=True)
    h=A@s.astype(float)
    q=np.bincount(inv,weights=cw*s[xs_i],minlength=no)
    E=float(2*np.sum(w*s[u]*s[v])-2*np.sum(np.abs(q)))
    for sweep in range(max_sweeps):
        changed=0
        for i in range(n):
            old=int(s[i])
            a,b=int(ptr[i]),int(ptr[i+1])
            ys=sy[a:b]; ws=sw[a:b]
            if len(ys):
                qnew=q[ys]-2.0*old*ws
                db=-2.0*(np.sum(np.abs(qnew))-np.sum(np.abs(q[ys])))
            else:
                qnew=None; db=0.0
            di=-4.0*old*float(h[i])
            de=di+db
            if de < -1e-14:
                s[i]=-old; changed+=1
                aa,bb=A.indptr[i],A.indptr[i+1]
                h[A.indices[aa:bb]] += -2.0*old*A.data[aa:bb]
                if len(ys): q[ys]=qnew
                E += de
        Echeck=float(2*np.sum(w*s[u]*s[v])-2*np.sum(np.abs(q)))
        print("SWEEP",sweep,"changed",changed,"E",Echeck,"O",overlap(s),flush=True)
        if abs(E-Echeck)>1e-7: print("ECHECK_MISMATCH",E,Echeck,flush=True)
        E=Echeck
        if changed==0: break
    return s,E

def marshall(ss):
    Aidx=np.array([x+L*y for y in range(L) for x in range(L) if (x+y)%2==0],dtype=np.uint64)
    parity=((ss[:,None]>>Aidx[None,:])&1).sum(axis=1)%2
    return np.where(parity==0,1,-1).astype(np.int8)

starts=[]
p0=solve_wgreedy(); starts.append(("free",p0))
starts.append(("marshall",marshall(arr)))
rng=np.random.default_rng(20270927)
for k in range(6): starts.append((f"random{k}",rng.choice(np.array([-1,1],dtype=np.int8),size=n)))

best=None
for name,s0 in starts:
    print("START",name,"E",energy_eff(s0),"O",overlap(s0),flush=True)
    s,e=integrated_descent(s0)
    print("FINAL",name,"E",e,"O",overlap(s),flush=True)
    if best is None or e<best[0]: best=(e,name,s.copy(),overlap(s))
print("BEST",best[1],"E",best[0],"O",best[3],flush=True)
