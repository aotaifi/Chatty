import math,time,os
import numpy as np
import scipy.sparse as sp
import jax, jax.numpy as jnp
import netket as nk, flax
import netket.jax as nkjax
from nqsmagic.models import ViT
from nqsmagic.utils import _logpsi_transl_2d

L=6; N=36; J2=.5; B=2
rng=np.random.default_rng(20260927)
hi=nk.hilbert.Spin(s=.5,N=N)
g=nk.graph.Hypercube(length=L,n_dim=2,pbc=True,max_neighbor_order=2)
model=ViT(num_layers=4,d_model=60,heads=10,L_eff=9,b=2,transl_invariant=True,two_dimensional=True)
apply=nkjax.HashablePartial(_logpsi_transl_2d,model.apply,2)
sam0=nk.sampler.MetropolisExchange(hi,graph=g,d_max=2,n_chains=6000,sweep_size=N)
v0=nk.vqs.MCState(sampler=sam0,apply_fun=apply,n_samples=6000,
 variables=model.init(jax.random.PRNGKey(1234),jnp.zeros((1,N))),n_discard_per_chain=10)
with open('vit_J2=0.50_N=6x6_k=0.mpack','rb') as f: v0=flax.serialization.from_bytes(v0,f.read())

def evalz(X,batch=2048):
    X=np.asarray(X,float); out=[]
    for i in range(0,len(X),batch): out.append(np.asarray(v0.log_value(jnp.asarray(X[i:i+batch]))))
    return np.concatenate(out)

def bonds():
    nn=[]; nnn=[]
    q=lambda x,y:(x%L)+L*(y%L)
    for y in range(L):
      for x in range(L):
        i=q(x,y); nn += [(i,q(x+1,y)),(i,q(x,y+1))]
        nnn += [(i,q(x+1,y+1)),(i,q(x+1,y-1))]
    return nn,nnn
NN,NNN=bonds()
def neigh(s):
    for bb,J in ((NN,1.0),(NNN,J2)):
      for i,j in bb:
        if ((s>>i)^(s>>j))&1: yield s^(1<<i)^(1<<j),J
def bits2x(ss):
    a=np.asarray(ss,dtype=np.uint64).reshape(-1,1)
    return 2*((a>>np.arange(N,dtype=np.uint64))&1).astype(float)-1
def x2bits(x):
    s=0
    for i,q in enumerate(x):
      if q>0:s|=1<<i
    return s
def grow(seed,K,p=.5):
    S={seed}
    def ch(s): return [t for t,_ in neigh(s) if t not in S and rng.random()<=p]
    fr=set(ch(seed))
    while len(S)<K and fr:
      a=list(fr);rng.shuffle(a);new=set()
      for s in a:
        if s in S:continue
        S.add(s)
        if len(S)>=K:break
        new.update(ch(s))
      fr=new-S
    return S
def extend(S,hops):
    S=set(S);fr=set(S)
    for _ in range(hops):
      new=set()
      for s in fr:new.update(t for t,_ in neigh(s))
      new-=S;S|=new;fr=new
    return S
def edges(arr,la):
    pos={int(s):i for i,s in enumerate(arr)};u=[];v=[];lw=[]
    for i,s in enumerate(arr):
      for t,J in neigh(int(s)):
        j=pos.get(t)
        if j is not None and i<j:
          u.append(i);v.append(j);lw.append(math.log(.5*J)+la[i]+la[j])
    u=np.asarray(u,np.int32);v=np.asarray(v,np.int32);lw=np.asarray(lw)
    w=np.exp(lw-lw.max()) if len(lw) else np.array([])
    return u,v,w
def solve(n,u,v,w):
    order=np.argsort(w)[::-1];par=np.arange(n,dtype=np.int32);rk=np.zeros(n,np.int8);px=np.zeros(n,np.int8)
    def find(x):
      y=x;p=0
      while par[y]!=y:p^=int(px[y]);y=int(par[y])
      return y,p
    for q in order:
      i=int(u[q]);j=int(v[q]);ri,pi=find(i);rj,pj=find(j)
      if ri==rj:continue
      z=pi^pj^1
      if rk[ri]<rk[rj]:par[ri]=rj;px[ri]=z
      else:
        par[rj]=ri;px[rj]=z
        if rk[ri]==rk[rj]:rk[ri]+=1
    s=np.ones(n,np.int8)
    for i in range(n): _,p=find(i);s[i]=-1 if p else 1
    A=sp.coo_matrix((np.r_[w,w],(np.r_[u,v],np.r_[v,u])),shape=(n,n)).tocsr();h=A@s.astype(float)
    for _ in range(30):
      changed=0
      for i in range(n):
        if s[i]*h[i]>1e-15:
          old=int(s[i]);s[i]=-old;changed+=1
          a,b=A.indptr[i],A.indptr[i+1];h[A.indices[a:b]]+=-2*old*A.data[a:b]
      if not changed:break
    return s
def hidden(z,la):
    w=np.exp(2*(la-la.max()));phi=.5*np.angle(np.sum(w*np.exp(2j*z.imag)))
    rel=np.angle(np.exp(1j*(z.imag-phi)));return np.where(np.cos(rel)>=0,1,-1).astype(np.int8),float(np.average(np.abs(np.sin(rel)),weights=w))
def overlap(p,t,la):
    w=np.exp(2*(la-la.max()));w/=w.sum();return abs(np.sum(w*p*t))

# flattened seeds: direct symmetric exchange Metropolis, p ~ |psi|^0.1
def flat_seeds(nc=8,burn=80,between=12,alpha=.1):
  ss=[]
  for _ in range(nc):
    pos=rng.choice(N,N//2,replace=False); q=0
    for i in pos:q|=1<<int(i)
    ss.append(q)
  z=evalz(bits2x(ss)); la=z.real
  out=[]
  for it in range(burn+between*6):
    cand=[]
    for q in ss:
      up=[i for i in range(N) if (q>>i)&1]; dn=[i for i in range(N) if not((q>>i)&1)]
      i=int(rng.choice(up));j=int(rng.choice(dn));cand.append(q^(1<<i)^(1<<j))
    z2=evalz(bits2x(cand));lb=z2.real
    acc=np.log(rng.random(nc))<np.minimum(0,alpha*(lb-la))
    for k in np.where(acc)[0]:ss[k]=cand[k]
    la=np.where(acc,lb,la)
    if it>=burn and (it-burn+1)%between==0: out.extend(ss)
  return out
seedbits=flat_seeds()
print('FLAT_SEEDS',len(seedbits),flush=True)

def run(K,hops,reps=6):
  ans=[]
  for r in range(reps):
    t0=time.time();core=grow(seedbits[r%len(seedbits)],K);ext=extend(core,hops)
    arr=np.array(sorted(ext),dtype=np.uint64);z=evalz(bits2x(arr));la=z.real
    tru,leak=hidden(z,la);u,v,w=edges(arr,la);pred=solve(len(arr),u,v,w)
    pos={int(s):i for i,s in enumerate(arr)};ci=np.array([pos[s] for s in core])
    O=overlap(pred[ci],tru[ci],la[ci])
    ans.append((O,len(arr),len(w),leak,time.time()-t0))
    print('RUN',K,hops,r,'O',O,'ext',len(arr),'edges',len(w),'leak',leak,'sec',ans[-1][-1],flush=True)
  a=np.array([q[0] for q in ans]); print('SUMMARY',K,hops,'medianO',np.median(a),'worstO',a.min(),'median_ext',np.median([q[1] for q in ans]),flush=True)


from scipy.sparse.csgraph import connected_components

def sparse_extend(core, order=2, reltol=1e-4):
    core=set(core); cur=set(core)
    stats=[]
    for step in range(order):
      cur=extend(cur,1)
      arr=np.array(sorted(cur),dtype=np.uint64); z=evalz(bits2x(arr)); la=z.real
      pos={int(q):i for i,q in enumerate(arr)}; u,v,w=edges(arr,la)
      ci=set(pos[q] for q in core)
      bothcore=np.array([(int(a) in ci and int(b) in ci) for a,b in zip(u,v)],dtype=bool)
      keep=bothcore | (w >= reltol*np.max(w))
      A=sp.coo_matrix((np.ones(2*np.sum(keep),dtype=np.int8),
          (np.r_[u[keep],v[keep]],np.r_[v[keep],u[keep]])),shape=(len(arr),len(arr))).tocsr()
      nc,lab=connected_components(A,directed=False)
      root=lab[pos[next(iter(core))]]
      cidx=np.array([pos[q] for q in core])
      if not np.all(lab[cidx]==root):
        print('DISCONNECTED_CORE',step,reltol,nc,flush=True)
      mask=(lab==root); cur=set(int(q) for q in arr[mask])
      stats.append((len(arr),int(np.sum(keep)),len(cur)))
    return cur,stats

from scipy.sparse.csgraph import connected_components

def sparse_extend(core, order=2, reltol=1e-4):
    core=set(core); cur=set(core)
    stats=[]
    for step in range(order):
      cur=extend(cur,1)
      arr=np.array(sorted(cur),dtype=np.uint64); z=evalz(bits2x(arr)); la=z.real
      pos={int(q):i for i,q in enumerate(arr)}; u,v,w=edges(arr,la)
      ci=set(pos[q] for q in core)
      bothcore=np.array([(int(a) in ci and int(b) in ci) for a,b in zip(u,v)],dtype=bool)
      keep=bothcore | (w >= reltol*np.max(w))
      A=sp.coo_matrix((np.ones(2*np.sum(keep),dtype=np.int8),
          (np.r_[u[keep],v[keep]],np.r_[v[keep],u[keep]])),shape=(len(arr),len(arr))).tocsr()
      nc,lab=connected_components(A,directed=False)
      root=lab[pos[next(iter(core))]]
      cidx=np.array([pos[q] for q in core])
      if not np.all(lab[cidx]==root):
        print('DISCONNECTED_CORE',step,reltol,nc,flush=True)
      mask=(lab==root); cur=set(int(q) for q in arr[mask])
      stats.append((len(arr),int(np.sum(keep)),len(cur)))
    return cur,stats


def solve_westerhout_greedy(n,u,v,w):
    # Source-faithful implementation of ising-glass-annealer greedySolve.
    # Process couplings largest-first. Cluster global sign is gauge, so union-by-size
    # is equivalent while avoiding the Haskell implementation's O(n) cluster scans.
    adj=[[] for _ in range(n)]
    for a,b,c in zip(u,v,w):
        a=int(a); b=int(b); c=float(c)
        adj[a].append((b,c)); adj[b].append((a,c))
    sign=np.ones(n,dtype=np.int8)
    cid=np.full(n,-1,dtype=np.int32)
    members={}
    nextcid=0
    order=np.argsort(w)[::-1]
    for q in order:
        a=int(u[q]); b=int(v[q]); c=float(w[q])
        ca=int(cid[a]); cb=int(cid[b])
        if ca>=0 and cb>=0:
            if ca==cb: continue
            frustrated=(int(sign[a])*int(sign[b])*c)>0
            # Merge smaller into larger. Flipping either side when frustrated differs
            # from Haskell only by a global sign of the merged component.
            if len(members[ca]) < len(members[cb]):
                ca,cb=cb,ca
            if frustrated:
                # Must re-evaluate which physical component is cb after possible swap:
                # flipping either component is equivalent, so flip chosen smaller cb.
                for x in members[cb]: sign[x]*=-1
            for x in members[cb]: cid[x]=ca
            members[ca].extend(members[cb]); del members[cb]
        elif ca>=0 or cb>=0:
            if ca>=0: cc=ca; x=b
            else: cc=cb; x=a
            m=0.0
            for j,cj in adj[x]:
                if int(cid[j])==cc: m += cj*int(sign[j])
            # Haskell: de=-sign[x]*2*m with sign[x]=+1; flip if de<0.
            if m>0: sign[x]=-1
            cid[x]=cc; members[cc].append(x)
        else:
            cc=nextcid; nextcid+=1
            cid[a]=cid[b]=cc; members[cc]=[a,b]
            sign[a]=1; sign[b]=-1 if c>0 else 1
    # Isolated vertices are not expected for the retained connected component.
    for i in range(n):
        if cid[i]<0:
            cc=nextcid; nextcid+=1; cid[i]=cc; members[cc]=[i]
    if len(members)!=1:
        raise RuntimeError(f'greedySolve expected one connected component, got {len(members)}')
    # Exact optimizeLocally: sequential scans, flip any spin with negative dE.
    A=sp.coo_matrix((np.r_[w,w],(np.r_[u,v],np.r_[v,u])),shape=(n,n)).tocsr()
    h=A@sign.astype(float)
    sweeps=0
    while True:
        changed=False
        for i in range(n):
            de=-4.0*float(sign[i])*float(h[i])
            if de < 0.0:
                old=int(sign[i]); sign[i]=-old; changed=True
                a0,a1=A.indptr[i],A.indptr[i+1]
                js=A.indices[a0:a1]; cs=A.data[a0:a1]
                h[js] += -2.0*old*cs
        if not changed: break
        sweeps += 1
    return sign,sweeps

def paper_trial_faithful(seed,K=50,order=2,reltol=1e-4):
  core=grow(seed,K); ext,st=sparse_extend(core,order,reltol)
  arr=np.array(sorted(ext),dtype=np.uint64);z=evalz(bits2x(arr));la=z.real
  tru,leak=hidden(z,la);u,v,w=edges(arr,la)
  pos={int(q):i for i,q in enumerate(arr)};ci=set(pos[q] for q in core)
  both=np.array([(int(a) in ci and int(b) in ci) for a,b in zip(u,v)],bool)
  keep=both | (w>=reltol*np.max(w))
  pred,ns=solve_westerhout_greedy(len(arr),u[keep],v[keep],w[keep])
  cii=np.array([pos[q] for q in core])
  O=overlap(pred[cii],tru[cii],la[cii])
  print('WGREEDY',K,order,reltol,'O',O,'size',len(arr),'kept_edges',int(np.sum(keep)),'local_sweeps',ns,'stages',st,'leak',leak,flush=True)
  return O


def solve_wgreedy_field(n,u,v,w,field):
    adj=[[] for _ in range(n)]
    for a,b,c in zip(u,v,w):
        a=int(a); b=int(b); c=float(c)
        adj[a].append((b,c)); adj[b].append((a,c))
    sign=np.ones(n,dtype=np.int8)
    cid=np.full(n,-1,dtype=np.int32); members={}; nextcid=0
    order=np.argsort(w)[::-1]
    for q in order:
        a=int(u[q]); b=int(v[q]); c=float(w[q])
        ca=int(cid[a]); cb=int(cid[b])
        if ca>=0 and cb>=0:
            if ca==cb: continue
            frustrated=(int(sign[a])*int(sign[b])*c)>0
            if len(members[ca]) < len(members[cb]): ca,cb=cb,ca
            if frustrated:
                for x in members[cb]: sign[x]*=-1
            for x in members[cb]: cid[x]=ca
            members[ca].extend(members[cb]); del members[cb]
        elif ca>=0 or cb>=0:
            if ca>=0: cc=ca; x=b
            else: cc=cb; x=a
            m=0.0
            for j,cj in adj[x]:
                if int(cid[j])==cc: m += cj*int(sign[j])
            if m>0: sign[x]=-1
            cid[x]=cc; members[cc].append(x)
        else:
            cc=nextcid; nextcid+=1
            cid[a]=cid[b]=cc; members[cc]=[a,b]
            sign[a]=1; sign[b]=-1 if c>0 else 1
    if len(members)!=1:
        raise RuntimeError(f'expected connected graph, got {len(members)} comps')
    A=sp.coo_matrix((np.r_[w,w],(np.r_[u,v],np.r_[v,u])),shape=(n,n)).tocsr()
    h=A@sign.astype(float)
    sweeps=0
    while True:
        changed=False
        for i in range(n):
            de=-4.0*float(sign[i])*float(h[i])-2.0*float(sign[i])*float(field[i])
            if de < -1e-15:
                old=int(sign[i]); sign[i]=-old; changed=True
                a0,a1=A.indptr[i],A.indptr[i+1]
                js=A.indices[a0:a1]; cs=A.data[a0:a1]
                h[js] += -2.0*old*cs
        if not changed: break
        sweeps+=1
        if sweeps>100: raise RuntimeError('local optimize failed to converge')
    return sign,sweeps


def local_opt_from(sign,u,v,w,field,maxs=100):
    sign=np.array(sign,dtype=np.int8,copy=True)
    n=len(sign)
    A=sp.coo_matrix((np.r_[w,w],(np.r_[u,v],np.r_[v,u])),shape=(n,n)).tocsr()
    h=A@sign.astype(float)
    ns=0
    while True:
        changed=0
        for i in range(n):
            de=-4.0*float(sign[i])*float(h[i])-2.0*float(sign[i])*float(field[i])
            if de < -1e-15:
                old=int(sign[i]); sign[i]=-old; changed+=1
                a0,a1=A.indptr[i],A.indptr[i+1]
                js=A.indices[a0:a1]; cs=A.data[a0:a1]
                h[js] += -2.0*old*cs
        if not changed: break
        ns+=1
        if ns>=maxs: break
    return sign,ns

def marshall_sign(ss):
    Aidx=np.array([x+L*y for y in range(L) for x in range(L) if (x+y)%2==0],dtype=np.uint64)
    q=((ss[:,None]>>Aidx[None,:])&1).sum(axis=1)
    return np.where((q%2)==0,1,-1).astype(np.int8)

def build_core_cache(rr=5,reltol=1e-4):
    name=f'boundary_cache_core{rr}.npz'
    if os.path.exists(name):
        d=np.load(name)
        print('CACHE_LOAD',name,flush=True)
        return {k:d[k] for k in d.files}
    dat=np.load('vit6x6_saved_cores.npz')
    core=set(int(x) for x in dat[f'core{rr}'])
    ext,st=sparse_extend(core,2,reltol)
    arr=np.array(sorted(ext),dtype=np.uint64)
    z=evalz(bits2x(arr)); la=z.real
    tru,_=hidden(z,la)
    pos={int(q):i for i,q in enumerate(arr)}
    u,v,w=edges(arr,la)
    ci=set(pos[q] for q in core)
    both=np.array([(int(a) in ci and int(b) in ci) for a,b in zip(u,v)],bool)
    keep=both | (w>=reltol*np.max(w))
    uk,vk,wk=u[keep],v[keep],w[keep]
    maxlw=-np.inf
    for i,s in enumerate(arr):
        for t,J in neigh(int(s)):
            j=pos.get(t)
            if j is not None and i<j:
                maxlw=max(maxlw,math.log(.5*J)+la[i]+la[j])
    xs_i=[]; xs_s=[]; xs_J=[]
    for i,s in enumerate(arr):
        for t,J in neigh(int(s)):
            if t not in pos:
                xs_i.append(i); xs_s.append(t); xs_J.append(J)
    xs_i=np.asarray(xs_i,np.int32); xs_s=np.asarray(xs_s,np.uint64); xs_J=np.asarray(xs_J,float)
    outside,inv=np.unique(xs_s,return_inverse=True)
    # Amplitudes only on exterior: phases/signs are not used by the cavity algorithm.
    lout=evalz(bits2x(outside)).real
    cw=np.exp(np.log(.5*xs_J)+la[xs_i]+lout[inv]-maxlw)
    cii=np.array([pos[q] for q in core],dtype=np.int32)
    np.savez(name,arr=arr,la=la,tru=tru,uk=uk,vk=vk,wk=wk,cii=cii,
             outside=outside,xs_i=xs_i,inv=inv.astype(np.int32),cw=cw)
    print('CACHE_SAVE',name,'inside',len(arr),'outside',len(outside),'cross',len(cw),flush=True)
    return dict(arr=arr,la=la,tru=tru,uk=uk,vk=vk,wk=wk,cii=cii,
                outside=outside,xs_i=xs_i,inv=inv.astype(np.int32),cw=cw)

def run_cavity(rr=5):
    d=build_core_cache(rr)
    arr=d['arr']; la=d['la']; tru=d['tru']; uk=d['uk']; vk=d['vk']; wk=d['wk']; cii=d['cii']
    outside=d['outside']; xs_i=d['xs_i']; inv=d['inv']; cw=d['cw']; n=len(arr); no=len(outside)
    p0,_=solve_wgreedy_field(n,uk,vk,wk,np.zeros(n))
    print('CAVITY_START',rr,'O',overlap(p0[cii],tru[cii],la[cii]),flush=True)

    # Fixed Marshall boundary: cheap global guide baseline.
    sm=marshall_sign(outside)
    fm=np.bincount(xs_i,weights=2*cw*sm[inv],minlength=n)
    pm,nm=local_opt_from(p0,uk,vk,wk,fm)
    print('MARSHALL_BOUNDARY',rr,'O',overlap(pm[cii],tru[cii],la[cii]),'sweeps',nm,flush=True)

    # Amplitude-only cavity: integrate the omitted one-shell spins by block coordinate descent.
    p=p0.copy()
    lastE=None
    for it in range(20):
        q=np.bincount(inv,weights=cw*p[xs_i],minlength=no)
        so=np.where(q>0,-1,1).astype(np.int8)
        field=np.bincount(xs_i,weights=2*cw*so[inv],minlength=n)
        pn,ns=local_opt_from(p,uk,vk,wk,field)
        Ein=2*np.sum(wk*pn[uk]*pn[vk])
        Ecross=2*np.sum(cw*pn[xs_i]*so[inv])
        E=float(Ein+Ecross)
        O=overlap(pn[cii],tru[cii],la[cii])
        changed=int(np.sum(pn!=p))
        print('CAVITY',rr,'it',it,'O',O,'changed',changed,'E',E,'local_sweeps',ns,flush=True)
        p=pn
        if changed==0: break
        if lastE is not None and E>lastE+1e-8:
            print('ENERGY_WARNING',E-lastE,flush=True)
        lastE=E
    return p

run_cavity(5)
