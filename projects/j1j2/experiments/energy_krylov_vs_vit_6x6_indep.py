import os, math, numpy as np
os.environ['ALPHA']='2.0'
src=open('krylov_scaling_6x6.py').read().split('trstates=sample')[0]
exec(src)
t=-14.75594475197843
KEDGE=16
RG=np.random.default_rng(202609292)
print('T_FIXED',t,'KEDGE',KEDGE,flush=True)

ALPHA=2.0
xs=sample(202609293,64,4,burn=300,between=64)  # 256, 64 independent chains x 4 widely spaced rounds
zx=evalz(bits2x(xs)); lax=zx.real

# all first neighbors: exact Marshall and ViT energies; also rM(x)
flat=[]; meta=[]; byx=[[] for _ in range(len(xs))]
for bi,s0 in enumerate(xs):
    s=int(s0)
    for i,j,J,mr in ALL:
        if ((s>>i)^(s>>j))&1:
            idx=len(flat); flat.append(s^(1<<i)^(1<<j)); meta.append((bi,J,mr)); byx[bi].append(idx)
flat=np.asarray(flat,np.uint64)
uy,inv=np.unique(flat,return_inverse=True)
zy=evalz(bits2x(uy),batch=2048); lay=zy.real
print('GRAPH',len(xs),len(flat),len(uy),flush=True)

diag=np.asarray([diag_one(int(s)) for s in xs],float)
eM=diag.copy(); eVA=diag.astype(complex); eVB=diag.copy()
# global binary phase for x+y
m=max(float(lax.max()),float(lay.max()))
c=np.sum(np.exp(2*(lax-m))*np.exp(2j*zx.imag))+np.sum(np.exp(2*(lay-m))*np.exp(2j*zy.imag))
phi=.5*np.angle(c)
hx=np.where(np.cos(np.angle(np.exp(1j*(zx.imag-phi))))>=0,1.,-1.)
hy=np.where(np.cos(np.angle(np.exp(1j*(zy.imag-phi))))>=0,1.,-1.)

for occ,(jj,(bi,J,mr)) in enumerate(zip(inv,meta)):
    amp=math.exp(lay[jj]-lax[bi])
    eM[bi] += .5*J*mr*amp
    eVA[bi] += .5*J*amp*np.exp(1j*(zy[jj].imag-zx[bi].imag))
    eVB[bi] += .5*J*amp*hy[jj]*hx[bi]
eVA=eVA.real
rmx=eM.copy(); qx=np.where(rmx<=t,1.,-1.)

# sample KEDGE connected edges per x; collect their y states
chosen=[]
for bi,inds in enumerate(byx):
    kk=min(KEDGE,len(inds))
    sel=RG.choice(np.asarray(inds),size=kk,replace=False)
    for occ in sel: chosen.append((bi,int(occ),len(inds)))
sy_occ=np.array([q[1] for q in chosen],int)
sy_states=flat[sy_occ]
sy_unique,sy_inv=np.unique(sy_states,return_inverse=True)
print('SAMPLED_EDGES',len(chosen),'unique_y',len(sy_unique),flush=True)

# rM only for sampled y
def rM_batch(states,chunk=2048):
    ans=[]
    for st in range(0,len(states),chunk):
        ss=np.asarray(states[st:st+chunk],np.uint64)
        z=evalz(bits2x(ss),batch=2048); la=z.real
        ff=[]; mm=[]
        for bi,s0 in enumerate(ss):
            s=int(s0)
            for i,j,J,mr in ALL:
                if ((s>>i)^(s>>j))&1:
                    ff.append(s^(1<<i)^(1<<j)); mm.append((bi,J,mr))
        ff=np.asarray(ff,np.uint64)
        uq,iv=np.unique(ff,return_inverse=True)
        lu=evalz(bits2x(uq),batch=2048).real
        rr=np.asarray([diag_one(int(s)) for s in ss],float)
        for jj,(bi,J,mr) in zip(iv,mm):
            rr[bi]+=.5*J*mr*math.exp(lu[jj]-la[bi])
        ans.append(rr)
        print('RMY',min(st+chunk,len(states)),'/',len(states),flush=True)
    return np.concatenate(ans)
rmyu=rM_batch(sy_unique)
qyu=np.where(rmyu<=t,1.,-1.)

# per-x paired edge estimator of K - comparator
dKA=np.zeros(len(xs)); dKB=np.zeros(len(xs)); dKM=np.zeros(len(xs))
counts=np.zeros(len(xs),int)
for kk,(bi,occ,deg) in enumerate(chosen):
    jj=inv[occ]; J=meta[occ][1]; mr=meta[occ][2]
    amp=math.exp(lay[jj]-lax[bi])
    qy=qyu[sy_inv[kk]]
    k_ratio=mr*qy*qx[bi]
    va_ratio=np.cos(zy[jj].imag-zx[bi].imag)
    vb_ratio=hy[jj]*hx[bi]
    mar_ratio=mr
    fac=.5*J*amp*deg
    dKA[bi]+=fac*(k_ratio-va_ratio)
    dKB[bi]+=fac*(k_ratio-vb_ratio)
    dKM[bi]+=fac*(k_ratio-mar_ratio)
    counts[bi]+=1
dKA/=counts; dKB/=counts; dKM/=counts
eK_A=eVA+dKA; eK_B=eVB+dKB; eK_M=eM+dKM

def stats(v,block=None):
    v=np.asarray(v,float)
    # sample order is 4 rounds x 64 independent chains; average within chain,
    # then use between-chain variance for a correlation-robust SE.
    cm=v.reshape(-1,64).mean(axis=0)
    return float(v.mean()),float(cm.std(ddof=1)/np.sqrt(len(cm))),len(cm)
for name,v in [('MarshallExact',eM),('ViTBinaryExact',eVB),('ViTActualExact',eVA),
               ('Krylov_fromActualDelta',eK_A)]:
    mu,se,nb=stats(v); print('ENERGY',name,mu,'SE',se,'per_site',mu/N,flush=True)
for name,v in [('KminusViTActual',dKA),('KminusViTBinary',dKB),('KminusMarshall',dKM)]:
    mu,se,nb=stats(v); print('DELTA',name,mu,'SE',se,'z',mu/se if se else np.nan,'per_site',mu/N,flush=True)
# bootstrap x and edge stochasticity is not separated; paired block SE is conservative for the sample stream.
np.savez_compressed('energy_krylov_vs_vit_6x6_indep.npz',states=xs,eM=eM,eVB=eVB,eVA=eVA,dKA=dKA,dKB=dKB,dKM=dKM,t=t)
