import os, math, numpy as np
# Load definitions only.
os.environ['ALPHA']='0.8'
src=open('krylov_scaling_6x6.py').read().split('trstates=sample')[0]
exec(src)

t=-14.75594475197843
print('T_FIXED_FROM_INDEPENDENT_UNLABELED_TRAIN',t,flush=True)

# 2) independent direct |a|^2 sample for physical energy
ALPHA=2.0
xs=sample(202609291,16,16,burn=200,between=12)  # 256
zx=evalz(bits2x(xs)); lax=zx.real
rmx=local_rM(xs,'PHYS_X')['r']

# enumerate every connected neighbor of every x
flat=[]; meta=[]
for bi,s0 in enumerate(xs):
    s=int(s0)
    for i,j,J,mr in ALL:
        if ((s>>i)^(s>>j))&1:
            flat.append(s^(1<<i)^(1<<j)); meta.append((bi,J,mr))
flat=np.asarray(flat,np.uint64)
uy,inv=np.unique(flat,return_inverse=True)
print('ENERGY_GRAPH x',len(xs),'neighbor_occ',len(flat),'unique_y',len(uy),flush=True)
zy=evalz(bits2x(uy)); lay=zy.real

# Compute r_M(y) in chunks, which needs one additional Hamiltonian shell.
def rM_many(states, chunk=4096):
    out=[]
    for st in range(0,len(states),chunk):
        ss=np.asarray(states[st:st+chunk],np.uint64)
        z=evalz(bits2x(ss),batch=4096); la=z.real
        f=[]; mm=[]
        for bi,s0 in enumerate(ss):
            s=int(s0)
            for i,j,J,mr in ALL:
                if ((s>>i)^(s>>j))&1:
                    f.append(s^(1<<i)^(1<<j)); mm.append((bi,J,mr))
        f=np.asarray(f,np.uint64)
        uq,iv=np.unique(f,return_inverse=True)
        lu=evalz(bits2x(uq),batch=4096).real
        rr=np.asarray([diag_one(int(s)) for s in ss],float)
        for jj,(bi,J,mr) in zip(iv,mm):
            rr[bi] += .5*J*mr*math.exp(lu[jj]-la[bi])
        out.append(rr)
        print('RMY',min(st+chunk,len(states)),'/',len(states),flush=True)
    return np.concatenate(out)
rmy=rM_many(uy,4096)

qx=np.where(rmx<=t,1.,-1.)
qy=np.where(rmy<=t,1.,-1.)

# Marshall local energy is r_M(x)
eM=rmx.copy()
eK=np.asarray([diag_one(int(s)) for s in xs],float)
eVB=eK.copy()
eVC=eK.astype(complex)

# binary ViT phase: estimate global binary phase from x+y union (only ratios matter)
m=max(float(lax.max()),float(lay.max()))
c=np.sum(np.exp(2*(lax-m))*np.exp(2j*zx.imag))+np.sum(np.exp(2*(lay-m))*np.exp(2j*zy.imag))
phi=.5*np.angle(c)
hx=np.where(np.cos(np.angle(np.exp(1j*(zx.imag-phi))))>=0,1.,-1.)
hy=np.where(np.cos(np.angle(np.exp(1j*(zy.imag-phi))))>=0,1.,-1.)

for jj,(bi,J,mr) in zip(inv,meta):
    amp=math.exp(lay[jj]-lax[bi])
    # sK_y/sK_x = Marshall ratio * qy/qx
    eK[bi] += .5*J*mr*amp*qy[jj]*qx[bi]
    eVB[bi] += .5*J*amp*hy[jj]*hx[bi]
    eVC[bi] += .5*J*amp*np.exp(1j*(zy[jj].imag-zx[bi].imag))

eVC=eVC.real

# paired block errors; sample order is round-major with 16 chains.
def stats(v,block=32):
    v=np.asarray(v,float)
    nb=len(v)//block
    bm=np.array([v[i*block:(i+1)*block].mean() for i in range(nb)])
    return float(v.mean()),float(bm.std(ddof=1)/np.sqrt(nb)),nb

for name,v in [('Marshall',eM),('KrylovSign',eK),('ViTBinary',eVB),('ViTActual',eVC)]:
    mu,se,nb=stats(v); print('ENERGY',name,mu,'SE_block',se,'nblock',nb,'per_site',mu/N,flush=True)
for name,v in [('KminusViTActual',eK-eVC),('KminusViTBinary',eK-eVB),('KminusMarshall',eK-eM)]:
    mu,se,nb=stats(v); print('DELTA',name,mu,'SE_block',se,'z',mu/se if se>0 else np.nan,'per_site',mu/N,flush=True)

np.savez_compressed('energy_krylov_vs_vit_6x6_fast.npz',states=xs,rmx=rmx,eM=eM,eK=eK,eVB=eVB,eVC=eVC,t=t)
