import os, math, numpy as np
BASE='/Users/aliotaifi/j1j2_vit_bench'
os.chdir(BASE)
src=open('true_k2_6x6.py').read().split("d=np.load('krylov_scaling_6x6_a1.20_tr4096_va2048.npz')")[0]
exec(src)

p='/Users/aliotaifi/Chatty/projects/j1j2/results/true_k2_6x6_3471990/true_k2_6x6.npz'
d=np.load(p)
allx=d['phys_states'].astype(np.uint64)
r0all=d['r0phys'].astype(float); r1all=d['r1phys'].astype(float)
bad=np.where(r1all>-15.810227701977523)[0]
rng=np.random.default_rng(20260930)
good=np.setdiff1d(np.arange(len(allx)),bad)
ctrl=rng.choice(good,size=27,replace=False)
idx=np.concatenate([bad,ctrl])
xs=allx[idx]
print('SUBSET',len(xs),'bad',bad.tolist(),flush=True)

first=[]
for x in xs:first.extend(y for y,J,mr in neigh(int(x)))
need=np.unique(np.concatenate([xs,np.asarray(first,np.uint64)]))
ensure_r0(need)
print('R0_READY',len(need),'zcache',len(zc),'r0cache',len(r0c),flush=True)

def q1(x):
    return 1.0 if T1-r0c[int(x)]>=0 else -1.0

def rproj(x):
    x=int(x); lx=zc[x].real; qx=q1(x); r=diag(x)
    for y,J,mr in neigh(x):
        r += .5*J*mr*q1(y)*qx*math.exp(zc[int(y)].real-lx)
    return float(r)

rp=np.asarray([rproj(x) for x in xs])
rt=r1all[idx]
qa=np.abs(T1-r0all[idx])
nb=len(bad)
print('BAD_TRUE_R1',rt[:nb].tolist(),flush=True)
print('BAD_PROJ_R1',rp[:nb].tolist(),flush=True)
print('CTRL_PROJ_Q',np.quantile(rp[nb:],[0,.1,.25,.5,.75,.9,1]).tolist(),flush=True)
for j in range(nb):
    pct=float(np.mean(rp[nb:]<=rp[j]))
    print('BAD_RANK',j,'proj',rp[j],'ctrl_cdf',pct,'qamp',qa[j],flush=True)
print('CORR_TRUE_LOGQ',float(np.corrcoef(np.log(qa),rt)[0,1]),flush=True)
print('CORR_PROJ_LOGQ',float(np.corrcoef(np.log(qa),rp)[0,1]),flush=True)
