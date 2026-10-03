import os,math,numpy as np
os.environ['ALPHA']='2.0'
src=open('krylov_scaling_6x6.py').read().split('trstates=sample')[0]
exec(src)
d=np.load('energy_krylov_vs_vit_6x6_edge32.npz')
xs=d['states'].astype(np.uint64); eM=d['eM']; eVA=d['eVA']; eVB=d['eVB']; t=float(d['t'])
dKA=d['dKA'].copy(); dKB=d['dKB'].copy()
sel=np.where(np.abs(dKA)>0.2)[0]
print('EXACTIFY',len(sel),'indices',sel.tolist(),flush=True)

sx=xs[sel]; zx=evalz(bits2x(sx),batch=2048); lax=zx.real
flat=[];meta=[]
for li,s0 in enumerate(sx):
 s=int(s0)
 for i,j,J,mr in ALL:
  if ((s>>i)^(s>>j))&1:
   flat.append(s^(1<<i)^(1<<j));meta.append((li,J,mr))
flat=np.asarray(flat,np.uint64); uy,inv=np.unique(flat,return_inverse=True)
zy=evalz(bits2x(uy),batch=2048); lay=zy.real
print('OUTLIER_GRAPH occ',len(flat),'unique_y',len(uy),flush=True)

def rM_batch(states,chunk=2048):
 ans=[]
 for st in range(0,len(states),chunk):
  ss=np.asarray(states[st:st+chunk],np.uint64)
  z=evalz(bits2x(ss),batch=2048);la=z.real
  ff=[];mm=[]
  for bi,s0 in enumerate(ss):
   s=int(s0)
   for i,j,J,mr in ALL:
    if ((s>>i)^(s>>j))&1:
     ff.append(s^(1<<i)^(1<<j));mm.append((bi,J,mr))
  ff=np.asarray(ff,np.uint64);uq,iv=np.unique(ff,return_inverse=True)
  lu=evalz(bits2x(uq),batch=2048).real
  rr=np.asarray([diag_one(int(s)) for s in ss],float)
  for jj,(bi,J,mr) in zip(iv,mm): rr[bi]+=.5*J*mr*math.exp(lu[jj]-la[bi])
  ans.append(rr);print('RMY',min(st+chunk,len(states)),'/',len(states),flush=True)
 return np.concatenate(ans)
rmy=rM_batch(uy); qy=np.where(rmy<=t,1.,-1.)
qx=np.where(eM[sel]<=t,1.,-1.)

eK=np.asarray([diag_one(int(s)) for s in sx],float)
for jj,(li,J,mr) in zip(inv,meta):
 amp=math.exp(lay[jj]-lax[li])
 eK[li]+=.5*J*mr*amp*qy[jj]*qx[li]
exactA=eK-eVA[sel];exactB=eK-eVB[sel]
print('OLD_NEW',[(int(ii),float(dKA[ii]),float(exactA[k])) for k,ii in enumerate(sel)],flush=True)
dKA[sel]=exactA;dKB[sel]=exactB

def report(name,x):
 x=np.asarray(x,float)
 print(name,'mean',x.mean(),'naiveSE',x.std(ddof=1)/np.sqrt(len(x)),
       'nonzero',np.sum(abs(x)>1e-12),'quant',np.quantile(x,[0,.01,.05,.5,.95,.99,1]).tolist(),flush=True)
 for b in [16,32,64]:
  nb=len(x)//b;bm=np.array([x[i*b:(i+1)*b].mean() for i in range(nb)])
  print(name,'block',b,'SE',bm.std(ddof=1)/np.sqrt(nb),'means',bm.tolist(),flush=True)
report('REFINED_KminusViTActual',dKA);report('REFINED_KminusViTBinary',dKB)
np.savez_compressed('energy_krylov_vs_vit_6x6_refined.npz',states=xs,dKA=dKA,dKB=dKB,sel=sel,exactA=exactA,exactB=exactB,t=t)
