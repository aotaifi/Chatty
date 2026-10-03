import numpy as np, time
exec(open("gfmc_validate_4x4_vec.py").read().split('run(10000,.001')[0])
r=np.asarray((H@(a0*sM))/(a0*sM));q=a0*a0;q/=q.sum()
def wq(x,w,p):
 o=np.argsort(x);c=np.cumsum(w[o]);return np.interp(p,c/c[-1],x[o])
edges=np.array([-np.inf]+[wq(r,q,p) for p in (.25,.5,.75)]+[np.inf])
b=np.clip(np.searchsorted(edges,r,side='right')-1,0,3)
qm=np.bincount(b,weights=q,minlength=4);qm/=qm.sum()
fm=np.bincount(b,weights=f_exact,minlength=4);fm/=fm.sum()
gt=np.log(fm/qm);gt-=np.sum(qm*gt)
print('TRUE',gt.tolist(),flush=True)
def evolve(M,steps,seed):
 rng=np.random.default_rng(seed);w=rng.choice(D,M,p=q)
 for it in range(steps):
  Eref=float(np.mean(Eloc[w]));st=1-.001*(diag[w]-Eref);mw=.001*rate[w];tot=st+mw.sum(1)
  u=rng.random(M)*tot;c=np.cumsum(np.c_[st,mw],1);k=(u[:,None]>c).sum(1);nxt=w.copy();mv=k>0
  nxt[mv]=nei[w[mv],k[mv]-1];w=nxt[sysidx(tot/tot.sum(),rng,M)]
 return w
def getg(cnt):
 ff=(cnt+.5)/(cnt.sum()+2);g=np.log(ff/qm);g-=np.sum(qm*g);return g
# independent terminal populations: restart from q every population
for M,R,steps in [(32,100,1000),(128,50,1000),(128,100,1000),(128,50,2000)]:
 gs=[];cnt=np.zeros(4)
 for j in range(R):
  w=evolve(M,steps,100000+1000*M+10*steps+j);cnt+=np.bincount(b[w],minlength=4)
  if (j+1)%10==0:gs.append(getg(cnt.copy()))
 g=getg(cnt)
 print('IND',M,R,steps,'Nterminal',M*R,'g',g.tolist(),'err',float(np.linalg.norm(g-gt)),flush=True)
 # jackknife-ish block variability: 10 groups
 group=[]
 for j0 in range(10):
  cc=np.zeros(4)
  for j in range(max(1,R//10)):
   w=evolve(M,steps,900000+j0*100+j);cc+=np.bincount(b[w],minlength=4)
  group.append(getg(cc))
 print('GROUPSD',np.std(group,axis=0,ddof=1).tolist(),flush=True)
