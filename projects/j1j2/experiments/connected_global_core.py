import math,time,json
import numpy as np
import scipy.sparse as sp
exec(open('bench_westerhout_greedy.py').read().split("for rr in range(6): paper_trial_faithful")[0])

ALPHA=.8; SEED=202609281; rng=np.random.default_rng(SEED)
Aidx=np.array([x+L*y for y in range(L) for x in range(L) if (x+y)%2==0],dtype=int)
def marshall_arr(X):
 ndown=np.sum(X[:,Aidx]<0,axis=1);return np.where(ndown%2==0,1,-1).astype(np.int8)
# equilibrium-ish root from the fixed alpha=.8 training sample
st=np.load('global_ticnn_states.npz'); root=int(st['train_states'][0])
nchains=32
ss=[root]*nchains
la=evalz(bits2x(ss)).real
ALL=[(i,j,J) for bb,J in ((NN,1.0),(NNN,J2)) for i,j in bb]
core={root}; prod=[]; visits=0; accepted=0
maxsteps=900; mincore=2200
for step in range(maxsteps):
    cand=[]; moved=[]
    for q in ss:
        i,j,J=ALL[int(rng.integers(len(ALL)))]
        if ((q>>i)^(q>>j))&1:
            cand.append(q^(1<<i)^(1<<j));moved.append(True)
        else:
            cand.append(q);moved.append(False)
    lb=evalz(bits2x(cand)).real
    ac=(np.log(rng.random(nchains))<np.minimum(0,ALPHA*(lb-la))) & np.asarray(moved)
    for k in np.where(ac)[0]:
        ss[k]=cand[k];core.add(int(cand[k]));accepted+=1
    la=np.where(ac,lb,la)
    if step>=100:
        prod.extend(ss);visits+=nchains
    if step%100==0:
        print('PATH',step,'core',len(core),'accept',accepted/max(1,(step+1)*nchains),flush=True)
    if step>=250 and len(core)>=mincore:break
print('CORE_DONE steps',step+1,'core',len(core),'prodvisits',len(prod),'accept',accepted/((step+1)*nchains),flush=True)
np.savez_compressed('connected_core_states.npz',core=np.array(sorted(core),np.uint64),prod=np.array(prod,np.uint64))

ext=extend(core,1)
arr=np.array(sorted(ext),np.uint64); print('EXT',len(arr),flush=True)
z=evalz(bits2x(arr));laa=z.real;tru,leak=hidden(z,laa)
u,v,w=edges(arr,laa); print('GRAPH edges',len(w),'leak',leak,flush=True)
pos={int(q):i for i,q in enumerate(arr)}
cm=np.zeros(len(arr),bool)
for q in core:cm[pos[int(q)]]=True
star=cm[u]|cm[v]
print('STAR edges',int(star.sum()),'INDUCED edges',len(w),flush=True)
# solve source-faithful greedy
pstar,ns1=solve_westerhout_greedy(len(arr),u[star],v[star],w[star])
print('SOLVED_STAR sweeps',ns1,flush=True)
pind,ns2=solve_westerhout_greedy(len(arr),u,v,w)
print('SOLVED_INDUCED sweeps',ns2,flush=True)
marr=marshall_arr(bits2x(arr))

def diag(s):
 e=0.
 for bb,J in ((NN,1.0),(NNN,J2)):
  for i,j in bb:e+=J*(.25 if (((s>>i)&1)==((s>>j)&1)) else -.25)
 return e

prod=np.asarray(prod,np.uint64)
# production sample is from q~a^alpha; map to union
pi=np.array([pos[int(q)] for q in prod],np.int32)
lwp=(2-ALPHA)*laa[pi];lwp-=lwp.max();iw=np.exp(lwp);iw/=iw.sum()
def score(sign):
 Oq=abs(float(np.mean(sign[pi]*tru[pi])))
 Op=abs(float(np.sum(iw*sign[pi]*tru[pi])))
 return Oq,Op
def energy(sign):
 vals=[]
 for q,ii in zip(prod,pi):
  e=diag(int(q));sx=int(sign[ii]);lx=laa[ii]
  for t,J in neigh(int(q)):
   jj=pos[int(t)]
   e += .5*J*math.exp(laa[jj]-lx)*sx*int(sign[jj])
  vals.append(e)
 vals=np.asarray(vals);E=float(np.sum(iw*vals))
 # paired uncertainty handled outside
 return E,vals
scores={}
for name,s in [('Marshall',marr),('Star',pstar),('Induced',pind),('Hidden',tru)]:
 Oq,Op=score(s);E,vals=energy(s);scores[name]=(Oq,Op,E,vals)
 print('RESULT',name,'Oq',Oq,'Ophys',Op,'E',E,flush=True)
base=scores['Marshall'][3]
ess=1/np.sum(iw*iw)
for name in ('Star','Induced','Hidden'):
 d=scores[name][3]-base;mu=float(np.sum(iw*d));se=float(np.sqrt(np.sum(iw*(d-mu)**2)/ess))
 print('DELTA',name,'dE',mu,'se',se,'ESS',ess,flush=True)
with open('connected_global_core_result.json','w') as f:
 json.dump({k:{'Oq':float(v[0]),'Ophys':float(v[1]),'E':float(v[2])} for k,v in scores.items()},f,indent=2)
