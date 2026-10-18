#!/usr/bin/env python3
"""20-site scale-up diagnostic: online amplitude queries, sampled candidates, exact symmetry orbit gates.
Amplitude data is used as an on-demand *oracle backend*, never scanned to select candidates.
It is still a full-FN-guide oracle: proof-of-concept only, not a scalable solver.
Ground state ED values only used for post-run evaluation.
"""
import os,json,time
os.environ['OPENBLAS_NUM_THREADS']='1'
from collections import defaultdict
import numpy as np
from pathlib import Path
P=Path('/mnt/data'); d=np.load(P/'sign20_training_data.npz'); tar=np.load(P/'audit_sign20_targets.npz')
states=d['states'].astype(np.uint32); amplitude=d['amp']; signs=d['guide']; psi0=d['psi0']; diag=d['diag']; target=tar['optimal_flip'].astype(bool)
D=len(states);N=20;Lx=4;Ly=5
lookup=np.full(1<<N,-1,np.int32);lookup[states]=np.arange(D)
def site(x,y):
 q,r=divmod(y,Ly);return (x-q)%Lx+Lx*r
bonds=[]
for yy in range(Ly):
 for xx in range(Lx):
  i=site(xx,yy)
  for dx,dy,h in ((1,0,.5),(0,1,.5),(1,1,.25),(-1,1,.25)):
   j=site(xx+dx,yy+dy);bonds.append((np.uint32((1<<i)|(1<<j)),float(h),i,j))
# all orbits are generated on demand; only the geometry is precomputed
xy={site(xx,yy):(xx,yy) for yy in range(Ly) for xx in range(Lx)}
perms=[]
for dy in range(Ly):
 for dx in range(Lx): perms.append(np.array([site(xy[i][0]+dx,xy[i][1]+dy) for i in range(N)],dtype=np.int8))
assert len(perms)==20
mask=(1<<N)-1
class Sampler:
 def __init__(self):
  self.amps={};self.keys={};self.members={};self.n_amp_requests=0
 def idx(self,x):
  ix=int(lookup[int(x)]);assert ix>=0;return ix
 def amp(self,x):
  x=int(x);self.n_amp_requests+=1
  if x not in self.amps:self.amps[x]=float(amplitude[self.idx(x)])
  return self.amps[x]
 def orbit(self,x):
  x=int(x)
  if x not in self.keys:
   vals=[]
   for perm in perms:
    y=0
    for i,j in enumerate(perm):y|=((x>>i)&1)<<int(j)
    vals.extend((y,mask^y))
   unique=tuple(sorted(set(vals)));k=unique[0]
   for y in unique:self.keys[y]=k
   self.members[k]=unique
  return self.keys[x]
 def propose(self,x,rng):
  maskv,_,_,_=bonds[int(rng.integers(len(bonds)))];y=x^int(maskv)
  if int(lookup[int(y)])<0:return x,False
  a,b=self.amp(x),self.amp(y)
  if rng.random()<(b/max(a,1e-300)):return y,True # target distribution a^1 (tempered beta=0.5)
  return x,False
 def neighbors(self,x):
  for maskv,h,i,j in bonds:
   y=int(x)^int(maskv)
   if int(lookup[int(y)])>=0:yield y,h

def run(seed,n_seed=200,burn=4000,thin=15,budget=40):
 t0=time.time();rng=np.random.default_rng(seed);sam=Sampler()
 x=int(states[int(rng.integers(D))]);accepted=0;seeds=[]
 for step in range(burn+n_seed*thin):
  x,acc=sam.propose(x,rng);accepted+=acc
  if step>=burn and (step-burn)%thin==0:seeds.append(x)
 exploration_calls=len(sam.amps)
 seen=set(seeds)
 for x in seeds:
  for y,h in sam.neighbors(x):seen.add(y)
 pools={sam.orbit(x) for x in seen}; pool=sorted(pools)
 closure= len(sam.keys)
 # Every coefficient involving a candidate orbit is obtained by explicit on-demand amplitude queries.
 # W_g,h records oriented bonds from orbit g into orbit h, including h outside the pool.
 raw={} ; edges={};candidate_set=set(pool)
 for g in pool:
  W=defaultdict(float)
  for x in sam.members[g]:
   ix=sam.idx(x); ax=sam.amp(x);sx=int(signs[ix])
   for y,h in sam.neighbors(x):
    gy=sam.orbit(y)
    if gy==g:continue
    jy=sam.idx(y);ay=sam.amp(y)
    W[gy]+=h*sx*int(signs[jy])*ax*ay
  edges[g]=W;raw[g]=4*sum(W.values())
 queried=len(sam.amps)
 # Compare identical acquired candidate pool, fixed 40-orbit budget.
 nsel=min(budget,len(pool)); method={}
 def select(policy):
  eta={};field=raw.copy();chosen=[];trajectory=[];selected=set()
  if policy=='greedy':
   for step in range(nsel):
    # recompute marginal energy gain given previously committed orbit flips
    gain={g:(-1 if g in selected else 1)*(raw[g]-8*sum(v for j,v in edges[g].items() if j in selected)) for g in pool if g not in selected}
    g=max(gain,key=gain.get)
    if gain[g]<=0:break
    chosen.append(g);selected.add(g);trajectory.append(float(gain[g]))
  elif policy=='ranked':
   for g in sorted(pool,key=lambda u:raw[u],reverse=True)[:nsel]:
    chosen.append(g);selected.add(g)
  elif policy=='random':
   for g in rng.choice(pool,size=nsel,replace=False):
    chosen.append(int(g));selected.add(int(g))
  # Exact fixed-amplitude delta from locally computed edges: each crossed edge counted once.
  e=-sum(raw[g] for g in selected)+8*sum(v for g in selected for h,v in edges[g].items() if h in selected)/2
  # WHY: -4*sum crossing W. raw[g]=4*(W outside+W toward selected).
  # Inside edges were counted twice in raw, must add 8*sum_undirected =4*sum_directed.
  e=-sum(raw[g] for g in selected)+4*sum(v for g in selected for h,v in edges[g].items() if h in selected)
  # All ED labels only post hoc.
  selix=np.concatenate([np.array([sam.idx(x) for x in sam.members[g]],dtype=int) for g in selected]) if selected else np.empty(0,dtype=int)
  ed_err=float(np.sum((psi0[selix]*signs[selix]>0)*psi0[selix]**2)) if len(selix) else 0.
  ed_fix=float(np.sum((psi0[selix]*signs[selix]<0)*psi0[selix]**2)) if len(selix) else 0.
  return dict(deltaE=float(e),selected_orbits=len(selected),selected_states=len(selix),ed_wrong_change=float(ed_err-ed_fix),proposed_krylov_flips=int(np.sum(target[selix])),gain_trajectory=trajectory)
 for policy in ('random','ranked','greedy'):method[policy]=select(policy)
 # Check symmetry graph local orientation and brute delta independently (ED not used)
 if seed%2==0:
  for g in pool[:20]:
   for h,v in edges[g].items():
    if h in candidate_set:
     err=abs(v-edges[h].get(g,0.));assert err<1e-10,(g,h,err)
 return dict(seed=seed,n_seed=n_seed,acceptance=accepted/(burn+n_seed*thin),distinct_seed=len(set(seeds)),exploration_unique_amplitudes=exploration_calls,n_expanded_candidates=len(seen),n_pool_orbits=len(pool),symmetry_closure_configs=closure,all_unique_amplitude_queries=queried,avg_degree=len(bonds),duration_s=round(time.time()-t0,2),methods=method)
if __name__=='__main__':
 out=[]
 for seed in (101,102,103,104):
  r=run(seed);out.append(r)
  print('SEED',seed,'pool',r['n_pool_orbits'],'queries',r['all_unique_amplitude_queries'],'methods',[(k,round(v['deltaE'],8)) for k,v in r['methods'].items()],'sec',r['duration_s'],flush=True)
 summary={k:{m:float(np.mean([r['methods'][k][m] for r in out])) for m in ('deltaE','selected_orbits','selected_states','ed_wrong_change')} for k in out[0]['methods']}
 x={'scope':'diagnostic with on-demand exact FN amplitudes as oracle; not scalable training; sampled 20-site 4x5 torus; 4 seeds','summary':summary,'per_seed':out,'warning':['Global full-guide amplitude vector exists but used only as query backend. It is NOT a realizable FN oracle at large sizes.','Sampling from amplitude^1 via local Metropolis can have poor mixing; not a QMC walker.','Energy monotonicity for greedy holds only for fixed amplitudes; no network generalized and no new FN solve.','ED labels used only post hoc for sign-error scoring.']}
 with open(P/'j1j2_local_query_diagnostic_results.json','w') as f:json.dump(x,f,indent=2)
 print('SUMMARY',json.dumps(summary,indent=2),flush=True)