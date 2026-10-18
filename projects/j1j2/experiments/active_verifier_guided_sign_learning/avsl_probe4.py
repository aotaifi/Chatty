#!/usr/bin/env python3
"""AVSL 4x4 on-demand analytic amplitude exploration. No ED/FN amplitude oracle.
Evaluates proposed symmetry-orbit flips at fixed guide amplitude, with strict query counts.
This is a method/software diagnostic, not evidence of large-size scaling.
"""
from __future__ import annotations
from collections import defaultdict
import json, math, argparse
from functools import lru_cache
from pathlib import Path
import numpy as np

L=4; N=L*L; FULL=(1<<N)-1; K=8
BONDS=[]
for y in range(L):
 for x in range(L):
  i=y*L+x
  for dx,dy,J in ((1,0,1.),(0,1,1.),(1,1,.5),(-1,1,.5)):
   j=((y+dy)%L)*L+((x+dx)%L)
   BONDS.append((i,j,J,.5*J,(1<<i)|(1<<j)))
assert len(BONDS)==64
PERMS=tuple(tuple(((y+dy)%L)*L+(x+dx)%L for y in range(L) for x in range(L)) for dy in range(L) for dx in range(L))

@lru_cache(maxsize=None)
def transformed(x,p):
 return sum(((x>>i)&1)<<j for i,j in enumerate(PERMS[p]))

@lru_cache(maxsize=None)
def orbit(x):
 z=set()
 for p in range(len(PERMS)):
  a=transformed(x,p);z.update((a,a^FULL))
 return tuple(sorted(z))

@lru_cache(maxsize=None)
def key(x):return orbit(x)[0]

class Oracle:
 """Callable analytic guide; each distinct state is counted once."""
 def __init__(self,kappa=.32):
  self.kappa=kappa; self.asked={}; self.neighbor_calls=0
 def evaluate(self,x):
  x=int(x)
  if x not in self.asked:
   diag=0.
   for i,j,J,hop,mask in BONDS:
    diag+=J*(.25 if ((x>>i)&1)==((x>>j)&1) else -.25)
   marshall=(-1)**sum((x>>(y*L+xx))&1 for y in range(L) for xx in range(L) if (xx+y)%2==0)
   a=math.exp(-self.kappa*diag)
   self.asked[x]=(a,marshall)
  return self.asked[x]
 def neighbors(self,x):
  self.neighbor_calls+=1
  for i,j,J,hop,mask in BONDS:
   if bool((x>>i)&1) != bool((x>>j)&1):yield x^mask,hop

@lru_cache(maxsize=None)
def first_bitstring(rngseed):
 return sum(1<<i for i in np.random.default_rng(rngseed).choice(N,K,replace=False))

def generate(seed=1,n_seeds=200,burn=700,thin=4,kappa=.32):
 rng=np.random.default_rng(seed);g=Oracle(kappa);x=first_bitstring(seed+8811);samples=[]
 for t in range(burn+n_seeds*thin):
  i,j,J,hop,mask=BONDS[int(rng.integers(len(BONDS)))]; xx=x^mask
  if (x>>i&1)!=(x>>j&1):
   a,_=g.evaluate(x);b,_=g.evaluate(xx)
   if rng.random()<min(1.,b/a):x=xx
  if t>=burn and (t-burn)%thin==0:samples.append(x)
 frontier=set(samples); scores=defaultdict(list)
 for x in sorted(set(samples)):
  ax,sx=g.evaluate(x)
  for y,h in g.neighbors(x):
   ay,sy=g.evaluate(y);frontier.add(y)
   if key(y)!=key(x):scores[key(y)].append(float(h*ax*ay*sx*sy))
 pool=sorted({key(x) for x in frontier})
 return g,pool,scores,dict(seed=seed,distinct_seeds=len(set(samples)),n_frontier=len(frontier),n_orbits=len(pool),common_queries=len(g.asked),common_neighbor_calls=g.neighbor_calls)

def score_orbit(g,gid):
 members=orbit(gid); a,s=g.evaluate(gid)
 out=defaultdict(float)
 for y,h in g.neighbors(gid):
  ky=key(y)
  if ky!=gid:
   b,sy=g.evaluate(y)
   out[ky]+=len(members)*float(h*a*b*s*sy)
 return 4*sum(out.values()),out

def evaluate_choice(base,pool,scores,method,seed,B=30,max_flips=12):
 g=Oracle(base.kappa);g.asked=dict(base.asked)
 rng=np.random.default_rng(seed+178121)
 if method=='random':order=list(rng.permutation(pool))
 elif method=='observed_signed':order=sorted(pool,key=lambda k:(-np.mean(scores[k]) if scores[k] else 1.e30,k))
 elif method=='max_violation':order=sorted(pool,key=lambda k:(-max(scores[k]) if scores[k] else 1.e30,k))
 else:raise ValueError(method)
 selected=[];cross={};raw={};increment=[]
 for k in order[:min(B,len(order))]:raw[k],cross[k]=score_orbit(g,k)
 for _ in range(max_flips):
  cand={k:raw[k]-8*sum(w for other,w in cross[k].items() if other in selected) for k in raw if k not in selected}
  if not cand:break
  k=max(cand,key=cand.get)
  if cand[k]<=0:break
  selected.append(int(k));increment.append(float(cand[k]))
 gain=float(sum(increment))
 via_edges=4*sum(w for k in selected for other,w in cross[k].items() if other not in selected)
 assert abs(via_edges-gain)<1e-9,(via_edges,gain)
 return dict(gain=gain,selected=len(selected),checked_orbits=min(B,len(pool)),extra_queries=len(g.asked)-len(base.asked),total_queries=len(g.asked),neighbor_calls=g.neighbor_calls,selected_keys=selected)

def run(seeds=range(101,113),B=30,n_seeds=200,kappa=.32):
 all_rows=[]
 for seed in seeds:
  base,pool,scores,meta=generate(seed,n_seeds=n_seeds,kappa=kappa)
  methods={m:evaluate_choice(base,pool,scores,m,seed,B=B) for m in ['random','observed_signed','max_violation']}
  all_rows.append(dict(**meta,methods=methods))
  print('SEED',seed,'pool',meta['n_orbits'],'common_calls',meta['common_queries'], 'gains', {m:round(methods[m]['gain'],6) for m in methods},flush=True)
 results={'scope':'4x4 analytic callable amplitude, Marshall signs, on-demand guide; no FN/ED oracle, fixed guide frozen-a variational energy',
 'B':B,'kappa':kappa,'n_seeds':n_seeds,'methods':{},'per_seed':all_rows,
 'limitations':['Tempered Metropolis, not FN/GFMC; translations + spin inversion only',
 'Analytic g0, no learning or uncertainty','Gains are UNNORMALIZED energy numerator, not normalized physical energy','Common frontier expansion cost is counted']}
 for m in ('random','observed_signed','max_violation'):
  gains=np.array([row['methods'][m]['gain'] for row in all_rows]); costs=np.array([row['methods'][m]['total_queries'] for row in all_rows])
  results['methods'][m]=dict(mean_gain=float(gains.mean()),std_gain=float(gains.std(ddof=1)),positive_trials=int((gains>0).sum()),
  mean_total_queries=float(costs.mean()),mean_gain_per_1000_queries=float(np.mean(gains/costs)*1000),
  win_vs_random=int(sum(row['methods'][m]['gain']>row['methods']['random']['gain'] for row in all_rows)))
 return results

if __name__=='__main__':
 ap=argparse.ArgumentParser()
 ap.add_argument('--kappa',type=float,default=.32);ap.add_argument('--B',type=int,default=30)
 ap.add_argument('--seeds',type=int,default=12);ap.add_argument('--seed-start',type=int,default=101)
 ap.add_argument('--n-seeds',type=int,default=200);ap.add_argument('--out',default='avsl_probe4_results.json')
 a=ap.parse_args()
 output=run(range(a.seed_start,a.seed_start+a.seeds),a.B,a.n_seeds,a.kappa)
 Path(a.out).write_text(json.dumps(output,indent=2));print('SUMMARY',json.dumps(output['methods'],indent=2))
