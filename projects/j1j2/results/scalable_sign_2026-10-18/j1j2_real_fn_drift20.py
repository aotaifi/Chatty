#!/usr/bin/env python3
"""Diagnose realistic *correlated* amplitude changes: two consecutive exact FN solves.
Old amplitude a is FN guide at iteration k; next amplitude a' is another FN relaxation with the same frozen signs. Candidate support and cheap bond screening use a. Compare actual energy improvement under a' with updates selected from a. No ED training labels or target amplitudes used in selection.
"""
import os,json
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
from collections import defaultdict
import j1j2_budgeted_orbit20 as B
import j1j2_local_query_diagnostic as M
NEXT=np.load('/mnt/data/sign20_suppression.npz')['next_amp']
class NextGuide(M.Sampler):
 def amp(self,x):
  x=int(x);self.n_amp_requests+=1
  if x not in self.amps:self.amps[x]=float(NEXT[self.idx(x)])
  return self.amps[x]

def eval_new(selected, s):
 sel=set(selected);res=0.
 for g in sel:
  ag=s.amp(g);sg=int(M.signs[s.idx(g)]);n=len(s.members[g])
  for y,h in s.neighbors(g):
   if s.orbit(y) in sel:continue
   res-=4*n*h*ag*s.amp(y)*sg*int(M.signs[s.idx(y)])
 return -float(res)

def choose(prio, s,maxflips=40):
 edges={g:B.exact_score_one(s,g)[1] for g in prio};chosen=[];sel=set();margin=[]
 for i in range(maxflips):
  top=None;gain=0.;pair=None
  for g in prio:
   if g in sel:continue
   vv=[(-w if h in sel else w) for h,w in edges[g].items()]
   p=sum(v for v in vv if v>0);q=-sum(v for v in vv if v<0)
   v=4*(p-q)
   if v>gain:gain=v;top=g;pair=(p,q)
  if top is None:break
  chosen.append(top);sel.add(top)
  p,q=pair;margin.append(float(.5*np.log(p/q)) if q else float('inf'))
 return chosen,margin

def run(seed,Budget=300):
 s,pool,amp,proxy,info=B.generate(seed)
 prio=sorted(pool,key=lambda g:(-proxy['observed_signed'][g],g))[:Budget]
 chosen,margins=choose(prio,s)
 ns=NextGuide();ns.keys=s.keys.copy();ns.members=s.members.copy()
 chosen_new,margins_new=choose(prio,ns)
 ev_old=eval_new(chosen,s)
 ev_forward=eval_new(chosen,ns)
 ev_new=eval_new(chosen_new,ns)
 # local-ratio changes: amp_next[y]/amp_next[g] relative to amp_old[y]/amp_old[g], across encountered priority orbit neighbor edges
 errs=[];weights=[];pair_orbits=set();stabilized=[]
 for g in prio:
  dg=float(np.log(ns.amp(g)/s.amp(g)))
  for y,h in s.neighbors(g):
   gy=s.orbit(y)
   if gy==g:continue
   pair=(min(g,gy),max(g,gy))
   if pair in pair_orbits:continue
   pair_orbits.add(pair)
   dgy=float(np.log(ns.amp(y)/s.amp(y)))
   errs.append(dgy-dg)
   weights.append(abs(h*s.amp(g)*s.amp(y)))
 errs=np.asarray(errs);weights=np.asarray(weights)
 # How many of the old-selected orbits would actually increase energy if flipped individually against the new-amplitude guide, in their original greedy order?
 sett=set();bad=0;newgains=[]
 for g in chosen:
  # per-orbit crossing weight using next amplitudes, with actual preceding selected
  w=B.exact_score_one(ns,g)[1]
  actual_gain=4*sum(-v if h in sett else v for h,v in w.items())
  newgains.append(float(actual_gain));bad+=(actual_gain<=0);sett.add(g)
 return dict(seed=seed,n_selected=len(chosen),gain_old=float(ev_old),gain_next_on_old_flips=float(ev_forward),gain_next_optimized=float(ev_new),n_bad_sequential_gains=int(bad),n_next_selected=len(chosen_new),old_margin_median=float(np.median(margins)) if margins else None,edge_drift_abs_median=float(np.median(abs(errs))),edge_drift_abs_p90=float(np.quantile(abs(errs),.9)),edge_drift_abs_p99=float(np.quantile(abs(errs),.99)),edge_drift_weighted_rms=float(np.sqrt(np.average(errs**2,weights=weights))),n_neighbor_pairs=len(errs),candidate_pool=len(pool),predicted_old_gain_recomputed=float(ev_old),sequential_next_gains=newgains)
if __name__=='__main__':
 rows=[]
 for seed in range(201,213):
  row=run(seed);rows.append(row)
  print('SEED',seed,'gains old,next,on next-opt',*[round(row[k],7) for k in ['gain_old','gain_next_on_old_flips','gain_next_optimized']], 'bad',row['n_bad_sequential_gains'],'ratios',round(row['edge_drift_abs_median'],4),round(row['edge_drift_abs_p90'],4),flush=True)
 summary={x:float(np.mean([r[x] for r in rows])) for x in ['gain_old','gain_next_on_old_flips','gain_next_optimized','n_bad_sequential_gains','n_selected','n_next_selected','old_margin_median','edge_drift_abs_median','edge_drift_abs_p90','edge_drift_abs_p99','edge_drift_weighted_rms']}
 summary['total_trials_negative_energy']=int(sum(r['gain_next_on_old_flips']<0 for r in rows))
 print('SUMMARY',summary)
 with open('/mnt/data/j1j2_real_fn_drift20_results.json','w') as f:json.dump({'summary':summary,'per_seed':rows,'caveats':['FN next amplitude is another exact frozen-sign FN relaxation, not an independently trained NQS.','Candidate pools, priorities, initial signs fixed by old amplitudes.','Full-space amplitude vectors served via an on-demand oracle.','All energy gains calculated at fixed signs and amplitudes, not post-FN reoptimization.']},f,indent=2)