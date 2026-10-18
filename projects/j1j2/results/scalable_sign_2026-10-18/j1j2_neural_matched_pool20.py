#!/usr/bin/env python3
"""Fairer local-guide ablation: two amplitude CNNs score the SAME walker-discovered
configurations and their one-hop neighborhoods. Walkers use the point-CNN guide;
only model predictions (no exact FN oracle) enter sign decisions. Exact FN
amplitudes are post hoc energy labels. Includes scorer/explorer query counts.
"""
import os,math,json,sys,time
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
import numpy as np,torch
import j1j2_neural_local20 as N
import j1j2_local_query_diagnostic as M
P=Path('/mnt/data')

def load(seed,arm):
 d=torch.load(P/f'j1j2_neural_guide20_{seed}_{arm}.pt',map_location='cpu',weights_only=False)
 net=N.CNN();net.load_state_dict(d['state_dict']);net.eval()
 return net,float(d['mean']),float(d['sd'])

def walkers(model,mu,sd,seed,n_seed=120,burn=1200,thin=12):
 rng=np.random.default_rng(seed);s=N.LearnedGuide(model,mu,sd)
 x=int(M.states[int(rng.integers(M.D))]);visited=[]
 for t in range(burn+n_seed*thin):
  x,_=s.propose(x,rng)
  if t>=burn and (t-burn)%thin==0:visited.append(x)
 seeds=sorted(set(visited));poolstates=set(seeds)
 for x in seeds:
  for y,h in s.neighbors(x):poolstates.add(y)
 pool=sorted({s.orbit(x) for x in poolstates})
 return s,seeds,pool

def score_model(model,mu,sd,seeds,pool,base_s,budget=300,policy='observed_signed'):
 s=N.LearnedGuide(model,mu,sd);s.keys=base_s.keys.copy();s.members=base_s.members.copy()
 edge={g:0. for g in pool};counts={g:0 for g in pool}
 for x in seeds:
  ax=s.amp(x);sx=int(M.signs[s.idx(x)])
  for y,h in s.neighbors(x):
   g=s.orbit(y);edge[g]+=h*sx*int(M.signs[s.idx(y)])*ax*s.amp(y);counts[g]+=1
 proxy={g:edge[g]/counts[g] if counts[g] else 0. for g in pool}
 prio=sorted(pool,key=lambda g:(-proxy[g],g))[:min(len(pool),budget)]
 return N.evaluate_selection(s,prio),dict(n_model_orbit_queries=len(s.orb_amps),candidate_orbits=len(pool))

def main():
 result=[]
 for seed in (0,1):
  point=load(seed,'point');edge=load(seed,'edge')
  for t in range(3):
   sid=3000+seed*100+t
   t0=time.time();s,visited,pool=walkers(*point,sid)
   obs=[]
   for arm in ('point','edge'):
    model,mu,sd=point if arm=='point' else edge
    r,meta=score_model(model,mu,sd,visited,pool,s)
    obs.append(dict(arm=arm,actual_gain=r['actual_gain'],predicted_gain=r['predicted_gain'],n_selected=r['n_selected'],n_model_orbit_queries=meta['n_model_orbit_queries']))
   row=dict(seed=seed,trial=t,explorer='point',n_seed_visited=len(visited),n_orbit_pool=len(pool),exploration_model_calls=s.batches,arms=obs,elapsed=round(time.time()-t0,2))
   result.append(row)
   print('TRIAL',seed,t,[(v['arm'],round(v['actual_gain'],8),v['n_selected']) for v in obs],flush=True)
   (P/'j1j2_neural_matched_pool20_results.json').write_text(json.dumps(dict(in_progress=True,per_trial=result),indent=2))
 summary={k:{a:float(np.mean([next(v[k] for v in r['arms'] if v['arm']==a) for r in result])) for a in ('point','edge')} for k in ('actual_gain','predicted_gain','n_selected','n_model_orbit_queries')}
 (P/'j1j2_neural_matched_pool20_results.json').write_text(json.dumps(dict(in_progress=False,summary=summary,per_trial=result,limitations=['Pointwise CNN defines the independent learned-guide walker candidate pool; both scoring models see identical seed configurations and neighbor candidates.','Training used exact FN amplitude labels from the finite-size oracle.','FN table is used for post-decision evaluation only, and exact sign guide still comes from stored finite-size table.','Sign-update energy is fixed-amplitude; neural output generalization or subsequent FN solve not measured.']),indent=2))
 print('SUMMARY',json.dumps(summary),flush=True)
if __name__=='__main__':main()
