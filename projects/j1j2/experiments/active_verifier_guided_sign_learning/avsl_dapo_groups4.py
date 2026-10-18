#!/usr/bin/env python3
"""Pre-RL GRPO/DAPO dynamic-group signal diagnostic, not a trained policy.

4x4 J1-J2 with analytic CALLABLE guide and explicit on-demand query counting.
The keep/no-flip action has reward zero. All rejected binary groups still cost queries.
Run: python avsl_dapo_groups4.py --trials 12 --budget 80 --out results.json
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import avsl_probe4 as p

def arm(seed, kappa, rank, budget=80, group_size=4, n_seeds=180):
    base, pool, scores, metadata=p.generate(seed=seed,n_seeds=n_seeds,kappa=kappa)
    rng=np.random.default_rng(431219+seed)
    randomized=list(rng.permutation(pool))
    ordered=sorted(pool,key=lambda k:(-np.mean(scores[k]) if scores[k] else float('inf'), k))
    if rank=='random':order=randomized
    elif rank=='observed_signed':order=ordered
    elif rank=='half_mix':
        order=[];used=set();a=b=0
        while len(order)<len(pool):
            src=ordered if len(order)%2==0 else randomized
            idx=a if src is ordered else b
            while idx<len(src) and src[idx] in used:idx+=1
            if idx>=len(src):
                src=randomized;idx=b
                while idx<len(src) and src[idx] in used:idx+=1
            if idx>=len(src):break
            val=src[idx];order.append(val);used.add(val)
            if src is ordered:a=idx+1
            else:b=idx+1
    else:raise ValueError(rank)
    g=p.Oracle(base.kappa);g.asked=dict(base.asked)
    n=min(budget,len(order))
    chosen=order[:n]
    records=[]
    for i in range(0, n,group_size):
        group=chosen[i:i+group_size]
        if len(group)!=group_size:break
        gains=[p.score_orbit(g,int(k))[0] for k in group]
        success=[gain>1.e-10 for gain in gains]
        # All-success/all-failure BINARY reward groups have zero group-relative advantage.
        mixed=any(success) and not all(success)
        # Continuous gain and an explicit no-flip reward (0) retain useful no-flip signal.
        keep_anchored=float(np.std([0.]+gains,ddof=0))>1e-10
        records.append(dict(orbits=[int(k) for k in group], gains=gains,
                            binary_mixed=mixed, anchored_var=keep_anchored,
                            positives=sum(success), max_gain=float(max(gains))))
    total_positive=sum(gr['positives'] for gr in records)
    all_negative=sum(gr['positives']==0 for gr in records)
    accepted=[gr for gr in records if gr['binary_mixed']]
    queries=len(g.asked)
    return dict(seed=int(seed),kappa=kappa,rank=rank,pool_orbits=len(pool),
                common_queries=len(base.asked),total_unique_queries=queries,
                extra_unique_queries=queries-len(base.asked),attempted_orbits=group_size*len(records),
                groups=len(records),binary_mixed_groups=len(accepted),
                all_negative_groups=all_negative,
                continuous_nonzero_variance_groups=sum(gr['anchored_var'] for gr in records),
                positive_orbits=total_positive,
                positives_in_retained=sum(gr['positives'] for gr in accepted),
                largest_positive_gain=max((max(0.,gr['max_gain']) for gr in records), default=0.),
                aggregate_positive_single_flip_gain=float(sum(max(0.,v) for gr in records for v in gr['gains'])),
                groups_detail=records)

def run(kappas=(.32,.9),seeds=range(101,113),budget=80,group_size=4,n_seeds=180):
    rows=[]
    for kappa in kappas:
      for seed in seeds:
       for rank in ['random','observed_signed','half_mix']:
        a=arm(seed,kappa,rank,budget,group_size,n_seeds)
        rows.append(a)
       print('RUN',kappa,seed,'positive',[(x['rank'],x['positive_orbits']) for x in rows[-3:]],flush=True)
    summary={}
    for kappa in kappas:
     summary[str(kappa)]={}
     for rank in ['random','observed_signed','half_mix']:
      s=[r for r in rows if r['kappa']==kappa and r['rank']==rank]
      mean=lambda f:float(np.mean([r[f] for r in s]))
      summary[str(kappa)][rank]=dict(mean_positive_orbits=mean('positive_orbits'),
        mean_binary_mixed_groups=mean('binary_mixed_groups'),
        mean_all_negative_groups=mean('all_negative_groups'),
        mean_continuous_anchored_informative_groups=mean('continuous_nonzero_variance_groups'),
        mean_unique_queries=mean('total_unique_queries'),
        mean_extra_queries=mean('extra_unique_queries'),
        mean_positive_per_1000_queries=float(np.mean([r['positive_orbits']/r['total_unique_queries']*1000 for r in s])),
        mean_positive_gain_sum=mean('aggregate_positive_single_flip_gain'),
        mean_positive_retained_by_filter=mean('positives_in_retained'))
    return dict(title='AVSL pre-RL group-reward diagnostic 4x4; no fitted policy',
        setup=dict(kappas=kappas,seeds=list(seeds),budget=budget,group_size=group_size,
                  n_seeds=n_seeds,reward='positive if full symmetry orbit frozen-a energy decreases',
                  scope='analytic callable a=exp(-kappa Hxx), Marshall signs, sampled frontier; no FN/ED oracle',
                  caveats=['NOT an RL training experiment; only group-information diagnostic.',
                           'Group successes ignore interactions between simultaneously proposed flips.',
                           'kappa is a toy analytic guide hyperparameter, not a physical FN iterate.',
                           'All scoring calls including filtered groups included in query budget.',
                           'Gains are unnormalized Hamiltonian-numerator gains; same guide only.',
                           '4x4 alone says nothing about asymptotic sample complexity.']),
         summary=summary,per_seed=rows)

if __name__=='__main__':
 ap=argparse.ArgumentParser()
 ap.add_argument('--budget',type=int,default=80)
 ap.add_argument('--groupsize',type=int,default=4)
 ap.add_argument('--trials',type=int,default=12)
 ap.add_argument('--out',default='avsl_dapo_groups4_results.json')
 a=ap.parse_args()
 result=run(seeds=range(101,101+a.trials),budget=a.budget,group_size=a.groupsize)
 Path(a.out).write_text(json.dumps(result,indent=2))
 print('SUMMARY',json.dumps(result['summary'],indent=2))
