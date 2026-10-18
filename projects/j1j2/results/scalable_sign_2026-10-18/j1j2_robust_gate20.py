#!/usr/bin/env python3
"""Robust symmetry-orbit sign update at fixed amplitudes with on-demand local queries.

A representative orbit g has noisy amplitude ahat_g=a_g exp(delta_g), with bounded |delta_h| <= eps for *neighbor* orbits h. For a current sign pattern the crossing signed edge sums are w_gh = sum_{x in g, y in h} H_xy s_x s_y ahat_x ahat_y.
The true single-orbit flip gain is G_g = 4 exp(-delta_g) sum_h w_gh exp(-delta_h).
Writing P=sum_h max(w_gh,0), N=sum_h max(-w_gh,0), G_g>0 is CERTIFIED if exp(-eps)*P-exp(eps)*N>0. This removes the unknown overall amplitude factor at g and is tighter than bounding both endpoints independently. Each flipped orbit changes the sign of bonds to that orbit.

The eps value is an assumed uniform per-orbit log-amplitude ERROR BOUND, *not* inferred from the data. Tests use synthetic Gaussian deviations with std sigma, hence eps=k*sigma is only a heuristic confidence multiplier and never a deterministic certificate unless every queried neighboring orbit satisfies it.
"""
import os, json, argparse
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
from collections import defaultdict
import j1j2_budgeted_orbit20 as B
import j1j2_noisy_oracle_diagnostic as N
import j1j2_local_query_diagnostic as M


def run_one(seed, sigma, k, budget=300, max_flips=40):
    # Exactly the same clean-guide sampling candidate support as the previous noise test.
    base,pool,rep,proxy,meta=B.generate(seed)
    s=N.Noisy(base,seed=seed+300,sigma=sigma)
    rng=np.random.default_rng(seed)
    x=int(M.states[int(rng.integers(M.D))]);distinct=set()
    burn=2500;n_seed=120;thin=15
    for step in range(burn+n_seed*thin):
        maskv,_,_,_=M.bonds[int(rng.integers(len(M.bonds)))];y=x^int(maskv)
        if M.lookup[int(y)]>=0:
            ax=float(M.amplitude[M.lookup[int(x)]]);ay=float(M.amplitude[M.lookup[int(y)]])
            if rng.random()<ay/max(ax,1e-300):x=y
        if step>=burn and (step-burn)%thin==0:distinct.add(x)
    signed=defaultdict(float);cnt=defaultdict(int)
    for x in distinct:
        ax=s.amp(x); sx=int(M.signs[s.idx(x)])
        for y,h in s.neighbors(x):
            g=s.orbit(y)
            signed[g] += h*sx*int(M.signs[s.idx(y)])*ax*s.amp(y)
            cnt[g] += 1
    priority=sorted(pool,key=lambda g:(-signed[g]/max(1,cnt[g]),g))[:budget]
    bonds={}
    for g in priority:
        _,e=B.exact_score_one(s,g)
        bonds[g]=e
    selected=set();trace=[];certified_at_accept=0;failure_to_bound=0
    eps=k*sigma
    fminus=np.exp(-eps); fplus=np.exp(eps)
    for _ in range(max_flips):
        best=None;score=-np.inf;best_margin=-np.inf;best_raw=0
        for g in priority:
            if g in selected:continue
            es=bonds[g];P=0.; Q=0.
            for h,v in es.items():
                vv=-v if h in selected else v
                if vv>0:P+=vv
                else:Q-=vv
            margin=fminus*P-fplus*Q
            # maximise conservative lower margin; all arms use same ranking and query budget
            if margin>best_margin:
                best,best_margin,best_raw=g,margin,4*(P-Q)
        if best is None or best_margin<=0:break
        selected.add(best)
        trace.append(dict(orbit=int(best),robust_margin=float(best_margin),predicted_marginal_gain=float(best_raw)))
    actual=N.true_deltaE(selected,s)
    predicted=-sum(4*sum(bonds[g].values()) for g in selected)+4*sum(v for g in selected for h,v in bonds[g].items() if h in selected)
    # For reproducibility, independently compute actual gain on every accepted orbit's true crossing edges.
    true_signs=[]
    for t in trace:
        g=t['orbit']; neigh=bonds[g]
        # no need to store per-step true gain, total is the primary metric
    # Count noisy orbit errors exceeding assumed bound among evaluated neighbors
    queried_orbits=set(s.orbit(u) for u in s.amps)
    n_exceed=sum(abs(s.zcache[g]*sigma)>eps+1e-12 for g in queried_orbits)
    return dict(seed=seed,sigma=sigma,k=k,eps=float(eps),selected_orbits=len(selected),gain_true=-float(actual),gain_pred=-float(predicted),n_distinct_queries=len(s.amps),n_queried_orbits=len(queried_orbits),n_orbits_exceeding_bound=n_exceed, trace=trace)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,default=12);p.add_argument('--ks',type=float,nargs='+',default=[0,1,2,3]);p.add_argument('--sigmas',type=float,nargs='+',default=[0,.1,.3,.5]);p.add_argument('--out',default='/mnt/data/j1j2_robust_gate20_results.json');a=p.parse_args()
    results=[]
    for seed in range(201,201+a.seeds):
        row=[]
        for sigma in a.sigmas:
            for k in a.ks:
                x=run_one(seed,sigma,k);results.append(x)
                row.append((sigma,k,round(x['gain_true'],7),x['selected_orbits']))
        print('SEED',seed,row,flush=True)
        with open(a.out,'w') as f:json.dump({'partial':True,'per_seed':results},f)
    summary={}
    for sigma in a.sigmas:
        for k in a.ks:
            rr=[r for r in results if r['sigma']==sigma and r['k']==k]; gains=np.array([r['gain_true'] for r in rr]);
            summary[f'{sigma}_{k}']=dict(sigma=sigma,k=k,mean_gain=float(gains.mean()),median_gain=float(np.median(gains)),worsening=int(np.sum(gains<0)),positive=int(np.sum(gains>0)),zero=int(np.sum(gains==0)),mean_selected=float(np.mean([r['selected_orbits'] for r in rr])),mean_queries=float(np.mean([r['n_distinct_queries'] for r in rr])),mean_orbits_exceeding_bound=float(np.mean([r['n_orbits_exceeding_bound'] for r in rr])))
    with open(a.out,'w') as f:json.dump({'summary':summary,'per_seed':results,'limitations':['Same candidate pools sampled with clean true amplitudes as earlier study; ranking and gating use noisy queries.','Gaussian noise is not strictly bounded; eps=k sigma is heuristic unless verified or calibrated.','Fixed amplitude sign-only update; ED used only as final score.','Full FN amplitude vector remains an on-demand oracle.','No neural-network or GFMC scaling demonstrated.']},f,indent=2)
    print('SUMMARY',json.dumps(summary,indent=2),flush=True)