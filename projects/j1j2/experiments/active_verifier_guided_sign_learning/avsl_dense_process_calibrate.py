#!/usr/bin/env python3
"""Heldout audit of physically signed 4-edge *intermediate* process reward.

Ranks intermediate-orbit and terminal-orbit positive fixed-a gains. Uses
disjoint seeds only and never modifies policy training.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata, spearmanr
from avsl_dapo_explorer20 import World,make_roots,DIM
from avsl_dense_process20 import rollout,signed_process,BONDS

def auc(reward,success):
    reward=np.asarray(reward,float);success=np.asarray(success,bool)
    n1=int(success.sum());n0=int((~success).sum())
    if min(n0,n1)==0:return None
    # Equivalent Wilcoxon positive-vs-negative rank probability.
    r=rankdata(reward,method='average')
    return float((r[success].sum()-n1*(n1+1)/2)/(n1*n0))

def local_mean(w,x):
    ax=w.o.a(x);sx=w.s(x)
    vals=[]
    for bits,h in BONDS:
        if (x&bits).bit_count()==1:
            y=x^bits
            vals.append(2*h*sx*w.s(y)*w.o.a(y)/max(ax,1e-250))
    return float(np.mean(vals))

def run(args):
    seeds=[81001,81002,81003,81004]
    roots,_=make_roots(args.data,seeds,per_seed=args.roots//4)
    w=World(args.data,'post_krylov',9981)
    rng=np.random.default_rng(87114)
    rows=[]
    for root in roots:
        tr=rollout(w,int(root),np.zeros(DIM),rng,.15,'dense_signed',4)
        first_g,first_gain,_=w.score(tr['first_state'])
        tr.pop('grads')
        rows.append(dict(root=int(root),first_state=tr['first_state'],
                         proxy=tr['raw'],first_gain=float(first_gain),
                         terminal_gain=tr['gain'],
                         first_positive=bool(first_gain>1e-12),
                         terminal_positive=tr['success']))
    pr=np.array([r['proxy'] for r in rows]);fi=np.array([r['first_positive'] for r in rows]);te=np.array([r['terminal_positive'] for r in rows])
    # A physical guard on the Monte Carlo sampled proxy: it must estimate the
    # true *single-configuration* averaged signed off-diagonal field.
    sanity=[]
    for x in [int(w.o.states[k]) for k in (120,321,1029)]:
        target=local_mean(w,x)
        vals=[signed_process(w,x,rng,4)[0] for _ in range(800)]
        v=np.asarray(vals)
        se=float(v.std(ddof=1)/np.sqrt(len(v)))
        sanity.append(dict(x=x,full_local_mean=target,draw_mean=float(v.mean()),
                           standard_error=se,z=float((v.mean()-target)/max(se,1e-20))))
    result=dict(label='AVSL07_independent_heldout_intermediate_proxy_calibration',
                n=len(rows),n_first_positive=int(fi.sum()),
                n_terminal_positive=int(te.sum()),
                first_auc=auc(pr,fi),terminal_auc=auc(pr,te),
                mean_proxy_first_positive=float(pr[fi].mean()) if fi.any() else None,
                mean_proxy_first_negative=float(pr[~fi].mean()) if (~fi).any() else None,
                mean_proxy_terminal_positive=float(pr[te].mean()) if te.any() else None,
                mean_proxy_terminal_negative=float(pr[~te].mean()) if (~te).any() else None,
                proxy_spearman_first_gain=float(spearmanr(pr,[r['first_gain'] for r in rows]).statistic),
                proxy_spearman_terminal_gain=float(spearmanr(pr,[r['terminal_gain'] for r in rows]).statistic),
                mean_sampled_proxy=float(pr.mean()),std_proxy=float(pr.std()),
                sanity_local_unbiased=sanity,
                guide_costs=w.o.counts(),seeds=seeds,
                caveat='AUC is exploratory on separately seeded toy roots; terminal positives rare, uncertainty large. The local 4-edge signed energy proxy has no certified relationship to the full orbit flip gain.',
                rows=rows)
    Path(args.out).write_text(json.dumps(result,indent=2))
    for k,v in result.items():
        if k not in ('rows','guide_costs'):print(k,v)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--roots',type=int,default=512)
    run(p.parse_args())
