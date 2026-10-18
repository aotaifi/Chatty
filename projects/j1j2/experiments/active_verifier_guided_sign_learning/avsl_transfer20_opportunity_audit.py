#!/usr/bin/env python3
"""Post-hoc full-orbit opportunity audit (evaluation oracle, NEVER a scalable selection policy)."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json,time,numpy as np
from pathlib import Path
import avsl_transfer20 as T
def run():
    t0=time.time(); db=T.Backend()
    groups=[]
    for x in T.STATES:
        k=db.key(int(x))
        if k==int(x):groups.append(int(x))
    assert sum(len(db.members[k]) for k in groups)==T.DIM
    d=json.load(open('/mnt/data/avsl_transfer20_results.json'))
    result={'n_states':int(T.DIM),'n_orbits':len(groups),'chambers':{}}
    norm=float(np.dot(T.AMPS,T.AMPS))
    for chamber in ('marshall','post_krylov'):
        scores={g:T.score(db,g,chamber)[0]/norm for g in groups}
        positive={g:v for g,v in scores.items() if v>1e-13}
        vals=np.array(sorted(positive.values(),reverse=True))
        rec={'n_positive_single_orbits':len(positive),'positive_fraction':len(positive)/len(groups),
             'sum_positive_independent_gains':float(vals.sum()),'top_10_sum':float(vals[:10].sum()),
             'top_22_sum':float(vals[:22].sum()),'top_40_sum':float(vals[:40].sum()),
             'max_single_gain':float(vals[0]) if len(vals) else 0.,
             'p50_positive_gain':float(np.median(vals)) if len(vals) else 0.}
        pool_stats=[]
        for t in d['test_episodes']:
            seed=t['seed']; db2,roots,pool,_=T.frontier(seed,n_root=d['args']['roots'])
            test=[g for g in pool if T.stable_split(g)==0]
            pool_stats.append({'seed':seed,'n_frontier':len(pool),'n_test':len(test),
                'positive_orbits_frontier':sum(g in positive for g in pool),
                'positive_orbits_holdout':sum(g in positive for g in test),
                'positive_gain_sum_frontier':sum(positive.get(g,0.) for g in pool),
                'positive_gain_sum_holdout':sum(positive.get(g,0.) for g in test),
                'positive_gain_sum_global':float(vals.sum()),
                'positive_orbit_coverage_frontier':sum(g in positive for g in pool)/max(1,len(positive)),
                'positive_orbit_coverage_holdout':sum(g in positive for g in test)/max(1,len(positive))})
        rec['test_pools_summary']={k:float(np.mean([row[k] for row in pool_stats])) for k in pool_stats[0] if k!='seed'}
        rec['test_pools']=pool_stats
        result['chambers'][chamber]=rec
        print(chamber,rec['n_positive_single_orbits'],rec['sum_positive_independent_gains'],rec['test_pools_summary'],flush=True)
    result['wall_s']=time.time()-t0
    Path('/mnt/data/avsl_transfer20_opportunity_audit.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__':run()
