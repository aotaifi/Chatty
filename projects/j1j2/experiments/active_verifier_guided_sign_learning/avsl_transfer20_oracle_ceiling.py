#!/usr/bin/env python3
"""Post-hoc oracle 22-score hindsight ranking ceiling; NOT a deployment baseline."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json,numpy as np
from pathlib import Path
import avsl_transfer20 as T
def run():
    d=json.load(open('/mnt/data/avsl_transfer20_results.json'))
    db=T.Backend();groups=[]
    for x in T.STATES:
        g=db.key(int(x))
        if g==int(x):groups.append(g)
    norm=float(np.dot(T.AMPS,T.AMPS))
    result={'n_orbits_global':len(groups),'note':'Global score oracle only used for post-hoc evaluation','chambers':{}}
    for chamber in ('marshall','post_krylov'):
        val={};edges={}
        for g in groups:val[g],edges[g]=T.score(db,g,chamber)
        rows=[]
        for t in d['test_episodes']:
            seed=t['seed'];b,roots,pool,_=T.frontier(seed,n_root=d['args']['roots'])
            test=[g for g in pool if T.stable_split(g)==0]
            top=sorted(test,key=lambda g:val[g],reverse=True)[:d['args']['verify_budget']]
            gain,selected=T.greedy(val,edges,top)
            oracle=gain/norm
            learned=t['arms'][chamber]['learned_bonds']['fixed_a_gain']
            bond=t['arms'][chamber]['observed_signed_bonds']['fixed_a_gain']
            rows.append({'seed':seed,'heldout_positive_orbits':sum(val[g]>0 for g in test),
                'oracle_top_22_gain':oracle,'oracle_selected':len(selected),
                'learned_gain':learned,'bond_gain':bond,
                'learned_vs_oracle':learned/oracle if oracle>1e-20 else None,
                'bond_vs_oracle':bond/oracle if oracle>1e-20 else None})
        summary={k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k!='seed'}
        result['chambers'][chamber]={'summary':summary,'trials':rows}
        print(chamber,summary,flush=True)
    Path('/mnt/data/avsl_transfer20_oracle_ceiling.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__':run()
