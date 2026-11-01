#!/usr/bin/env python3
"""Post-hoc full-orbit opportunity coverage audit (NEVER candidate acquisition).
Potentially enumerate all 4654 orbit keys, only on the 20-site reference.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from avsl_cooperative20 import Oracle, orbit_boundary


def main(args):
    o=Oracle(args.data)
    # This is explicitly a full-orbit 20-site evaluator; no global oracle on 6x6.
    for x in o.states:
        o.key(int(x))
    global_keys=sorted(o.members)
    gain={}
    for g in global_keys:
        w=orbit_boundary(o,g)
        gain[g]=4*sum(w.values())/o.norm
    positive={k:v for k,v in gain.items() if v>1e-12}
    rows=[]
    union=set()
    for law in ('tempered_a','a2','a_half','uniform_states'):
        visited=set()
        positive_episode=[]
        seeds=[]
        for seed in args.seeds:
            name=f'cooperative20_{"" if law=="tempered_a" else law+"_"}{seed}.json'
            path=Path(args.directory)/name
            rec=json.loads(path.read_text())
            seedset={int(k) for r in rec['records'] for k in r['orbit_keys']}
            seeds.append(dict(seed=seed,n_visited=len(seedset),n_positive=len(positive.keys() & seedset),
                              positive_gain_sum=sum(positive[k] for k in seedset & positive.keys())))
            visited |= seedset
        union |= visited
        rows.append(dict(law=law,n_visited=len(visited),
                         n_positive_visited=len(visited&positive.keys()),
                         fraction_positive_orbits_visited=len(visited&positive.keys())/len(positive),
                         fraction_sum_positive_single_gains_visited=sum(positive[k] for k in visited&positive.keys())/sum(positive.values()),
                         seeds=seeds))
    result=dict(n_states=len(o.states), n_orbits=len(global_keys),
                n_profitable_singles=len(positive),
                sum_positive_single_gains=sum(positive.values()),
                max_positive_single_gain=max(positive.values()),
                root_laws=rows,
                union_n_visited=len(union),
                union_positive_orbit_fraction=len(union&positive.keys())/len(positive),
                union_positive_gain_mass_fraction=sum(positive[k] for k in union&positive.keys())/sum(positive.values()),
                costs=o.counts(),
                caveat='POST HOC whole-sector amplitude and orbit enumeration forbidden in deployed walker policy.')
    Path(args.out).write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='root_laws' and k!='costs'},indent=2))
    for row in rows: print(row['law'], row['n_visited'],row['n_positive_visited'],
                           round(row['fraction_positive_orbits_visited'],4),
                           round(row['fraction_sum_positive_single_gains_visited'],4))
    assert len(global_keys)==4654, len(global_keys)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True);p.add_argument('--directory',required=True)
    p.add_argument('--out',required=True);p.add_argument('--seeds',type=int,nargs='+',default=[30260,30261,30262,30263])
    main(p.parse_args())
