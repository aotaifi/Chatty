#!/usr/bin/env python3
"""Independent full-H / symmetry spot audit for bounded 20-site AVSL patches.

No optimizer/training. Reference data and full sparse H used ONLY to audit
the exploratory patch results, never candidate discovery/ranking.
"""
import argparse, json, math, numpy as np
from scipy.sparse import coo_matrix
from pathlib import Path
from avsl_cooperative20 import Oracle, N, BONDS, MASK, cut_gain, orbit_boundary

def full_offdiag(o):
    rows=[];cols=[];data=[]
    for bit,h in BONDS:
        ys=np.bitwise_xor(o.states, np.uint32(bit))
        js=o.look[ys]
        idx=np.flatnonzero(js>=0)
        rows.append(idx.astype(np.int32))
        cols.append(js[idx].astype(np.int32))
        data.append(np.full(len(idx),h,dtype=np.float64))
    return coo_matrix((np.concatenate(data), (np.concatenate(rows),np.concatenate(cols))),shape=(len(o.states),)*2).tocsr()

def energy_off(o,H,s):
    ps=o.a_data*s
    return float(np.dot(ps,H@ps)/o.norm)

def main(args):
    o=Oracle(args.data)
    records=[]
    for path in args.results:
        obj=json.loads(Path(path).read_text())
        records.extend((str(path),q) for q in obj['records'])
    rng=np.random.default_rng(31524)
    pick=rng.choice(len(records),size=min(args.ncheck,len(records)),replace=False)
    H=full_offdiag(o)
    E0=energy_off(o,H,o.s_data)
    checks=[]
    for k in pick:
        path, rec=records[int(k)]
        keys=rec['orbit_keys']
        g=int(rec['oracle_mask'])
        # Evaluate both oracle-optimal set and an entirely random joint set.
        for tag,mask in [('oracle',g),('random',int(rng.integers(1,1<<len(keys))))]:
            subset=[int(keys[j]) for j in range(len(keys)) if (mask>>j)&1]
            altered=o.s_data.copy()
            for key in subset:
                if key not in o.members: o.key(key)
                for x in o.members[key]:
                    altered[o.look[x]] *= -1
            delta=E0-energy_off(o,H,altered)
            exact=cut_gain(subset,{t:orbit_boundary(o,t) for t in subset},o.norm)
            # Cross-check mathematical identity and stored optimizer's G.
            if tag=='oracle': stored=rec['oracle_best']
            else: stored=exact
            checks.append({'file':path,'tag':tag,'mask':mask,'expected':exact,
                           'stored':stored,'direct':delta,'absolute_error':abs(delta-exact),
                           'stored_error':abs(stored-delta)})
    # Check translating every orbit member preserves a and s (sign-only
    # spin-flip symmetry, translation group) within exact reference precision.
    trials=sorted(set(int(k) for _,r in records[:30] for k in r['orbit_keys']))
    worst_amp=worst_sign=0.
    for g in trials:
        if g not in o.members: o.key(g)
        for x in o.members[g]:
            worst_amp=max(worst_amp,abs(o.a(x)-o.a(g)))
            worst_sign=max(worst_sign,int(o.s(x)!=o.s(g)))
    summary={'n_fullH_checks':len(checks),'max_abs_error':max(q['absolute_error'] for q in checks),
             'max_stored_error':max(q['stored_error'] for q in checks),
             'n_symmetry_orbits':len(trials),'max_amp_symmetry_error':worst_amp,
             'n_sign_symmetry_violations_indicator':worst_sign,
             'E0_offdiag':E0,'notes':'Offdiagonal full Hamiltonian expectation independently agrees with crossing scores; diagonal cancels for sign-only flips.'}
    Path(args.out).write_text(json.dumps({'summary':summary,'checks':checks},indent=2))
    print(json.dumps(summary,indent=2))
    assert summary['max_abs_error']<2e-11
    assert summary['max_stored_error']<2e-11
    assert worst_sign==0
    assert worst_amp<1e-8

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--ncheck',type=int,default=24)
    ap.add_argument('results',nargs='+')
    main(ap.parse_args())
