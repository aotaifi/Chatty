#!/usr/bin/env python3
"""Bounded, gain-blind 20-site cooperative-orbit AVSL diagnostic.

WARNING: reference EXACT post-Krylov FN amplitudes and signs are an oracle.
The sampler is toy a(x)-Metropolis, NOT a 128-walker FN mixed population.
NO labels or scores are used to build candidate patches. No training.

Scopes:
 - Discovery: local one-hop Hamiltonian-neighbour graph, no energy labels.
 - Evaluation: exact guide amplitude/sign oracle scores joint orbit cuts.
 - Counterfactual: within-patch best-subset oracle (NOT an implementable ranker).
"""
import argparse
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

N, LX, LY = 20, 4, 5
MASK = (1 << N) - 1


def site(x, y):
    q, r = divmod(y, LY)
    return ((x - q) % LX) + LX * r


BONDS = [
    ((1 << site(x, y)) | (1 << site(x+dx, y+dy)), h)
    for y in range(LY) for x in range(LX)
    for dx,dy,h in ((1,0,.5),(0,1,.5),(1,1,.25),(-1,1,.25))
]
XY = {site(x,y): (x,y) for y in range(LY) for x in range(LX)}
PERMS = [
    tuple(site(XY[i][0]+dx,XY[i][1]+dy) for i in range(N))
    for dy in range(LY) for dx in range(LX)
]


class Oracle:
    def __init__(self, path):
        d = np.load(path)
        self.states = d["states"].astype(np.uint32)
        self.a_data = d["amp"].astype(float)
        self.s_data = d["guide"].astype(np.int8)
        self.look = np.full(1 << N, -1, np.int32)
        self.look[self.states] = np.arange(len(self.states),dtype=np.int32)
        self.norm = float(np.dot(self.a_data,self.a_data))
        self.cache_a = {}
        self.cache_s = {}
        self.keys = {}
        self.members = {}
        self.n_amp_calls = 0
        self.n_sign_calls = 0
        self.n_neigh_probes = 0
        self.n_orbit_expanded = 0

    def a(self,x):
        x=int(x); self.n_amp_calls+=1
        if x not in self.cache_a: self.cache_a[x]=float(self.a_data[self.look[x]])
        return self.cache_a[x]

    def s(self,x):
        x=int(x); self.n_sign_calls+=1
        if x not in self.cache_s: self.cache_s[x]=int(self.s_data[self.look[x]])
        return self.cache_s[x]

    def key(self,x):
        x=int(x)
        if x in self.keys: return self.keys[x]
        vals=[]
        for p in PERMS:
            q=0
            for i,j in enumerate(p): q |= ((x>>i)&1) << j
            vals.extend((q,q^MASK))
        vals=tuple(sorted(set(vals)))
        k=vals[0]
        for v in vals: self.keys[v]=k
        self.members[k]=vals
        self.n_orbit_expanded+=len(vals)
        return k

    def neigh(self,x):
        x=int(x)
        for bits,h in BONDS:
            self.n_neigh_probes+=1
            y=x^bits
            if self.look[y]>=0:
                yield int(y), h

    def counts(self):
        return dict(amp_requests=self.n_amp_calls,
                    sign_requests=self.n_sign_calls,
                    unique_amp=len(self.cache_a),
                    unique_sign=len(self.cache_s),
                    neighbor_probes=self.n_neigh_probes,
                    orbit_members_expanded=self.n_orbit_expanded,
                    unique_orbits=len(self.members))


def sample_roots(o, seed, n=256, burn=2200, thin=13, law="tempered_a"):
    rng=np.random.default_rng(seed)
    if law == "uniform_states":
        return [int(x) for x in o.states[rng.integers(len(o.states),size=n)]], None
    gamma={"tempered_a":1.0,"a2":2.0,"a_half":0.5}[law]
    x=int(o.states[rng.integers(len(o.states))])
    roots=[]
    accepted=0
    for t in range(burn+n*thin):
        bits,h=BONDS[int(rng.integers(len(BONDS)))]
        y=x^bits
        o.n_neigh_probes+=1
        if o.look[y]>=0 and rng.random()<min(1.,(o.a(y)/max(o.a(x),1e-300))**gamma):
            x=y;accepted+=1
        if t>=burn and (t-burn)%thin==0:
            roots.append(int(x))
    return roots,accepted/(burn+n*thin)


def patch(o,root,seed,k=8):
    """Gain-blind random Hamiltonian-neighbour exploration of quotient graph."""
    rng=np.random.default_rng(seed)
    selected=[o.key(root)]
    visited=set(selected)
    # grow an equal-size connected orbit patch using no amplitude or energy score
    for t in range(1,k):
        choices=[]
        for g in selected:
            for y,h in o.neigh(g):
                v=o.key(y)
                if v not in visited: choices.append(v)
        if not choices: break
        nxt=int(choices[rng.integers(len(choices))])
        selected.append(nxt);visited.add(nxt)
    return selected


def orbit_boundary(o,g):
    """Summed oriented W_gh for exact symmetry orbit g (evaluation ONLY)."""
    a=o.a(g);s=o.s(g)
    m=len(o.members[g])
    weights=defaultdict(float)
    for y,h in o.neigh(g):
        ky=o.key(y)
        if ky != g:
            weights[ky]+=m*h*s*o.s(y)*a*o.a(y)
    return weights


def cut_gain(keys, boundaries, norm):
    inside=set(keys)
    return 4.0/norm*sum(
        w for g in inside for h,w in boundaries[g].items()
        if h not in inside
    )


def best_subset(keys, boundaries, norm):
    """In-patch oracle; may not be used by explorer or trained NN labels."""
    vals=[4*sum(boundaries[g].values())/norm for g in keys]
    matrix=np.zeros((len(keys),len(keys)))
    for i,g in enumerate(keys):
        for j,h in enumerate(keys):
            if j>i:
                matrix[i,j]=boundaries[g].get(h,0.)
    best_gain=0.
    best_mask=0
    for mask in range(1,1<<len(keys)):
        chosen=[i for i in range(len(keys)) if (mask>>i)&1]
        gain=sum(vals[i] for i in chosen)
        gain -= 8/norm * sum(matrix[i,j] for idx,i in enumerate(chosen)
                               for j in chosen[idx+1:])
        if gain>best_gain+1e-14:
            best_gain,best_mask=gain,mask
    return best_gain,best_mask,vals


def evaluate(o,keys,boundaries,rng):
    z=o.norm
    prefix={}
    for k in (1,2,4,8):
        if len(keys)>=k:
            prefix[str(k)]=cut_gain(keys[:k],boundaries,z)
    best_gain, mask, vals=best_subset(keys,boundaries,z)
    single=max(0.,*vals)
    # budget-aware current-chamber comparator; uses identical scored keys
    positives=[i for i,v in enumerate(vals) if v>1e-14]
    greedy_mask=0;greedy_gain=0.
    for _ in range(len(keys)):
        top=None;top_gain=greedy_gain
        for i in positives:
            if greedy_mask&(1<<i): continue
            cand=greedy_mask|(1<<i)
            gc=cut_gain([keys[j] for j in range(len(keys)) if (cand>>j)&1],boundaries,z)
            if gc>top_gain+1e-14: top,top_gain=i,gc
        if top is None: break
        greedy_mask |= 1<<top;greedy_gain=top_gain
    chosen=[i for i in range(len(keys)) if (mask>>i)&1]
    cooperative = (best_gain > 1e-12 and len(chosen)>=2 and
                   all(vals[i] <= 1e-12 for i in chosen))
    # Predefined size-matched random subset control, not gain-guided
    random_mask=int(rng.integers(1, 1<<len(keys)))
    random_gain=cut_gain(
        [keys[i] for i in range(len(keys)) if (random_mask>>i)&1],
        boundaries,z
    )
    pair_gain=0.; negative_singles_pair=False
    for i in range(len(keys)):
        for j in range(i+1,len(keys)):
            gg=cut_gain((keys[i],keys[j]),boundaries,z)
            if gg>pair_gain:
                pair_gain=gg
            if gg>1e-12 and vals[i]<=0 and vals[j]<=0:
                negative_singles_pair=True
    return dict(prefix_gain=prefix,oracle_best=float(best_gain),
                oracle_n_chosen=len(chosen),
                oracle_mask=mask,positive_singles=sum(v>1e-12 for v in vals),
                best_single=single,greedy_single=greedy_gain,
                random_mask_gain=random_gain,negative_singles_pair=negative_singles_pair,
                cooperative_all_singles_bad=bool(cooperative),
                any_positive=bool(best_gain>1e-12),
                orbit_keys=[int(g) for g in keys],
                # Keep the individual gains for independent post-hoc verification.
                single_gains=[float(x) for x in vals])


def run(args):
    start=time.monotonic()
    o=Oracle(args.data)
    roots,acceptance=sample_roots(o,args.seed,args.anchors,args.burn,args.thin,args.root_law)
    after_sampler=o.counts()
    rng=np.random.default_rng(args.seed+917)
    patches=[patch(o,x,args.seed+100003+i,args.max_patch) for i,x in enumerate(roots)]
    after_discovery=o.counts()
    unique_keys=sorted({g for keys in patches for g in keys})
    boundaries={g:orbit_boundary(o,g) for g in unique_keys}
    after_eval=o.counts()
    records=[evaluate(o,p,boundaries,rng) for p in patches]
    gains=np.asarray([r['oracle_best'] for r in records])
    greedy=np.asarray([r['greedy_single'] for r in records])
    singles=np.asarray([r['best_single'] for r in records])
    random=np.asarray([r['random_mask_gain'] for r in records])
    union_positive=np.asarray([r['any_positive'] for r in records],int)
    synergy=np.asarray([r['negative_singles_pair'] for r in records],int)
    cooperative=np.asarray([r['cooperative_all_singles_bad'] for r in records],int)
    total_orbit_keys=sum(len(p) for p in patches)
    unique_orbit_keys=len(unique_keys)
    result={
        "label":"20-site frozen exact post-Krylov FN oracle diagnostic, not learned on-policy FN",
        "args":vars(args),"n_states":len(o.states),"norm":o.norm,
        "n_patches":len(records),"n_unique_root_orbits":len(set(o.key(x) for x in roots)),
        "n_total_patch_orbit_occurrences":total_orbit_keys,
        "n_unique_patch_orbits":unique_orbit_keys,
        "n_evaluation_scores":len(unique_keys),
        "acceptance":acceptance,
        "costs":{"sampler":after_sampler,"after_discovery":after_discovery,
                 "after_eval":after_eval,"wall_s":time.monotonic()-start},
        "summary":{
            "positive_patch_fraction":float(union_positive.mean()),
            "negative_single_positive_pair_fraction":float(synergy.mean()),
            "oracle_best_with_only_bad_singles_fraction":float(cooperative.mean()),
            "mean_oracle_best_gain":float(gains.mean()),
            "mean_best_single_gain":float(singles.mean()),
            "mean_greedy_gain":float(greedy.mean()),
            "mean_random_mask_gain":float(random.mean()),
            "fraction_patch_oracle_better_greedy":float(np.mean(gains>greedy+1e-12)),
            "excess_oracle_over_greedy":float(np.mean(gains-greedy)),
            "prefix_mean_gains":{str(k):float(np.mean(
                [r["prefix_gain"][str(k)] for r in records if str(k) in r["prefix_gain"]
            ])) for k in (1,2,4,8)},
            "null_keep_gain":0.0,
            "n_bad_single_positive_pair":int(synergy.sum()),
            "n_all_bad_singles_oracle_positive":int(cooperative.sum())
        },
        "records":records,
        "caveats":[
            "Exact stored 20-site FN amplitude and sign arrays, evaluator only except toy Metropolis root proposals.",
            "Toy pi~a Metropolis roots are not FN mixed walkers, so no walker-only scaling established.",
            "Within-patch oracle enumerates 2^8 candidate subsets AFTER discovery; this is NOT a deployable selector.",
            "Patches share the same physical sign chamber and can overlap: patches are not independent systems.",
            "One symmetry-orbit flip need not be a size-consistent local sign rule.",
            "No compression/network generalization is tested by this stage. See follow-up rule study."
        ]
    }
    Path(args.out).write_text(json.dumps(result,indent=2))
    print(json.dumps({"summary":result["summary"],"costs":result["costs"],
                      "n_unique_patch_orbits":unique_orbit_keys,
                      "n_unique_root_orbits":result["n_unique_root_orbits"]},indent=2))
    assert all(r["oracle_best"]>=r["greedy_single"]-1e-12 for r in records)
    assert all(r["oracle_best"]>=r["best_single"]-1e-12 for r in records)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--data",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--seed",type=int,default=30260)
    p.add_argument("--anchors",type=int,default=256)
    p.add_argument("--burn",type=int,default=2200)
    p.add_argument("--thin",type=int,default=13)
    p.add_argument("--max-patch",type=int,default=8)
    p.add_argument("--root-law",choices=("tempered_a","a2","a_half","uniform_states"),default="tempered_a")
    run(p.parse_args())
