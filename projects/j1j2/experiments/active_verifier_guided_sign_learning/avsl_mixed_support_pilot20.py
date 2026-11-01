#!/usr/bin/env python3
"""AVSL-08: real FN-mixed walker support + gain-blind search pilot.

TESTS: do valuable single-orbit corrections lie outside FN root support,
and does a small on-policy *exploration* policy find them better than
equal-number verifier attempts from walker roots, random and tempered hops?

FN walker sampler is continuous-time importance sampled H_FN via Feynman-Kac
branching; no exact FN phi enters dynamics. Exact 20-site FN phi and global
orbit gain table are held separate for posthoc auditing only.

Network learns a 2-hop *proposal distribution*, not signs, not FN amplitudes.
The budget accounting includes expensive source RL training labels and action
queries; terminal score is exact fixed-a SINGLE-ORBIT energy gain, not K1 label
or joint reprojected FN energy. This is a support + feasibility pilot only.
"""
import argparse,json,time
from pathlib import Path
from collections import Counter
import numpy as np
from avsl_fn_mixed20 import MixedFN
from avsl_dapo_explorer20 import World,softmax,rng_choice,trajectory,DIM,BONDS

def full_orbit_oracle(data,reference):
    """Off-policy audit table; never passed to the train/test policy."""
    w=World(data,'post_krylov',899)
    for x in w.o.states:w.o.key(int(x))
    positive={}
    gains={}
    masses={}
    f=np.load(reference)['mixed']
    for g,mem in w.o.members.items():
        _,gain,_=w.score(g)
        gains[g]=float(gain)
        masses[g]=float(sum(f[int(w.o.look[x])] for x in mem))
        if gain>1e-12:positive[g]=gain
    return gains,masses,w.o.counts()

def learn_policy(data,roots,seed,budget=400,lr=.23,scale=1e-5):
    """REINFORCE: signed continuous *energy improvement*, including failures."""
    w=World(data,'post_krylov',seed)
    rng=np.random.default_rng(seed)
    theta=np.zeros(DIM)
    stats=Counter()
    trace=[]
    for t in range(budget//4):
        x=int(roots[int(rng.integers(len(roots)))])
        rr=[trajectory(w,x,theta,'learned',rng,epsilon=.15) for _ in range(4)]
        raw=np.array([z['gain'] for z in rr],float)
        rew=np.tanh(raw/scale) # signed, bounded, keep=0; no positive reward for harmful flips
        gradient=np.mean([float(adv)*z['grad'] for adv,z in zip(rew,rr)],axis=0)
        theta+=lr*np.clip(gradient,-1,1)
        stats['attempted']+=4
        stats['positive']+=int(sum(z['positive'] for z in rr))
        stats['negative']+=int(sum(z['gain']<0 for z in rr))
        stats['unique_verified']+=int(sum(z['fresh'] for z in rr))
        stats['groups_all_nonpositive']+=int(all(z['gain']<=1e-12 for z in rr))
        stats['updates']+=1
        if (t+1)%25==0: trace.append(dict(groups=t+1,positives=stats['positive'],theta_norm=float(np.linalg.norm(theta))))
    return theta,dict(counts=dict(stats),trajectory=trace,
                      unique_amp=w.o.counts()['unique_amp'],amp_requests=w.o.counts()['amp_requests'],
                      unique_sign=w.o.counts()['unique_sign'],n_hops=w.policy_moves,
                      n_edge_probes=w.o.counts()['neighbor_probes'],
                      unique_verified_orbits=w.evaluations,policy_gradient_updates=stats['updates'],
                      train_verified_orbit_keys=sorted(map(int,w.scores)),
                      train_positive_orbit_keys=sorted(int(g) for g,v in w.scores.items() if v>1e-12))

def try_proposal(w,x,method,theta,rng,depth=2):
    x=int(x)
    if method=='walker':
        return x
    if method=='tempered':
        # Metropolis using symmetric, all-bonds physical swaps, g(x)^(0.5)
        # Unlike a pure-a Metropolis chain this is a search process initialized
        # at actual FN walkers. Proposal attempts 8, including invalid swaps.
        for step in range(8):
            bits,h=BONDS[int(rng.integers(len(BONDS)))]
            w.o.n_neigh_probes+=1
            if (x&bits).bit_count()!=1:continue
            y=x^bits
            r=w.o.a(y)/max(w.o.a(x),1e-250)
            if rng.random()<min(1,r**.5):x=y
        return x
    if method=='learned':
        return int(trajectory(w,x,theta,'learned',rng,epsilon=.08,depth=depth)['endpoint'])
        # NOTE trajectory above scores immediately; this is deliberate only
        # because same verifier score was budgeted in all arms in evaluate().
    for k in range(depth):
        opts=w.options(x,k)
        if not opts:break
        f=np.array([z[1] for z in opts])
        if method=='random':
            j=int(rng.integers(len(opts)))
        elif method=='geometry':
            j=rng_choice(rng,softmax(f[:,2]*4))
        else:raise ValueError(method)
        x=opts[j][0]
    return int(x)

def evaluate(data,roots,methods,theta,seed,positive_reference,
             all_gains,orbit_masses,B=160,extra_control=0,train_verified=None):
    fn_observed=set(World(data,'post_krylov',0).o.key(int(x)) for x in roots)
    w_obs=World(data,'post_krylov',0)
    fn_observed=set(w_obs.o.key(int(x)) for x in roots)
    # Common root stream: controls may be allowed more terminal attempts to
    # offset the 400 expensive source policy-verification labels.
    max_budget=B+extra_control
    pick=np.random.default_rng(seed+553).choice(len(roots),size=max_budget,replace=True)
    root_samples=np.asarray(roots)[pick]
    result={}
    train_verified=set(train_verified or ())
    for j,method in enumerate(methods):
        arm_B=B if method=='learned' else B+extra_control
        w=World(data,'post_krylov',seed+19+j)
        rng=np.random.default_rng(seed+991+j)
        hits=set();positives=set();endpoint_keys=set();all_eps=[]
        for root in root_samples[:arm_B]:
            xx=try_proposal(w,root,method,theta,rng)
            g,gain,_=w.score(xx)
            endpoint_keys.add(int(g))
            if gain>1e-12:
                positives.add(int(g));all_eps.append(float(gain))
                if g not in fn_observed:hits.add(int(g))
        count=w.o.counts()
        result[method]=dict(
            n_verifier_attempts=arm_B,n_distinct_endpoints=len(endpoint_keys),
            positives_unseen_in_train_labels=len(hits-train_verified),
            positives_total=len(positives),positives_unseen_from_fn=len(hits),
            newly_found_valuable_gains_sum=float(sum(positive_reference[g] for g in hits)),
            best_positive=float(max(all_eps,default=0)),
            fn_unseen_keys=sorted(hits),
            n_guide_unique=count['unique_amp'],n_guide_requests=count['amp_requests'],
            n_sign_unique=count['unique_sign'],
            n_bond_probes=count['neighbor_probes'],
            n_orbit_expanded=count['orbit_members_expanded'],
            total_verifier_queries=w.evaluations)
    return result,dict(
         n_fn_unique_orbits=len(fn_observed),
         n_fn_positive_orbits=len(fn_observed & set(positive_reference)),
         n_fn_unseen_positive_orbits=len(set(positive_reference)-fn_observed),
         n_fn_unseen_positive_gain_mass=float(sum(positive_reference[g] for g in positive_reference if g not in fn_observed)),
         n_fn_unseen_positive_mixed_mass=float(sum(orbit_masses[g] for g in positive_reference if g not in fn_observed)))

def main(args):
    tic=time.perf_counter()
    oracle,masses,ref_cost=full_orbit_oracle(args.data,args.reference)
    positives={g:v for g,v in oracle.items() if v>1e-12}
    train=MixedFN(args.data,'post_krylov')
    roots_train,train_diag=train.sample(seed=args.seed+1101,n_walkers=args.walkers,beta=args.beta,burn=args.burn,record_every=args.record)
    theta,tstat=learn_policy(args.data,roots_train,args.seed+44,budget=args.train_budget)
    methods=['walker','random','tempered','geometry','learned']
    report=[]
    for i in range(args.test_seeds):
        sampler=MixedFN(args.data,'post_krylov')
        roots,meta=sampler.sample(args.seed+4000+i*983,n_walkers=args.walkers,beta=args.beta,
                                   burn=args.burn,record_every=args.record)
        extra=int(np.ceil(args.train_budget/args.test_seeds)) if args.match_training_labels else 0
        arms,support=evaluate(args.data,roots,methods,theta,args.seed+818+i,
                              positives,oracle,masses,B=args.budget,
                              extra_control=extra,train_verified=tstat['train_verified_orbit_keys'])
        if args.save_roots:
            np.savez_compressed(Path(args.out).with_suffix('').as_posix()+f'_testroots_{i}.npz',roots=roots)
        report.append(dict(seed=i,fn_sampler=meta,support=support,arms=arms))
        print('TEST',i,'support',support, 'arms', {k:v['positives_unseen_from_fn'] for k,v in arms.items()},flush=True)
    means={}
    for name in methods:
        means[name]={k:float(np.mean([run['arms'][name][k] for run in report])) for k in (
          'positives_total','positives_unseen_from_fn','positives_unseen_in_train_labels','n_verifier_attempts','best_positive',
          'n_guide_unique','n_bond_probes','total_verifier_queries')}
    out=dict(label='AVSL08 actual-FN-mixed exploratory search 20sites',
             args=vars(args),validity=[
              'FN walker transitions use importance-sampled continuous time jump rates and path-integrated branching resampling.',
              'Only the FN-mixed sampler, NOT a^gamma Metropolis, generated baseline and train/test roots.',
              'Global exact 20-site orbit scorer is hidden audit; not an online learner input.',
              'The learned exploration network has 9 geometry-only features and updates by signed tanh of exact frozen-a single-orbit energy gain.',
              'Training costs are NOT amortized by B heldout proposals. No sign or NN update / Krylov threshold chosen or post-FN energy check.',
              'Newly discovered is defined as absent from the FULL heldout recorded FN roots, not just from B selected roots.',
              'No root supports across different FN sign chambers; six walker populations share one 20-site FN chamber.',
              'The tempered search is 8 MH steps targeting a^0.5, starting from FN root, not a stationary tempered run.'
              ],
             audit=dict(n_positive_orbits=len(positives),n_total_orbits=len(oracle),
                 sum_positive_single_energy=float(sum(positives.values())),
                 audit_oracle_cost=ref_cost),
             training_fn_sampler=train_diag, training_policy=tstat,
             policy_weights=theta.tolist(),
             tests=report,summary=means,seconds=time.perf_counter()-tic)
    Path(args.out).write_text(json.dumps(out,indent=2))
    print('SUMMARY',json.dumps(means,indent=2),'WALL',out['seconds'],flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True);p.add_argument('--reference',required=True);p.add_argument('--out',required=True)
    p.add_argument('--seed',type=int,default=28109)
    p.add_argument('--walkers',type=int,default=128)
    p.add_argument('--beta',type=float,default=10.)
    p.add_argument('--burn',type=float,default=4.)
    p.add_argument('--record',type=int,default=4)
    p.add_argument('--train-budget',type=int,default=400)
    p.add_argument('--budget',type=int,default=160)
    p.add_argument('--test-seeds',type=int,default=6)
    p.add_argument('--match-training-labels',action='store_true')
    p.add_argument('--save-roots',action='store_true')
    main(p.parse_args())
