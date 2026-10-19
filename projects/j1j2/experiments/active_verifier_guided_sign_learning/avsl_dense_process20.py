#!/usr/bin/env python3
"""AVSL-07: dense process-reward on two-hop RL sign-correction exploration.

Physics reward (terminal): exact frozen-a single symmetry-orbit sign-flip gain,
only from scored endpoints of short exploratory trajectories. NOT FN energy.
Process reward (intermediate only): unbiased four-edge randomized estimator
of a configuration's local signed offdiagonal Hamiltonian sum, with frozen
positive guide amplitudes/signs, WITHOUT using any full terminal-orbit gain.
All oracle guide queries for that feedback are charged per policy.
No ED ground-state labels, global sign classifier, or model deployment.
Compare against terminal-only group RL, geometry-only process, and
shuffled signed process (identical oracle feedback cost, information destroyed).
"""
import argparse,json,math,time
from collections import Counter
from pathlib import Path
import numpy as np
from avsl_dapo_explorer20 import World,make_roots,softmax,rng_choice,eval_arms,DIM,BONDS

MODES=('binary','dense_signed','dense_shuffled','dense_geometry','keep_relative')
K=4

def grad_action(features,theta,epsilon,rng):
    q=softmax(features@theta)
    p=(1-epsilon)*q+epsilon/len(q)
    j=rng_choice(rng,p)
    grad=(1-epsilon)*q[j]/p[j]*(features[j]-(q@features))
    return j,grad

def signed_process(w,x,rng,edges=4):
    # Uniform without-replacement subsampling of incident legal hops; same
    # exact amplitude ratio and sign terms as H local energy, but no symmetry
    # orbit expansion and no terminal gain oracle. Unbiased before tanh.
    s=int(w.s(x));ax=w.o.a(x);available=[]
    for bits,h in BONDS:
        w.o.n_neigh_probes+=1
        if (int(x)&bits).bit_count()==1:available.append((x^bits,h))
    ix=rng.choice(len(available),size=min(edges,len(available)),replace=False)
    terms=[]
    for i in ix:
        y,h=available[int(i)]
        terms.append(2*h*s*w.s(y)*w.o.a(y)/max(ax,1e-250))
    return float(np.mean(terms)),len(terms)

def rollout(w,root,theta,rng,epsilon,mode,process_edges=4):
    x=int(root);grads=[];states=[];feats=[];option_cost=0
    for step in range(2):
        opts=w.options(x,step)
        if not opts:break
        ff=np.array([f for _,f in opts])
        j,g=grad_action(ff,theta,epsilon,rng)
        x=int(opts[j][0]);w.visited.add(x);w.policy_moves+=1
        grads.append(g);states.append(x);feats.append(ff[j])
        option_cost+=len(opts)
    raw=0.;nproc=0
    if len(states) and mode in ('dense_signed','dense_shuffled'):
        raw,nproc=signed_process(w,states[0],rng,edges=process_edges)
    elif len(states) and mode=='dense_geometry':
        # Physical site geometry as no-guide-call feature-only ablation.
        raw=float(feats[0][2])
    g,gain,fresh=w.score(x)
    return dict(root=int(root),endpoint=x,first_state=int(states[0]) if states else x,orbit=g, gain=float(gain),
                success=bool(gain>1e-12),fresh=bool(fresh),
                grads=grads,raw=raw,nproc=nproc,option_cost=option_cost)

def normalized(values):
    v=np.asarray(values,dtype=float)
    return (v-v.mean())/(v.std()+1e-8)

def train_policy(world,roots,seed,mode,budget=1200,
                 epsilon=.15,lr=.2,process_weight=.5,
                 edges=4,clip=1.):
    if budget%K:raise ValueError('require 4-group budget')
    rng=np.random.default_rng(seed);theta=np.zeros(DIM)
    counts=Counter();traces=[];proc_raw=[];labels=[]
    for idx in range(budget//K):
        x=int(roots[int(rng.integers(len(roots)))])
        g=[rollout(world,x,theta,rng,epsilon,mode,edges) for _ in range(K)]
        success=np.array([int(z['success']) for z in g],float)
        mixed=bool(0<sum(success)<K)
        counts['groups']+=1
        counts['attempted']+=K
        counts['positive']+=int(sum(success))
        counts['unique_scored']+=sum(z['fresh'] for z in g)
        counts['all_negative']+=int(sum(success)==0)
        counts['all_positive']+=int(sum(success)==K)
        counts['mixed_binary']+=int(mixed)
        counts['process_bond_samples']+=sum(z['nproc'] for z in g)
        if mode in ('dense_signed','dense_shuffled','dense_geometry'):
            pp=normalized([z['raw'] for z in g])
            if mode=='dense_shuffled':
                pp=pp[rng.permutation(K)]
            counts['nonzero_process_groups']+=int(np.std(pp)>1e-6)
            # Process reward shapes the *first action only* (before final
            # orbit scorer); binary outcome shapes both actions.
            # A process "good" score does not qualify for physical acceptance.
            bin_adv=normalized(success) if mixed else np.zeros(K)
            gradient=np.zeros(DIM)
            for j,z in enumerate(g):
                if z['grads']:
                    gradient+=process_weight*pp[j]*z['grads'][0]/K
                    gradient+=.5*bin_adv[j]*sum(z['grads'])/K
            theta+=lr*np.clip(gradient,-clip,clip)
            counts['updates']+=1
        elif mode=='binary':
            if mixed:
                advantages=normalized(success)
                gradient=np.mean([advantages[j]*sum(z['grads']) for j,z in enumerate(g)],axis=0)
                theta+=lr*np.clip(gradient,-clip,clip)
                counts['updates']+=1
        elif mode=='keep_relative':
            # Same terminal binary rewards as the existing baseline:
            # modest penalty for failed exploration but no unsafe acceptance.
            r=np.where(success>0,1.,-.1)
            gradient=np.mean([r[j]*sum(z['grads']) for j,z in enumerate(g)],axis=0)
            theta+=lr*np.clip(gradient,-clip,clip)
            counts['updates']+=1
        proc_raw.extend(z['raw'] for z in g)
        labels.extend(z['success'] for z in g)
        if (idx+1)%100==0:
            traces.append(dict(group=idx+1,positives=counts['positive'],
                  mixed=counts['mixed_binary'],updates=counts['updates'],
                  model_norm=float(np.linalg.norm(theta)),
                  distinct_amplitudes=world.o.counts()['unique_amp']))
    pp=np.asarray(proc_raw,dtype=float);lab=np.asarray(labels,dtype=bool)
    # Descriptive calibration on SAME train experiences, not a heldout claim.
    qlo=float(np.mean(pp[~lab])) if (~lab).any() else None
    qhi=float(np.mean(pp[lab])) if lab.any() else None
    return theta,dict(counts),traces,dict(mean_process_if_failed=qlo,
          mean_process_if_succeeded=qhi,
          nonzero_process_fraction=float(np.mean(np.abs(pp)>1e-10)),
          p_success=float(lab.mean()))

def main(args):
    if args.budget>2500 or args.eval_attempts>240 or args.budget%4:
        raise ValueError('hard budget cap')
    seeds_train=[41001+11*(args.seed-10019)+i for i in range(4)]
    seeds_test=[51001+11*(args.seed-10019)+i for i in range(4)]
    t0=time.perf_counter()
    train_roots,trstat=make_roots(args.data,seeds_train,per_seed=args.roots//4)
    test_roots,tsstat=make_roots(args.data,seeds_test,per_seed=args.eval_attempts//4)
    trained={};policies={}
    for k,mode in enumerate(MODES):
        w=World(args.data,args.chamber,args.seed+100+k)
        t=time.perf_counter()
        th,counts,trace,cal=train_policy(
          w,train_roots,args.seed+77+101*k,mode,
          budget=args.budget,process_weight=args.process_weight,edges=args.edges)
        cost=dict(guide_amplitude_unique=w.o.counts()['unique_amp'],
            guide_amplitude_requests=w.o.counts()['amp_requests'],
            sign_unique=w.o.counts()['unique_sign'],
            sign_requests=w.o.counts()['sign_requests'],
            neighbor_probes=w.o.counts()['neighbor_probes'],
            symmetry_orbits=w.o.counts()['unique_orbits'],
            score_oracle_calls=w.evaluations,
            feature_queries=w.feature_queries,
            wall_s=time.perf_counter()-t)
        trained[mode]=dict(counts=counts,policy_weights=th.tolist(),
             trace=trace,calibration_train_only=cal,cost=cost)
        policies[mode]=th
    # The old evaluator calls each learned policy generically, without process
    # oracle queries at deployment. Independent per-arm caches, matched roots.
    policies.update({z:np.zeros(DIM) for z in ('random','geometry','novelty','signed_bond')})
    evaluation=eval_arms(args.data,test_roots,policies,args.seed+800,
                 B=args.eval_attempts,chamber=args.chamber)
    out=dict(label='AVSL-07_20site_dense_process_oracle_assisted',
         args=vars(args),seeds_train=seeds_train,seeds_test=seeds_test,
         training=trained,evaluation=evaluation,
         sampler_train=trstat,sampler_test=tsstat,
         seconds=time.perf_counter()-t0,
         caveats=[
           'All sign/amplitude feedback from exact 20-site guide oracle; not deployable NQS inference.',
           'Process reward is a subsampled signed-bond local-energy proxy, not independent proof of improvement.',
           'Train-only success association does not constitute heldout calibration.',
           'Terminal positives are fixed-a single symmetry-orbit gains, not consistent joint sign policy or FN energy.',
           'Two-hop trajectory score and group-gradient imitate selected DAPO ideas; not exact DAPO clipping or full GRPO implementation.',
           'Test roots disjoint seed streams but may revisit train symmetry orbits.',
           'Comparators get matched terminal attempts but some process modes pay additional guide evaluation cost.'])
    Path(args.out).write_text(json.dumps(out,indent=2))
    print('TRAIN CHAMBER',args.chamber,'SEED',args.seed,'TIME',round(out['seconds'],2),flush=True)
    for mode,t in trained.items():
        print('TRAIN',mode,'positive',t['counts']['positive'],'mixed',t['counts']['mixed_binary'],
             'all_negative',t['counts']['all_negative'],'updates',t['counts']['updates'],
             'amplitudes',t['cost']['guide_amplitude_unique'],'oracles',t['cost']['score_oracle_calls'],
             'cal',t['calibration_train_only'],flush=True)
    for name,r in evaluation.items():
        print('EVAL',name,'pos',r['positive_distinct_orbits'],'best',r['best_single'],
              'guide',r['costs']['o']['unique_amp'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True);p.add_argument('--out',required=True)
    p.add_argument('--seed',type=int,default=10019)
    p.add_argument('--chamber',choices=('marshall','post_krylov'),default='post_krylov')
    p.add_argument('--budget',type=int,default=1200)
    p.add_argument('--roots',type=int,default=256)
    p.add_argument('--eval-attempts',type=int,default=160)
    p.add_argument('--process-weight',type=float,default=.5)
    p.add_argument('--edges',type=int,default=4)
    main(p.parse_args())
