#!/usr/bin/env python3
"""AVSL-06: stochastic *frontier-generation* policy, DAPO-inspired group RL.

NO training on ED signs. Exact 20-site (post-K1 FN) amplitudes/signs enter only
the sampler + terminal gain oracle; the learned policy uses geometry and search
history but never guide amplitudes, K1 fields or oracle gains as *inputs*.
Fully exploratory result only: not an implementable M128 FN rollout.
K=4 independent action trajectories per root, depth two, terminal orbit score.
Both all-failed and all-success groups consume queries. No positive updates
on harmful proposals. Distinct per-arm score caches; no test-time training.
"""
import argparse,json,math,time,hashlib
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from avsl_cooperative20 import Oracle,BONDS,N,MASK,sample_roots

DIM=9

def marshall(x,mask):
    return 1 if (int(x)&mask).bit_count()%2==0 else -1

class World:
    def __init__(self,npz, chamber,seed):
        self.o=Oracle(npz);self.chamber=chamber;self.rng=np.random.default_rng(seed)
        self.mask=int(np.load(npz)['maskA'][0])
        self.scores={};self.edges={};self.evaluations=0;self.oracle_cache_hit=0
        self.feature_queries=0;self.feature_cache={}
        self.boundary_queries=0
        self.orbit_calls=0
        self.policy_moves=0;self.revisit_endpoints=0;self.visited=set()
    def s(self,x):
        if self.chamber=='post_krylov': return self.o.s(x)
        return marshall(x,self.mask)
    def features(self,x,y,h,depth,n_seen):
        # Site-geometry ONLY; all NN/NNN occupations locally visible from x, y.
        # Per-action feature cost is explicit; no amplitude or expensive sign call.
        self.feature_queries+=1
        y=int(y)
        if y not in self.feature_cache:
            nn=nnn=0
            for bits,w in BONDS:
                if (y&bits).bit_count()==1:
                    if w==.5:nn+=1
                    else:nnn+=1
            self.feature_cache[y]=(nn,nnn)
        nn,nnn=self.feature_cache[y]
        return np.array([1.,(nn-20)/20.,(nnn-20)/20.,float(h==.5),
                         float(depth),float(min(n_seen,5))/5.,
                         float((int(x)&y).bit_count())/N,
                         float((y&self.mask).bit_count()-N/4)/10.,
                         float((nn-nnn)/40.)],float)
    def options(self,x,depth):
        opts=[]
        for bits,h in BONDS:
            y=int(x)^bits
            self.o.n_neigh_probes+=1
            if (int(x)&bits).bit_count()==1:
                f=self.features(x,y,h,depth,int(y in self.visited))
                opts.append((y,f))
        return opts
    def score(self,x):
        self.orbit_calls+=1
        g=self.o.key(x)
        if g in self.scores:
            self.oracle_cache_hit+=1
            return g,self.scores[g],False
        a=self.o.a(g);s=self.s(g)
        mult=len(self.o.members[g])
        w=0.
        # Single symmetry orbit boundary. Exact 20-site oracle is isolated here.
        for bits,h in BONDS:
            y=int(g)^bits
            self.o.n_neigh_probes+=1
            if (int(g)&bits).bit_count()==1:
                ky=self.o.key(y)
                if ky!=g:
                    w+=mult*h*s*self.s(y)*a*self.o.a(y)
        gain=float(4*w/self.o.norm)
        self.scores[g]=gain;self.boundary_queries+=1;self.evaluations+=1
        return g,gain,True

def make_roots(npz,seeds,per_seed=128):
    # Reference amplitude used only to draw toy walker-like roots, same for
    # every method/chamber. Seeds remain disjoint for train vs evaluation.
    roots=[];stats=[]
    for seed in seeds:
        world=World(npz,'marshall',seed)
        rr,acc=sample_roots(world.o,seed,n=per_seed,burn=1000,thin=11)
        roots.extend(rr)
        stats.append(dict(seed=seed,acceptance=acc,sampler_costs=world.o.counts()))
    return roots,stats

def rng_choice(rng,p):
    return min(len(p)-1,int(np.searchsorted(np.cumsum(p),rng.random(),side='right')))

def softmax(z):
    v=z-np.max(z);a=np.exp(v);return a/a.sum()

def trajectory(w,root,theta,mode,rng,epsilon=.15,depth=2):
    x=int(root);gr=np.zeros(DIM);n_generated=0
    for k in range(depth):
        opts=w.options(x,k)
        if not opts:break
        f=np.array([v for _,v in opts])
        if mode=='random':
            p=np.full(len(opts),1/len(opts))
        elif mode=='geometry':
            # Nonlearned, simple, fully observed geometry priority:
            # favor local frustrated-next-NN environments.
            p=softmax(4*f[:,2])
        elif mode=='novelty':
            # Geometry with novelty based on actual encountered states.
            p=softmax(2.0*f[:,2]+1.5*f[:,3]-2.0*f[:,5])
        elif mode=='signed_bond':
            # Signed-edge local-field baseline. This queries the exact 20-site
            # amplitude/sign oracle on every candidate edge, so charges MORE
            # calls than geometry-only or learned policies.
            ax=w.o.a(x);sx=w.s(x)
            signed=np.array([2*w.s(y)*sx*ax*w.o.a(y)*(.5 if f0[3]>0.5 else .25)
                             for y,f0 in opts],float)
            # Absolute amplitudes arbitrary; normalize by root amplitude^2.
            signed=signed/max(ax*ax,1e-240)
            p=softmax(np.clip(signed,-8.,8.))
        else:
            p=(1-epsilon)*softmax(f@theta)+epsilon/len(opts)
        j=rng_choice(rng,p)
        if mode not in ('random','geometry','novelty','signed_bond'):
            # exact log probability gradient for an epsilon-mixture softmax
            q=softmax(f@theta)
            grad=(1-epsilon)*q[j]/p[j]*(f[j]-(q@f))
            gr+=grad
        x=opts[j][0]
        w.visited.add(x)
        w.policy_moves+=1
        n_generated+=len(opts)
    g,gain,fresh=w.score(x)
    if not fresh:w.revisit_endpoints+=1
    return dict(root=int(root),endpoint=x,orbit=g,gain=gain,positive=gain>1e-12,
                grad=gr,fresh=fresh,n_options=n_generated)

def train(w,roots,seed,method,budget,group_size=4,epsilon=.15,lr=.2, clip=1.0):
    rng=np.random.default_rng(seed)
    theta=np.zeros(DIM,dtype=float)
    total=Counter();record=[]
    for group in range(budget//group_size):
        root=int(roots[int(rng.integers(len(roots)))])
        samples=[trajectory(w,root,theta,method,rng,epsilon) for _ in range(group_size)]
        rewards=np.array([int(z['positive']) for z in samples],float)
        gains=np.array([z['gain'] for z in samples])
        total['attempted']+=len(samples);total['fresh']+=sum(z['fresh'] for z in samples)
        total['positives']+=sum(z['positive'] for z in samples)
        mixed=bool(0<sum(rewards)<group_size)
        total['mixed_groups']+=int(mixed)
        total['all_failed']+=int(sum(rewards)==0)
        total['all_success']+=int(sum(rewards)==group_size)
        if method in ('dapo','grpo'):
            # All binary-homogeneous groups give no standardized advantage.
            # Explicit anchor is zero reward for keep; no policy acceptance
            # without gain>0, irrespective of advantage.
            if mixed:
                adv=(rewards-rewards.mean())/(rewards.std()+1e-12)
                gradient=np.mean([a*z['grad'] for a,z in zip(adv,samples)],axis=0)
                theta+=lr*np.clip(gradient,-clip,clip)
                total['updates']+=1
            # 'dapo' and 'grpo' have the SAME binary-reward, mixed-group update
            # under a matched attempted-oracle budget; the two labels here
            # are independent stochastic REPLICAS, not methods that can be
            # fairly contrasted. The complete DAPO clipped surrogate and
            # token-normalized loss are NOT implemented.
        elif method=='keep_relative':
            # Transferable process-level objective that trains also on failed
            # proposals; avoid GRPO relative ranking of harmful actions.
            r=np.where(rewards>0,1.,-.10)
            gradient=np.mean([a*z['grad'] for a,z in zip(r,samples)],axis=0)
            theta+=lr*np.clip(gradient,-clip,clip)
            total['updates']+=1
        if group%100==0:
            record.append(dict(group=group,attempted=total['attempted'],
                  positives=total['positives'],mixed=total['mixed_groups'],
                  unique_scores=w.evaluations,theta_norm=float(np.linalg.norm(theta))))
    return theta,dict(total),record

def eval_arms(npz,root_sets,policies,seed,B=160,eps=.08,chamber='post_krylov'):
    # A per-arm evaluator, no gradient updates. Roots episode-matched.
    # Fairness: each policy spends exactly B attempted terminal scoring
    # slots, plus actually instrumented independent feature/orbit costs.
    result={}
    for idx,(name,theta) in enumerate(policies.items()):
        w=World(npz,chamber,seed+202+idx)
        rng=np.random.default_rng(seed+30000+idx)
        records=[]
        accepted=[];accepted_set=set();energy=0.
        # Deliberately a common ordered list of roots across all policy arms.
        all_roots=np.array(root_sets[:B],dtype=np.uint32)
        for root in all_roots:
            tr=trajectory(w,int(root),theta,
                name if name in ('random','geometry','novelty','signed_bond') else 'learned',rng,eps)
            tr.pop('grad')
            g=tr['orbit']
            # Never sum positive single gains as actual joint energy! This is
            # an optimistic raw-single opportunity monitor only.
            if tr['gain']>1e-12 and g not in accepted_set:
                accepted_set.add(g); accepted.append(g)
            records.append(tr)
        d=dict(positive_draws=sum(z['positive'] for z in records),
               positive_distinct_orbits=sum(w.scores[g]>1e-12 for g in w.scores),
               new_orbits=w.evaluations,
               attempted=len(records),repeated=len(records)-w.evaluations,
               best_single=float(max([0]+list(w.scores.values()))),
               sum_positive_singles=float(sum(max(0,v) for v in w.scores.values())),
               # The joint gain is recomputed in a separate independent audit.
               costs=dict(o=w.o.counts(),feat=w.feature_queries,
                 feature_distinct_configs=len(w.feature_cache),
                 moves=w.policy_moves,orbit_accesses=w.orbit_calls,
                 oracle_score_calls=w.evaluations,repeated_score_hits=w.oracle_cache_hit),
               proposals=records,positive_orbits=accepted)
        result[name]=d
    return result

def main(args):
    ts=time.perf_counter()
    if args.budget%args.group!=0:raise ValueError('budget multiple of group')
    if args.budget>12000 or args.eval_attempts>512 or args.roots>512:
        raise ValueError('hard cap')
    train_seeds=[25001+10*(args.seed-10019)+i for i in range(4)]
    test_seeds=[35001+10*(args.seed-10019)+i for i in range(4)]
    train_roots,sampler_train=make_roots(args.data,train_seeds,
                                     per_seed=args.roots//4)
    test_roots,sampler_eval=make_roots(args.data,test_seeds,
                                     per_seed=args.eval_attempts//4)
    # Train on selected chamber; test both on disjoint-root trajectories.
    train={};policies={}
    for j,mode in enumerate(('dapo','grpo','keep_relative')):
        world=World(args.data,args.train_chamber,args.seed+98+j)
        theta,counts,trace=train_fn(world,train_roots,args.seed+j*977,mode,
                  args.budget,args.group,args.epsilon,args.lr)
        train[mode]=dict(counts=counts,weights=theta.tolist(),
             costs=dict(o=world.o.counts(),feature_queries=world.feature_queries,
                        oracle_score_calls=world.evaluations,
                        orbit_accesses=world.orbit_calls,
                        policy_moves=world.policy_moves),
             trace=trace)
        policies[mode]=theta
    for name in ('random','geometry','novelty','signed_bond'):policies[name]=np.zeros(DIM)
    evaluations={}
    for chamber in ('marshall','post_krylov'):
        evaluations[chamber]=eval_arms(args.data,test_roots,policies,args.seed+700,
                B=args.eval_attempts,chamber=chamber)
    output=dict(description=__doc__,args=vars(args),
          result_type='DAPO_INSPIRED_POLICY_GRADIENT_TOY_20SITE',
          cost_warning='Exact 20-site amplitudes used in root sampler and terminal verifier; no GFMC/NQS physics claim',
          split=dict(train_seeds=train_seeds,
                     test_seeds=test_seeds,
                     chamber_train=args.train_chamber,
                     chamber_eval=['marshall','post_krylov'],
                     caveat='test orbit IDs not explicitly heldout; roots disjoint RNG streams, with possible orbit overlap'),
          training=train,eval=evaluations,
          sampler=dict(train=sampler_train,eval=sampler_eval),
          seconds=time.perf_counter()-ts)
    Path(args.out).write_text(json.dumps(output,indent=2))
    print('TIME',output['seconds'])
    for chamber,arms in evaluations.items():
        print('TEST',chamber)
        for name,r in arms.items():
            print(' ',name,'positive',r['positive_draws'],'unique_pos',r['positive_distinct_orbits'],
                  'unique_orbits',r['new_orbits'],'best_gain',round(r['best_single'],9),
                  'amp_distinct',r['costs']['o']['unique_amp'],
                  'probes',r['costs']['o']['neighbor_probes'])
    print('TRAIN',[(k,v['counts'],v['costs']['oracle_score_calls']) for k,v in train.items()])

train_fn=train
if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--budget',type=int,default=2000)
    ap.add_argument('--train-chamber',choices=('marshall','post_krylov'),default='marshall')
    ap.add_argument('--group',type=int,default=4)
    ap.add_argument('--roots',type=int,default=256)
    ap.add_argument('--eval-attempts',type=int,default=160)
    ap.add_argument('--epsilon',type=float,default=.15)
    ap.add_argument('--lr',type=float,default=.2)
    ap.add_argument('--seed',type=int,default=10019)
    main(ap.parse_args())
