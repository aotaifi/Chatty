#!/usr/bin/env python3
"""AVSL-08: REAL M128 importance-sampled fixed-node branching + 20-site support.

A frozen, approximately FN-optimized guide for post-Krylov signs is the 20-site reference.
The true stationary mixed distribution is f(x) proportional to
a_guide(x)*phi_FN(x), which need NOT equal a_guide(x)^2.
The guide is not certified converged; FN beta mixing remains to be audited.
Unlike earlier AVSL toys, the observed snapshots come from the repository's
actual importance-sampled, systematic-reconfiguration FN projector, not an
a^gamma Metropolis root chain. a^2 Metropolis is used ONLY for initialization,
which is NOT guaranteed to be stationary for the approximate guide.

Existing FN–Krylov benchmarks are UNMODIFIED. This measures detection of
profitable single symmetry-orbit flips of frozen a (not true post-FN energy).
The full-basis exact oracle is used ONLY for post-hoc opportunity diagnostics.
Model training NEVER sees exact full-orbit labels outside queried candidates.
"""
import argparse, json, math, time, sys
from collections import Counter
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from callable_fn_loop import CachedAmplitude,K1FNEngine
from avsl_cooperative20 import Oracle,BONDS,sample_roots
from avsl_dapo_explorer20 import World,DIM,softmax,rng_choice,trajectory

class Tilted20:
    def __init__(self,oracle):
        self.o=oracle
        self.diag=np.load(oracle.data_path)['diag'] if hasattr(oracle,'data_path') else None
        self.neighbor_calls=0
        self.neighbor_returned=0
    def diag_energy(self,x):
        if self.diag is None:
            # computed from spin-dependent zz interactions, same as native ED
            z=0.
            for bits,h in BONDS:
                anti=(int(x)&bits).bit_count()==1
                z+=(h*2)*(-.25 if anti else .25)
            return z
        return float(self.diag[self.o.look[int(x)]])
    def neigh(self,x):
        self.neighbor_calls+=1
        out=[]
        for bits,h in BONDS:
            y=int(x)^bits
            if (int(x)&bits).bit_count()==1:
                out.append((y,2*h,0.))
        self.neighbor_returned+=len(out)
        return out

class ExactSignFN(K1FNEngine):
    def __init__(self,lat,amplitude,oracle):
        super().__init__(lat,amplitude)
        self.oracle=oracle
    def ensure_sign(self,states):
        for x in np.unique(np.asarray(states,np.uint64)):
            v=int(x)
            if v not in self.sign_cache:self.sign_cache[v]=self.oracle.s(v)

def make_fn_walkers(data,seed,M=128,beta=.45,burn=.10,
                    tau=.025,n_init=1024,init_burn=2400,init_thin=9):
    """Return actual *post-reconfiguration* walker snapshots, with costs."""
    o=Oracle(data)
    # Guide-squared a^2 preconditioning only; NOT stationary when phi_FN != a.
    pool,ar=sample_roots(o,seed+6000,n=n_init,burn=init_burn,
                         thin=init_thin,law='a2')
    init_cost=o.counts().copy()
    guide=CachedAmplitude(lambda xs: np.log(
        np.asarray([o.a(int(x)) for x in xs],dtype=float)))
    lat=Tilted20(o)
    eng=ExactSignFN(lat,guide,o)
    ti=time.perf_counter()
    walkers,Es=eng.run_fn(pool,M=M,beta_target=beta,
          burn_beta=burn,tau_max=tau,seed=seed+9000,progress_every=0)
    dur=time.perf_counter()-ti
    if not len(walkers) or any(int(x).bit_count()!=10 for x in walkers):
        raise AssertionError('bad FN snapshot')
    ref=np.asarray([eng.local_cache[int(x)][1] for x in walkers])
    if not np.isfinite(ref).all():raise AssertionError('nonfinite local energy')
    info=dict(
        seed=seed,source='genuine_K1FNEngine_FN_M128_systematic_reconfiguration',
        M=M,beta_target=beta,beta_burn=burn,tau_max=tau,
        snapshots=len(Es),n_walkers=len(walkers),
        n_unique_configs=len(set(map(int,walkers))),
        n_unique_orbits=len({o.key(int(x)) for x in walkers}),
        initial_a2_MH_acceptance=float(ar),
        warning_mixed_not_converged="beta sensitivity, population control bias and autocorrelation not yet established",
        guide_is_converged_FN_eigenvector=False,
        fn_wall_s=dur,fn_energy_mean=float(Es.mean()),
        fn_energy_spread=float(Es.std()),
        local_energy_mean=float(ref.mean()),local_energy_spread=float(ref.std()),
        init_oracle_cost=init_cost,
        all_oracle_cost=o.counts(),fn_unique_guide_states=guide.neval,
        fn_lattice_neighbor_calls=lat.neighbor_calls,
        fn_lattice_neighbor_edges=lat.neighbor_returned)
    return [int(x) for x in walkers],info

def oracle_opportunities(data):
    # Complete 20-site enumeration for *diagnostics only*, never an action list.
    o=Oracle(data)
    mass={};orbits={}
    for x,amp in zip(o.states,o.a_data):
        g=o.key(int(x));mass[g]=mass.get(g,0.)+float(amp*amp)
        orbits[g]=len(o.members[g])
    w=World(data,'post_krylov',5)
    gains={}
    for g in mass:
        key,gain,fresh=w.score(g)
        gains[key]=gain
    assert len(gains)==4654,len(gains)
    positive={g:v for g,v in gains.items() if v>1e-12}
    return o,mass,gains,positive,dict(n_orbits=len(mass),
         n_positive_orbits=len(positive),fraction_positive=len(positive)/len(mass),
         independent_positive_gains_sum=float(sum(positive.values())),
         largest_positive_single_orbit_gain=float(max(positive.values())),
         note='sum of positive single-flip gains NOT a joint energy gain')

def root_support(o,roots,positive,mass):
    hit={o.key(int(x)) for x in roots}
    pos=set(positive);unseen=pos-hit;covered=pos&hit
    return dict(n_roots=len(roots),n_hit_orbits=len(hit),
       n_positive_hit=len(covered),n_positive_missed=len(unseen),
       fraction_positive_missed=len(unseen)/len(pos),
       positive_gain_missed_sum=float(sum(positive[g] for g in unseen)),
       positive_gain_hit_sum=float(sum(positive[g] for g in covered)),
       positive_mixed_mass_missed=float(sum(mass[g] for g in unseen)),
       # Mass of all missed positive orbits, NOT physical gain.
       oracle_diagnostic_only=True)

def one_hop_frontier(o,roots):
    keys={o.key(int(x)) for x in roots}
    for x in roots:
        for bits,h in BONDS:
            if (int(x)&bits).bit_count()==1:
                keys.add(o.key(int(x)^bits))
    return keys

def learned_path(w,root,theta,rng,epsilon=.18,depth=2):
    x=int(root);gr=np.zeros(DIM)
    for step in range(depth):
        opts=w.options(x,step)
        if not opts:break
        ff=np.array([f for _,f in opts])
        q=softmax(ff@theta);p=(1-epsilon)*q+epsilon/len(q)
        j=rng_choice(rng,p)
        gr+=(1-epsilon)*q[j]/p[j]*(ff[j]-q@ff)
        x=int(opts[j][0]);w.visited.add(x)
        w.policy_moves+=1
    g,gain,fresh=w.score(x)
    return g,float(gain),bool(fresh),gr

def train_continuous(data,roots,seed,B=400,lr=.06,scale=.0001):
    """On-policy raw signed fixed-a energy reward, keep=0, no binary labels.

    Exact process gradient of trajectory log probability; updates on
    both positive and negative rewards (negative repels harmful locations).
    Clip only the UPDATE magnitude; raw, signed gain is recorded.
    """
    w=World(data,'post_krylov',seed)
    rng=np.random.default_rng(seed+1000)
    theta=np.zeros(DIM,float)
    counts=Counter()
    for i in range(B):
        x=int(roots[int(rng.integers(len(roots)))])
        g,gain,fresh,gradient=learned_path(w,x,theta,rng)
        # No positive reward for harmful action even if group centered.
        # Clipping protects rare steep gradients from destabilizing online SGD.
        # This clips scalar gain but preserves sign and ranking in the normal range.
        reward=float(np.clip(gain/scale,-2.,2.))
        theta+=lr*np.clip(reward*gradient,-.4,.4)
        counts['positive']+=int(gain>1e-12)
        counts['negative']+=int(gain< -1e-12)
        counts['near_zero']+=int(abs(gain)<=1e-12)
        counts['new_scores']+=int(fresh)
    return theta,dict(counts),dict(oracle_cost=w.o.counts(),
       score_calls=w.evaluations,feature_queries=w.feature_queries,
       policy_moves=w.policy_moves,raw_scale=scale)

def benchmark_arm(data,roots,arm,seed,B,theta=None,unique_budget=False):
    w=World(data,'post_krylov',seed+500)
    rng=np.random.default_rng(seed+100000)
    positives={};visited=[];gains=[]
    for root in (roots if unique_budget else roots[:B]):
        if unique_budget and w.evaluations>=B:break
        if arm=='walker':
            g,gain,fresh=w.score(int(root))
        elif arm=='random2':
            t=trajectory(w,int(root),np.zeros(DIM),'random',rng)
            g,gain,fresh=t['orbit'],t['gain'],t['fresh']
        elif arm=='learned':
            g,gain,fresh,_=learned_path(w,int(root),theta,rng)
        else:raise ValueError(arm)
        visited.append(g);gains.append(float(gain))
        if gain>1e-12:positives[g]=float(gain)
    return dict(
      B=len(visited),distinct_scores=w.evaluations,
      matched_unique_verifier_budget=bool(unique_budget),
      positive_distinct_orbits=len(positives),
      positive_draws=int(sum(v>1e-12 for v in gains)),
      best_single_gain=float(max([0]+gains)),
      positive_singles_sum_NOT_joint=float(sum(positives.values())),
      keys=[int(g) for g in visited],
      unique_positive_keys=[int(g) for g in positives],
      oracle_queries=w.o.counts(),
      feature_queries=w.feature_queries,
      policy_moves=w.policy_moves)

def tempered_proposals(data,seed,B,burn=1200,thin=7,gamma=.5):
    o=Oracle(data)
    rng=np.random.default_rng(seed+71000)
    x=int(o.states[rng.integers(len(o.states))])
    roots=[];accepted=0
    for t in range(burn+thin*B):
        bits,h=BONDS[int(rng.integers(len(BONDS)))]
        y=x^bits;o.n_neigh_probes+=1
        if (x&bits).bit_count()==1:
            if np.log(max(1e-300,rng.random()))<min(0.,gamma*np.log(o.a(y)/o.a(x))):
                x=y;accepted+=1
        if t>=burn and (t-burn)%thin==0:roots.append(x)
    return roots,dict(burn=burn,thin=thin,gamma=gamma,
         accept_rate=accepted/(burn+B*thin),oracle_queries=o.counts())

def experiment(args):
    ts=time.perf_counter()
    if args.M!=128 or args.beta>4.0 or args.n_init>2048 or args.train_budget>1200 or args.eval_budget>512:
        raise ValueError('hard pilot safety cap')
    trainroots,trainfn=make_fn_walkers(args.data,args.seed+77,args.M,args.beta,
        args.burn,args.tau,args.n_init)
    evalroots,evalfn=make_fn_walkers(args.data,args.seed+555,args.M,args.beta,
        args.burn,args.tau,args.n_init)
    rng=np.random.default_rng(args.seed)
    rng.shuffle(evalroots)
    # Training uses separate FN walker runs; no reuse of random seed streams.
    theta,tstat,tcost=train_continuous(args.data,trainroots,args.seed+999,
                     args.train_budget,lr=args.lr,scale=args.scale)
    B=min(args.eval_budget,len(evalroots))
    arms={}
    for j,arm in enumerate(('walker','random2','learned')):
        arms[arm]=benchmark_arm(args.data,evalroots,arm,args.seed+400+j,B,theta,unique_budget=args.unique_budget)
    temper,tempcost=tempered_proposals(args.data,args.seed,B*8 if args.unique_budget else B)
    arms['tempered']=benchmark_arm(args.data,temper,'walker',args.seed+421,B,unique_budget=args.unique_budget)
    arms['tempered']['proposal_chain_cost']=tempcost
    # Score orbits not found in any of the actual FN walker snapshots, not just
    # in randomly selected B initial walkers.
    o,mass,gains,positive,full=oracle_opportunities(args.data)
    real=root_support(o,evalroots,positive,mass)
    fkeys=one_hop_frontier(o,evalroots)
    unseen=set(positive)-fkeys
    fstats=dict(n_orbits=len(fkeys),n_positive_in_frontier=len(set(positive)&fkeys),
          n_positive_unseen=len(unseen),
          positive_gain_unseen_sum=float(sum(positive[g] for g in unseen)))
    rootseen={o.key(x) for x in evalroots}
    for name,r in arms.items():
        missed=[g for g in r['unique_positive_keys'] if g not in rootseen]
        r['positive_outside_actual_fn_walker_support']=len(missed)
        r['outside_positive_gains_sum_NOT_joint']=float(sum(positive[g] for g in missed))
    result=dict(label='AVSL08_real_FN_M128_missed_correction_pilot',
       data_source=args.data,args=vars(args),train_FN=trainfn,
       heldout_FN=evalfn,full_opportunity_audit=full,
       heldout_full_fn_support=real,heldout_one_hop_frontier=fstats,
       training=dict(stats=tstat,training_cost=tcost,weights=theta.tolist()),
       evaluation=arms,
       seconds=time.perf_counter()-ts,
       caveats=[
        'M128 snapshots are from actual finite-tau K1FNEngine projection and systematic branching, not toy Metropolis. Seed pool was pre-equilibrated by independent a^2 MH.',
        'Stored 20-site guide a is approximate, NOT an exact FN eigenvector; mixed f=a*phiFN is generally different from a^2. Projector beta convergence, M128 population bias and autocorrelation NOT certified.',
        'Initial a^2 sampler and FN projector query exact global 20-site amplitude oracle, not scalable NQS.',
        'Full-basis oracle only used for post-hoc diagnostic; sign/orbit gain labels never enter policy features.',
        'Learned policy trained on SAME post-Krylov frozen chamber; heldout seeds only, orbit identities can overlap.',
        'Binary success is frozen-a single orbit flip >1e-12; no joint accepted eta(x) or new FN projection.',
        'Training and FN preparation cost reported separately; learned arm did not receive free training.',
        'Tempered proposal sampler is an independent amplitude-ratio Metropolis chain (NOT FN dynamics), extra cost reported.',
        'No evidence of physically important rare sign corrections until missed positive total gain and FN-supported comparison are reported.' ])
    Path(args.out).write_text(json.dumps(result,indent=2))
    print('FN',args.seed,'runtime_s',round(result['seconds'],2),
        'FN snapshots',len(evalroots),'fn unique',evalfn['n_unique_configs'],
        'FN local std',round(evalfn['local_energy_spread'],10))
    print('SUPPORT',real,'FRONTIER',fstats)
    for arm,r in arms.items():
        print('ARM',arm,'pos',r['positive_distinct_orbits'],
              'outside FN',r['positive_outside_actual_fn_walker_support'],
              'gain',r['positive_singles_sum_NOT_joint'],
              'guide states',r['oracle_queries']['unique_amp'])
    print('TRAIN',tstat,tcost)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--seed',type=int,default=17019)
    p.add_argument('--M',type=int,default=128)
    p.add_argument('--n-init',type=int,default=768)
    p.add_argument('--beta',type=float,default=.38)
    p.add_argument('--burn',type=float,default=.12)
    p.add_argument('--tau',type=float,default=.025)
    p.add_argument('--train-budget',type=int,default=400)
    p.add_argument('--eval-budget',type=int,default=160)
    p.add_argument('--lr',type=float,default=.06)
    p.add_argument('--scale',type=float,default=.0001)
    p.add_argument('--unique-budget',action='store_true')
    experiment(p.parse_args())
