#!/usr/bin/env python3
"""Bounded AVSL Stage-0 interface preflight; NO neural training or physical claim.

Uses the actual repository K1FNEngine and M=128 mixed walker projector, with
a *synthetic analytic*, unnormalized positive Jastrow guide. Explicitly NOT
a real NQS checkpoint, true on-policy FN optimisation, or an FN benchmark.

Validation runs separate independent-seed a^2 VMC chains for paired physical
frozen-guide energy differences (keep/global/K1-to-Marshall). Mixed FN walkers
are NEVER substituted for those chains. Counters are append-only increments.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))
from callable_fn_loop import CachedAmplitude, K1FNEngine, SquareJ1J2
from avsl_onpolicy_core import (CostLedger, MeteredCallable, paired_fixed_a_audit,
                                paired_local_difference)


class CountingLattice(SquareJ1J2):
    def __init__(self,L,J2):
        super().__init__(L,J2)
        self.neigh_calls=0
        self.neigh_returned=0

    def neigh(self,x):
        ls=super().neigh(x)
        self.neigh_calls+=1
        self.neigh_returned+=len(ls)
        return ls


class AnalyticJastrow:
    """Simple translation-covariant test callable, no checkpoint and no ML.

    log a(x) = g1 * number NN antiparallel bonds + g2 * NNN antiparallel.
    Amplitudes are positive, nonconstant, and fully callable on 36-bit states.
    """
    def __init__(self,lat,g1=0.045,g2=-0.025):
        self.nn=[(i,j) for i,j,J,_ in lat.bonds if J==1.0]
        self.nnn=[(i,j) for i,j,J,_ in lat.bonds if J!=1.0]
        self.g1=float(g1);self.g2=float(g2)

    def log_amplitude_bits(self,states):
        return [self.g1*sum(((int(x)>>i)^(int(x)>>j))&1 for i,j in self.nn)
                +self.g2*sum(((int(x)>>i)^(int(x)>>j))&1 for i,j in self.nnn)
                for x in states]


def random_half_filled(N,rng):
    bits=rng.choice(N,size=N//2,replace=False)
    return sum(1<<int(j) for j in bits)


def independent_a2_chains(lat,log_a,ledger,*,seed_base,chains,
                          burn,thin,draws,phase):
    """Symmetric bond-proposal Metropolis, fixed Sz=0; independent initial rngs.

    Each attempted bond exchange is a single micro-proposal, including rejected
    same-spin bonds. A complete sweep is N micro-proposals, not one proposal.
    This proposal preserves exact pi∝a^2 detailed balance.
    """
    all_chains=[]
    rate=[]
    attempts=chains*(burn+thin*draws)
    start=time.perf_counter()
    nb=len(lat.bonds)
    for i in range(chains):
        rng=np.random.default_rng(seed_base+i)
        x=random_half_filled(lat.N,rng)
        accepted=0;roots=[]
        lx=log_a(x)
        for t in range(burn+thin*draws):
            j,k,_,_=lat.bonds[int(rng.integers(nb))]
            if ((x>>j)^(x>>k)) & 1:
                y=x^(1<<j)^(1<<k)
                ly=log_a(y)
                if math.log(max(rng.random(),1e-300))<min(0.,2*(ly-lx)):
                    x,lx=y,ly
                    accepted+=1
            if t>=burn and (t-burn)%thin==0:
                roots.append(x)
        all_chains.append(roots)
        rate.append(accepted/(burn+thin*draws))
    ledger.record(phase,"independent_a2_Metropolis",
                  wall_s=time.perf_counter()-start,
                  n_neighbor_hops=attempts,
                  n_vmc_sweeps=attempts/lat.N)
    return all_chains,rate


def np_dump_summary(data):
    def clean(v):
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,(np.integer,np.floating)):return v.item()
        if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
        if isinstance(v,(tuple,list)):return [clean(x) for x in v]
        return v
    return clean(data)


def preflight(args):
    if args.L!=6: raise ValueError("Stage-0 scope fixed to L=6")
    if args.M!=128: raise ValueError("Stage-0 uses exactly M=128")
    if args.fn_beta_target>0.12 or args.burn>1000 or args.draws>256 or args.chains>8:
        raise ValueError("hard budget cap exceeded")
    if min(args.thin,args.draws,args.chains,args.M)<=0 or args.burn<0:
        raise ValueError("invalid sampler budget")

    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    ledger_path=out/"events.jsonl"
    if ledger_path.exists():
        raise FileExistsError(f"refuse append into preexisting run: {ledger_path}")
    code_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    rid=f"avsl05-stage0-seed{args.seed}"
    lid=dict(run_id=rid,code_commit=code_sha,
             chamber_id="synthetic_Jastrow_initial_K1_not_NQS_FN",
             arm="no_learning_stage0",
             device=f"{platform.system()}-{platform.machine()}-Python{platform.python_version()}",
             model_hash="ANALYTIC_JASTROW_g1_0.045_g2_-0.025",
             walker_seed=args.seed+4400,proposal_seed=args.seed+1200,
             training_seed=-1)
    ledger=CostLedger(ledger_path,**lid)
    start=time.perf_counter()
    lat=CountingLattice(args.L,args.J2)
    backend=AnalyticJastrow(lat)
    metered=MeteredCallable(backend.log_amplitude_bits,role="guide",
                           ledger=ledger,phase="all_guide_calls")
    # Keep K1FNEngine's own correctly deduplicated front-end: it uses the same
    # public batched, callable interface as NQS checkpoint backends.
    amp=CachedAmplitude(lambda x: metered.many([int(y) for y in x]))
    eng=K1FNEngine(lat,amp)
    tstart=time.perf_counter()

    # Separate, fully recorded threshold roots; verifier has disjoint seeds.
    train,atrain=independent_a2_chains(lat,metered,ledger,
        seed_base=args.seed+1200,chains=args.chains,
        burn=args.burn,thin=args.thin,draws=args.draws,
        phase="threshold_roots")
    trainroots=np.asarray([x for chain in train for x in chain],dtype=np.uint64)
    T,r,info=eng.fit_threshold(trainroots)
    threshold_duration=time.perf_counter()-tstart
    if not np.isfinite(T) or not np.all(np.isfinite(r)):
        raise AssertionError("bad K1 threshold")
    eng.ensure_sign(trainroots)
    k1_ratio=np.mean([eng.sign_cache[int(x)]!=lat.marshall(int(x)) for x in trainroots])

    # Actual M128 mixed FN walker engine, deliberately tiny projection beta.
    amp_before=amp.neval
    lat_before=(lat.neigh_calls,lat.neigh_returned)
    tstart=time.perf_counter()
    mixed, E=eng.run_fn(trainroots,M=args.M,beta_target=args.fn_beta_target,
                      burn_beta=0.0,tau_max=args.tau_max,
                      seed=args.seed+4400,progress_every=0)
    fn_duration=time.perf_counter()-tstart
    if len(E)==0 or len(mixed)!=len(E)*args.M:
        raise AssertionError(("incomplete short FN projector",len(E),len(mixed)))
    if not np.all(np.isfinite(E)):raise AssertionError("non-finite FN local energies")
    if any(int(x).bit_count()!=lat.N//2 for x in mixed):
        raise AssertionError("walker escaped Sz=0")
    ledger.record("fn_projection","short_real_M128_FN_engine",
                  n_fn_reprojection_steps=len(E),
                  n_walker_steps=len(E)*args.M,
                  n_neighbor_hops=lat.neigh_returned-lat_before[1])
    # Deliberately do not use mixed for a^2 variational verifier!

    chains,aval=independent_a2_chains(lat,metered,ledger,
       seed_base=args.seed+90000,chains=args.chains,
       burn=args.burn,thin=args.thin,draws=args.draws,
       phase="heldout_independent_a2")

    def sign(x):
        eng.ensure_sign([x])
        return eng.sign_cache[int(x)]

    metered_sign=MeteredCallable(
        lambda states: [sign(x) for x in states],
        role="sign",ledger=ledger,phase="evaluation_sign_calls")
    def s(x):return metered_sign(int(x))

    neigh=lambda x: [(y,0.5*J) for y,J,_ in lat.neigh(x)]

    def e_keep(x):return 1
    def e_global(x):return -1
    def e_marshall(x):return int(lat.marshall(x)*s(x))

    def paired(e):
        return paired_fixed_a_audit(chains,neighbors=neigh,
                                   log_a=metered,sign=s,eta=e,
                                   sites=lat.N,sampling_law="a2")
    pkeep,pglobal,pmarshall=paired(e_keep),paired(e_global),paired(e_marshall)
    if pkeep.delta_energy!=0 or pglobal.delta_energy!=0:
        raise AssertionError("keep/global sign invariance failed")

    # Validate joint rule on every input pair; both delta=0 controls cannot
    # magically certify mixing or the energies for a nontrivial candidate.
    ledger.record("verification","control_keep_global_marshall",
                  n_verifier_candidates_attempted=3)
    outdata=dict(
      result_type="synthetic_interface_validation_NOT_real_NQS",
      code_sha256=code_sha,params=vars(args),
      fn=dict(n_time_slices=len(E),M=args.M,n_mixed_snapshots=len(mixed),
          n_unique_mixed_states=len(set(map(int,mixed))),
          guide_misses_during_FN=int(amp.neval-amp_before),
          runtime_s=fn_duration,
          raw_E_mean=float(E.mean()),raw_E_final=float(E[-1]),
          warning="Projector beta intentionally tiny, not converged. Mixed snapshots are not a2 samples."),
      threshold=dict(value=float(T),n_training_roots=len(trainroots),
          k1_disagreement_fraction_with_marshall=float(k1_ratio),
          fit_time_s=threshold_duration,
          quantiles=np.quantile(r,[0,.1,.5,.9,1]).tolist()),
      paired={name:vars(item) for name,item in
              [('keep',pkeep),('global_sign',pglobal),('K1_to_Marshall',pmarshall)]},
      chains=dict(n_chains=args.chains,n_samples_per_chain=args.draws,
          threshold_acceptance_per_chain=atrain,
          verifier_acceptance_per_chain=aval,
          seeds_threshold=[args.seed+1200+j for j in range(args.chains)],
          seeds_verifier=[args.seed+90000+j for j in range(args.chains)],
          warning="No integrated autocorrelation or equilibrium diagnostic; SE provisional."),
      costs=dict(
        model_calls=metered.model_calls,model_distinct_configs=len(metered.cache),
        model_requests=metered.requests,
        sign_model_calls=metered_sign.model_calls,
        sign_distinct_configs=len(metered_sign.cache),
        lat_neigh_calls=lat.neigh_calls,
        lat_neigh_edges=lat.neigh_returned,
        fn_wall_s=fn_duration,
        total_wall_s=time.perf_counter()-start),
      device=lid["device"],
      limitations=[
        "Synthetic analytic log-amplitudes, NOT checkpoint NQS. No NQS inference timing.",
        "FN mixed walkers used only as projector smoke check.",
        "Independent a2 VMC samples have no burn/thinning autocorrelation proof.",
        "K1 to Marshall is a reversible reference candidate, not a learned improvement.",
        "Short beta no physical convergence or FN accuracy.",
      ])
    (out/"result.json").write_text(json.dumps(np_dump_summary(outdata),indent=2))
    # No bulky walker snapshots: physical FN here is only an interface smoke test.
    print(json.dumps({'threshold':outdata['threshold'],'fn':outdata['fn'],
        'paired':outdata['paired'],'costs':outdata['costs']},indent=2))
    return outdata


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",required=True)
    p.add_argument("--seed",type=int,default=240519)
    p.add_argument("--L",type=int,default=6)
    p.add_argument("--J2",type=float,default=0.5)
    p.add_argument("--M",type=int,default=128)
    p.add_argument("--chains",type=int,default=4)
    p.add_argument("--draws",type=int,default=64)
    p.add_argument("--burn",type=int,default=256)
    p.add_argument("--thin",type=int,default=5)
    p.add_argument("--fn-beta-target",type=float,default=0.05)
    p.add_argument("--tau-max",type=float,default=0.025)
    preflight(p.parse_args())
