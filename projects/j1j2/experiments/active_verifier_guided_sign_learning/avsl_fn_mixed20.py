#!/usr/bin/env python3
"""20-site ACTUAL importance-sampled fixed-node branching walkers.

The continuous-time Feynman-Kac process represents f(x,t)=a(x) phi_FN(x,t):
  jump y <- x at k_xy = h_xy a(y)/a(x), when s(x)s(y)<0
  birth/death potential E_L(x)=H_FN a(x)/a(x)
                        = H_xx+sum_y h_xy s(x)s(y) a(y)/a(x).
We propagate exact continuous-time jumps within each resampling interval and
apply path weight exp[-int E_L(x_t)dt]. The global shift cancels when resampling.
No exact FN eigenvector enters sampler; a is the current *guide*, NOT phi_FN.
A separate exact diagonalization can audit the resulting mixed distribution.
Do not claim walker independence: population control and genealogy correlate.
"""
import argparse, json, time
from collections import Counter
from pathlib import Path
import numpy as np
from avsl_cooperative20 import Oracle,BONDS,sample_roots

class MixedFN:
    def __init__(self, path, sign_mode='post_krylov'):
        self.o=Oracle(path)
        self.d=np.load(path)
        self.guide=self.d['amp']
        self.diag=self.d['diag']
        self.sign=np.asarray(self.d['guide'],np.int8)
        if sign_mode=='marshall':
            ma=int(self.d['maskA'][0])
            bits=np.array([(int(x)&ma).bit_count()%2 for x in self.o.states],np.int8)
            self.sign=np.where(bits==0,1,-1).astype(np.int8)
        self.sign_mode=sign_mode
        self.cache={}
        self.n_transition_builds=0
        self.n_edges=0
        self.n_a_ratio=0
        self.n_s_query=0
        self.n_jumps=0
        self.n_interval=0
        self.n_weight_log=0
    def transitions(self,x):
        x=int(x)
        if x in self.cache:return self.cache[x]
        ix=int(self.o.look[x])
        if ix<0:raise ValueError('Outside half-filling sector')
        ax=self.o.a(x);sx=int(self.sign[ix])
        rates=[];dest=[];elist=float(self.diag[ix])
        for bits,h in BONDS:
            self.o.n_neigh_probes+=1
            if (x&bits).bit_count()!=1:continue
            y=x^bits; iy=int(self.o.look[y])
            ratio=self.o.a(y)/ax
            z=sx*int(self.sign[iy]);self.n_a_ratio+=1;self.n_s_query+=2
            elist+=h*z*ratio
            if z<0:
                rates.append(h*ratio);dest.append(y)
        rates=np.asarray(rates,float);dest=np.asarray(dest,np.uint32)
        total=float(np.sum(rates))
        self.cache[x]=(dest,rates,np.cumsum(rates),total,elist)
        self.n_transition_builds+=1;self.n_edges+=len(dest)
        return self.cache[x]
    def interval(self,x,dt,rng,max_jumps=10000):
        remain=float(dt);acc=0.;n=0
        while remain>1e-14:
            dest,rates,cs,total,e=self.transitions(x)
            if total<=0:
                acc+=e*remain;break
            hold=rng.exponential(1/total)
            if hold>=remain:
                acc+=e*remain;break
            acc+=e*hold;remain-=hold
            x=int(dest[min(int(np.searchsorted(cs,rng.random()*total)),len(dest)-1)])
            n+=1
            if n>max_jumps:raise RuntimeError('Excessive Gillespie jump rate; check guide')
        self.n_jumps+=n
        self.n_interval+=1
        return x,acc
    def sample(self,seed,n_walkers=128,beta=12.,dt=.15,burn=6.,
               record_every=3,init_burn=2200):
        rng=np.random.default_rng(seed)
        starts,acceptance=sample_roots(self.o,seed+179,n=n_walkers,burn=init_burn,
                                      thin=11,law='a2')
        x=np.asarray(starts,np.uint32)
        steps=int(np.ceil(beta/dt))
        samples=[];unique_visited=set(map(int,x))
        ess=[];log_norm=[];genealogy=[]
        ids=np.arange(n_walkers,dtype=np.int64)
        for k in range(steps):
            lweights=np.empty(n_walkers)
            for j in range(n_walkers):
                xx,integ=self.interval(int(x[j]),dt,rng)
                x[j]=xx;lweights[j]=-integ
            m=float(np.max(lweights))
            w=np.exp(lweights-m);wn=w/w.sum()
            ess.append(float(1/np.dot(wn,wn)))
            log_norm.append(float(m+np.log(w.mean())))
            # Systematic resampling, unbiased expected counts but correlated descendants.
            p0=rng.random()/n_walkers
            c=np.cumsum(wn)
            idx=np.searchsorted(c,p0+np.arange(n_walkers)/n_walkers,side='left')
            x=x[idx].copy(); ids=ids[idx]
            unique_visited.update(map(int,x))
            if (k+1)*dt>=burn and (k+1)%record_every==0:
                samples.extend(map(int,x))
                genealogy.append(int(np.unique(ids).size))
        return np.asarray(samples,np.uint32),dict(
            seed=seed,walker_count=n_walkers,steps=steps,
            dt=float(dt),beta=float(beta),burn=float(burn),
            record_every=record_every,n_recorded=len(samples),
            n_unique_recorded=len(set(map(int,samples))),
            n_unique_walkers_seen=len(unique_visited),
            n_independent_ancestor_ids_final=int(np.unique(ids).size),
            median_ancestor_ids=float(np.median(genealogy)) if genealogy else None,
            acceptance_initial_a2=float(acceptance),
            mean_effective_sample_size=float(np.mean(ess)),
            min_effective_sample_size=float(np.min(ess)),
            mean_log_norm_per_step=float(np.mean(log_norm)),
            n_jumps=self.n_jumps,unique_a=self.o.counts()['unique_amp'],
            n_amp_requests=self.o.counts()['amp_requests'],
            n_sign_requests=self.n_s_query,n_neigh_probes=self.o.n_neigh_probes,
            n_local_energy_evals=self.n_transition_builds,
            n_transition_edges=self.n_edges)

def ref_fn(data,out,source='/tmp/avsl05-20site-structural-data/fn_collapse_exact20_krylov_v2.py',sign_mode='post_krylov'):
    # Exact 20-site verification: ED solely for audit, not for walker dynamics.
    src=Path(source).read_text()
    namespace={'__name__':'__avsl20_verifier__'}
    exec(src[:src.index('import sys\nNITER=')],namespace)
    q=np.load(data)
    a=q['amp'].copy(); s=q['guide'].copy()
    if sign_mode=='marshall':s=namespace['sM'].copy()
    en,phi,info=namespace['fn_step'](a,s,'heldout exact FN verification',tol=2e-9)
    pmix=a*phi;pmix/=pmix.sum()
    ppure=phi**2;ppure/=ppure.sum()
    np.savez_compressed(out,phi=phi,mixed=pmix,pure=ppure,fn_energy=en)
    return dict(fn_energy=float(en),residual=info['resid'],
                mixed_norm=float(pmix.sum()),a2_vs_mixed_tv=float(.5*np.abs(a*a/np.sum(a*a)-pmix).sum()))

def audit(mixed,roots,sim,expected_energy=None):
    oracle=sim.o
    w=np.bincount(oracle.look[roots],minlength=len(oracle.states))
    w=w/w.sum()
    a=oracle.a_data
    # Diagonal observables: nearest-neighbor antiparallel counts, density
    # of states where post-Krylov root-level small-a score is under threshold.
    n1=np.zeros(len(a),float)
    for bits,h in BONDS:
        if h==.5:n1+=np.array([(int(z)&bits).bit_count()==1 for z in oracle.states],float)
    n1true=float(mixed@n1); n1mc=float(w@n1)
    indices=np.argsort(a); cut=int(.1*len(a))
    tail=indices[:cut]
    seen,counts=np.unique(roots,return_counts=True)
    est_energy=float(sum(n*sim.transitions(int(x))[4] for x,n in zip(seen,counts))/len(roots))
    return dict(fn_energy_mixed_walker_estimator=est_energy,
                fn_energy_exact=expected_energy,
                fn_energy_abs_error=abs(est_energy-expected_energy) if expected_energy is not None else None,
                nn_domainwalls_exact=n1true,nn_domainwalls_mc=n1mc,
                nn_domainwalls_abs_error=abs(n1true-n1mc),
                rare_low_a_fraction_exact=float(mixed[tail].sum()),
                rare_low_a_fraction_mc=float(w[tail].sum()),
                n_distinct_recorded_states=int(np.sum(w>0)),
                n_samples=int(len(roots)))

def main(args):
    ts=time.perf_counter()
    ref=None
    if args.make_ref:
        ref=ref_fn(args.data,args.ref_out,sign_mode=args.sign_mode)
    results=[]
    for j in range(args.seeds):
        sim=MixedFN(args.data,args.sign_mode)
        roots,diag=sim.sample(args.seed+991*j,args.walkers,args.beta,args.dt,args.burn,args.record_every)
        out=Path(args.out).with_suffix('')
        np.savez_compressed(str(out)+f'_roots_{args.seed+991*j}.npz',roots=roots)
        if Path(args.ref_out).exists():
            q=np.load(args.ref_out)
            diag['reference_audit']=audit(q['mixed'],roots,sim,expected_energy=float(q['fn_energy']))
        results.append(diag)
        print('FN mixed',j,diag,flush=True)
    output=dict(description=__doc__,reference=ref,args=vars(args),
               sign_chamber=args.sign_mode,simulations=results,
               seconds=time.perf_counter()-ts)
    Path(args.out).write_text(json.dumps(output,indent=2))
    print('SAVED',args.out,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--ref-out',required=True);ap.add_argument('--make-ref',action='store_true')
    ap.add_argument('--sign-mode',choices=('marshall','post_krylov'),default='post_krylov')
    ap.add_argument('--walkers',type=int,default=128)
    ap.add_argument('--beta',type=float,default=12.)
    ap.add_argument('--burn',type=float,default=6.)
    ap.add_argument('--dt',type=float,default=.15)
    ap.add_argument('--record-every',type=int,default=4)
    ap.add_argument('--seeds',type=int,default=2)
    ap.add_argument('--seed',type=int,default=28019)
    main(ap.parse_args())
