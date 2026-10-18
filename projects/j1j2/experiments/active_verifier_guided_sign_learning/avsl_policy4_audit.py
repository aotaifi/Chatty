#!/usr/bin/env python3
"""Post-hoc normalized variational energies, paired seed-bootstrap and full-H checks.
Never used for policy training or acquisition. Runs after rollout JSON is frozen.
"""
import json,itertools,sys,argparse
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import avsl_probe4 as p
from avsl_probe4_validation import full_energy

def normalization(kappa):
 g=p.Oracle(kappa);tot=0.
 for spins in itertools.combinations(range(p.N),p.K):
  bits=sum(1<<i for i in spins);amp,_=g.evaluate(bits);tot+=amp*amp
 return float(tot)

def audit(path, out, full_checks=4):
 d=json.load(open(path));Z={str(k):normalization(k) for k in sorted({r['kappa'] for r in d['per_eval']})}
 rows=[];methods=['learned','signed','random','max_violation']
 for row in d['per_eval']:
  z=Z[str(row['kappa'])];r={k:row[k] for k in ['seed','kappa']}
  for method in methods:r[method]=row['methods'][method]['gain_numerator']/z
  rows.append(r)
 arr={m:np.array([r[m] for r in rows]) for m in methods}
 delta=arr['learned']-arr['signed'];dr=arr['learned']-arr['random']
 # cluster bootstrap by seed; kappa points for a given seed are not independent
 seeds=sorted({r['seed'] for r in rows});random=np.random.default_rng(8891)
 meanbyseed={s:np.mean([r['learned']-r['signed'] for r in rows if r['seed']==s]) for s in seeds}
 boot=np.array([np.mean([meanbyseed[s] for s in random.choice(seeds,size=len(seeds),replace=True)]) for i in range(12000)])
 checks=[]
 for row in d['per_eval'][:full_checks]:
  ener0,z=full_energy([],row['kappa'])
  for method in ['learned','signed','random']:
   actual,_=full_energy(row['methods'][method]['selected_orbit_keys'],row['kappa'])
   predicted=row['methods'][method]['gain_numerator']/Z[str(row['kappa'])]
   err=abs((ener0-actual)-predicted)
   if err>2e-10:raise RuntimeError(f'full-H disagreed: {err}')
   checks.append({'seed':row['seed'],'kappa':row['kappa'],'method':method,'discrepancy':float(err)})
 result={'source':str(path),'normalizers':Z,'n_eval':len(rows),
  'mean_energy_improvement':{m:float(arr[m].mean()) for m in methods},
  'mean_gain_learned_minus_signed':float(delta.mean()),
  'mean_gain_learned_minus_random':float(dr.mean()),
  'seed_cluster_bootstrap_95pct_mean_diff_learned_minus_signed':list(map(float,np.quantile(boot,[.025,.975]))),
  'wins_ties_losses_vs_signed':[int(np.sum(delta>1e-11)),int(np.sum(abs(delta)<=1e-11)),int(np.sum(delta<-1e-11))],
  'all_positive_tested':{m:int(sum(r['methods'][m]['positive_tested'] for r in d['per_eval'])) for m in methods},
  'per_kappa':{str(k):{m:float(np.mean([r[m] for r in rows if r['kappa']==k])) for m in methods} for k in sorted({r['kappa'] for r in rows})},
  'full_hamiltonian_verification':checks,
  'train_total_guide_queries':d['total_train_guide_queries'],
  'train_extra_verifier_queries':d['total_train_verifier_queries'],
  'mean_common_test_queries':float(np.mean([r['common_queries'] for r in d['per_eval']])),
  'mean_test_total_queries':{m:float(np.mean([r['methods'][m]['total_guide_queries'] for r in d['per_eval']])) for m in methods},
  'caveats':['Z uses full Sz=0 enumeration post-hoc solely to normalize energy; not used for candidate selection/training.',
    'Seed-level bootstrap accounts for three correlated kappa tests per seed, not independent cross-model replication.',
    'No demonstrated favorable amortized training cost or extrapolation beyond 4x4.']}
 Path(out).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('input');parser.add_argument('output');parser.add_argument('--checks',type=int,default=3)
 a=parser.parse_args();audit(a.input,a.output,a.checks)
