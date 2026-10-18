#!/usr/bin/env python3
"""Distribution shift: re-score candidates after initial learned orbit corrections.

Reuses the *original* trained candidate model, then replaces the Marshall signs
by signs with the earlier selected symmetry orbits flipped. No new network
training or FN relaxation. Tests whether exploration policy follows the current
sign chamber or just the analytic 4x4 orbit geometry. Existing ground-truth
labels are not leaked to model training: prior accepted flips came from the
original model's already costed verifier.
"""
import json,torch,numpy as np
from pathlib import Path
import avsl_probe4 as p
import avsl_learned_policy4 as L

def main():
 d=json.loads(Path('/mnt/data/avsl_learned_policy4_seed42.json').read_text())
 checkpoint=torch.load('/mnt/data/avsl_learned_policy4_seed42.pt',map_location='cpu',weights_only=False)
 model=L.Policy(len(checkpoint['mean']));model.load_state_dict(checkpoint['state']);model.eval()
 baseclass=p.Oracle
 class Changed(baseclass):
  flips=set()
  def evaluate(self,x):
   amp,sgn=super().evaluate(x)
   return (amp,-sgn if p.key(x) in Changed.flips else sgn)
 p.Oracle=Changed
 out=[]
 for r in d['per_eval']:
  f=r['methods']['learned']['selected_orbit_keys'];Changed.flips=set(f)
  z=L.evaluate_model(r['seed'],r['kappa'],model,checkpoint['mean'],checkpoint['std'],B=22,n_seeds=160,mode='new_seeds')
  z['previously_flipped_orbits']=list(f)
  out.append(z)
 summary={}
 for m in ['learned','signed','random','max_violation']:
  vals=[r['methods'][m] for r in out]
  summary[m]={'mean_positive_tested':float(np.mean([v['positive_tested'] for v in vals])),
    'mean_gain_numerator':float(np.mean([v['gain_numerator'] for v in vals])),
    'mean_queries':float(np.mean([v['total_guide_queries'] for v in vals])),
    'mean_selected_orbits':float(np.mean([v['selected_orbits'] for v in vals]))}
 print('ROUND 2 SUMMARY',json.dumps(summary,indent=2))
 result={'scope':'4x4 frozen amplitude, modified signs from first learned correction; initial network frozen; no FN refresh','summary':summary,'per_eval':out,
  'limitations':['Not a full self-consistent FN–Krylov loop','Sign chambers built from same models first accepted orbit flips; geometry shifts only through current signs','The reward is exact with respect to frozen analytic guide but not exact ground state']}
 Path('/mnt/data/avsl_policy4_second_round_all_results.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()
