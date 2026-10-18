#!/usr/bin/env python3
"""Extra holdout tests using the already trained 4x4 policy; no retraining."""
from pathlib import Path
import numpy as np,torch,json,argparse
from avsl_learned_policy4 import Policy,evaluate_model

def run(checkpoint,out,n_seeds=40,n_test_seeds=12):
 torch.set_num_threads(2)
 d=torch.load(checkpoint,map_location='cpu',weights_only=False)
 net=Policy(len(d['mean']));net.load_state_dict(d['state']);net.eval()
 res=[]
 for B in (8,12,18):
  for k in (.7,.8,.9,1.0,1.2):
   for seed in range(3100,3100+n_test_seeds):
    r=evaluate_model(seed,k,net,d['mean'],d['std'],B=B,n_seeds=n_seeds)
    if r is not None:res.append(dict(B=B,**r))
 summary={}
 for B in (8,12,18):
  summary[str(B)]={}
  for k in (.7,.8,.9,1.0,1.2):
   sub=[r for r in res if r['B']==B and r['kappa']==k]
   if not sub: continue
   summary[str(B)][str(k)]={m:{'avg_positive_tested':float(np.mean([r['methods'][m]['positive_tested'] for r in sub])),
      'mean_gain_numerator':float(np.mean([r['methods'][m]['gain_numerator'] for r in sub])),
      'mean_total_queries':float(np.mean([r['methods'][m]['total_guide_queries'] for r in sub]))}
      for m in ['learned','signed','random']}
   summary[str(B)][str(k)]['learned_vs_signed_wins_ties_losses']=[
    int(sum(r['methods']['learned']['gain_numerator']>r['methods']['signed']['gain_numerator']+1e-9 for r in sub)),
    int(sum(abs(r['methods']['learned']['gain_numerator']-r['methods']['signed']['gain_numerator'])<=1e-9 for r in sub)),
    int(sum(r['methods']['learned']['gain_numerator']+1e-9<r['methods']['signed']['gain_numerator'] for r in sub))]
 result={'setup':{'checkpoint':checkpoint,'n_seeds':n_seeds,'n_test_seeds':n_test_seeds,'B':[8,12,18],'kappas':[.7,.8,.9,1.0,1.2]},'summary':summary,'per_eval':res}
 Path(out).write_text(json.dumps(result,indent=2));print('B/kappa, learned positives, signed positives, wins-ties-losses')
 for b,d in summary.items():
  for k,v in d.items():print(b,k,round(v['learned']['avg_positive_tested'],2),round(v['signed']['avg_positive_tested'],2),v['learned_vs_signed_wins_ties_losses'])
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',default='/mnt/data/avsl_learned_policy4_seed42.pt');ap.add_argument('--out',default='/mnt/data/avsl_learned_policy4_budgetcheck.json');ap.add_argument('--n-seeds',type=int,default=40);ap.add_argument('--test-runs',type=int,default=12)
 a=ap.parse_args();run(a.checkpoint,a.out,a.n_seeds,a.test_runs)
