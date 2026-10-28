#!/usr/bin/env python3
"""Train-only negative-score quantile abstention sweep, with exact FN audit.
No heldout K1 oracle labels or FN energy used to choose these thresholds."""
import argparse,json,time
import numpy as np
from pathlib import Path
from avsl10_compact_local_rule20 import feature_matrix,apply_rule_to_all
from avsl09_k1_fn_loop20 import fn_exact_reference

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--data',required=True);ap.add_argument('--rule',required=True)
 ap.add_argument('--out',required=True)
 args=ap.parse_args()
 d=json.load(open(args.rule))
 rule={k:np.asarray(v,dtype=float) if k in ('coefs','mean','std') else v for k,v in d['learned_rule'].items()}
 keys=np.array(d['train_keys'],dtype=np.uint32)
 yy=np.array(d['train_labels'],dtype=int)
 dd=np.load(args.data)
 lookup=np.full(1<<20,-1,dtype=np.int32)
 lookup[dd['states']]=np.arange(len(dd['states']))
 amplitudes=dd['amp'][lookup[keys]] if len(rule['coefs'])==43 else None
 train=(feature_matrix(keys,amplitudes=amplitudes)-rule['mean'])/rule['std']
 train_scores=train@rule['coefs']+rule['bias']
 ns=fn_exact_reference(args.data)
 baseline=float(d['initial_FN_energy'])
 result=dict(model=args.rule,cut_strategy='negative K1 training-label quantiles only',cuts=[])
 for q in (.995,.999,.9995,.9999,1.):
  cut=max(float(np.quantile(train_scores[yy==0],q)),float(np.quantile(train_scores[yy==1],.25)))
  pred_train=train_scores>cut
  physics,_=apply_rule_to_all(args.data,rule,ns,baseline,threshold_override=cut)
  item=dict(negative_quantile=q,cut=cut,train_precision=float(np.sum(pred_train*(yy==1))/max(1,np.sum(pred_train))),
            train_recall=float(np.sum(pred_train*(yy==1))/max(1,np.sum(yy))),n_train_selected=int(pred_train.sum()),**physics)
  result['cuts'].append(item)
  print('CUT',q,'train precision',item['train_precision'],'FN dE',item['post_FN_energy_gain'],'flipped orbits',item['n_flip_symmetry_orbits'],flush=True)
 Path(args.out).write_text(json.dumps(result,indent=2))
 print('SAVED',args.out)
if __name__=='__main__':main()
