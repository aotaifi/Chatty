#!/usr/bin/env python3
"""Local log-amplitude accuracy needed to certify orbit corrections at fixed a.
At current signs, for candidate orbit g let P=sum_{h:w_gh>0}w_gh and N=sum_{h:w_gh<0}-w_gh, with w_gh=sum_{x in g,y in h}H_xy s_x s_y a_x a_y.
If neighbor log-amplitude error <= eps, a sufficient sign-safe flip criterion is eps < 0.5*log(P/N).  Across successive flips, w_gh changes sign if h already selected. The criterion only needs relative neighbor amplitude information; the multiplicative error at g cancels.
"""
import json, numpy as np
import j1j2_budgeted_orbit20 as B

def one(seed,budget=300,max_flips=40):
 s,pool,amps,proxies,info=B.generate(seed)
 eligible=sorted(pool,key=lambda g:(-proxies['observed_signed'][g],g))[:budget]
 scored={g:B.exact_score_one(s,g)[1] for g in eligible}
 chosen=set();tr=[]
 for t in range(max_flips):
  best=None;gain=-float('inf');pn=None
  for g in eligible:
   if g in chosen:continue
   vals=[(-v if h in chosen else v) for h,v in scored[g].items()];P=sum(v for v in vals if v>0);N=-sum(v for v in vals if v<0);z=4*(P-N)
   if z>gain:best=g;gain=z;pn=(P,N)
  if best is None or gain<=0:break
  P,N=pn
  eps=float(0.5*np.log(P/N)) if N>0 else float('inf')
  tr.append(dict(i=t,orbit=int(best),gain=float(gain),P=float(P),N=float(N),eps_max=eps,relative_margin=float((P-N)/(P+N)) if P+N>0 else float('nan')))
  chosen.add(best)
 return dict(seed=seed,n_selected=len(chosen),trace=tr,n_candidate_orbits=len(pool),n_queries=len(s.amps))
if __name__=='__main__':
 rows=[one(seed) for seed in range(201,213)]
 margins=[t['eps_max'] for r in rows for t in r['trace']]; m=np.array(margins)
 print('accepted',len(m),'by seed',[(r['seed'],r['n_selected']) for r in rows]);print('epsilon_max quantiles',np.quantile(m,[0,.1,.25,.5,.75,.9,1]));print('gain positive counts if eps:',[(e,sum(x>e for x in m)) for e in [0.0001,.001,.003,.01,.03,.05,.1,.2,.3]]);print('relative margin median',np.median([t['relative_margin'] for r in rows for t in r['trace']]))
 path='/mnt/data/j1j2_margin_precision20_results.json'
 with open(path,'w') as f:json.dump({'per_seed':rows,'summary':{'n_selected':len(m),'eps_q':{str(k):float(v) for k,v in zip(['min','p10','p25','median','p75','p90','max'],np.quantile(m,[0,.1,.25,.5,.75,.9,1]))},'n_with_eps_gt':{str(e):int(np.sum(m>e)) for e in [0.0001,.001,.003,.01,.03,.05,.1,.2,.3]}},'method':'fixed exact amplitudes, visited local-query candidate pool; selection uses observed signed bond proxy, then exact current marginal energy, max 40 orbit flips'},f,indent=2)