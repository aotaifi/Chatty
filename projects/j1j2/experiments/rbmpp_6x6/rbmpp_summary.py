"""Collect the RBM+PP guide numbers and loop trajectories into results/rbmpp_6x6/summary.json (run on ws1).
  python rbmpp_summary.py OUT.json RUNS_JSON
RUNS_JSON maps guide tag -> dict(train=run dir, table=table json, loop=wt_loop run dir, chop=[cx_chop2 run dirs],
ws=rbmpp_ws json). Missing entries are skipped.
"""
import json, os, sys
import numpy as np
E0 = -0.50380965
cfg = json.load(open(sys.argv[2]))
out = {}
for tag, c in cfg.items():
    o = {}
    if c.get('train'):
        r = json.load(open(os.path.join(c['train'], 'train.json')))
        h = r['hist']
        E = np.array([x['E'] for x in h]); V = np.array([x['var'] for x in h])
        w = [x['wall'] for x in h]
        o['train'] = dict(steps=len(h), wall_h=r.get('train_sec', w[-1] if w else 0) / 3600, eval=r.get('eval'),
                          last500=float(E[-500:].mean()) if len(E) >= 500 else None,
                          prev500=float(E[-1000:-500].mean()) if len(E) >= 1000 else None,
                          last500_se=float(E[-500:].std() / np.sqrt(500)) if len(E) >= 500 else None,
                          var_last500=float(V[-500:].mean()) if len(V) >= 500 else None,
                          var_prev500=float(V[-1000:-500].mean()) if len(V) >= 1000 else None)
    if c.get('table'):
        o['table'] = json.load(open(c['table']))
    if c.get('loop'):
        r = json.load(open(os.path.join(c['loop'], 'wt_loop.json')))
        o['loop'] = dict(guide_H=r.get('guide_H'), guide_lanczos=r.get('guide_lanczos'),
                         iters=[dict(it=x['it'], H=x['H_kry'], E_FN=x.get('E_FN_next'), frac=x['frac'],
                                     L1=x.get('lanczos_H'), L2=x.get('lanczos_H_p2'), L1_E_FN=x.get('lanczos_E_FN'),
                                     sec=x['sec']) for x in r['iters']], sec=r.get('sec'))
    if c.get('chop'):
        o['chop'] = []
        for d in c['chop']:
            r = json.load(open(os.path.join(d, 'cx_chop2.json')))
            o['chop'].append(dict(run=os.path.basename(d), refs=r['refs'], sec=r.get('sec'),
                                  iters=[dict(it=x['it'], H=x['H'], H_w=x['H_w'], E_FN=x['E_FN'], frac=x['frac'],
                                              L1=x['lanczos_p1'], L2=x['lanczos_p2'], ret=x['retention_step'],
                                              nets=x['n_small_nets']) for x in r['iters']]))
    if c.get('ws'):
        o['ws'] = json.load(open(c['ws']))
    out[tag] = o
json.dump(out, open(sys.argv[1], 'w'), indent=1, default=float)
print(json.dumps({k: list(v) for k, v in out.items()}))
