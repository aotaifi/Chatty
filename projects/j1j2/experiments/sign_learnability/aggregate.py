#!/usr/bin/env python3
"""Collect evals/N*/*.json (sl_eval.py) into results/sign_learnability/summary.json (+ evals tarball by the caller).
Usage: python aggregate.py EVALS_DIR DATA_META_DIR OUT_JSON
summary = dict(meta={N: exact chain data}, runs=[scalar record per run], hists={name: margin/decade/logit hists}
for the reference runs (beta 0.5, seed 0, nbr 1, lam 0: default protocol n=1e4 size M; 8e4-step protocol n=3e4 sizes M/L/XL)})."""
import json, glob, os, sys

ev, metad, out = sys.argv[1:4]
SCAL = ['name', 'N', 'D', 'target', 'k', 'nparams', 'n_train', 'nsteps', 't_train', 't_eval', 't_hop', 'device', 'final_bce',
        'w_triv', 'w_lab', 'w_net', 'w_hop', 'w_hop2', 'eps_net', 'eps_hop', 'eps_hop2', 'p0_seen', 'frac_orbits_seen',
        'w_lab_seen', 'w_lab_unseen', 'err_train_unweighted', 'frac_orbits_wrong', 'n_wrong_orbits',
        'w_ref_k', 'w_ref_k1', 'w_ref_k2', 'd_hop', 'hop_repaired', 'hop_residual', 'hop_new', 'd_hop2']
runs, hists = [], {}
for f in sorted(glob.glob(f'{ev}/N*/*.json')):
    d = json.load(open(f))
    r = {k: d.get(k) for k in SCAL}
    A = d['args']
    for k in ('beta', 'seed', 'n', 'nbr', 'size', 'lam', 'epochs', 'min_steps', 'max_steps', 'kern', 'init_seed', 'tag'):
        r[k] = A.get(k)
    m = d['margin']; lg = d['logit']
    r.update(m_q_all=m['q_all'], m_q_wrong=m['q_wrong'], m_frac_wrong_below=m['frac_wrong_below'],
             m_frac_all_below=m['frac_all_below'], m_frac_flip_below=m['frac_flip_below'],
             lg_q_abs_all=lg['q_abs_all'], lg_q_abs_wrong=lg['q_abs_wrong'], lg_frac_wrong_below1=lg['frac_wrong_abs_below1'],
             lg_frac_all_below1=lg['frac_all_abs_below1'], lg_corr_delta=lg['corr_logit_delta'])
    runs.append(r)
    default = A['epochs'] == 10 and A['max_steps'] == 40000 and A['n'] == 10000
    conv = A['max_steps'] == 80000 and A.get('min_steps') == 80000 and A['n'] == 30000
    if A['beta'] == 0.5 and A['seed'] == 0 and A['nbr'] == 1 and A['lam'] == 0 and A['size'] in ('M', 'L', 'XL') and (default or conv):
        hists[d['name'] + f"@N{d['N']}"] = dict(decades=d['decades'], margin=m, logit=lg)
meta = {}
for f in sorted(glob.glob(f'{metad}/N*/meta.json')):
    m = json.load(open(f)); meta[m['N']] = m
json.dump(dict(meta=meta, runs=runs, hists=hists), open(out, 'w'))
print(len(runs), 'runs;', len(hists), 'hist sets ->', out, os.path.getsize(out) / 1e6, 'MB')
