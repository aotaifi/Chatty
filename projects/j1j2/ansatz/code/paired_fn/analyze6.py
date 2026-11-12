#!/usr/bin/env python3
"""analyze6.py (paired_fn): combine the ensembles of each 6x6 pair in a run directory (run6.py outputs) and report against the exact Delta.
Per pair: per-ensemble delta +- sigma (binning), combined (inverse variance; error = max(weighted, scatter)), z = (est - exact)/sigma,
sigma / (|Delta|/3), quotable.py verdict on the ensembles' E_D.npy, long-run rho and kappa_work (standalone sa_<guide> runs if present).
usage: analyze6.py RUNDIR [--sa SA_RUNDIR] [--prefix P]   -> RUNDIR/analysis6.json and a table on stdout"""
import argparse, glob, json, os, sys
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import analysis as AN, guides6 as G
ap = argparse.ArgumentParser(); ap.add_argument('run'); ap.add_argument('--sa', default=None); ap.add_argument('--prefix', default='')
a = ap.parse_args(); SA = a.sa or a.run; NS = 36
groups = {}
for d in sorted(glob.glob(f'{a.run}/{a.prefix}*_e*/summary.json')):
    tag = os.path.basename(os.path.dirname(d)).rsplit('_e', 1)[0]; groups.setdefault(tag, []).append(os.path.dirname(d))
def sa_series(name):
    p = f'{SA}/sa_{ {"fni_parent": "T0B0_c47"}.get(name, name)}'
    if not os.path.exists(f'{p}/E_A.npy'): return None, None
    s = json.load(open(f'{p}/summary.json')); return np.load(f'{p}/E_A.npy') * NS, s['sec_per_block']
out = {}
for tag, dirs in groups.items():
    S = [json.load(open(f'{d}/summary.json')) for d in dirs]; pair = S[0]['args']['pair']; ex = S[0]['delta_exact_site']
    m = [s['pair']['delta_site'] for s in S]; e = [s['pair']['sigma_site'] for s in S]; c = AN.combine(m, e)
    q = AN.quotable([f'{d}/E_D.npy' for d in dirs], f'Delta/site {tag}', plot=f'{a.run}/{tag}_E_D.png')
    rec = dict(pair=pair, exact=ex, n_ens=len(S), ens=[dict(dir=d, delta=s['pair']['delta_site'], sigma=s['pair']['sigma_site'], z=s['z'], rho=s['pair']['rho'],
               blocks=s['blocks'], start=s['args']['start'], EA_minus_exact=s['EA_minus_exact_site'], EB_minus_exact=s.get('EB_minus_exact_site'),
               same_post=s.get('same_post'), alpha=s.get('alpha'), sep_rate=s.get('sep_rate'), cut_rate=s.get('cut_rate'), F_weight_A=s.get('F_weight_A'),
               F_ess_frac_A=s.get('F_ess_frac_A'), sec_per_block=s['sec_per_block']) for d, s in zip(dirs, S)],
               combined=c, z=(c['mean'] - ex) / c['err'], sigma_over_third=c['err'] / (abs(ex) / 3), quotable=q['verdict'], quotable_exit=q['exit'],
               rho_mean=float(np.mean([s['pair']['rho'] for s in S])))
    A, B = G.PAIRS[pair]; sA, cA = sa_series(A); sB, cB = sa_series(B)
    if sA is not None and sB is not None:
        kw = []
        for d, s in zip(dirs, S):
            eA = np.load(f'{d}/E_A.npy') * NS; eB = np.load(f'{d}/E_B.npy') * NS
            kw.append(AN.pair_stats(eA, eB, s['sec_per_block'], sA, sB, cA, cB, NS=NS))
        rec['kappa_work'] = [k['kappa_work'] for k in kw]; rec['kappa_work_opt'] = [k['kappa_work_opt'] for k in kw]; rec['r_cost'] = [k['r_cost'] for k in kw]
        rec['S_A_sa'] = kw[0]['S_A_sa']; rec['S_B_sa'] = kw[0]['S_B_sa']
    out[tag] = rec
    print(f'{tag:10s} exact {ex:+.3e} est {c["mean"]:+.3e} +- {c["err"]:.2e} z {rec["z"]:+.2f} sigma/(|D|/3) {rec["sigma_over_third"]:.2f} rho {rec["rho_mean"]:.3f} '
          f'kappa {np.mean(rec.get("kappa_work", [np.nan])):.3g} | {q["verdict"][:90]}')
json.dump(out, open(f'{a.run}/analysis6.json', 'w'), indent=1, default=float)
