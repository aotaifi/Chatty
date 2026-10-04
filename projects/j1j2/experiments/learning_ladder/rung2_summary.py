#!/usr/bin/env python3
"""Summarise rung-2 JSONs (rung2_sampled_sr_loop_4x4.py) next to the exact 4x4 loops.

Usage: rung2_summary.py <results_dir> [--its 5 20] [--glob 'prod_*.json']
Groups runs by (mode, budget); reports mean and min..max over seeds of w_s, eps, E_FN at the
requested loop iterations and at the plateau (median of the last --tail iterations per run).
"""
import argparse, glob, json, os
import numpy as np

KS = '/Users/aliotaifi/Chatty-organize/projects/j1j2/krylov_sign_structure/results'


def exact_refs():
    out = {}
    d = json.load(open(f'{KS}/anderson_loop/full_4x4_plain.json'))
    out['exact plain a<-phi_FN'] = {r['it']: (r['w_s'], r['eps'], r.get('E_FN')) for r in d['history']}
    g = json.load(open(f'{KS}/closed_fn_krylov_4x4_J2p5_J2zero_init_halfstep.json'))
    E0 = g['E0']
    out['exact geometric mean'] = {r['it']: ((1 - r['O_sign']) / 2, r['E_error'] / abs(E0), None)
                                   for r in g['history'] if r}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dir')
    ap.add_argument('--glob', default='prod_*.json')
    ap.add_argument('--its', type=int, nargs='+', default=[5, 20])
    ap.add_argument('--tail', type=int, default=5)
    args = ap.parse_args()
    groups = {}
    for f in sorted(glob.glob(os.path.join(args.dir, args.glob))):
        d = json.load(open(f))
        m = d['meta']['args']
        pref = os.path.basename(f).split('_b')[0]
        groups.setdefault((pref, m['budget']), []).append((f, d))
    fmt = lambda v: f'{np.mean(v):.2e} [{np.min(v):.1e},{np.max(v):.1e}]'
    print('| run | n_seeds | ' + ' | '.join(f'w_s @{k} | eps @{k}' for k in args.its) +
          ' | w_s plateau | eps plateau | E_FN plateau | iters |')
    print('|' + '---|' * (2 + 2 * len(args.its) + 4))
    for name, h in exact_refs().items():
        cells = []
        for k in args.its:
            cells += [f'{h[k][0]:.2e}', f'{h[k][1]:.2e}']
        print(f'| {name} | 1 | ' + ' | '.join(cells) + ' | (converges to 0) | (->0) | | |')
    for (mode, budget), runs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        cells = []
        for k in args.its:
            ws = [r[1]['history'][k]['w_s'] for r in runs if len(r[1]['history']) > k]
            ep = [r[1]['history'][k]['eps'] for r in runs if len(r[1]['history']) > k]
            cells += [fmt(ws) if ws else '-', fmt(ep) if ep else '-']
        pw = [np.median([x['w_s'] for x in r[1]['history'][-args.tail:]]) for r in runs]
        pe = [np.median([x['eps'] for x in r[1]['history'][-args.tail:]]) for r in runs]
        pf = [np.median([x['eps_FN'] for x in r[1]['history'][-args.tail:] if 'eps_FN' in x]) for r in runs]
        nit = [len(r[1]['history']) - 1 for r in runs]
        sm = [np.mean([x.get('samples_used', budget) for x in r[1]['history'][1:]]) for r in runs]
        print(f'| {mode} N={budget} | {len(runs)} | ' + ' | '.join(cells) +
              f' | {fmt(pw)} | {fmt(pe)} | eps_FN {fmt(pf)} | {min(nit)}-{max(nit)} (samples/loop-it ~{np.mean(sm):.1e}) |')


if __name__ == '__main__':
    main()
