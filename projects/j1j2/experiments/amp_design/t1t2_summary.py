#!/usr/bin/env python3
"""Summary table + figure for T1/T2 (4x4 ViT, net H start). Reads results/amp_design/vit/*.json.
Usage: python t1t2_summary.py   (writes results/amp_design/vit_t1t2.{png,pdf} and vit_t1t2_table.md)"""
import json, glob
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / 'results/amp_design/vit'
OUT = ROOT / 'results/amp_design'
C = dict(loop='#2a78d6', cplx='#eb6834', net='#1baf7a', sr='#eda100', ref='#52514e')


def load(pat):
    return [json.load(open(f)) for f in sorted(glob.glob(str(R / pat)))]


def curve(runs, key):
    """mean over seeds, aligned by record index; cpu_h averaged."""
    n = min(len(r['history']) for r in runs)
    cpu = np.mean([[h['cpu_h'] for h in r['history'][:n]] for r in runs], 0)
    y = np.array([[h.get(key, np.nan) for h in r['history'][:n]] for r in runs])
    return cpu, y.mean(0), y.min(0), y.max(0)


def srvmc():
    f = R / 'T2_srvmc_ckpt.json'
    if not f.exists(): return None
    d = json.load(open(f)); h = d['history']
    return np.array([x['cpu_h'] for x in h]), np.array([x['eps'] for x in h]), np.array([x['eps_guide'] for x in h]), d


def interp_at(cpu, y, c):
    if c < cpu[0] or c > cpu[-1]: return np.nan
    return float(np.exp(np.interp(c, cpu, np.log(y))))


start = load('T1_k1_rgntr_N10000_s0.json')[0]['history'][0]
eps0_guide, eps0_fn = start['eps_guide'], start['eps_FN']
eps0_psi = 3.629191390127996e-04          # net H, complex psi (training_summary.json)

# ---------------- table ----------------
rows = []
def add(name, runs, keys):
    if not runs: return
    h = [r['history'][-1] for r in runs]
    cpu = np.mean([x['cpu_h'] for x in h])
    vals = {k: (np.mean([x[k] for x in h]), np.min([x[k] for x in h]), np.max([x[k] for x in h])) for k in keys if k in h[0]}
    rows.append((name, len(runs), cpu, vals))

for opt in ('rgntr', 'hybrid', 'sr', 'c1'):
    for N in (10000, 100000):
        add(f'T1 Krylov signs, {opt}, N={N:g}, 6 steps', load(f'T1_k1_{opt}_N{N}_s*.json'), ['eps_guide', 'eps_FN', 'w_s'])
for opt in ('rgntr', 'hybrid'):
    for N in (10000, 100000):
        add(f'T1 net signs, {opt}, N={N:g}, 6 steps', load(f'T1_k0_{opt}_N{N}_s*.json'), ['eps_guide', 'eps_FN', 'w_s'])
add('T2 loop (Krylov sign + 3 RGN-TR)/iter, 10 iters', load('T2_loop_spi3_s*.json'), ['eps_guide', 'eps_FN', 'w_s'])
add('T2 loop (Krylov sign + 1 RGN-TR)/iter, 30 iters', load('T2_loop_spi1_s*.json'), ['eps_guide', 'eps_FN', 'w_s'])
add('T2 net signs fixed, RGN-TR, 30 steps', load('T2_netsign_s*.json'), ['eps_guide', 'eps_FN', 'w_s'])
add('T2 complex ViT (sign+amp), RGN-TR', load('T2_complex_s*.json'), ['eps_psi', 'eps_guide', 'eps_FN', 'w_s'])
add('T2 loop, adaptive lam/N (Krylov sign + 3 RGN-TR)/iter', load('T2_loopA_spi3_s*.json'), ['eps_guide', 'eps_FN', 'w_s'])
add('T2 net signs fixed, RGN-TR adaptive', load('T2_netsignA_s*.json'), ['eps_guide', 'eps_FN', 'w_s'])
add('T2 complex ViT (sign+amp), RGN-TR adaptive', load('T2_complexA_s*.json'), ['eps_psi', 'eps_guide', 'eps_FN', 'w_s'])

lines = ['| arm | seeds | CPU-h | eps<H> (guide or psi) | reduction vs net | eps_FN | reduction vs net FN | w_s |', '|---|---|---|---|---|---|---|---|']
for name, n, cpu, v in rows:
    ev = v.get('eps_psi', v['eps_guide'])
    e0 = eps0_psi if 'eps_psi' in v else eps0_guide
    if ev[0] > 10 * e0:
        lines.append(f"| {name} | {n} | {cpu:.2f} | diverged ({ev[0]:.1e}) | - | {v['eps_FN'][0]:.1e} | - | - |"); continue
    lines.append(f"| {name} | {n} | {cpu:.2f} | {ev[0]:.2e} [{ev[1]:.1e},{ev[2]:.1e}] | {100*(1-ev[0]/e0):.0f}% | "
                 f"{v['eps_FN'][0]:.2e} | {100*(1-v['eps_FN'][0]/eps0_fn):.0f}% | {v['w_s'][0]:.1e} |")
s = srvmc()
if s is not None:
    cpu, e, eg, d = s
    sc = load('T2_srvmc_score.json')
    efn = sc[0]['history'][0]['eps_FN'] if sc else float('nan')
    lines.append(f"| T2 standard VMC (SR, complex ViT, N=4000, lr 0.05, 600 steps) | 1 | {cpu[-1]:.2f} | {e[-1]:.2e} | {100*(1-e[-1]/eps0_psi):.0f}% | "
                 f"{efn:.2e} | {100*(1-efn/eps0_fn):.0f}% | {d['history'][-1]['w_s']:.1e} |")
old = json.load(open(ROOT / 'results/net_start_4x4/vmccont_H_N4000_lr05.json'))
oc = np.array(old['cpu']) - old['cpu'][0]; oe = np.array(old['eps_net'])
for c in (5, 20, 50):
    v = interp_at(oc[1:], oe[1:], c)
    lines.append(f"| reference: same SR VMC, long run on 16-core `cluster` nodes, at {c} CPU-h | 1 | {c} | {v:.2e} | {100*(1-v/eps0_psi):.0f}% | | | |")

# equal-CPU-h comparison
lines += ['', '**At equal CPU-h** (seed mean, log-interpolated; variational eps: loop/net-sign = <H> of the guide, VMC = <H> of psi)', '',
          '| CPU-h | loop (adaptive, spi3) | loop spi1 | net signs RGN-TR | complex RGN-TR (all runs) | SR VMC |', '|---|---|---|---|---|---|']
arms = {'loop spi3': ('T2_loopA_spi3_s*.json', 'eps_guide'), 'loop spi1': ('T2_loop_spi1_s*.json', 'eps_guide'),
        'net': ('T2_netsignA_s*.json', 'eps_guide'), 'cplx': ('T2_complex*_s*.json', 'eps_psi')}
def eq_cost(pat, key, c):
    vals = []
    for r in load(pat):
        h = r['history']; v = interp_at(np.array([x['cpu_h'] for x in h]), np.array([x[key] for x in h]), c)
        if not np.isnan(v): vals.append(v)
    return (float(np.exp(np.mean(np.log(vals)))), len(vals)) if vals else (np.nan, 0)

for c in (0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0):
    cells = []
    for k in ('loop spi3', 'loop spi1', 'net', 'cplx'):
        e0k = eps0_psi if k == 'cplx' else eps0_guide
        v, n = eq_cost(arms[k][0], arms[k][1], c)
        cells.append('-' if np.isnan(v) else f'{v:.2e} ({100*(1-v/e0k):.0f}%, n={n})')
    if s is not None:
        v = interp_at(s[0], s[1], c); cells.append('-' if np.isnan(v) else f'{v:.2e} ({100*(1-v/eps0_psi):.0f}%)')
    lines.append(f'| {c:g} | ' + ' | '.join(cells) + ' |')
(OUT / 'vit_t1t2_table.md').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))

# ---------------- figure ----------------
plt.rcParams.update({'font.size': 9, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
                     'axes.edgecolor': '#52514e', 'axes.labelcolor': '#0b0b0b', 'xtick.color': '#52514e', 'ytick.color': '#52514e'})
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.3), constrained_layout=True)
mk = dict(lw=0.9, ms=3)

# (a) T1: eps_FN vs optimiser step at Krylov signs (N=1e4)
a = ax[0]
for opt, col, lab in (('rgntr', C['loop'], 'RGN, trust region'), ('hybrid', C['net'], 'edge-preconditioned gradient'),
                      ('sr', C['sr'], 'SR'), ('c1', C['cplx'], 'FN surrogate (C1)')):
    runs = load(f'T1_k1_{opt}_N10000_s*.json')
    if not runs: continue
    n = min(len(r['history']) for r in runs)
    y = np.array([[h['eps_FN'] for h in r['history'][:n]] for r in runs])
    a.plot(np.arange(n), y.mean(0), 'o-', color=col, label=lab, **mk)
runs = load('T1_k0_rgntr_N10000_s*.json')
if runs:
    y = np.array([h['eps_FN'] for h in runs[0]['history']])
    a.plot(np.arange(len(y)), y, 'o--', color=C['loop'], mfc='white', label='RGN, net signs', **mk)
a.set_ylim(2.5e-5, 6e-4)
a.annotate('FN surrogate diverges\n(no trust region)', xy=(2.6, 5.6e-4), xytext=(3.1, 3.9e-4), fontsize=7, color=C['cplx'],
           arrowprops=dict(arrowstyle='->', color=C['cplx'], lw=0.6))
a.set_yscale('log'); a.set_xlabel('optimiser step (N = 10$^4$ samples)'); a.set_ylabel(r'$\epsilon$ of $E_{\rm FN}$ of the guide')
a.set_title('T1: one amplitude refresh', fontsize=9, loc='left'); a.legend(frameon=False, fontsize=7.5)

# (b), (c) T2 vs CPU-h
for panel, key_loop, key_cplx, ylab in ((ax[1], 'eps_guide', 'eps_psi', r'$\epsilon$ of $\langle H\rangle$'),
                                         (ax[2], 'eps_FN', 'eps_FN', r'$\epsilon$ of $E_{\rm FN}$ of the guide')):
    for k, pat, key, col, lab in (('loop', 'T2_loopA_spi3_s*.json', key_loop, C['loop'], 'loop: Krylov sign + RGN'),
                                  ('net', 'T2_netsignA_s*.json', key_loop, C['net'], 'RGN, net signs fixed'),
                                  ('cplx', 'T2_complexA_s*.json', key_cplx, C['cplx'], 'RGN, complex ViT (VMC)')):
        runs = load(pat)
        if not runs: continue
        if k == 'cplx':      # different lengths: plot each run, label once
            for i, r in enumerate(runs):
                cpu = np.maximum([h['cpu_h'] for h in r['history']], 1e-2); y = [h[key] for h in r['history']]
                panel.plot(cpu, y, 'o-', color=col, label=lab if i == 0 else None, **mk)
            continue
        cpu, m, lo, hi = curve(runs, key)
        cpu = np.maximum(cpu, 1e-2)
        panel.plot(cpu, m, 'o-', color=col, label=lab, **mk)
        if len(runs) > 1: panel.fill_between(cpu, lo, hi, color=col, alpha=0.15, lw=0)
    if s is not None and panel is ax[1]:
        cpu, e, eg, d = s
        sel = cpu > 0
        panel.plot(cpu[sel][::60], e[sel][::60], 'o-', color=C['sr'], label='SR, complex ViT (standard VMC)', **mk)
        panel.plot(oc[1::10], oe[1::10], color=C['sr'], lw=0.8, ls='--', label='SR VMC, long run (16-core nodes)')
    if s is not None and panel is ax[2]:
        sc = load('T2_srvmc_score.json')
        if sc: panel.plot([s[0][-1]], [sc[0]['history'][0]['eps_FN']], 'o', color=C['sr'], ms=4, label='SR VMC (end)')
    e0 = eps0_guide if panel is ax[1] else eps0_fn
    panel.axhline(e0, color=C['ref'], lw=0.6, ls=':'); panel.text(0.012, e0 * 1.06, 'trained net', fontsize=7, color=C['ref'])
    panel.set_xscale('log'); panel.set_yscale('log'); panel.set_xlabel('CPU-h after training (12-core node)'); panel.set_ylabel(ylab)
ax[1].set_title('T2: loop vs VMC at equal cost', fontsize=9, loc='left'); ax[1].legend(frameon=False, fontsize=6.8, loc='lower left', handlelength=1.6, labelspacing=0.3)
ax[2].set_title('T2: fixed-node energy', fontsize=9, loc='left')
for i, a in enumerate(ax):
    a.text(-0.16, 1.04, '(%s)' % 'abc'[i], transform=a.transAxes, fontsize=10, fontweight='bold')
    a.grid(True, which='major', lw=0.3, color='#d9d8d4'); a.set_axisbelow(True)
    for sp_ in ('top', 'right'): a.spines[sp_].set_visible(False)
fig.savefig(OUT / 'vit_t1t2.png', dpi=200); fig.savefig(OUT / 'vit_t1t2.pdf')
print('figure written')
