#!/usr/bin/env python3
"""AMP6 gate figure + table (6x6, large-N fixed-sign amplitude steps). Reads results/amp_design/amp6/*.json."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / 'results/amp_design/amp6'; OUT = ROOT / 'results/amp_design'
C = {'B_zM': '#2a78d6', 'B_N1': '#1baf7a', 'B_vit': '#eb6834', 'ref': '#52514e'}
LAB = {'B_zM': 'zero-hop composite net zM (w_s 6e-5)', 'B_N1': 'zero-hop composite net N1 (w_s 1.1e-4)',
       'B_vit': "ViT's own sign (control D)"}
runs = {k: json.load(open(R / f'amp6_{k}.json')) for k in ('B_zM', 'B_N1', 'B_vit') if (R / f'amp6_{k}.json').exists()}

lines = ['| arm | start <H>-E_ViT | end <H>-E_ViT | accepted steps (verified dE/site, z) | GPU-h |', '|---|---|---|---|---|']
for k, r in runs.items():
    cs, ce = r.get('cert_start', {}).get('delta_vs_vit_site'), r.get('cert_end', {}).get('delta_vs_vit_site')
    acc = []
    for s in r['steps']:
        c = s['cands'][s['best']]
        acc.append(f"{c['meas_site']:+.1e} (z {c['meas_site'] / c['se_site']:.1f}{'' if s['accepted'] else ', rej.'})")
    f = lambda d: f'{d[0]:+.2e} +- {d[1]:.1e}' if d else '-'
    lines.append(f"| {LAB[k]} | {f(cs)} | {f(ce)} | {'; '.join(acc)} | {r['gpu_h']['total']:.2f} |")
(OUT / 'AMP6_table.md').write_text('\n'.join(lines) + '\n'); print('\n'.join(lines))

plt.rcParams.update({'font.size': 9, 'axes.linewidth': 0.6, 'axes.edgecolor': '#52514e', 'xtick.color': '#52514e', 'ytick.color': '#52514e'})
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.3), constrained_layout=True)
# (a) generalisation: fresh/train RMS ratio of every candidate vs N
a = ax[0]
sm = json.load(open(R / 'amp6_smoke.json'))
for k, r in [('smoke', sm)] + list(runs.items()):
    for s in r['steps']:
        for c in s['cands']:
            col = C.get(k, C['ref']); mk = 'o' if c['kind'] == 'gd' else 's'
            a.plot(s['n'] * (1 + 0.05 * np.random.default_rng(hash(c['kind']) % 1000).normal()), c['gen_ratio'], mk,
                   color=col, ms=3.5, alpha=0.8, mfc=col if c['kind'] == 'gd' else 'white')
a.axhline(1, color=C['ref'], lw=0.6, ls=':')
a.set_xscale('log'); a.set_ylim(0, 2); a.set_xlabel('samples per step N'); a.set_ylabel('RMS(δ log a): fresh / train')
a.set_title('Steps generalise (o gradient, □ SR)', fontsize=9, loc='left')
# (b) verified decrease of the best candidate per step
b = ax[1]
for i, (k, r) in enumerate(runs.items()):
    x = np.arange(1, len(r['steps']) + 1) + 0.12 * (i - 1)
    m = np.array([s['cands'][s['best']]['meas_site'] for s in r['steps']]); se = np.array([s['cands'][s['best']]['se_site'] for s in r['steps']])
    accd = np.array([s['accepted'] for s in r['steps']])
    b.errorbar(x, 1e6 * m, yerr=1e6 * se, fmt='none', ecolor=C[k], lw=0.9, capsize=2)
    b.plot(x[accd], 1e6 * m[accd], 'o', color=C[k], ms=5, label=LAB[k])
    b.plot(x[~accd], 1e6 * m[~accd], 'o', color='white', mec=C[k], ms=5)
b.axhline(0, color=C['ref'], lw=0.6, ls=':'); b.set_xticks([1, 2, 3])
b.set_xlabel('amplitude step (N = 1e5; open = not accepted)'); b.set_ylabel(r'verified $\Delta E$ of best step [$10^{-6}$/site]')
b.set_title('Fresh-sample verified gains', fontsize=9, loc='left'); b.legend(frameon=False, fontsize=6.5, loc='lower right')
# (c) guide energy vs ViT, start -> end
c_ = ax[2]
for i, (k, r) in enumerate(runs.items()):
    pts = [r.get('cert_start', {}).get('delta_vs_vit_site'), r.get('cert_end', {}).get('delta_vs_vit_site')]
    xs = [0 + 0.08 * (i - 1), 1 + 0.08 * (i - 1)]
    ok = [(x_, p) for x_, p in zip(xs, pts) if p]
    if not ok: continue
    c_.errorbar([o[0] for o in ok], [1e5 * o[1][0] for o in ok], yerr=[1e5 * o[1][1] for o in ok], fmt='o-', color=C[k], lw=0.9, ms=4, capsize=2)
c_.axhline(0, color=C['ref'], lw=0.6, ls=':'); c_.text(0.3, -0.6, 'ViT (VMC)', fontsize=7, color=C['ref'])
c_.set_xticks([0, 1]); c_.set_xticklabels(['start (|ViT| amplitude)', 'after 3 steps'])
c_.set_ylabel(r'$\langle H\rangle_G - E_{\rm ViT}$ [$10^{-5}$/site]'); c_.set_title('Guide energy (paired certificate)', fontsize=9, loc='left')
for i, a_ in enumerate(ax):
    a_.text(-0.16, 1.04, '(%s)' % 'abc'[i], transform=a_.transAxes, fontsize=10, fontweight='bold')
    a_.grid(True, which='major', lw=0.3, color='#d9d8d4'); a_.set_axisbelow(True)
    for sp_ in ('top', 'right'): a_.spines[sp_].set_visible(False)
fig.savefig(OUT / 'amp6_gate.png', dpi=200); fig.savefig(OUT / 'amp6_gate.pdf'); print('figure written')
