#!/usr/bin/env python3
"""T3 (6x6) summary figure + table from results/amp_design/t3/*.json (copies of the ws1 run outputs)."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / 'results/amp_design/t3'
OUT = ROOT / 'results/amp_design'
C = dict(loop='#2a78d6', ref='#52514e', sr='#eda100', cplx='#eb6834', net='#1baf7a')
E_VIT = (-0.5036542608124052, 2.131044877124918e-05)

rgn1 = json.load(open(R / 'rgn_it1.json')); rgn2 = json.load(open(R / 'rgn_it2.json'))
dist = json.load(open(R / 'distill_dist1.json')); sign = json.load(open(R / 'sign_sign2.json'))

# ---- sign quality (ED w_s, x ~ psi0^2)
ws = [('ViT', 1.3e-4, 0.2e-4), ('K1 net\n+ hop(|ViT|)\n(start)', 6.0e-5, 1.2e-5),
      ('composite\nnet N1', dist['ED']['stored']['w'], dist['ED']['stored']['err']),
      ('N1\n+ hop(a1)\n(it 2)', sign['ED']['new']['w'], sign['ED']['new']['err'])]
# ---- energy vs ViT VMC (paired certificates)
cert = [('start\n(|ViT|, s1)', rgn1.get('cert_start') or dict(delta_vs_vit_site=[4.591296476243025e-06, 7.867230622348863e-06])),
        ('after it1\n(a1, s1)', rgn1['cert_end']),
        ('sign step 2\n(a1, s2)', dict(delta_vs_vit_site=sign['vs_vit']['delta_new_site'])),
        ('after it2\n(a2 = a1, s2)', rgn2['cert_end'])]
# ---- per-step best verified decrease
steps = []
for it, r in ((1, rgn1), (2, rgn2)):
    for s in r['steps']:
        b = s['cands'][s['best']]
        steps.append((it, s['step'], s['n'], b['meas_site'], b['se_site'], s['accepted'], b['kind']))

lines = ['| stage | <H>_G - E_ViT (per site, paired) | <H>_G |', '|---|---|---|']
for name, c in cert:
    d = c['delta_vs_vit_site']
    lines.append(f"| {name.replace(chr(10), ' ')} | {d[0]:+.2e} +- {d[1]:.1e} | {E_VIT[0] + d[0]:.6f}({int(round(1e6 * np.hypot(E_VIT[1], d[1])))}) |")
lines += ['', '| iteration | step | N | best candidate | verified dE/site | SE | accepted |', '|---|---|---|---|---|---|---|']
for it, st, n, m, se, acc, kind in steps:
    lines.append(f'| {it} | {st} | {n} | {kind} | {m:+.2e} | {se:.1e} | {acc} |')
lines += ['', '| sign | ED w_s |', '|---|---|'] + [f"| {n.replace(chr(10), ' ')} | {w:.1e} +- {e:.1e} |" for n, w, e in ws]
(OUT / 'T3_table.md').write_text('\n'.join(lines) + '\n'); print('\n'.join(lines))

plt.rcParams.update({'font.size': 9, 'axes.linewidth': 0.6, 'axes.edgecolor': '#52514e', 'xtick.color': '#52514e', 'ytick.color': '#52514e'})
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.3), constrained_layout=True)
a = ax[0]
x = np.arange(len(ws))
a.errorbar(x, [w for _, w, _ in ws], yerr=[e for *_, e in ws], fmt='o-', color=C['loop'], lw=0.9, ms=4, capsize=2)
a.set_yscale('log'); a.set_xticks(x); a.set_xticklabels([n for n, *_ in ws], fontsize=7, rotation=0)
a.set_ylabel(r'sign error $w_s$ (ED, $x\sim\psi_0^2$)'); a.set_title('Signs: composite net + one hop', fontsize=9, loc='left')
a = ax[1]
x = np.arange(len(cert)); y = np.array([c['delta_vs_vit_site'][0] for _, c in cert]); e = np.array([c['delta_vs_vit_site'][1] for _, c in cert])
a.errorbar(x, 1e5 * y, yerr=1e5 * e, fmt='o-', color=C['loop'], lw=0.9, ms=4, capsize=2)
a.axhline(0, color=C['ref'], lw=0.6, ls=':'); a.text(2.45, 0.25, 'ViT (VMC)', fontsize=7, color=C['ref'])
a.set_xticks(x); a.set_xticklabels([n for n, _ in cert], fontsize=7)
a.set_ylabel(r'$\langle H\rangle_G - E_{\rm ViT}$  [$10^{-5}$ per site]'); a.set_title('Guide energy vs ViT (paired)', fontsize=9, loc='left')
a = ax[2]
xs = np.arange(1, len(steps) + 1)
m = np.array([s[3] for s in steps]); se = np.array([s[4] for s in steps]); acc = np.array([s[5] for s in steps])
a.errorbar(xs, 1e5 * m, yerr=1e5 * se, fmt='none', ecolor=C['loop'], lw=0.9, capsize=2)
a.plot(xs[acc], 1e5 * m[acc], 'o', color=C['loop'], ms=5, label='accepted (z > 2)')
a.plot(xs[~acc], 1e5 * m[~acc], 'o', color='white', mec=C['loop'], ms=5, label='rejected')
a.axhline(0, color=C['ref'], lw=0.6, ls=':')
a.axvline(3.5, color=C['ref'], lw=0.5, ls='--'); a.text(3.6, a.get_ylim()[1] * 0.85 if a.get_ylim()[1] > 0 else 0.2, 'sign step 2', fontsize=7, color=C['ref'])
a.set_xlabel('amplitude step (it1: 1-3, it2: 4-6)'); a.set_ylabel(r'best verified $\Delta E$  [$10^{-5}$ per site]')
a.set_title('Amplitude steps (fresh-sample check)', fontsize=9, loc='left'); a.legend(frameon=False, fontsize=7)
for i, a in enumerate(ax):
    a.text(-0.16, 1.04, '(%s)' % 'abc'[i], transform=a.transAxes, fontsize=10, fontweight='bold')
    a.grid(True, which='major', lw=0.3, color='#d9d8d4'); a.set_axisbelow(True)
    for sp_ in ('top', 'right'): a.spines[sp_].set_visible(False)
fig.savefig(OUT / 't3_6x6.png', dpi=200); fig.savefig(OUT / 't3_6x6.pdf'); print('figure written')
