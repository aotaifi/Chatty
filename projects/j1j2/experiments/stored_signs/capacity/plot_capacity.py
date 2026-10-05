#!/usr/bin/env python3
import json, glob, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
R = '/Users/aliotaifi/Chatty-organize/projects/j1j2/results/stored_signs'
ex = json.load(open(f'{R}/runs_4x4/main2_exact.json'))['history']
def load(name):
    ds = [json.load(open(f)) for f in sorted(glob.glob(f'{R}/capacity/runs/cap_{name}_s?.json'))]
    h = lambda k: np.array([[r[k] for r in d['history']] for d in ds])
    return np.log10(h('w_s')), np.log10(h('eps'))
curves = [('5k_b0.5', r'5k params, $\beta$=0.5 (baseline)', '#4C72B0', 'o'),
          ('200k_b0.5', r'200k params, $\beta$=0.5', '#C44E52', 's'),
          ('5k_b0.3', r'5k params, $\beta$=0.3', '#DD8452', '^'),
          ('200k_b0.3', r'200k params, $\beta$=0.3', '#8172B3', 'v'),
          ('5k_nbr_b0.5', r'5k params, $\beta$=0.5, +nbr', '#2A9D5C', 'D')]
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.2))
its = np.arange(0, 41)
for ax, k, yl in ((axs[0], 0, r'$\log_{10}\, w_s$'), (axs[1], 1, r'$\log_{10}\,\epsilon_{E}$')):
    ax.plot(its, np.log10([r['w_s' if k == 0 else 'eps'] for r in ex]), 'k--', lw=1.2, label='exact recursive loop', zorder=1)
    for name, lab, col, mk in curves:
        y = load(name)[k]; ym = y.mean(axis=0)
        ax.plot(its, ym, '-', color=col, lw=0.7, zorder=2)
        ax.plot(its, ym, mk, color=col, ms=3.2, mew=0, label=lab, zorder=3)
    ax.set_xlabel('iteration'); ax.set_ylabel(yl); ax.set_xlim(0, 41)
    ax.grid(alpha=0.25, lw=0.5)
axs[0].set_ylim(-6.5, -1.5); axs[1].set_ylim(-5, -0.9)
axs[0].legend(loc='upper right', frameon=False, fontsize=8.5, handletextpad=0.4, borderaxespad=0.3)
for ax, t in zip(axs, 'ab'): ax.text(-0.17, 1.03, f'({t})', transform=ax.transAxes, fontsize=12, fontweight='bold')
fig.tight_layout()
fig.savefig(f'{R}/capacity/fig_capacity_4x4.png', dpi=200); fig.savefig(f'{R}/capacity/fig_capacity_4x4.pdf')
