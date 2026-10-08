#!/usr/bin/env python3
"""Figure for results/tail_vs_N: cumulative share of the frozen-FN gain vs log10 phi^2 per N, + trend panels."""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze import load, cum_curve, metrics, RES

plt.rcParams.update({'font.size': 9.5, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.8,
                     'font.family': 'DejaVu Sans'})
INK, MUTED, GRID = '#1f2328', '#6a737d', '#e4e7eb'
NS = [16, 20, 24, 28, 32, 36]
COL = {n: matplotlib.colormaps['viridis'](v) for n, v in zip(NS, np.linspace(0.0, 0.80, 6))}
MK = dict(zip(NS, ['o', 's', '^', 'D', 'v', 'P']))

data = load()
bycl = {}
for cl, d in data.items():
    bycl[d['N']] = dict(Ncfg=d['states'][0]['N_cfg'], st={(s['k'] if s['k'] is not None else 'vit'): s for s in d['states']})

fig, axs = plt.subplots(2, 3, figsize=(14, 7.6), constrained_layout=True)


def draw(ax, key, norm, vit=False):
    for n in NS:
        if n not in bycl or key not in bycl[n]['st']: continue
        s = bycl[n]['st'][key]
        thr, cum = cum_curve(s['hist'], 'c_exact')
        x = thr + (np.log10(bycl[n]['Ncfg']) if norm else 0.0)
        pts = slice(7, 8 * 26, 8)                                 # one marker per decade (bin lower edge)
        ax.plot(x, cum, color=COL[n] if key != 'vit' else INK, lw=0.9, alpha=0.9, zorder=2,
                ls='--' if key == 'vit' else '-')
        ax.plot(x[pts], cum[pts], MK[n], color=COL[n] if key != 'vit' else INK, ms=4.2, mec='white', mew=0.5,
                ls='none', zorder=3, label=f'N={n}' if key != 'vit' else 'ViT $\\psi_P$ (N=36)')
    for lev in (0.5, 0.2):
        ax.axhline(lev, color=MUTED, lw=0.6, ls=':', zorder=1)
    ax.set_xlim((-1, -17) if not norm else (4, -8)); ax.set_ylim(-0.02, 1.02); ax.grid(color=GRID, lw=0.5, zorder=0)
    ax.set_ylabel('share of gain on configurations with $\\phi^2 \\geq$ threshold')
    ax.set_xlabel('$\\log_{10}$ threshold of per-configuration $\\phi^2/\\phi^2_{\\mathrm{unif}}$' if norm
                  else '$\\log_{10}$ threshold of per-configuration $\\phi_{FN}^2$ (decreasing $\\rightarrow$)')


for row, norm in enumerate((False, True)):
    draw(axs[row, 0], 1, norm); draw(axs[row, 1], 3, norm)
    ax = axs[row, 2]
    for key, ls, lab in ((1, '-', 'loop k=1'), (3, '--', 'loop k=3'), (8, '-.', 'loop k=8')):
        xs, ys = [], []
        for n in NS:
            if n in bycl and key in bycl[n]['st']:
                m = metrics(bycl[n]['st'][key])['c_exact']
                xs.append(n); ys.append(m['xt50'] if norm else m['t50'])
        ax.plot(xs, ys, ls=ls, color=INK, lw=0.9, label=lab); ax.plot(xs, ys, 'o', color=INK, ms=4.5, mec='white', mew=0.5)
        for n, y in zip(xs, ys): ax.plot([n], [y], MK[n], color=COL[n], ms=6, mec='white', mew=0.6, zorder=4)
    if 36 in bycl and 'vit' in bycl[36]['st']:
        m = metrics(bycl[36]['st']['vit'])['c_exact']
        ax.plot([36], [m['xt50'] if norm else m['t50']], 'X', color='#c0392b', ms=8, mec='white', mew=0.6, zorder=5,
                label='ViT $\\psi_P$')
    if not norm:
        ns_ = [n for n in NS if n in bycl]
        ax.plot(ns_, [-np.log10(bycl[n]['Ncfg']) for n in ns_], color=MUTED, lw=0.8, ls=':', label='uniform level $1/N_{cfg}$')
    ax.set_xlabel('number of sites N'); ax.grid(color=GRID, lw=0.5)
    ax.set_ylabel('$\\log_{10}\\phi^2$ below which half the gain sits' if not norm else
                  '$\\log_{10}(\\phi^2/\\phi^2_{\\mathrm{unif}})$ below which half the gain sits')
axs[0, 0].set_title('loop state after k = 1 FN/Krylov iteration', color=INK, fontsize=10.5, loc='left')
axs[0, 1].set_title('after k = 3 iterations', color=INK, fontsize=10.5, loc='left')
axs[0, 2].set_title('gain-halving threshold vs N', color=INK, fontsize=10.5, loc='left')
if 'vit' in bycl.get(36, {}).get('st', {}):
    draw(axs[0, 1], 'vit', False); draw(axs[1, 1], 'vit', True)
axs[0, 0].legend(frameon=False, loc='lower right', fontsize=8.5, ncol=2)
axs[0, 1].legend(frameon=False, loc='lower right', fontsize=8.5, ncol=2)
axs[0, 2].legend(frameon=False, loc='lower left', fontsize=8.5)
fig.suptitle('Tail concentration of one frozen-FN iteration (J$_2$/J$_1$=0.5, exact symmetric sector): '
             'cumulative share of the gain, node decomposition of the exact identity', fontsize=10.5, color=INK, x=0.01, ha='left')
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(RES, f'tail_vs_N.{ext}'), dpi=170)
print('saved')
