"""Figure for results/signs_8x8: (a) arm V and (b) arm M paired energies on RB; (c) held-out disagreement with K1 during training.
  python s8_figure.py energy.json train_V.json train_M.json OUTBASE
"""
import sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

E = json.load(open(sys.argv[1])); TV = json.load(open(sys.argv[2])); TM = json.load(open(sys.argv[3])); out = sys.argv[4]
rb = E['RB']
plt.rcParams.update({'font.size': 9, 'axes.linewidth': 0.8})
fig, ax = plt.subplots(1, 3, figsize=(11, 3.3), gridspec_kw=dict(width_ratios=[1, 1.15, 1]))

# (a) arm V: Delta <H> vs ViT sign, units 1e-6 per site
names = [('K1V', 'exact one-hop\nKrylov K1'), ('nV', 'stored net'), ('nV+hop', 'stored net\n+ one hop')]
y = [rb['vs_V'][k][0] * 1e6 for k, _ in names]; e = [rb['vs_V'][k][1] * 1e6 for k, _ in names]
xs = np.arange(len(names))
ax[0].errorbar(xs, y, yerr=e, fmt='o', color='C0', ms=5, capsize=3, lw=1)
ax[0].axhline(0, color='0.5', lw=0.8, ls='--')
ax[0].set_xticks(xs); ax[0].set_xticklabels([n for _, n in names], fontsize=8)
ax[0].set_xlim(-0.5, len(names) - 0.5)
ax[0].set_ylabel(r'$\langle H\rangle - \langle H\rangle_{\rm ViT\ sign}$  [$10^{-6}$ per site]')
ax[0].set_title('(a) from the ViT sign (arm V)', loc='left', fontsize=9)

# (b) arm M: <H> - <H>(ViT sign), log scale
names = [('M', 'Marshall'), ('K1M', 'exact one-hop\nKrylov K1'), ('nM', 'stored net'), ('nM+hop', 'stored net\n+ one hop')]
xs = np.arange(len(names))
for i, (k, _) in enumerate(names):
    m, s = rb['vs_V'][k]
    if m > 0:
        ax[1].errorbar([i], [m], yerr=[[min(s, 0.9 * m)], [s]], fmt='o', color='C3', ms=5, capsize=3, lw=1)
    else:
        ax[1].annotate(f'{m:.1e}\n({s:.0e})', (i, 1e-5), ha='center', fontsize=7)
ax[1].set_yscale('log')
ax[1].set_xticks(xs); ax[1].set_xticklabels([n for _, n in names], fontsize=8); ax[1].set_xlim(-0.5, len(names) - 0.5)
ax[1].set_ylabel(r'$\langle H\rangle - \langle H\rangle_{\rm ViT\ sign}$  per site')
ax[1].set_title('(b) from the Marshall sign (arm M)', loc='left', fontsize=9)

# (c) held-out disagreement with K1 vs training step
for T, lab, c in ((TV, 'arm V', 'C0'), (TM, 'arm M', 'C3')):
    h = T['hist']; ax[2].plot([q['step'] for q in h], [q['held_err'] for q in h], 'o-', ms=3, lw=0.8, color=c, label=lab)
    ax[2].axhline(T['label_flip_frac_held_nbr'] * 0 + (T['triv_held_x'] * 0.2 + T['triv_held_nbr'] * 0.8), color=c, lw=0.6, ls=':')
ax[2].set_yscale('log'); ax[2].set_xlabel('training step'); ax[2].set_ylabel('held-out disagreement with K1')
ax[2].set_title('(c) training (dotted: no-flip predictor)', loc='left', fontsize=9)
ax[2].legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(out + '.png', dpi=200); fig.savefig(out + '.pdf')
