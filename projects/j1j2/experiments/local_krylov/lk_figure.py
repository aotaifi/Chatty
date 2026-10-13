"""Figure for checks (a0) and (a8): (a) per-site gain of one global Krylov sign step on M decoupled 4x4 copies,
relative to one copy, vs N = 16 M (local rule = 1 by construction); (b) 8x8 cross-fitted gains, global G vs local C_3,
C_4, from the Marshall sign and from the ViT sign.
  python lk_figure.py a0_copies.json a8_analysis.json OUT_STEM
"""
import sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

A0 = json.load(open(sys.argv[1])); A8 = json.load(open(sys.argv[2])); OUT = sys.argv[3]
BLUE, ORANGE, INK, MUTED = '#2a78d6', '#eb6834', '#222222', '#888888'
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK})
fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw=dict(width_ratios=[1.15, 1]))

c = A0['curve']; Nn = np.array([r['N'] for r in c]); rat = np.array([r['ratio'] for r in c])
ax.plot(Nn, rat, '-', color=BLUE, lw=1, zorder=2)
ax.plot(Nn, rat, 'o', color=BLUE, ms=3, zorder=3, label='global rule G (best T)')
ax.axhline(1.0, color=ORANGE, lw=1.5, label='local rule (exact, any N)')
ax.axhline(0.5, color=MUTED, lw=0.6, ls=':')
ax.axvline(64, color=MUTED, lw=0.6, ls='--'); ax.text(68, 0.6, '8x8', color=MUTED, fontsize=8)
ax.set_xscale('log', base=2); ax.set_xticks([16, 64, 256, 1024]); ax.set_xticklabels(['$2^{4}$', '$2^{6}$', '$2^{8}$', '$2^{10}$'])
ax.set_ylim(0, 1.08); ax.set_xlabel('N = 16 M (sites)'); ax.set_ylabel('gain per site / gain of one copy')
ax.legend(frameon=False, loc='lower left', bbox_to_anchor=(0.0, 0.12), fontsize=8)
ax.text(-0.16, 1.02, 'a', transform=ax.transAxes, fontweight='bold', fontsize=11)

rules = [('G', 'G'), ('C3', '$C_3$'), ('C4', '$C_4$')]
for k, (base, lab, col, off) in enumerate([('M', 'from Marshall', BLUE, -0.12), ('V', 'from ViT sign', ORANGE, 0.12)]):
    y = np.array([A8[base][r]['xfit_gain'] for r, _ in rules]); e = np.array([A8[base][r]['se'] for r, _ in rules])
    x = np.arange(len(rules)) + off
    bx.errorbar(x, -y * 1e3, yerr=2 * e * 1e3, fmt='o', color=col, ms=4, capsize=0, elinewidth=1, label=lab)
bx.axhline(0, color=MUTED, lw=0.6)
bx.set_xticks(range(len(rules))); bx.set_xticklabels([l for _, l in rules])
bx.set_ylabel(r'energy lowered per site ($10^{-3}$)'); bx.legend(frameon=False, fontsize=8, loc='upper right')
bx.text(-0.24, 1.02, 'b', transform=bx.transAxes, fontweight='bold', fontsize=11)
fig.tight_layout()
fig.savefig(OUT + '.png', dpi=200); fig.savefig(OUT + '.pdf')
