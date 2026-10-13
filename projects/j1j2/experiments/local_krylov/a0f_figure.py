"""Figure for follow-up F1: gain per site relative to one copy vs N for (a) the global Krylov sign step and (b) one
clipped-Lanczos step, on decoupled 4x4 copies, four guides from sign-poor (Marshall) to NQS-like.
  python a0f_figure.py a0f_dilution.json OUT_STEM
"""
import sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = json.load(open(sys.argv[1])); OUT = sys.argv[2]
GUIDES = [('MAR', 'Marshall', '#2a78d6'), ('P45s', "sign of GS(J2'=0.45)", '#eb6834'),
          ('P48s', "sign of GS(J2'=0.48)", '#1baf7a'), ('P48f', "full GS(J2'=0.48)", '#eda100')]
MUTED = '#888888'
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': MUTED})
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
for ax, step, title, letter in ((axs[0], 'K', 'Krylov sign step', 'a'), (axs[1], 'Lc', 'clipped Lanczos step', 'b')):
    ax.axvspan(64, 196, color='#eeeeee', zorder=0); ax.text(80, 0.04, '8x8-14x14', color=MUTED, fontsize=7)
    ax.axhline(0.5, color=MUTED, lw=0.6, ls=':')
    for key, lab, col in GUIDES:
        c = R[key][step]['curve']; N = np.array([r['N'] for r in c]); y = np.array([r['ratio'] for r in c])
        ax.plot(N, y, '-', color=col, lw=1); ax.plot(N, y, 'o', color=col, ms=2.5, label=lab)
    ax.set_xscale('log'); ax.set_xlim(12, 9e4); ax.set_ylim(0, 1.08)
    ax.set_xticks([1e2, 1e3, 1e4]); ax.set_xticklabels(['$10^{2}$', '$10^{3}$', '$10^{4}$'])
    ax.set_xlabel('N (sites)'); ax.set_title(title, fontsize=9)
    ax.text(-0.14 if letter == 'a' else -0.06, 1.04, letter, transform=ax.transAxes, fontweight='bold', fontsize=11)
axs[0].set_ylabel('gain per site / gain of one copy')
axs[1].legend(frameon=False, fontsize=7, loc='lower left', bbox_to_anchor=(0.0, 0.08))
fig.tight_layout(); fig.savefig(OUT + '.png', dpi=200); fig.savefig(OUT + '.pdf')
