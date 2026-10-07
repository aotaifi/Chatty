"""Figure + fit table from fncal_table.json (numpy/matplotlib only).  Usage: python fncal_plot.py TABLE.json OUTDIR"""
import sys, json, os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fncal_analyze import fit_bias

GUIDES = {'vitvit': 'ViT amp + ViT sign', 'vitex': 'ViT amp + exact sign', 'exvit': 'exact amp + ViT sign'}
COL = {'vitvit': '#2a78d6', 'vitex': '#eb6834', 'exvit': '#1baf7a'}
MK = {'vitvit': 'o', 'vitex': 's', 'exvit': '^'}
rows = json.load(open(sys.argv[1])); out = sys.argv[2]
WIN = os.environ.get('FNCAL_WIN', 'b2.4')
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3dd', '#fcfcfb'
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': INK2,
                     'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2, 'figure.facecolor': SURF, 'axes.facecolor': SURF})
fig, axs = plt.subplots(1, 2, figsize=(12.2, 4.9), sharey=True)
panels = [('step', 'referee as used so far: step-averaged', 128), ('mean', 'beta-time-averaged (recommended)', 8)]
fits = {}
for ax, (est, title, Mmin) in zip(axs, panels):
    ax.axhspan(-3, 3, color='#d9d8d2', alpha=0.5, lw=0, zorder=0)
    ax.axhline(0, color=INK2, lw=0.8, zorder=1)
    for g in GUIDES:
        rr = sorted([r for r in rows if r['guide'] == g and r['window'] == WIN and r['M'] >= Mmin], key=lambda r: r['M'])
        if not rr: continue
        M = np.array([r['M'] for r in rr]); b = np.array([r[est] for r in rr]) * 1e6; se = np.array([r[est + '_se'] for r in rr]) * 1e6
        ax.errorbar(1 / M, b, se, color=COL[g], lw=1.0, marker=MK[g], ms=5.5, mec=SURF, mew=0.8, capsize=2, elinewidth=0.9,
                    label=GUIDES[g], zorder=3)
        fits[f'{est}|{g}'] = fit_bias(M, b, se, Mmin=Mmin)
    ax.set_title(title, fontsize=10.5, loc='left', color=INK)
    ax.set_xlabel('1 / M   (walkers per population)')
    ax.grid(axis='y', color=GRID, lw=0.6); ax.set_axisbelow(True)
axs[0].set_xlim(-0.0003, 1 / 128 * 1.06)
axs[0].set_xticks([0, 1 / 1024, 1 / 512, 1 / 256, 1 / 128]); axs[0].set_xticklabels(['0', '1/1024', '1/512', '1/256', '1/128'], fontsize=8.5)
axs[1].set_xlim(-0.003, 1 / 8 * 1.05)
axs[1].set_xticks([0, 1 / 64, 1 / 32, 1 / 16, 1 / 8]); axs[1].set_xticklabels(['0', '1/64', '1/32', '1/16', '1/8'], fontsize=8.5)
# dashed a + b/M fits on the time-averaged panel
for g in GUIDES:
    f = fits.get(f'mean|{g}', {}).get('free')
    if f:
        xs = np.linspace(0, 1 / 8, 50); axs[1].plot(xs, f['a'] + f['b'] * xs, color=COL[g], lw=0.6, ls=(0, (3, 3)), alpha=0.8, zorder=2)
axs[0].set_ylabel('DMC - exact FN   (1e-6 per site)')
axs[0].legend(frameon=False, loc='center right', fontsize=9, bbox_to_anchor=(1.0, 0.5))
axs[0].text(0.02, 0.97, 'grey band: pre-registered +-3e-6 / site', transform=axs[0].transAxes, ha='left', va='top', fontsize=8.5, color=INK2)
axs[1].text(0.98, 0.66, 'dashed: fit a + b/M over all M', transform=axs[1].transAxes, ha='right', va='top', fontsize=8.5, color=INK2)
# inset zoom for M >= 128 on the right panel
ins = axs[1].inset_axes([0.07, 0.58, 0.50, 0.34])
ins.axhspan(-3, 3, color='#d9d8d2', alpha=0.5, lw=0); ins.axhline(0, color=INK2, lw=0.6)
for g in GUIDES:
    rr = sorted([r for r in rows if r['guide'] == g and r['window'] == WIN and r['M'] >= 128], key=lambda r: r['M'])
    if not rr: continue
    M = np.array([r['M'] for r in rr]); b = np.array([r['mean'] for r in rr]) * 1e6; se = np.array([r['mean_se'] for r in rr]) * 1e6
    ins.errorbar(1 / M, b, se, color=COL[g], lw=0.8, marker=MK[g], ms=3.5, capsize=1.2, elinewidth=0.7)
ins.set_xlim(-0.0003, 1 / 128 * 1.08); ins.set_ylim(-5, 5)
ins.set_xticks([0, 1 / 1024, 1 / 256]); ins.set_xticklabels(['0', '1/1024', '1/256'], fontsize=7); ins.tick_params(labelsize=7)
ins.set_title('zoom, M >= 128', fontsize=7.5, color=INK2, loc='left'); ins.spines['top'].set_visible(False); ins.spines['right'].set_visible(False)
fig.suptitle('Lattice FN referee vs exact symmetric-sector FN, 6x6 J1-J2 (J2/J1 = 0.5), tau_max = 0.025, beta window 0.8-2.4',
             x=0.01, ha='left', fontsize=11, color=INK)
fig.tight_layout()
for ext in ('png', 'pdf'): fig.savefig(os.path.join(out, f'fig_fncal_bias_vs_invM.{ext}'), dpi=170, facecolor=SURF)
json.dump(fits, open(os.path.join(out, 'fncal_fits.json'), 'w'), indent=1)
print(json.dumps(fits, indent=1))
