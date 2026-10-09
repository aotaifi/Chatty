"""Figure for results/compress_6x6: (a) P1 sensitivity, (b) P2 shares per arm, (c) held-out log-ratio error per decade.
  python cx_figure.py RESULTS_DIR      (reads RESULTS_DIR/runs/*.json, writes fig_compress_6x6.{png,pdf})
"""
import glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = sys.argv[1]
rep = json.load(open(os.path.join(R, 'runs', 'cx_replay.json')))
p1 = list(rep['p1']) + json.load(open(os.path.join(R, 'runs', 'cx_p1ext.json')))['p1']
frac0 = {it['it']: it['frac'] for it in rep['iters'] if it['it'] <= 3}
eta = json.load(open(os.path.join(R, 'eta_star.json')))
runs = {}
for f in sorted(glob.glob(os.path.join(R, 'runs', 'cx_distill_*.json'))):
    r = json.load(open(f))
    if r['spec'].get('eval', True) and 'eval' in r:
        runs[os.path.basename(f)[len('cx_distill_'):-5]] = r

plt.rcParams.update({'font.size': 9, 'axes.linewidth': 0.7})
fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
ARMS = [('Aw_adam', "A' warm ViT, Adam", '#1f5fa8', 'o'), ('Aw_gn', "A' warm ViT, Gauss-Newton", '#16a085', 'D'),
        ('As', 'A ViT from scratch', '#7f8c8d', 'v'), ('B', 'B bond-ratio net', '#c0392b', 's')]

# (a) P1: frac(eta)/frac(0), mean and range over iterations 1-3
for model, mk, col, lab in (('VW', 'o', '#1f5fa8', 'iid log-noise on V, W'),
                            ('bond', 's', '#c0392b', 'white log-a noise (rms per bond)')):
    etas = sorted({q['eta'] for q in p1 if q['model'] == model})
    v = [[q['frac'] / frac0[q['it']] for q in p1 if q['model'] == model and q['eta'] == e] for e in etas]
    m = np.array([np.mean(a) for a in v]); lo = np.array([min(a) for a in v]); hi = np.array([max(a) for a in v])
    ax[0].errorbar(etas, m, yerr=[m - lo, hi - m], fmt=mk + '-', ms=4, lw=0.8, capsize=2, color=col, label=lab)
ax[0].axhline(0.9, color='k', lw=0.6, ls='--'); ax[0].axhline(0, color='k', lw=0.4)
ax[0].set_xscale('log'); ax[0].set_ylim(-1.0, 1.1)
ax[0].set_xlabel(r'noise $\eta$'); ax[0].set_ylabel('kept share of one write-back step')
ax[0].text(0.035, -0.9, r'$\eta\geq 0.03$: below $-1$', fontsize=7)
ax[0].legend(frameon=False, fontsize=7, loc='lower left', bbox_to_anchor=(0.0, 0.08))

# (b) shares of the stack gain per arm (same sign s_k)
for i, (tag, lab, col, mk) in enumerate(ARMS[:3]):
    for k, x0 in ((3, 0), (6, 1)):
        for name, r in runs.items():
            if name.startswith(tag + '_k%d' % k):
                x = x0 + 0.22 * (i - 1)
                for key, fill in (('share_H', col), ('share_EFN', 'none')):
                    v = r['eval'][key]
                    if v < -0.28:                              # off scale: marker on the edge + value
                        ax[1].plot(x, -0.28, mk, ms=5, color=col, mfc=fill, clip_on=False)
                        if key == 'share_H': ax[1].text(x + 0.05, -0.26, '%.0f' % v, fontsize=6.5, color=col)
                    else:
                        ax[1].plot(x, v, mk, ms=5, color=col, mfc=fill)
ax[1].axhline(0.7, color='k', lw=0.6, ls='--'); ax[1].axhline(0, color='k', lw=0.4)
ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(['k = 3', 'k = 6']); ax[1].set_xlim(-0.6, 1.6)
ax[1].set_ylim(-0.3, 1.0); ax[1].set_ylabel('share of the stack gain kept')
ax[1].text(1.45, 0.72, 'pass bar', ha='right', fontsize=7)
for tag, lab, col, mk in ARMS[:3]:
    ax[1].plot([], [], mk, color=col, ms=5, label=lab)
ax[1].plot([], [], 'ko', ms=5, label=r'filled: $\langle H\rangle$, open: $E_{FN}$', mfc='k')
ax[1].legend(frameon=False, fontsize=7, loc='upper left', bbox_to_anchor=(0.0, 0.68))
bc = [r['eval']['continuation']['frac_rel_table'] for n, r in runs.items() if n.startswith('B_')]
if bc:
    ax[1].text(0.5, 0.90, 'B (ratios only): continuation step\n%.0f to %.0f x the table step' % (max(bc), min(bc)),
               ha='center', va='top', fontsize=7, color='#c0392b')

# (c) held-out |H|a_y/a_x-weighted rms log-ratio error per decade, k = 6, seed 0, and psi_P (the warm start)
def curve(fid):
    d = sorted([q for q in fid if q['split'] == 'test' and q['decade'] >= -14], key=lambda q: -q['decade'])
    return [q['decade'] for q in d], [q['ratio_rms_w'] for q in d]
x, y = curve(rep['refs']['6']['base_fidelity'])
ax[2].plot(x, y, 'k^-', ms=4, lw=0.8, label=r'$\psi_P$ (warm start, untrained)')
for tag, lab, col, mk in ARMS:
    r = runs.get(tag + '_k6_s0')
    if r: x, y = curve(r['eval']['fidelity']); ax[2].plot(x, y, mk + '-', ms=3.5, lw=0.8, color=col, label=lab)
ax[2].axhline(eta['bond'], color='k', lw=0.6, ls='--'); ax[2].text(-14, eta['bond'] * 1.15, r'P1 tolerance $\eta^*$', fontsize=7, ha='right')
ax[2].set_yscale('log'); ax[2].set_xlabel(r'decade of $a_6^2$ per configuration ($\log_{10}$)')
ax[2].set_ylabel('held-out rms bond log-ratio error'); ax[2].invert_xaxis(); ax[2].set_ylim(2e-3, 5)
ax[2].legend(frameon=False, fontsize=7, loc='upper left')
for a, l in zip(ax, 'abc'):
    a.text(-0.17, 1.02, l, transform=a.transAxes, weight='bold', fontsize=11)
fig.tight_layout()
for e in ('png', 'pdf'): fig.savefig(os.path.join(R, f'fig_compress_6x6.{e}'), dpi=200)
