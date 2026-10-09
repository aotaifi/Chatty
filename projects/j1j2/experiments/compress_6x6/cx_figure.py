"""Figure for results/compress_6x6: (a) P1 sensitivity, (b) P2 shares per arm, (c) held-out log-ratio error per decade.
  python cx_figure.py RESULTS_DIR      (reads RESULTS_DIR/runs/*.json, writes fig_compress_6x6.{png,pdf})
"""
import glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = sys.argv[1]
rep = json.load(open(glob.glob(os.path.join(R, 'runs', 'cx_replay*.json'))[0]))
p1 = list(rep['p1'])
for f in glob.glob(os.path.join(R, 'runs', 'cx_p1ext*.json')):
    p1 += json.load(open(f))['p1']
frac0 = {it['it']: it['frac'] for it in rep['iters'] if it['it'] <= 3}
runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(R, 'runs', 'cx_distill_*.json')))]
runs = [r for r in runs if 'eval' in r and r['spec'].get('eval', True)]

plt.rcParams.update({'font.size': 9, 'axes.linewidth': 0.7})
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.2))
C = {'VW': '#1f5fa8', 'bond': '#c0392b', 'vit_warm': '#1f5fa8', 'vit_scratch': '#7f8c8d', 'bond_arm': '#c0392b'}

# (a) P1: frac(eta)/frac(0), mean and range over iterations 1-3
for model, mk in (('VW', 'o'), ('bond', 's')):
    etas = sorted({q['eta'] for q in p1 if q['model'] == model})
    m, lo, hi = [], [], []
    for e in etas:
        v = [q['frac'] / frac0[q['it']] for q in p1 if q['model'] == model and q['eta'] == e]
        m.append(np.mean(v)); lo.append(min(v)); hi.append(max(v))
    m, lo, hi = map(np.array, (m, lo, hi))
    ax[0].errorbar(etas, m, yerr=[m - lo, hi - m], fmt=mk + '-', ms=4, lw=0.8, capsize=2, color=C[model],
                   label={'VW': 'iid log-noise on V, W', 'bond': 'white log-a noise (bond rms)'}[model])
ax[0].axhline(0.9, color='k', lw=0.6, ls='--'); ax[0].axhline(0, color='k', lw=0.4)
ax[0].set_xscale('log'); ax[0].set_ylim(-1.0, 1.1)
ax[0].set_xlabel(r'noise $\eta$'); ax[0].set_ylabel('kept share of the write-back step')
ax[0].legend(frameon=False, fontsize=7, loc='lower left'); ax[0].text(0.02, 0.93, 'a', transform=ax[0].transAxes, weight='bold')

# (b) shares per arm and k
lab = {'vit_warm': "A' warm ViT", 'vit_scratch': 'A ViT scratch', 'bond': 'B ratio net'}
xs = {('vit_warm', 3): 0, ('vit_warm', 6): 1, ('vit_scratch', 3): 2.5, ('vit_scratch', 6): 3.5, ('bond', 3): 5, ('bond', 6): 6}
for r in runs:
    a, k = r['spec']['arm'], r['spec']['k']; x = xs[(a, k)] + 0.12 * (r['spec'].get('seed', 0) - 0.5)
    ev = r['eval']
    if a == 'bond':
        if 'continuation' in ev:
            ax[1].plot(x, ev['continuation']['frac_rel_table'], 'D', ms=4, color=C['bond_arm'])
    else:
        ax[1].plot(x, ev['share_H'], 'o', ms=4, color=C[a]); ax[1].plot(x, ev['share_EFN'], 's', ms=4, mfc='none', color=C[a])
ax[1].axhline(0.7, color='k', lw=0.6, ls='--'); ax[1].axhline(0, color='k', lw=0.4)
ax[1].set_xticks([0, 1, 2.5, 3.5, 5, 6]); ax[1].set_xticklabels(["A'\nk=3", "A'\nk=6", 'A\nk=3', 'A\nk=6', 'B\nk=3', 'B\nk=6'])
ax[1].set_ylabel('share of the stack gain'); ax[1].set_ylim(-0.5, 1.1)
ax[1].plot([], [], 'ko', ms=4, label=r'$\langle H\rangle$'); ax[1].plot([], [], 'ks', mfc='none', ms=4, label=r'$E_{FN}$')
ax[1].plot([], [], 'D', color=C['bond_arm'], ms=4, label='B: continuation / table')
ax[1].legend(frameon=False, fontsize=7, loc='lower right'); ax[1].text(0.02, 0.93, 'b', transform=ax[1].transAxes, weight='bold')

# (c) held-out weighted rms log-ratio error per decade, k = 6, seed 0, plus the warm-start (psi_P) error
def curve(fid):
    d = sorted([q for q in fid if q['split'] == 'test'], key=lambda q: -q['decade'])
    return [q['decade'] for q in d], [q['ratio_rms_w'] for q in d]
bf = rep['refs']['6']['base_fidelity']
x, y = curve(bf); ax[2].plot(x, y, 'k^-', ms=3.5, lw=0.8, label=r'$\psi_P$ (warm start)')
for r in runs:
    if r['spec']['k'] == 6 and r['spec'].get('seed', 0) == 0 and 'fidelity' in r['eval']:
        a = r['spec']['arm']; x, y = curve(r['eval']['fidelity'])
        ax[2].plot(x, y, 'o-', ms=3.5, lw=0.8, color=C['bond_arm' if a == 'bond' else a], label=lab[a])
eta_b = json.load(open(os.path.join(R, 'eta_star.json')))['bond'] if os.path.exists(os.path.join(R, 'eta_star.json')) else None
if eta_b: ax[2].axhline(eta_b, color='k', lw=0.6, ls='--')
ax[2].set_yscale('log'); ax[2].set_xlabel(r'decade of $a_6^2$ per configuration ($\log_{10}$)')
ax[2].set_ylabel('held-out rms log-ratio error'); ax[2].invert_xaxis()
ax[2].legend(frameon=False, fontsize=7); ax[2].text(0.02, 0.93, 'c', transform=ax[2].transAxes, weight='bold')
fig.tight_layout()
for e in ('png', 'pdf'): fig.savefig(os.path.join(R, f'fig_compress_6x6.{e}'), dpi=200)
