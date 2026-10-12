"""Round-4 figure: robust residual-re-base loop, 3 seeds. (a) <H>, (b) E_FN of the callable guide vs the table loop,
(c) guide + 1 / 2 exact Lanczos steps vs psi_P + p steps at equal hop count.   python cx_figure4.py RESULTS_DIR
"""
import json, os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = sys.argv[1]
ld = lambda f: json.load(open(os.path.join(R, 'runs', f)))
runs = [ld(f'cx_chop2_s{s}.json') for s in range(3)]
lz = ld('cx_lanczos_ref.json') if os.path.exists(os.path.join(R, 'runs', 'cx_lanczos_ref.json')) else None
H0, F0 = 1.3193875e-4, 1.0183074e-4
tl = runs[0]['table_loop']
cols = ['#c0392b', '#e67e22', '#8e44ad']
plt.rcParams.update({'font.size': 9, 'axes.linewidth': 0.7})
fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
for a, key, y0, lab in ((ax[0], 'H', H0, r'$\langle H\rangle$'), (ax[1], 'E_FN', F0, r'$E_{FN}$')):
    a.plot([0] + [t['it'] for t in tl], [y0] + [t[key] for t in tl], 'ko-', ms=3.5, lw=0.9, label='table loop (8-hop table)')
    for s, (r, c) in enumerate(zip(runs, cols)):
        a.plot([0] + [i['it'] for i in r['iters']], [y0] + [i[key] for i in r['iters']], 'o-', color=c, ms=3, lw=0.8,
               label=f'callable guide, seed {s}')
    a.axhline(4.47e-5, color='#7f8c8d', lw=0.6, ls=':'); a.text(0.2, 4.2e-5, 'RBM+PP <H>', fontsize=7, ha='left', color='#7f8c8d')
    a.set_yscale('log'); a.set_xlabel('loop iteration'); a.set_ylabel(lab + ' per site above $E_0$')
ax[0].axhline(H0 - 0.7 * (H0 - 4.03e-5), color='k', lw=0.5, ls='--'); ax[0].text(0.15, 6.95e-5, 'pass bar (it 6)', fontsize=7)
ax[0].legend(frameon=False, fontsize=6.5, loc='upper right')
# (c) equal hops: guide G_k costs one hop of psi_P; + p Lanczos steps -> 1 + p hops
for s, (r, c) in enumerate(zip(runs, cols)):
    it8 = r['iters'][-1]
    ax[2].plot([1, 2, 3], [it8['H'], it8['lanczos_p1'], it8['lanczos_p2']], 's-', color=c, ms=4, lw=0.8, label=f'guide (it 8) + 0, 1, 2 steps, seed {s}')
if lz:
    ax[2].plot([d['p'] for d in lz], [d['H'] for d in lz], 'ko-', ms=4, lw=0.9, label=r'$\psi_P$ + p Lanczos steps')
ax[2].set_yscale('log'); ax[2].set_xlabel('hops of $\\psi_P$ per evaluation'); ax[2].set_ylabel(r'$\langle H\rangle$ per site above $E_0$')
ax[2].set_xticks([0, 1, 2, 3, 4]); ax[2].legend(frameon=False, fontsize=6.5, loc='upper right')
for a, l in zip(ax, 'abc'):
    a.text(-0.17, 1.02, l, transform=a.transAxes, weight='bold', fontsize=11)
fig.tight_layout()
for e in ('png', 'pdf'): fig.savefig(os.path.join(R, f'fig_round4_6x6.{e}'), dpi=200)
