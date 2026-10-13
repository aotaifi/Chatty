"""Step-size stage-2 figure: callable loop with the clipped-Lanczos target (cx_chop3) vs stage 1 (exact tables),
round 4 (semi-implicit target, callable) and psi_P + p Lanczos steps.
(a) <H> - E0 per site vs loop iteration; (b) share of the exact clipped-Lanczos step kept per iteration;
(c) equal hop count: guide after 4 iterations + 0 / 1 / 2 Lanczos steps vs psi_P + p steps.
  python cx_figure5.py RESULTS_DIR
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = sys.argv[1]
P = lambda f: os.path.join(R, 'runs', f)
ld = lambda f: json.load(open(P(f)))
net = [(s, ld(f'cx_chop3_s{s}.json')) for s in (0, 1) if os.path.exists(P(f'cx_chop3_s{s}.json'))]
direct = ld('cx_chop3_direct_s0.json') if os.path.exists(P('cx_chop3_direct_s0.json')) else None
r4 = [ld(f'cx_chop2_s{s}.json') for s in range(3)]
st1 = ld('cx_step.json')
lz = ld('cx_lanczos_ref.json')
H0 = lz[0]['H']
cn = ['#c0392b', '#e67e22']
plt.rcParams.update({'font.size': 9, 'axes.linewidth': 0.7})
fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.5))
# (a) error vs iteration
a = ax[0]
ks = [0] + [r['k'] for r in st1['runs']['Lc_0.3']]
a.plot(ks, [H0] + [r['err'] for r in st1['runs']['Lc_0.3']], 'ks--', ms=4, lw=0.8, mfc='none',
       label='clipped-Lanczos step, exact tables (stage 1)')
for (s, r), c in zip(net, cn):
    a.plot([0] + [i['it'] for i in r['iters']], [H0] + [i['H'] for i in r['iters']], 'o-', color=c, ms=3.5, lw=0.9,
           label=f'callable loop, clipped-Lanczos target, seed {s}')
if direct:
    a.plot([0] + [i['it'] for i in direct['iters']], [H0] + [i['H'] for i in direct['iters']], 'D-', color='#2471a3',
           ms=3, lw=0.8, label='same, re-base fits exact target (no write-back net)')
r4m = np.mean([[i['H'] for i in r['iters']] for r in r4], 0)
a.plot(range(0, 9), [H0] + list(r4m), 'v-', color='#7f8c8d', ms=3, lw=0.8, label='round 4 (semi-implicit target), 3-seed mean')
a.axhline(4.47e-5, color='#7f8c8d', lw=0.5, ls=':'); a.text(5.1, 4.7e-5, 'RBM+PP', fontsize=7, color='#7f8c8d')
a.set_yscale('log'); a.set_xlabel('loop iteration k'); a.set_ylabel(r'$\langle H\rangle - E_0$ per site')
a.set_xticks(range(0, 9)); a.legend(frameon=False, fontsize=6.3, loc='lower left')
# (b) per-iteration share of the exact step (capture x retention)
a = ax[1]
w = 0.18
for j, ((s, r), c) in enumerate(zip(net, cn)):
    k = np.array([i['it'] for i in r['iters']])
    a.plot(k + (j - 0.5) * w, [i['iter_share'] for i in r['iters']], 'o', color=c, ms=5, label=f'share kept, seed {s}')
    a.plot(k + (j - 0.5) * w, [i['capture'] for i in r['iters']], '^', color=c, ms=4, mfc='none', label=f'write-back capture, seed {s}')
if direct:
    k = np.array([i['it'] for i in direct['iters']])
    a.plot(k + 1.5 * w, [i['iter_share'] for i in direct['iters']], 'D', color='#2471a3', ms=4, label='share kept, direct')
a.axhline(0.7, color='k', lw=0.6, ls='--'); a.text(0.6, 0.72, 'pass bar 0.70', fontsize=7)
a.set_ylim(min(-0.05, a.get_ylim()[0]), 1.05)
a.set_xlabel('loop iteration k'); a.set_ylabel('share of the exact step kept'); a.set_xticks([1, 2, 3, 4])
a.legend(frameon=False, fontsize=6.3, loc='lower right')
# (c) equal hops
a = ax[2]
for (s, r), c in zip(net, cn):
    it = r['iters'][-1]
    a.plot([1, 2, 3], [it['H'], it['lanczos_p1'], it['lanczos_p2']], 's-', color=c, ms=4, lw=0.8,
           label=f'callable guide (k = {it["it"]}) + 0, 1, 2 steps, seed {s}')
r4k = [r['iters'][3] for r in r4]
a.plot([1, 2, 3], [np.mean([i[q] for i in r4k]) for q in ('H', 'lanczos_p1', 'lanczos_p2')], 'v-', color='#7f8c8d',
       ms=4, lw=0.8, label='round 4 guide (k = 4) + 0, 1, 2 steps')
a.plot([d['p'] for d in lz], [d['H'] for d in lz], 'ko-', ms=4, lw=0.9, label=r'$\psi_P$ + p Lanczos steps')
a.set_yscale('log'); a.set_xlabel(r'hops of $\psi_P$ per evaluation'); a.set_ylabel(r'$\langle H\rangle - E_0$ per site')
a.set_xticks([0, 1, 2, 3, 4]); a.legend(frameon=False, fontsize=6.3, loc='lower left')
for a, l in zip(ax, 'abc'):
    a.text(-0.17, 1.02, l, transform=a.transAxes, weight='bold', fontsize=11)
fig.tight_layout()
for e in ('png', 'pdf'): fig.savefig(os.path.join(R, f'fig_step2_6x6.{e}'), dpi=200)
