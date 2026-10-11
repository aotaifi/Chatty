"""Round-2 figure: (a) constant-hop loop vs table loop, (b) structured vs white tolerance, (c) re-base retention.
  python cx_figure2.py RESULTS_DIR    (reads runs/cx_chop_s*.json, cx_struct.json, cx_rebase_diag.json)
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = sys.argv[1]
ld = lambda f: json.load(open(os.path.join(R, 'runs', f)))
c0, c1 = ld('cx_chop_s0.json'), ld('cx_chop_s1.json')
st = ld('cx_struct.json')
dg = ld('cx_rebase_diag.json') if os.path.exists(os.path.join(R, 'runs', 'cx_rebase_diag.json')) else None
H0 = 1.3193875e-4
plt.rcParams.update({'font.size': 9, 'axes.linewidth': 0.7})
fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))

# (a) <H> per iteration
tl = c0['table_loop']
ax[0].plot([0] + [t['it'] for t in tl], [H0] + [t['H'] for t in tl], 'ko-', ms=3.5, lw=0.8, label='table loop (wtLOOP8, 8-hop table)')
for c, col, lab in ((c0, '#1f5fa8', 'constant-hop, seed 0'), (c1, '#16a085', 'constant-hop, seed 1')):
    its = [0] + [i['it'] for i in c['iters']]
    ax[0].plot(its, [H0] + [i['H'] for i in c['iters']], 'o-', color=col, ms=3.5, lw=0.8, label=lab + ' (re-based guide)')
    ax[0].plot([i['it'] for i in c['iters']], [i['H_w'] for i in c['iters']], 'x', color=col, ms=4)
ax[0].axhline(4.47e-5, color='#7f8c8d', lw=0.6, ls=':'); ax[0].text(8.2, 4.55e-5, 'RBM+PP', fontsize=7, ha='right', color='#7f8c8d')
ax[0].axhline(2.55e-5, color='#c0392b', lw=0.6, ls='--'); ax[0].text(8.2, 2.62e-5, r'$\psi_P$ + 1 Lanczos', fontsize=7, ha='right', color='#c0392b')
ax[0].axhline(H0 - 0.7 * (H0 - 4.03e-5), color='k', lw=0.5, ls='--'); ax[0].text(0.2, 6.95e-5, 'pass bar (it 6)', fontsize=7)
ax[0].set_yscale('log'); ax[0].set_xlabel('loop iteration'); ax[0].set_ylabel(r'$\langle H\rangle$ per site above $E_0$')
ax[0].plot([], [], 'kx', ms=4, label='x: write-back table before re-base')
ax[0].legend(frameon=False, fontsize=6.5, loc='upper right')

# (b) kept share vs |H|-weighted bond rms
sty = {'white': ('k', 'o', 'white noise'), 'Aw_k3': ('#1f5fa8', 's', "A' student error"), 'Aw_k6': ('#1f5fa8', 's', None),
       'As_k6': ('#7f8c8d', 'v', 'A student error'), 'fb_k3': ('#e67e22', 'D', 're-base (base-feature net) error'),
       'fb_k6': ('#e67e22', 'D', None)}
seen = set()
for k, ls in ((3, '-'), (6, '--')):
    for fld, (col, mk, lab) in sty.items():
        pts = sorted([p for p in st['points'] if p['field'] == fld and p['k'] == k], key=lambda p: p['bond_rms'])
        if not pts: continue
        ax[1].plot([p['bond_rms'] for p in pts], [max(p['kept'], -1.05) for p in pts], mk + ls, color=col, ms=4,
                   lw=0.8, mfc=col if k == 3 else 'none', label=(lab if lab and lab not in seen else None))
        if lab: seen.add(lab)
ax[1].axhline(0.9, color='k', lw=0.5, ls=':'); ax[1].axhline(0, color='k', lw=0.4)
ax[1].set_xscale('log'); ax[1].set_ylim(-1.1, 1.1)
ax[1].set_xlabel(r'$|H|a_xa_y$-weighted rms bond log-ratio error'); ax[1].set_ylabel('kept share of the write-back step')
ax[1].plot([], [], 'k-', label='filled/solid: k = 3'); ax[1].plot([], [], 'k--', label='open/dashed: k = 6')
ax[1].legend(frameon=False, fontsize=6.5, loc='lower left')
ax[1].text(0.98, 0.80, 'values < -1 clipped', transform=ax[1].transAxes, fontsize=6.5, ha='right')

# (c) re-base retention per iteration (+ diagnostic variants at iteration 6)
for c, col, mk in ((c0, '#1f5fa8', 'o'), (c1, '#16a085', 's')):
    its = [i['it'] for i in c['iters'] if abs(i['retention_step']) < 5]
    ax[2].plot(its, [i['retention_step'] for i in c['iters'] if abs(i['retention_step']) < 5], mk + '-', color=col, ms=4, lw=0.8)
    ax[2].plot([i['it'] for i in c['iters']], [i['retention_acc'] for i in c['iters']], mk + ':', color=col, ms=3, lw=0.8, mfc='none')
if dg:
    for j, v in enumerate(dg['variants']):
        ax[2].plot(6.15 + 0.15 * j, v['retention_step'], '*', ms=8, color=['#c0392b', '#8e44ad', '#e67e22'][j],
                   label=v['tag'].replace('_', ' '))
ax[2].axhline(0.7, color='k', lw=0.5, ls='--'); ax[2].axhline(0, color='k', lw=0.4)
ax[2].set_xlabel('loop iteration'); ax[2].set_ylabel('re-base retention')
ax[2].plot([], [], 'k-', label="solid: this iteration's gain"); ax[2].plot([], [], 'k:', label='dotted: gain since $\\psi_P$')
ax[2].legend(frameon=False, fontsize=6.5, loc='lower left'); ax[2].set_ylim(-0.3, 1.1)
for a, l in zip(ax, 'abc'):
    a.text(-0.17, 1.02, l, transform=a.transAxes, weight='bold', fontsize=11)
fig.tight_layout()
for e in ('png', 'pdf'): fig.savefig(os.path.join(R, f'fig_round2_6x6.{e}'), dpi=200)
