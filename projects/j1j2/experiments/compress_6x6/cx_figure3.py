"""Round-3 figure: (a) constant-hop loop with residual re-bases vs scalar re-bases vs the table loop; (b) re-base
pre-tests at iteration 6 (share of the iteration's gain kept).   python cx_figure3.py RESULTS_DIR
"""
import json, os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = sys.argv[1]
ld = lambda f: json.load(open(os.path.join(R, 'runs', f)))
H0 = 1.3193875e-4
plt.rcParams.update({'font.size': 9, 'axes.linewidth': 0.7})
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.5), gridspec_kw=dict(width_ratios=[1.15, 1]))
tl = ld('cx_chop_s0.json')['table_loop']
ax[0].plot([0] + [t['it'] for t in tl], [H0] + [t['H'] for t in tl], 'ko-', ms=3.5, lw=0.9, label='table loop (8-hop table)')
for f, col, mk, lab in (('cx_chop_inc_s0.json', '#c0392b', 'o', 'residual re-base, seed 0'),
                        ('cx_chop_inc_s1.json', '#e67e22', 's', 'residual re-base, seed 1'),
                        ('cx_chop_s0.json', '#1f5fa8', 'o', 'full re-base (round 2), seed 0'),
                        ('cx_chop_s1.json', '#16a085', 's', 'full re-base (round 2), seed 1')):
    c = ld(f)['iters']
    ax[0].plot([0] + [i['it'] for i in c], [H0] + [i['H'] for i in c], mk + '-', color=col, ms=3.5, lw=0.8,
               mfc=col if 'residual' in lab else 'none', label=lab)
ax[0].axhline(4.47e-5, color='#7f8c8d', lw=0.6, ls=':'); ax[0].text(8.2, 4.53e-5, 'RBM+PP', fontsize=7, ha='right', color='#7f8c8d')
ax[0].axhline(H0 - 0.7 * (H0 - 4.03e-5), color='k', lw=0.5, ls='--'); ax[0].text(0.15, 6.95e-5, 'pass bar (it 6)', fontsize=7)
ax[0].set_yscale('log'); ax[0].set_xlabel('loop iteration'); ax[0].set_ylabel(r'$\langle H\rangle$ per site above $E_0$ (callable guide)')
ax[0].legend(frameon=False, fontsize=6.5, loc='upper right')

rows = []
d = ld('cx_rebase_diag.json')
rows.append(('full, scalar psi_P feats (chop)', d['retention_chop'], '#1f5fa8'))
for v in d['variants']:
    rows.append(({'warm_r5_20k_lr1e-3': 'full, warm start', 'fresh_60k_lr1e-3': 'full, 60k steps',
                  'C64_40k_lr1e-3': 'full, 111k params'}[v['tag']], v['retention_step'], '#1f5fa8'))
rows.append(('full, per-bond psi_P ratios', ld('cx_pt3_rebase_bond.json')['arms'][0]['retention_step'], '#8e44ad'))
rows.append(('full, + G_3 features (2 hops)', ld('cx_pt3_rebase_rollG3.json')['arms'][0]['retention_step'], '#7f8c8d'))
rows.append(('full, + G_5 features', ld('cx_pt3_rebase_rollG5.json')['arms'][0]['retention_step'], '#7f8c8d'))
inc = ld('cx_pt3_rebase_inc.json')['arms']
rows.append(('residual, scalar psi_P feats', inc[0]['retention_step'], '#c0392b'))
rows.append(('residual, + G_5 features', inc[1]['retention_step'], '#e67e22'))
y = list(range(len(rows)))[::-1]
ax[1].barh(y, [r[1] for r in rows], color=[r[2] for r in rows], height=0.6)
ax[1].set_yticks(y); ax[1].set_yticklabels([r[0] for r in rows], fontsize=7)
ax[1].axvline(0.7, color='k', lw=0.6, ls='--'); ax[1].axvline(0, color='k', lw=0.4)
ax[1].set_xlim(-0.15, 1.05); ax[1].set_xlabel("re-base: share of iteration 6's gain kept")
for a, l in zip(ax, 'ab'):
    a.text(-0.12 if l == 'a' else -0.55, 1.02, l, transform=a.transAxes, weight='bold', fontsize=11)
fig.tight_layout()
for e in ('png', 'pdf'): fig.savefig(os.path.join(R, f'fig_round3_6x6.{e}'), dpi=200)
