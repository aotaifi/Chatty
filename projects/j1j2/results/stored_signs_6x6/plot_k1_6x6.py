"""6x6 stored-sign test, first Krylov step: stored nets vs exact one-hop recursion (fixed |psi_ViT|)."""
import json
import matplotlib.pyplot as plt
d = json.load(open('k1_summary.json'))
ref = d['L_b0.3']
rows = [('exact recursion', ref['ED']['onehop_rec'], ref['ENERGY']['onehop_rec'], 'k', 'o')]
lab = {'S': 'small net (28k)', 'L': 'large net'}
for key, c, m in [('S_b0.5', '#7fa7d6', 's'), ('S_b0.3', '#7fa7d6', 'D'), ('L_b0.5', '#1f4e8c', 's'), ('L_b0.3', '#1f4e8c', 'D')]:
    n, b = key.split('_b')
    rows.append((f'{lab[n]}, β={b}', d[key]['ED']['stored'], d[key]['ENERGY']['stored'], c, m))
plt.rcParams.update({'font.size': 11})
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
y = list(range(len(rows)))[::-1]
for yi, (name, ed, en, c, m) in zip(y, rows):
    ax[0].errorbar(ed['w'] * 1e3, yi, xerr=ed['err'] * 1e3, fmt=m, color=c, ms=7, capsize=3, lw=1)
    ax[1].errorbar(en['dE_site'] * 1e4, yi, xerr=en['se_site'] * 1e4, fmt=m, color=c, ms=7, capsize=3, lw=1)
for a, v in [(ax[0], ref['ED']['onehop_rec']['w'] * 1e3), (ax[1], ref['ENERGY']['onehop_rec']['dE_site'] * 1e4)]:
    a.axvline(v, color='k', lw=0.8, ls='--', alpha=0.6)
    a.grid(axis='x', alpha=0.3); a.spines[['top', 'right']].set_visible(False)
ax[0].set_yticks(y); ax[0].set_yticklabels([r[0] for r in rows])
ax[0].set_xlabel(r'wrong-sign probability $w_s$ vs ED  [$10^{-3}$]')
ax[1].set_xlabel(r'$E - E_{\rm ViT}$ per site  [$10^{-4}$]')
ax[0].set_title('(a)', loc='left', fontweight='bold'); ax[1].set_title('(b)', loc='left', fontweight='bold')
fig.suptitle('6×6, J2/J1 = 0.5, ViT amplitude: first Krylov step from Marshall '
             r'(Marshall: $w_s$ = 2.0$\times10^{-2}$, +6.6$\times10^{-3}$/site)', fontsize=11)
fig.tight_layout()
fig.savefig('fig_k1_6x6.png', dpi=200); fig.savefig('fig_k1_6x6.pdf')
