"""Figure for results/loop_8x8: (a) paired energy change vs the projected ViT guide, (b) absolute E/N vs references.
  python l8_figure.py results/loop_8x8/eval.json results/loop_8x8/fig_loop_8x8
"""
import sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

E = json.load(open(sys.argv[1])); out = sys.argv[2]
arms = {a['arm']: a for a in E['arms'] if a['label'] == 'best'}
C_FN, C_VMC, C_REF, C_GREY = '#1f5fa8', '#c0392b', '#2e8b57', '#7f7f7f'

rows = []                                                    # (label, Delta, se, color)
for tag, a in arms.items():
    if a['kind'] == 'fn': rows.append((tag, a['Delta_site'], a['Delta_se_site'], C_FN))
valid = [a for a in arms.values() if a['kind'] == 'vmc' and a['ess_frac'] > 0.5 and a['step'] > 0]
for a in sorted(valid, key=lambda a: a['Delta_site'])[:4]:
    rows.append((a['arm'] + ' (control)', a['Delta_site'], a['Delta_se_site'], C_VMC))
rows.append(('target exp(T)\n(unprojected, 2 hops)', E['target_SI']['Delta_site'], E['target_SI']['Delta_se_site'], C_GREY))
rows.append(('one Lanczos step', E['lanczos1']['Delta_site'], E['lanczos1']['Delta_se_site'], C_REF))

fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw=dict(width_ratios=[1.15, 1]))
y = np.arange(len(rows))[::-1]
for yi, (lab, d, se, c) in zip(y, rows):
    ax.errorbar(d * 1e5, yi, xerr=se * 1e5, fmt='o', color=c, ms=5, capsize=2.5, lw=1)
ax.axvline(0, color='k', lw=0.8)
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
ax.set_xlabel(r'$\Delta\langle H\rangle$ per site vs projected ViT  [$10^{-5}$]')
ax.set_title('(a) paired referee, same samples', loc='left', fontsize=10)
ax.grid(axis='x', lw=0.3, alpha=0.5)

g = E['guide']; fn = arms.get('FN-Adam')
pts = [('ViT (paper)', -0.498823, 3.5e-5, C_GREY), ('projected ViT\n(this run)', g['E_site'], g['se'], C_GREY)]
if fn: pts.append(('+ FN write-back', fn['E_site'], fn['E_se_site'], C_FN))
best_c = min(valid, key=lambda a: a['Delta_site'], default=None)
if best_c: pts.append(('+ best VMC control', best_c['E_site'], best_c['E_se_site'], C_VMC))
pts.append(('+ one Lanczos step', E['lanczos1']['E_site'], E['lanczos1']['se'], C_REF))
pts += [('Hu VMC p=2', -0.49886, 1e-5, 'k'), ('RBM+PP', -0.498886, 1e-6, 'k')]
x = np.arange(len(pts))
for xi, (lab, e, se, c) in zip(x, pts):
    bx.errorbar(xi, e, yerr=se, fmt='o', color=c, ms=5, capsize=2.5, lw=1)
bx.axhspan(-0.49906 - 1e-5, -0.49906 + 1e-5, color='k', alpha=0.08)
bx.text(-0.3, -0.49906 + 1.2e-5, 'extrapolated E0 -0.49906(1)', fontsize=7.5, va='bottom', ha='left')
bx.set_xticks(x); bx.set_xticklabels([p[0] for p in pts], rotation=35, ha='right', fontsize=8)
bx.set_ylabel('E / N'); bx.set_title('(b) absolute energy, 8x8, J2/J1 = 0.5', loc='left', fontsize=10)
bx.ticklabel_format(axis='y', useOffset=False); bx.grid(axis='y', lw=0.3, alpha=0.5)
fig.tight_layout()
fig.savefig(out + '.png', dpi=180); fig.savefig(out + '.pdf')
print('saved', out)
