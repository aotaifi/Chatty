"""Figure for the 6x6 learned-loop iteration-2 study (results/learned_loop_6x6_it2).
(a) lattice-FN energy minus the ViT VMC energy for each guide (M=128 beta 1.2 circles, M=512 beta 2.4 squares);
(b) true sign error w_s (ED, x ~ psi0^2) of the guide signs.
Reads summary_fn.json (it2_summary.py output, merged tags). Usage: python plot_it2.py"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

E0 = -0.5038096538908783; EV = -0.5036542608124052
S = json.load(open('summary_fn.json'))
U = 1e-5
# (label, M128 key, M512 key)
rows = [
    ('ViT guide', 'vitall|M128|b1.2', 'vitM512all|M512|b2.4'),
    (r'$|$ViT$|$, exact signs', 'orc_vitamp_exsign|M128|b1.2', None),
    (r'$|\psi_0|$, ViT signs', 'orc_examp_vitsign|M128|b1.2', None),
    (r'$a_1$, exact signs', 'orc_a1amp_exsign|M128|b1.2', None),
    (r'$a_2$, exact signs', 'orc_a2amp_exsign|M128|b1.2', None),
    (r'G1: $a_1$, K1 net + hop', 'G1hop|M128|b1.2', None),
    (r'K2: $|$ViT$|$, K1 net + hop', 'K2vit|M128|b1.2', None),
    (r'K3: $a_1$, D2 net + hop', 'K3a1|M128|b1.2', 'K3a1M512|M512|b2.4'),
    (r'K3: $|$ViT$|$, K2 net + hop', 'K3vit|M128|b1.2', 'K3vitM512|M512|b2.4'),
    (r'G2: $a_2$, D2 net + hop', 'G2|M128|b1.2', None),
]
rows = [r for r in rows if r[1] in S]
ws = [  # (label, w_s, err) ED on 4e5 psi0^2 samples
    ('Marshall', 1.99e-2, 3.1e-4), ('K1 recursion', 1.125e-3, 5.3e-5), ('K1 net', 9.98e-4, 5.0e-5),
    ('ViT', 1.3e-4, 1.8e-5), ('G1 recursive', 8.0e-5, 2.0e-5), ('G1: K1 net + hop', 6.5e-5, 1.3e-5),
    ('D2 net (stored G1)', 4.9e-4, 3.5e-5), ('K2: K1 net + hop', 6.0e-5, 1.2e-5),
    ('K3: D2 net + hop ($a_1$)', 4.25e-5, 1.0e-5), ('K3: K2 net + hop ($|$ViT$|$)', 4.25e-5, 1.0e-5),
]
plt.rcParams.update({'font.size': 9, 'axes.linewidth': 0.7})
fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 3.6), gridspec_kw=dict(width_ratios=[1.25, 1]))
y = np.arange(len(rows))[::-1]
for yi, (lab, k1, k5) in zip(y, rows):
    r = S[k1]; d, e = r['minus_vit_vmc']
    ax.errorbar(d / U, yi + 0.12, xerr=e / U, fmt='o', ms=3.5, color='C0', lw=0.8, capsize=1.5)
    if k5 and k5 in S:
        d5, e5 = S[k5]['minus_vit_vmc']
        ax.errorbar(d5 / U, yi - 0.12, xerr=e5 / U, fmt='s', ms=3.5, color='C3', lw=0.8, capsize=1.5)
ax.axvline(0, color='k', lw=0.6); ax.axvline((E0 - EV) / U, color='0.5', lw=0.6, ls='--')
ax.text((E0 - EV) / U + 0.4, y[-1] - 0.6, '$E_0$', color='0.4', fontsize=8)
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=7.5)
ax.set_ylim(y[-1] - 0.9, y[0] + 0.7)
ax.set_xlabel(r'$(E_{\rm FN} - E_{\rm ViT})/N$  [$10^{-5}$]')
ax.plot([], [], 'o', color='C0', ms=3.5, label=r'$M=128,\ \beta=1.2$'); ax.plot([], [], 's', color='C3', ms=3.5, label=r'$M=512,\ \beta=2.4$')
ax.legend(frameon=False, fontsize=7, loc='upper left', bbox_to_anchor=(0.02, 1.0))
ax.text(-0.02, 1.02, '(a)', transform=ax.transAxes, fontweight='bold')
yb = np.arange(len(ws))[::-1]
for yi, (lab, w, e) in zip(yb, ws):
    bx.errorbar(np.log10(w), yi, xerr=[[np.log10(w) - np.log10(max(w - e, 1e-9))], [np.log10(w + e) - np.log10(w)]],
                fmt='o', ms=3.5, color='C2', lw=0.8, capsize=1.5)
bx.set_yticks(yb); bx.set_yticklabels([w[0] for w in ws], fontsize=7.5)
bx.yaxis.tick_right()
bx.set_xlabel(r'$\log_{10} w_s$ (ED)')
bx.set_xlim(-4.7, -1.4)
bx.text(-0.02, 1.02, '(b)', transform=bx.transAxes, fontweight='bold')
fig.tight_layout()
fig.savefig('fig_it2_6x6.pdf'); fig.savefig('fig_it2_6x6.png', dpi=200)
print('ok')
