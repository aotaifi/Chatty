"""Figure + decomposition table for the 6x6 stall diagnosis (results/stall_6x6/).
Usage: python st6_summary.py RESULTS_DIR
(a) E_FN error vs loop iteration: exact loop vs loops whose amplitude is projected onto a restricted family.
(b) energy cost of rough (iid, per-orbit) log-amplitude errors vs their rms, with the gain of one ideal iteration.
(c) exact energy ladder of the states in the decomposition.
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = sys.argv[1]
J = lambda f: json.load(open(os.path.join(R, f)))
C1, C2, C3, C4, C5 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4'
INK, MUTED = '#222222', '#77756b'
RBMPP, DMRG = 4.5e-5, 4.9e-6           # dE per site vs E0 = -0.50380965 (RBM+PP -0.503765, DMRG -0.503805)

xl = J('exact_loop_from_projvit.json'); xr = J('exact_loop_from_vit.json')
hop = J('feat_hop.json'); cl = J('feat_clusters44.json'); l2 = J('proj_loop_l2.json')
it_ = J('interp_fn_update.json'); sym = J('sym_test0.json'); base = J('base_validation.json')
t1 = J('t1_fit_exact_vit.json')

plt.rcParams.update({'font.size': 8.5, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
                     'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK})
fig, ax = plt.subplots(1, 3, figsize=(10.8, 3.5), gridspec_kw=dict(width_ratios=[1.15, 1.0, 1.25]))

# ---------------- (a) loops
a = ax[0]
ex = [(r['it'] - 1, r['dE_FN_prev']) for r in xl if r['kind'] == 'exact' and r['it'] >= 1]
hp = [(r['it'] - 1, r['dE_FN_prev']) for r in hop['hoploop']] + [(len(hop['hoploop']), hop['hoploop_final_FN']['dE_FN_site'])]
cp = [(0, 1.0183073736779057e-04), (1, 9.419941814956288e-05)]          # cluster-projected loop (job 16859765 log)
lp = [(r['it'] - 1, r['dE_FN_prev']) for r in l2 if r['kind'] == 'proj' and r['it'] >= 1] + \
     [(max(r['it'] for r in l2 if r['kind'] == 'proj'), [r for r in l2 if r['kind'] == 'proj_final_fn'][0]['dE_FN_prev'])]
for pts, col, lab, mk in ((ex, C1, 'exact loop', 'o'), (hp, C2, 'projected: one-hop features', 's'),
                          (cp, C3, 'projected: 44 symmetric clusters', '^'), (lp, C4, 'projected: L2 fit (rep-ViT start)', 'D')):
    x, y = np.array(pts).T
    a.plot(x, np.log10(y), '-', color=col, lw=0.9, marker=mk, ms=3.6, label=lab)
for v, t in ((RBMPP, 'RBM+PP'), (DMRG, 'DMRG')):
    a.axhline(np.log10(v), color=MUTED, lw=0.6, ls='--')
    a.text(12.2, np.log10(v) + 0.03, t, color=MUTED, fontsize=7.5, ha='right', va='bottom')
a.set_xlabel('loop iteration k'); a.set_ylabel(r'$\log_{10}\,[(E_{\rm FN}(a_k,s_k)-E_0)/N]$')
a.set_xlim(-0.4, 12.4); a.set_ylim(-5.45, -3.2)
a.legend(frameon=False, fontsize=7.2, loc='upper right')

# ---------------- (b) accuracy needed
b = ax[1]
sg = np.array([r['sigma'] for r in it_['noise']])
c_phi = np.array([r['phi1_plus_noise'] for r in it_['noise']]) - it_['phi1_kry']['dE_site']
c_psi = np.array([r['psiP_plus_noise'] for r in it_['noise']]) - it_['steps'][0]['H_own']
gain = it_['steps'][0]['H_kry1'] - it_['phi1_kry']['dE_site']
ss = np.logspace(-2.7, -1.4, 50)
b.plot(np.log10(ss), np.log10(0.33 * ss ** 2), '-', color=MUTED, lw=0.6)
b.text(-1.62, np.log10(0.33 * 0.03 ** 2) - 0.5, r'$0.33\,\sigma^2$', color=MUTED, fontsize=7.5)
b.plot(np.log10(sg), np.log10(c_phi), 'o', color=C1, ms=4.5, label=r'noise on $\phi_{\rm FN}$')
b.plot(np.log10(sg), np.log10(c_psi), 's', color=C2, ms=4.0, mfc='none', mew=1.0, label=r'noise on $\psi_P$')
b.axhline(np.log10(gain), color=C3, lw=0.9)
b.text(-2.68, np.log10(gain) + 0.06, 'gain of one ideal iteration', color=INK, fontsize=7.5)
for xv, t in ((it_['std_delta'], r'rms of the FN update $\delta$'), (0.0122, 'L2-projection error')):
    b.axvline(np.log10(xv), color=MUTED, lw=0.6, ls=':')
b.text(np.log10(it_['std_delta']) - 0.03, -6.2, r'rms $\delta$', rotation=90, color=MUTED, fontsize=7.2, ha='right', va='bottom')
b.text(np.log10(0.0122) + 0.03, -6.2, 'L2 proj. error', rotation=90, color=MUTED, fontsize=7.2, ha='left', va='bottom')
b.set_xlabel(r'$\log_{10}$ rms of iid log-amplitude error $\sigma$'); b.set_ylabel(r'$\log_{10}\,\Delta E/N$')
b.set_xlim(-2.7, -1.4); b.set_ylim(-6.3, -3.0)
b.legend(frameon=False, fontsize=7.2, loc='upper left')

# ---------------- (c) ladder
c = ax[2]
pa = sym['proj_abs']; hc = hop['hopcap']
rows = [
    ('ViT, rep-evaluated', base['vit']['H[vit]']['dE_site'], C4),
    ('ViT, full basis (VMC)', 1.56e-4, C4),
    (r'$\psi_P$ = D4$\times$flip projected ViT', sym['proj_complex']['dE_site'], C1),
    (r'$\psi_P$ + Krylov sign', pa['H[kry]']['dE_site'], C1),
    (r'$\psi_P$ + 44 clusters, exact opt.', cl['alt'][-1]['after_kry']['dE_site'], C3),
    (r'$\psi_P$ + one-hop features', hc['kry']['H']['dE_site'], C2),
    (r'$|\psi_0|$ + ViT sign (rep)', base['psi0_amp']['H[vit]']['dE_site'], MUTED),
    ('exact loop from $\\psi_P$, 1 it.', xl[1]['H_dE_site'], C5),
    ('exact loop from $\\psi_P$, 3 it.', xl[3]['H_dE_site'], C5),
    ('RBM+PP (lit.)', RBMPP, INK),
    ('exact loop from $\\psi_P$, 12 it.', xl[12]['H_dE_site'], C5),
    ('DMRG (lit.)', DMRG, INK),
]
rows = sorted(rows, key=lambda r: -r[1])
for i, (lab, v, col) in enumerate(rows):
    c.plot([np.log10(v)], [i], 'o', color=col, ms=5)
    c.text(np.log10(v) + 0.05, i, lab, va='center', fontsize=7.3, color=INK)
c.set_yticks([]); c.invert_yaxis()
c.set_xlabel(r'$\log_{10}\,[(\langle H\rangle-E_0)/N]$  (exact)')
c.set_xlim(-5.45, -3.1)
for s_ in ('left', 'right', 'top'): c.spines[s_].set_visible(False)
for s_ in ('right', 'top'):
    a.spines[s_].set_visible(False); b.spines[s_].set_visible(False)
for i, x in enumerate(ax):
    x.text(-0.13 if i < 2 else -0.03, 1.03, '(' + 'abc'[i] + ')', transform=x.transAxes, fontsize=10, fontweight='bold')
fig.tight_layout(w_pad=1.2)
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(R, f'stall_6x6.{ext}'), dpi=200)
print('wrote', os.path.join(R, 'stall_6x6.png'))
