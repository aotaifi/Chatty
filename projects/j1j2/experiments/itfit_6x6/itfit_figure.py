"""Figure for results/itfit_6x6: (a) stiffness: exact captured fraction of one FN iteration vs number of exact steps;
(b) captured fraction per loop iteration of the trained arms (zero-hop network and one-hop composite guide);
(c) loop: <H> of the guide (Krylov sign) per loop iteration vs the ideal loop and RBM+PP.
  python itfit_figure.py RESULTS_DIR
"""
import json, os, sys, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = sys.argv[1]
ex = json.load(open(os.path.join(R, 'itfit_exact.json')))
exl = json.load(open(os.path.join(R, 'itfit_exactloop.json')))
runs = {}
for f in sorted(glob.glob(os.path.join(R, 'runs', '*.json'))):
    r = json.load(open(f)); runs[r.get('tag', os.path.basename(f)[:-5])] = r

C = dict(blue='#2a78d6', orange='#eb6834', aqua='#1baf7a', yellow='#eda100', magenta='#e87ba4', green='#008300',
         violet='#4a3aa7', red='#e34948', ink='#0b0b0b', grey='#8a8985')
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6,
                     'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'legend.frameon': False})
fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.5), constrained_layout=True)

# (a) stiffness
a = ax[0]
pick = [('semi tau=1.0', 'semi-implicit, $\\tau=1$ (local)', C['blue'], 'o'),
        ('implicit tau=0.3', 'implicit, $\\tau=0.3$ (global solve)', C['aqua'], 's'),
        ('Krylov 2-hop RR', 'Krylov 2-hop RR', C['violet'], '^'),
        ('optimal explicit (1-hop RR)', 'explicit, optimal $\\tau$ (1-hop RR)', C['orange'], 'D'),
        ('explicit tau=0.9 tau_stab', 'explicit, $\\tau=0.9\\,\\tau_{stab}$', C['red'], 'v')]
for key, lab, col, mk in pick:
    sc = [s for s in ex['schemes'] if s['scheme'].startswith(key)][0]
    n = np.array([r['n'] for r in sc['rows'] if r['n'] > 0]); fr = np.array([r['frac'] for r in sc['rows'] if r['n'] > 0])
    a.plot(n, fr, '-', color=col, lw=1.0, marker=mk, ms=3.5, mfc=col, mec='white', mew=0.4, label=lab)
a.axhline(0.9, color=C['grey'], lw=0.6, ls='--'); a.text(1.1, 0.92, '90%', color=C['grey'], fontsize=8)
a.set_xscale('log'); a.set_xlim(0.8, 1.5e3); a.set_ylim(-0.02, 1.02)
a.set_xlabel('exact steps $n$'); a.set_ylabel('captured fraction of one FN iteration')
a.set_title('(a) stiffness of $F$ ($\\lambda_{max}=%.1f\\times10^{6}$)' % (ex['stiffness']['lambda_max'] / 1e6), loc='left', fontsize=9)
a.legend(fontsize=7, loc='center right')

# (b) captured fraction per loop iteration
b = ax[1]
order = [k for k in ['ITE-Q-i', 'ITE-Q-ii', 'ITE-Q-iii', 'ITE-P-iii', 'C-P-i', 'VMC-i', 'VMC-iii', 'ORACLE'] if k in runs]
cols = dict(zip(['ITE-Q-i', 'ITE-Q-ii', 'ITE-Q-iii', 'ITE-P-iii', 'C-P-i', 'VMC-i', 'VMC-iii', 'ORACLE'],
                [C['blue'], C['orange'], C['aqua'], C['yellow'], C['magenta'], C['green'], C['violet'], C['ink']]))
for j, k in enumerate(order):
    r = runs[k]; comp = r['spec'].get('composite')
    fr = [lp['end']['frac_composite' if comp else 'frac'] for lp in r['loops'] if 'end' in lp]
    it = np.arange(1, len(fr) + 1) + (j - len(order) / 2) * 0.06
    b.plot(it, fr, '-', color=cols[k], lw=1.0, marker='o', ms=4, mec='white', mew=0.4, label=k)
el = [v for v in exl['variants'] if v['tau'] == 1.0 and v['K'] == 2][0]
b.plot([1, 2, 3], [x['frac'] for x in el['loop']], '--', color=C['grey'], lw=0.8, label='exact ITE, K=2 (no fit)')
b.axhline(0.5, color=C['red'], lw=0.6, ls=':'); b.text(2.6, 0.52, 'pass bar', color=C['red'], fontsize=8)
b.set_xticks([1, 2, 3]); b.set_xlabel('loop iteration $k$'); b.set_ylabel('captured fraction $\\mathrm{frac}_k$')
b.set_ylim(-0.05, 1.02); b.set_title('(b) write-back per loop iteration', loc='left', fontsize=9)
b.legend(fontsize=7, loc='center right')

# (c) loop energies
c = ax[2]
ideal = [1.3193875e-4, 7.840082e-5, 5.584382e-5, 4.297001e-5]
c.plot([0, 1, 2, 3], np.array(ideal) * 1e5, '-', color=C['ink'], lw=1.0, marker='s', ms=3.5, label='ideal FN loop')
c.plot([0, 1, 2, 3], np.array([ideal[0]] + [x['H_kry_dE_site'] for x in el['loop']]) * 1e5, '--', color=C['grey'],
       lw=0.8, label='exact ITE, K=2 (no fit)')
for k in order:
    r = runs[k]; comp = r['spec'].get('composite')
    h = [r['loops'][0]['start']['H_guide_dE_site']] + [lp['end']['H_composite_dE_site' if comp else 'H_kry_dE_site']
                                                       for lp in r['loops'] if 'end' in lp]
    c.plot(np.arange(len(h)), np.array(h) * 1e5, '-', color=cols[k], lw=1.0, marker='o', ms=3.5, mec='white', mew=0.4, label=k)
c.axhline(4.47, color=C['red'], lw=0.6, ls=':'); c.text(0.05, 4.6, 'RBM+PP', color=C['red'], fontsize=8)
c.set_xticks([0, 1, 2, 3]); c.set_xlabel('loop iteration $k$')
c.set_ylabel('$(\\langle H\\rangle - E_0)/N$  [$10^{-5}$]'); c.set_title('(c) guide energy (Krylov sign)', loc='left', fontsize=9)
for f_ in ('png', 'pdf'):
    fig.savefig(os.path.join(R, f'itfit_6x6.{f_}'), dpi=200)
print('saved')
