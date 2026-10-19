"""Figure for results/deliverable_C: (a) 8x8 energy vs variance (p = 0, 1, two-point extrapolation) with literature and
the FN-DMC bound; (b) exact 6x6 calibration of the same extrapolation (p = 0, 1, 2 optimal and repeated step).
  python dc_figure.py RESULTS_DIR
"""
import sys, os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = sys.argv[1]
ex = json.load(open(os.path.join(R, 'extrap_p01.json')))
p6 = json.load(open(os.path.join(R, 'p2rep_6x6.json')))
fnp = os.path.join(R, 'fn8_analysis.json'); fn = json.load(open(fnp)) if os.path.exists(fnp) else None
plt.rcParams.update({'font.size': 9, 'axes.linewidth': 0.6})
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))

a = ax[0]; S = 1e4; Y = 1e5; E00 = -0.4988                       # x in 1e-4, y = (E/N - E00) in 1e-5
v = np.array([ex['var0'], ex['var1']]); e = np.array([ex['E0'], ex['E1']])
ve = np.array([ex['var0_boot_se'], ex['var1_boot_se']]); ee = np.array([ex['E0_boot_se'], ex['E1_boot_se']])
xx = np.linspace(0, v[0] * 1.05, 50)
a.plot(xx * S, (ex['E_extrap'] + ex['slope'] * xx - E00) * Y, '-', lw=0.7, color='C0')
a.errorbar(v * S, (e - E00) * Y, xerr=ve * S, yerr=ee * Y, fmt='o', ms=4, color='C0', lw=0.8, capsize=2, label='ViT, Lanczos p = 0, 1')
a.errorbar([0], [(ex['E_extrap'] - E00) * Y], yerr=[np.hypot(ex['E_extrap_boot_se'], 8e-6) * Y], fmt='s', ms=4, mfc='white',
           color='C0', lw=0.8, capsize=2, label='variance extrapolation')
for val, err, lab, c, ls in ((-0.498886, 1e-6, 'RBM+PP (Nomura-Imada)', 'C2', '--'), (-0.49906, 1e-5, 'Hu et al. 2013, extrap.', 'C3', ':')):
    a.axhspan((val - err - E00) * Y, (val + err - E00) * Y, color=c, alpha=0.25, lw=0)
    a.axhline((val - E00) * Y, color=c, ls=ls, lw=0.8, label=lab)
if fn:
    q = fn['quoted']; a.errorbar([2.5], [(q['E'] - E00) * Y], yerr=[q['se'] * Y], fmt='D', ms=4, color='C1', lw=0.8, capsize=2,
                                 label='FN-DMC, ViT guide')
a.set_xlabel(r'variance $\sigma^2/N$  ($10^{-4}$)'); a.set_ylabel(r'$E/N + 0.4988$  ($10^{-5}$)')
a.set_title('(a) 8x8', loc='left', fontsize=9); a.legend(fontsize=6.5, frameon=False, loc='center right')

b = ax[1]; E0 = -0.50380965
pts = [('p0', 'o', 'C0'), ('p1', 'o', 'C0'), ('p2_opt', 'o', 'C0'), ('p2_repeated', '^', 'C1')]
for k, m, c in pts:
    b.plot(p6[k]['var_site'] * S, p6[k]['dE_site'] * Y, m, ms=4, color=c, mfc='white' if k == 'p2_repeated' else c)
vv = np.array([p6['p0']['var_site'], p6['p1']['var_site']]); dd = np.array([p6['p0']['dE_site'], p6['p1']['dE_site']])
sl = (dd[0] - dd[1]) / (vv[0] - vv[1]); x2 = np.linspace(0, vv[0] * 1.05, 50)
b.plot(x2 * S, (dd[1] + sl * (x2 - vv[1])) * Y, '-', lw=0.7, color='C0', label='two-point line (p = 0, 1)')
b.axhline(0, color='k', lw=0.6, label='exact E0')
b.plot([], [], 'o', ms=4, color='C0', label='Lanczos p = 0, 1, 2 (optimal)')
b.plot([], [], '^', ms=4, color='C1', mfc='white', label='p = 2, repeated step')
b.set_xlabel(r'variance $\sigma^2/N$  ($10^{-4}$)'); b.set_ylabel(r'$(E - E_0)/N$  ($10^{-5}$)')
b.set_title('(b) 6x6, exact (calibration)', loc='left', fontsize=9); b.legend(fontsize=6.5, frameon=False, loc='upper left')
fig.tight_layout()
for ext in ('png', 'pdf'): fig.savefig(os.path.join(R, f'fig_deliverableC_8x8.{ext}'), dpi=200)
