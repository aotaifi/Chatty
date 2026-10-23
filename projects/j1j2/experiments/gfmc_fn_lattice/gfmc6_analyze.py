"""Analysis of the M1.1 6x6 GFMC chains.

  python gfmc6_analyze.py RUNROOT [--target 5.5738332e-06] [--burn_tau 10]
RUNROOT contains one directory per chain (blocks.npz, result.json from gfmc6_run.py; names s<seed>_M<M>_dt<dtau>).
Per chain: mixed and growth estimator, error from blocking (plateau of the last 4 levels), integrated autocorrelation time
(blocks and tau), block-mean drift in fifths (relaxation check, qmc-statistics).  Then, per seed pair at the reference point
(M_ref, dtau_ref): chains of the two seeds are independent (different init seed and RNG; nothing shared) and their difference is
tested.  Bias: linear fits of the mixed estimator in dtau (at M_ref) and 1/M (at dtau_ref).  Final line: the value at the reference
point, the extrapolation (dtau -> 0, M -> inf) and the 2-sigma comparison with the target (exact E_FN of A + 1 signed hop).
"""
import sys, os, json, glob, re
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gfmc_fn_lib import blocked_error, iat

root = sys.argv[1]; target = 5.5738332e-06; burn_tau = 10.0
if '--target' in sys.argv: target = float(sys.argv[sys.argv.index('--target') + 1])
if '--burn_tau' in sys.argv: burn_tau = float(sys.argv[sys.argv.index('--burn_tau') + 1])
N = 36
rows = []
for d in sorted(glob.glob(os.path.join(root, '*'))):
    f = os.path.join(d, 'blocks.npz')
    if not os.path.exists(f): continue
    z = np.load(f); dt = float(z['dtau']); M = int(z['M']); E0 = float(z['E0'])
    nb0 = int(round(burn_tau / dt))
    r = dict(name=os.path.basename(d), M=M, dtau=dt, nblocks=len(z['emix']), seed=int(re.search(r'_s(\d+)', d).group(1)) if re.search(r'_s(\d+)', d) else -1)
    for key in ('emix', 'egrow'):
        x = (z[key][nb0:] - E0) / N
        m, errs, tb = blocked_error(x)
        r[key] = dict(mean=m, err=max(errs[-4:]), err0=errs[0], iat_blocks=iat(x), iat_tau=iat(x) * dt)
        fifths = [float(np.mean(c)) for c in np.array_split(x, 5)]
        r[key]['fifths'] = fifths
    r['sec_block'] = float(np.median(z['sec'])); r['evals_block'] = float(np.mean(z['evals'])); r['ess_mean'] = float(np.mean(z['ess'][nb0:]))
    rows.append(r)
for r in rows:
    e = r['emix']
    print(f"{r['name']:28s} M={r['M']:6d} dtau={r['dtau']:.3f} nb={r['nblocks']:6d} dE_mix {e['mean']:.4e} +- {e['err']:.1e} (iat {e['iat_blocks']:.1f} blk = {e['iat_tau']:.2f} tau; "
          f"fifths drift {(e['fifths'][-1]-e['fifths'][0])/max(e['err'],1e-30):+.1f} sig)  grow {r['egrow']['mean']:.4e} +- {r['egrow']['err']:.1e}  s/block {r['sec_block']:.1f} ESS/M {r['ess_mean']/r['M']:.2f}")
json.dump(rows, open(os.path.join(root, 'analysis_chains.json'), 'w'), indent=1, default=float)
if not rows: sys.exit(0)
# bias fits (weighted linear), mixed estimator
def wfit(x, y, s):
    A = np.vstack([np.ones_like(x), x]).T; W = 1 / s ** 2
    cov = np.linalg.inv(A.T @ (A * W[:, None])); p = cov @ (A.T @ (W * y)); return p, cov
Mref = max(set(r['M'] for r in rows), key=lambda m: sum(1 for r in rows if r['M'] == m))
dref = max(set(r['dtau'] for r in rows if r['M'] == Mref), key=lambda d: sum(1 for r in rows if r['M'] == Mref and r['dtau'] == d))
def comb(sel):
    v = np.array([r['emix']['mean'] for r in sel]); s = np.array([r['emix']['err'] for r in sel]); w = 1 / s ** 2
    return float(np.sum(w * v) / w.sum()), float(1 / np.sqrt(w.sum()))
ref = [r for r in rows if r['M'] == Mref and r['dtau'] == dref]
m_ref, e_ref = comb(ref)
print(f'reference point M={Mref} dtau={dref}: {len(ref)} chains (seeds {[r["seed"] for r in ref]}) combined dE_mix {m_ref:.4e} +- {e_ref:.1e}')
if len(ref) >= 2:
    a, b = ref[0]['emix'], ref[1]['emix']
    print(f'seed difference {(a["mean"]-b["mean"]):.2e} = {(a["mean"]-b["mean"])/np.hypot(a["err"], b["err"]):+.1f} sigma')
res = dict(ref=dict(M=Mref, dtau=dref, mean=m_ref, err=e_ref), target=target)
dts = sorted(set(r['dtau'] for r in rows if r['M'] == Mref))
if len(dts) >= 2:
    xs = np.array(dts); ys, ss = zip(*[comb([r for r in rows if r['M'] == Mref and r['dtau'] == d]) for d in dts])
    p, cov = wfit(xs, np.array(ys), np.array(ss)); res['dtau_slope'] = (p[1], float(np.sqrt(cov[1, 1]))); res['dtau0'] = (p[0], float(np.sqrt(cov[0, 0])))
    print(f'dtau scan at M={Mref}: ' + ', '.join(f'{d}: {y:.3e}+-{s:.1e}' for d, y, s in zip(dts, ys, ss)) + f'; slope {p[1]:.2e}+-{np.sqrt(cov[1,1]):.1e} per tau, dtau->0 {p[0]:.4e}+-{np.sqrt(cov[0,0]):.1e}')
Ms = sorted(set(r['M'] for r in rows if r['dtau'] == dref))
if len(Ms) >= 2:
    xs = np.array([1.0 / m for m in Ms]); ys, ss = zip(*[comb([r for r in rows if r['dtau'] == dref and r['M'] == m]) for m in Ms])
    p, cov = wfit(xs, np.array(ys), np.array(ss)); res['invM_slope'] = (p[1], float(np.sqrt(cov[1, 1]))); res['invM0'] = (p[0], float(np.sqrt(cov[0, 0])))
    print(f'M scan at dtau={dref}: ' + ', '.join(f'{m}: {y:.3e}+-{s:.1e}' for m, y, s in zip(Ms, ys, ss)) + f'; slope {p[1]:.2e}+-{np.sqrt(cov[1,1]):.1e} (per 1/M), M->inf {p[0]:.4e}+-{np.sqrt(cov[0,0]):.1e}')
# final estimate: reference + (dtau->0 shift) + (M->inf shift); errors added in quadrature
est, err = m_ref, e_ref
if 'dtau0' in res and 'dtau_slope' in res:
    est += -res['dtau_slope'][0] * dref; err = float(np.hypot(err, res['dtau_slope'][1] * dref))
if 'invM0' in res:
    est += -res['invM_slope'][0] / Mref; err = float(np.hypot(err, res['invM_slope'][1] / Mref))
res['final'] = (est, err)
nsig = (est - target) / err
print(f'FINAL M1.1: extrapolated dE_FN {est:.4e} +- {err:.1e} vs target {target:.4e}: {nsig:+.2f} sigma -> ' + ('PASS (within 2 sigma)' if abs(nsig) <= 2 else 'FAIL'))
json.dump(res, open(os.path.join(root, 'analysis_final.json'), 'w'), indent=1, default=float)
