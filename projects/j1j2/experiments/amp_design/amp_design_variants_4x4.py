#!/usr/bin/env python3
"""Follow-up to amp_design_4x4.py on the net-H frozen problem (k0, tab model): which estimator/regulariser
makes a SAMPLED second-order step work, and what replaces the exploitable validation gate?

Variants (all from N samples per step, 3 seeds, gain = exact frozen gain fraction):
  gn_lam      : (A_hat + lam*mean(diag) I) d = -g_hat, x ~ b^2                        (lam scan)
  rgn_mu      : (A_hat + mu*(trA/trS) S_hat + 1e-6 mean(diag) I) d = -g_hat (Webber-Lindsey RGN form)
  gn_pi       : as gn_lam but x ~ pi(x) = b_x (|F_off| b)_x  (edge-marginal: over-samples FN-wall configs),
                self-normalised weights b_x^2/pi(x)
  gn_temp     : x ~ b^{2 beta}, beta = 0.5, weights b^{2-2beta}
  sr_qstep    : SR direction from N/2 samples, step length t* = -g.d / d^T A d from the other N/2 (no gate)
  sr_oracle   : reference (exact line search), from amp_design_4x4
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, json, time, argparse
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import amp_design_4x4 as M

ap = argparse.ArgumentParser()
ap.add_argument('--state', default='k0')
ap.add_argument('--Ns', default='10000,100000')
ap.add_argument('--seeds', type=int, default=3)
ap.add_argument('--out', default=None)
ap.add_argument('--set', default='scan1')
ap.add_argument('--model', default='tab')
ap.add_argument('--rfK', type=int, default=1000)
args = ap.parse_args()

H, diag, ei, ej, hij = M.build_H(0.5)
E0, _ = M.ground(H, tol=1e-12)
st = np.load(M.ROOT / 'results/net_start_4x4/startH.npz')
a = np.asarray(st['a'], float); a /= np.linalg.norm(a); s = M.canonical(st['s'])
if args.state == 'k1':
    F0 = M.build_fn(diag, ei, ej, hij, a, s); _, phi = M.ground(F0, v0=a, tol=1e-11); phi = np.abs(phi) / np.linalg.norm(phi)
    s = M.krylov_update(H, diag, ei, ej, hij, phi, s); a = phi
fz = M.Frozen(M.build_fn(diag, ei, ej, hij, a, s), a)
oid = M.orbit_ids(); tab = M.tab_model(oid); no = tab.P
model = tab if args.model == 'tab' else M.rf_model(oid, args.rfK, 7)
def lsolve(Mt, g, lam):
    return model.J @ M.solve(model.lift(Mt), model.J.T @ g, lam)   # orbit-space update
Fabs_off = (-fz.Foff).tocsr()


def draw(q, n, rng, u):
    """n samples from q (unnormalised over D); self-normalised weights towards b^2."""
    p = q / q.sum(); smp = rng.choice(M.D, size=n, p=p)
    U, C = np.unique(smp, return_counts=True)
    b = fz.b(u); w = C * (b[U] ** 2) / p[U]; return U, w / w.sum()


def qdist(kind, u, beta=0.5):
    b = fz.b(u)
    if kind == 'b2': return b * b
    if kind == 'pi': return b * (Fabs_off @ b)
    if kind == 'temp': return b ** (2 * beta)
    raise ValueError(kind)


def gn(N, rng, lam=1e-3, mu=None, kind='b2', steps=2, beta=0.5):
    u = np.zeros(M.D); out = []
    for _ in range(steps):
        U, w = draw(qdist(kind, u, beta), N, rng, u)
        g, S, A, _ = M.tab_stats(fz, oid, u, U, w)
        Mx = A.copy()
        if mu is not None:
            Mx = Mx + mu * (np.trace(A) / max(np.trace(S), 1e-300)) * S
        d = lsolve(Mx, g, lam)
        u = u + d[oid]; out.append(fz.gain(u))
    return out


def sr_oracle(N, rng, lam=1e-3, steps=10, kind='b2', beta=0.5):
    u = np.zeros(M.D); out = []
    for st in range(steps):
        U, w = draw(qdist(kind, u, beta), N, rng, u)
        g, S, _, _ = M.tab_stats(fz, oid, u, U, w, edges=False)
        d = lsolve(S, g, lam)[oid]
        ts = np.geomspace(1e-8, 1e3, 120); es = [fz.energy(u + t * d) for t in ts]
        if min(es) < fz.energy(u): u = u + ts[int(np.argmin(es))] * d
        if st + 1 in (2, 4, 10): out.append(fz.gain(u))
    return out


def sr_qstep(N, rng, lam=1e-3, steps=10):
    u = np.zeros(M.D)
    for _ in range(steps):
        U1, w1 = draw(qdist('b2', u), N // 2, rng, u); U2, w2 = draw(qdist('b2', u), N - N // 2, rng, u)
        g1, S1, _, _ = M.tab_stats(fz, oid, u, U1, w1, edges=False)
        d = lsolve(S1, g1, lam)
        g2, _, A2, _ = M.tab_stats(fz, oid, u, U2, w2)
        curv = float(d @ A2 @ d); slope = float(g2 @ d)
        if curv <= 0 or slope >= 0: continue
        u = u + (-slope / curv) * d[oid]
    return fz.gain(u)


VSET = {
 'scan1': {
    'gn_lam1e-3': lambda N, r: gn(N, r, lam=1e-3),
    'gn_lam1e-2': lambda N, r: gn(N, r, lam=1e-2),
    'gn_lam1e-1': lambda N, r: gn(N, r, lam=1e-1),
    'gn_lam1': lambda N, r: gn(N, r, lam=1.0),
    'rgn_mu0.1': lambda N, r: gn(N, r, lam=1e-6, mu=0.1),
    'rgn_mu1': lambda N, r: gn(N, r, lam=1e-6, mu=1.0),
    'gn_pi_lam1e-2': lambda N, r: gn(N, r, lam=1e-2, kind='pi'),
    'gn_pi_lam1e-1': lambda N, r: gn(N, r, lam=1e-1, kind='pi'),
    'gn_temp_lam1e-2': lambda N, r: gn(N, r, lam=1e-2, kind='temp'),
    'sr_qstep': lambda N, r: [sr_qstep(N, r)]},
 'scan4': {
    **{f'gn_b{b}_lam{l:g}': (lambda b, l: (lambda N, r: gn(N, r, lam=l, kind='temp', beta=b, steps=4)))(b, l)
       for b in (0.25, 0.5) for l in (1e-6, 1e-5, 1e-4)},
    'sr_oracle_b0.5_lam1e-5': lambda N, r: sr_oracle(N, r, lam=1e-5, kind='temp', beta=0.5)},
 'scan3': {
    **{f'gn_b{b}_lam{l:g}': (lambda b, l: (lambda N, r: gn(N, r, lam=l, kind='temp', beta=b, steps=4)))(b, l)
       for b in (0.25, 0.5, 1.0) for l in (1e-3, 1e-2)},
    'sr_oracle_b2': lambda N, r: sr_oracle(N, r),
    'sr_oracle_b0.5': lambda N, r: sr_oracle(N, r, kind='temp', beta=0.5)},
 'scan2': {
    **{f'gn_b{b}_lam{l:g}': (lambda b, l: (lambda N, r: gn(N, r, lam=l, kind='temp', beta=b, steps=4)))(b, l)
       for b in (0.0, 0.25, 0.5, 0.75) for l in (1e-3, 1e-2, 1e-1)},
    'sr_oracle_b2': lambda N, r: sr_oracle(N, r),
    'sr_oracle_b0.5': lambda N, r: sr_oracle(N, r, kind='temp', beta=0.5)},
}
variants = VSET[args.set]
res = dict(state=args.state, G=fz.G, variants={})
t0 = time.time()
for N in [int(x) for x in args.Ns.split(',')]:
    for name, f in variants.items():
        vals = [f(N, np.random.default_rng([sd, N, 23])) for sd in range(args.seeds)]
        arr = np.array(vals)
        m = arr.mean(0).tolist(); sd = arr.std(0).tolist()
        res['variants'][f'{name}_N{N}'] = dict(mean=m, std=sd)
        print(f'{args.state} N={N:>7d} {name:16s} gain/step mean {np.round(m, 3)} std {np.round(sd, 3)}  {time.time()-t0:.0f}s', flush=True)
out = args.out or str(M.ROOT / f'results/amp_design/amp_design_variants_{args.state}_{args.set}_{args.model}.json')
Path(out).write_text(json.dumps(res, indent=1))
