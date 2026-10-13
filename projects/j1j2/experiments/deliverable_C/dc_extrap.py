"""Deliverable C: variance extrapolation of the 8x8 Lanczos sequence p = 0, 1 (and 2 when available).

Inputs are the raw per-sample local energies already on ws1 (copied locally):
  p = 0: plain VMC of psi_P  (l8vmc_17138725/vmc.npz: EL per sample, chain id)              -> E_0, var_0
  p = 1: direct VMC of psi_1 (l8dira direct.npz, rvD1-3 direct2.npz, rvL2-3 late.npz)       -> E_1, var_1
         E_L1 = ((1 - a Er) E_L + a E_L2) / (1 + a (E_L - Er)), a = -0.160951, Er = -31.92677568
var_p = <E_L^2> - <E_L>^2 under |psi_p|^2 = <psi_p|H^2|psi_p> - E_p^2 (exact identity for the local-energy variance).
Extrapolation: E = E_inf + A var (linear, Hu-Becca-Parola-Sorella 2013).  Per-site units: E/N and var/N.
Errors: chain bootstrap (resample chains inside each dataset), plus heavy-tail sensitivity (top-k trimming).
  python dc_extrap.py DATA_DIR OUT.json
"""
import sys, json, os
import numpy as np

N = 64; AL = -0.160951; ER = -31.92677568
D_, OUT = sys.argv[1], sys.argv[2]


def el1(z):
    return ((1 - AL * ER) * z['EL'] + AL * z['EL2']) / (1 + AL * (z['EL'] - ER))


v = np.load(os.path.join(D_, 'vmc', 'vmc.npz')); E0 = v['EL']; c0 = v['chain']
parts = []
z = np.load(os.path.join(D_, 'dira', 'direct.npz')); parts.append((el1(z), z['chain'], 'run_a'))
for k in (1, 2, 3):
    z = np.load(os.path.join(D_, f'rvD{k}', 'direct2.npz')); parts.append((el1(z), z['chain'] + 10000 * k, f'D{k}'))
for k in (2, 3):
    z = np.load(os.path.join(D_, f'rvL{k}', 'late.npz')); ok = np.isfinite(z['EL2'])
    parts.append((el1(z)[ok], z['chain'][ok] + 10000 * k, f'L{k}'))
E1 = np.concatenate([p[0] for p in parts]); c1 = np.concatenate([p[1] for p in parts])


def ev(E):
    return E.mean() / N, E.var() / N


def extrap(e0, v0, e1, v1):
    A = (e0 - e1) / (v0 - v1)
    return e1 - A * v1, A


def groups(E, c):
    ids, inv = np.unique(c, return_inverse=True)
    return [E[inv == i] for i in range(len(ids))]


rg = np.random.default_rng(7)
g0 = groups(E0, c0); g1 = groups(E1, c1)


def boot(gs, B):
    out = []
    n = len(gs)
    for _ in range(B):
        idx = rg.integers(n, size=n)
        out.append(np.concatenate([gs[i] for i in idx]))
    return out


B = 1000
b0 = [ev(E) for E in boot(g0, B)]; b1 = [ev(E) for E in boot(g1, B)]
e0, v0 = ev(E0); e1, v1 = ev(E1); Ex, A = extrap(e0, v0, e1, v1)
bx = np.array([extrap(*b0[i], *b1[i])[0] for i in range(B)])
res = dict(E0=e0, var0=v0, E1=e1, var1=v1, n0=len(E0), n1=len(E1), nch0=len(g0), nch1=len(g1),
           slope=A, E_extrap=Ex, E_extrap_boot_se=float(bx.std()), E_extrap_boot_q=[float(q) for q in np.quantile(bx, [.025, .16, .5, .84, .975])],
           var0_boot_se=float(np.std([b[1] for b in b0])), var1_boot_se=float(np.std([b[1] for b in b1])),
           E0_boot_se=float(np.std([b[0] for b in b0])), E1_boot_se=float(np.std([b[0] for b in b1])),
           ratio_var1_var0=v1 / v0)
# heavy-tail sensitivity: drop the k most extreme samples (both ends, by |E - median|) from each set
sens = {}
for k in (0, 5, 10, 30, 100):
    def trim(E):
        if k == 0: return E
        o = np.argsort(np.abs(E - np.median(E)))
        return E[o[:-k]]
    a = ev(trim(E0)); b = ev(trim(E1)); sens[k] = dict(E0=a[0], var0=a[1], E1=b[0], var1=b[1], E_extrap=extrap(*a, *b)[0])
res['trim_sensitivity'] = sens
# alternative psi_P variance estimator check is done on referee data (<(H^2 psi)/psi> - h1^2), see README
json.dump(res, open(OUT, 'w'), indent=1, default=float)
print(json.dumps(res, indent=1, default=float))
