"""Amendment L-1: error of E_L(x^b) of psi_P under the light-cone (exact within rho, additive beyond), 8x8 two-hop data.
  python s8_elcone.py OUT.json RUN_DIR
"""
import sys, os, glob, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s8_pairint_geom import BI, BJ, MASKS, JB, DB, NB, ONE
OUT, RUN = sys.argv[1], sys.argv[2]
RHOS = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
diag0 = 0.25 * JB.sum()
acc = {r: [0.0, 0.0, 0.0] for r in RHOS}; ELx_all = []; T0 = time.time(); n = 0
for f in sorted(glob.glob(os.path.join(RUN, 'chunk_*.npz'))):
    d = np.load(f); U2 = d['U2']; la = d['la']; s = d['s'].astype(np.float64)
    look = lambda c: np.clip(np.searchsorted(U2, c), 0, len(U2) - 1)
    for x in d['X']:
        v = (((x >> BI) ^ (x >> BJ)) & ONE).astype(bool)
        ix = look(np.array([x], np.uint64))[0]; lx, sx = la[ix], s[ix]
        ib = look(x ^ MASKS); lb, sb = la[ib], s[ib]
        ELx_all.append(diag0 - 0.5 * JB[v].sum() + np.sum(np.where(v, 0.5 * JB * sx * sb * np.exp(np.clip(lb - lx, -60, 60)), 0)))
        bs = np.nonzero(v)[0]
        xb = (x ^ MASKS[bs])[:, None]                                   # (nb, 1)
        vy = (((xb >> BI[None]) ^ (xb >> BJ[None])) & ONE).astype(bool)  # valid bonds of y = x^b
        ibb = look((xb ^ MASKS[None]).reshape(-1)).reshape(len(bs), NB)
        lyy, syy = la[ibb], s[ibb]
        ly, sy = lb[bs][:, None], sb[bs][:, None]
        ex = 0.5 * JB[None] * sy * syy * np.exp(np.clip(lyy - ly, -60, 60))
        lad = lb[None, :] + lb[bs][:, None] - lx; sad = sb[None, :] * sb[bs][:, None] * sx
        ad = 0.5 * JB[None] * sy * sad * np.exp(np.clip(lad - ly, -60, 60))
        disj = DB[bs] > 0
        wb = 0.5 * JB[bs] * np.exp(np.clip(lb[bs] - lx, -60, 60))
        for r in RHOS:
            use_add = disj & (DB[bs] > r + 1e-6) & vy
            err = np.sum(np.where(use_add, ad - ex, 0.0), 1)
            acc[r][0] += np.sum(err ** 2); acc[r][1] += np.sum(wb * err ** 2); acc[r][2] += np.sum(wb)
        n += len(bs)
    print(os.path.basename(f), n, f'{time.time() - T0:.0f}s', flush=True)
sdE = float(np.std(ELx_all))
res = dict(run=RUN, n_pairs_xb=n, sd_EL=sdE, levels=[dict(rho=r, rms=float(np.sqrt(a[0] / n)), wrms=float(np.sqrt(a[1] / a[2])),
                                                         rms_over_sd=float(np.sqrt(a[0] / n) / sdE), wrms_over_sd=float(np.sqrt(a[1] / a[2]) / sdE))
                                                    for r, a in acc.items()])
json.dump(res, open(OUT, 'w'), indent=1); print(json.dumps(res, indent=1)); print('DONE')
