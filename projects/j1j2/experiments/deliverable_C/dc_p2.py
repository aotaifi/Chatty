"""Deliverable C, E3: three-hop local data for the p = 2 Lanczos state (pilot P2-0, PREREG.md).

For a configuration x and a guide psi (log a, sign) we evaluate psi on the three-hop ball B3(x) and return, with
A = H - E_ref and a_k(z) = (A^k psi)(z)/psi(z):
  a1, a2, a3 at x;  a1, a2 on the one-hop neighbours (for diagnostics).
Recursion (exact): a1(z) = D'(z) + sum_w h(z,w),  a_{k+1}(z) = D'(z) a_k(z) + sum_{w in N(z)} h(z,w) a_k(w),
  D' = H_zz - E_ref, h(z,w) = J/2 s(z) s(w) a(w)/a(z).
States:  psi_1 = psi f, f = 1 + alpha a1;   psi_2 = psi g, g = 1 + b1 a1 + b2 a2;
  E_L[psi_1](x) = E_ref + (a1 + alpha a2)/f;   E_L[psi_2](x) = E_ref + (a1 + b1 a2 + b2 a3)/g.

  pilot:  python dc_p2.py OUT SPEC.json   SPEC: samples [npz paths with X, chain, sweep], sweep, n, seed, alpha, Eref, phi, batch
"""
import os, sys, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dc_fn as F


def threehop(lat, guide, x, Er):
    x = np.asarray([x], np.uint64)
    nb0, v0 = lat.neighbors(x); Y1 = np.unique(np.concatenate([x, nb0[v0]]))
    nb1, v1 = lat.neighbors(Y1); Y2 = np.unique(np.concatenate([Y1, nb1[v1]]))
    nb2, v2 = lat.neighbors(Y2); Y3 = np.unique(np.concatenate([Y2, nb2[v2]]))
    la, s = guide(Y3)
    i2 = np.searchsorted(Y3, Y2)

    def level(Z, nbZ, vZ):
        """D' and the (row, col-in-Y3, h) lists for the configurations Z (all neighbours of Z are in Y3)."""
        iz = np.searchsorted(Y3, Z)
        o, b = np.nonzero(vZ); w = np.searchsorted(Y3, nbZ[o, b])
        h = 0.5 * lat.JB[b] * s[iz][o] * s[w] * np.exp(np.clip(la[w] - la[iz][o], -60, 60))
        return lat.diag(vZ) - Er, o, w, h
    # a1 on Y2 (needs psi on Y3)
    Dp2, o2, w2, h2 = level(Y2, nb2, v2)
    a1_Y2 = Dp2 + np.bincount(o2, weights=h2, minlength=len(Y2))
    a1_Y3 = np.full(len(Y3), np.nan); a1_Y3[i2] = a1_Y2
    # a2 on Y1 (needs a1 on Y2)
    Dp1, o1, w1, h1 = level(Y1, nb1, v1)
    a1_Y1 = a1_Y3[np.searchsorted(Y3, Y1)]
    a2_Y1 = Dp1 * a1_Y1 + np.bincount(o1, weights=h1 * a1_Y3[w1], minlength=len(Y1))
    a2_Y3 = np.full(len(Y3), np.nan); a2_Y3[np.searchsorted(Y3, Y1)] = a2_Y1
    # a3 at x (needs a2 on Y1)
    Dp0, o0, w0, h0 = level(x, nb0, v0)
    a1x = a1_Y3[np.searchsorted(Y3, x)][0]; a2x = a2_Y3[np.searchsorted(Y3, x)][0]
    a3x = Dp0[0] * a2x + np.sum(h0 * a2_Y3[w0])
    ix1 = np.searchsorted(Y1, x)[0]
    return dict(a1=a1x, a2=a2x, a3=a3x, nb_a1=np.delete(a1_Y1, ix1), nb_a2=np.delete(a2_Y1, ix1),
                n1=len(Y1), n2=len(Y2), n3=len(Y3))


def ritz3(a1, a2, a3, w, mode='balanced'):
    """3x3 Ritz in span{psi, A psi, A^2 psi} from samples with weights w ~ psi^2/pi.  mu_k = <A^k>.  Returns the lowest
    eigenvalue shift (relative to E_ref) and the coefficients (1, b1, b2)."""
    a = [np.ones_like(a1), a1, a2, a3]
    w = w / w.sum()
    if mode == 'balanced':
        pr = {0: (0, 0), 1: (0, 1), 2: (1, 1), 3: (1, 2), 4: (2, 2), 5: (2, 3)}
    else:
        pr = {0: (0, 0), 1: (0, 1), 2: (0, 2), 3: (0, 3), 4: (1, 3), 5: (2, 3)}
    mu = {k: float(np.sum(w * a[i] * a[j])) for k, (i, j) in pr.items()}
    S = np.array([[mu[i + j] for j in range(3)] for i in range(3)])
    Hm = np.array([[mu[i + j + 1] for j in range(3)] for i in range(3)])
    from scipy.linalg import eigh
    ev, vec = eigh(Hm, S)
    c = vec[:, 0] / vec[0, 0]
    return float(ev[0]), float(c[1]), float(c[2]), mu


def main():
    OUT = sys.argv[1]; SPEC = json.load(open(sys.argv[2])); os.makedirs(OUT, exist_ok=True); T0 = time.time()
    sys.path.insert(0, os.path.join(HERE, '..', 'loop_8x8'))
    import l8_core as C
    g = C.Guide('vit_J2=0.50_N=8x8_k=0.mpack', sym=True, batch=SPEC.get('batch', 1024), phi=SPEC['phi'])
    guide = lambda S: (lambda la, ph: (la, g.sign(ph)))(*g.eval(S))
    lat = F.Lattice(8); al = SPEC['alpha']; Er = SPEC['Eref']
    rg = np.random.default_rng(SPEC['seed'])
    Xs = []
    for k, p in enumerate(SPEC['samples']):
        z = np.load(p); m = z['sweep'] == SPEC['sweep']; Xs.append(z['X'][m])
    X = np.concatenate(Xs); X = X[rg.choice(len(X), SPEC['n'], replace=False)]
    rows = []
    for i, x in enumerate(X):
        t = time.time(); n0 = g.neval
        d = threehop(lat, guide, x, Er)
        rows.append(dict(a1=d['a1'], a2=d['a2'], a3=d['a3'], n3=d['n3'], sec=time.time() - t, evals=g.neval - n0))
        C.log(f'{i + 1}/{len(X)}', {k: v for k, v in rows[-1].items()})
        np.savez(os.path.join(OUT, 'p2.npz'), X=X[:i + 1], **{k: np.array([r[k] for r in rows]) for k in rows[0]})
    a1 = np.array([r['a1'] for r in rows]); a2 = np.array([r['a2'] for r in rows]); a3 = np.array([r['a3'] for r in rows])
    f = 1 + al * a1; w = 1.0 / f ** 2                      # samples ~ psi_1^2 -> psi_P^2 weights
    EL1 = Er + (a1 + al * a2) / f
    rb = ritz3(a1, a2, a3, w, 'balanced'); ru = ritz3(a1, a2, a3, w, 'unbalanced')
    res = dict(spec=SPEC, n=len(rows), sec_per_sample=float(np.mean([r['sec'] for r in rows])),
               evals_per_sample=float(np.mean([r['evals'] for r in rows])), n3_mean=float(np.mean([r['n3'] for r in rows])),
               E1_check_site=float(EL1.mean() / 64), ritz_balanced=rb[:3], ritz_unbalanced=ru[:3], mu_balanced=rb[3], mu_unbalanced=ru[3],
               repeated_step=(2 * al, al * al), sec=time.time() - T0)
    for name, (b1, b2) in (('balanced', rb[1:3]), ('unbalanced', ru[1:3]), ('repeated', (2 * al, al * al))):
        gg = 1 + b1 * a1 + b2 * a2; EL2 = Er + (a1 + b1 * a2 + b2 * a3) / gg
        res['psi2_' + name] = dict(b1=b1, b2=b2, sd_EL2_site=float(EL2.std() / 64), sd_EL1_site=float(EL1.std() / 64),
                                   sd_diff_site=float((EL2 - EL1).std() / 64), mean_diff_site_unweighted=float((EL2 - EL1).mean() / 64),
                                   g_over_f_range=[float((gg / f).min()), float((gg / f).max())])
    C.dump(os.path.join(OUT, 'p2.json'), res); C.log('DONE', json.dumps(res, default=float))


if __name__ == '__main__':
    main()
