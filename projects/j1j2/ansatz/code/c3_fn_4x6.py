"""C3-FN on 4x6 (PLAN_ansatz.md "C3-FN"), after C3-T0-1.
  python3 c3_fn_4x6.py OUTDIR NPROC C3T0_1_JSON
Rebuilds the best state of each span of the C3-T0-1 run (S Ritz vector; V and R' seeds at the largest common rank)
from that run's json (greedy order, sector parameters, string order; R' factors replayed from the same seeds),
checks the rebuilt Ritz energies against the json, then computes lattice FN energies (ten Haaf; sign and guide
from the state), the Marshall-sign FN energy (guide |psi0|) and the FN energy of (1 + aH) psi_S (a by min E_var).
"""
import sys, os, json
OUT, NPROC, SRC = sys.argv[1], int(sys.argv[2]), sys.argv[3]
sys.argv = [sys.argv[0], OUT, str(NPROC)]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from multiprocessing import Pool
import c3_t0_4x6 as M                                   # module level: basis, H, psi0, Slater tables (no __main__)
from c3_t0_4x6 import log, H, psi0, E0, N, Dim, LX, LY, BONDS, NB, sN, energy
src = json.load(open(SRC))

if __name__ == '__main__':
    M.POOL = Pool(NPROC)
    res = dict(source=SRC)
    def dump(): json.dump(res, open(os.path.join(OUT, 'c3_fn_4x6.json'), 'w'), indent=1, default=str)
    MAPS = M.build_maps(); FLIP = M.LUT[M.basis ^ ((1 << N) - 1)]
    CHI = np.sign([float(psi0 @ psi0[m]) for m in MAPS]); chiF = float(psi0 @ psi0[FLIP])
    def symmetrise(v):
        w = np.zeros(Dim)
        for ch, m in zip(CHI, MAPS): w += ch * v[m]
        if abs(chiF) > 0.999: w = w + np.sign(chiF) * w[FLIP]
        return w
    # ---- strings, verbatim order from c3_t0_4x6.py
    def seg_cross(p1, p2, q1, q2):
        def orient(a, b, c): return np.sign((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        return orient(p1, p2, q1) * orient(p1, p2, q2) < 0 and orient(q1, q2, p1) * orient(q1, q2, p2) < 0
    EPS = (0.1, 0.03)
    def string_mask(dx, dy, xfirst=True):
        a = (0.5 + EPS[0], 0.5 + EPS[1]); b = (a[0] + dx, a[1]) if xfirst else (a[0], a[1] + dy); c = (a[0] + dx, a[1] + dy)
        mask = np.zeros(NB, int)
        for k, (i, j, kind, x, y, ddx, ddy) in enumerate(BONDS):
            n = 0
            for (p, q) in ((a, b), (b, c)):
                for mx in range(-3, 4):
                    for my in range(-3, 4):
                        if seg_cross((x + mx * LX, y + my * LY), (x + ddx + mx * LX, y + ddy + my * LY), p, q): n += 1
            mask[k] = n & 1
        return mask
    CLASSES = [(dx, dy) for dx in range(3) for dy in range(4) if (dx, dy) != (0, 0)]
    strings = []
    for (dx, dy) in CLASSES:
        if dx and dy: variants = [('xfirst', dx, dy, True), ('yfirst', dx, dy, False)]
        elif dx: variants = [('short', dx, 0, True), ('long', dx - LX, 0, True)]
        else: variants = [('short', 0, dy, False), ('long', 0, dy - LY, False)]
        for (nm, ddx, ddy, xf) in variants: strings.append(((dx, dy), nm, string_mask(ddx, ddy, xf)))
    best = {tuple(int(c) for c in k): v for k, v in src['single_by_bc'].items()}
    secs = [k for k in sorted(best) if best[k]['usable']]
    S_states = [M.state(best[k]['params'], k)[0] for k in secs]
    items = [(bc, cl, nm, m) for bc in secs for (cl, nm, m) in strings]
    for i, (bc, cl, nm, m) in enumerate(items):           # consistency with the stored metadata order
        mm = src['arm_V']['meta'][i]; assert list(bc) == mm[0] and list(cl) == mm[1] and nm == mm[2], (i, mm)
    def ritz_vec(vecs, cut=1e-10):
        Phi = np.array([v / np.linalg.norm(v) for v in vecs]).T
        S = Phi.T @ Phi; Hm = Phi.T @ (H @ Phi)
        s, U = np.linalg.eigh(S); keep = s > cut * s.max()
        X = U[:, keep] / np.sqrt(s[keep]); e, c = np.linalg.eigh(X.T @ Hm @ X)
        vg = Phi @ (X @ c[:, 0]); return int(keep.sum()), float((e[0] - E0) / N), vg / np.linalg.norm(vg)
    _C = H.tocoo(); _off = _C.row != _C.col
    _R, _Cc, _Hh = _C.row[_off].astype(np.int64), _C.col[_off].astype(np.int64), _C.data[_off]; _HD = H.diagonal(); del _C, _off
    FNINFO = {}
    def fn_energy(v, zero_rel=1e-14, tag=None):
        """lattice FN (ten Haaf), sign and guide from v. Configurations with |v| < zero_rel * max|v| lie on the node
        (infinite wall) and are removed. Lowest eigenvalue by LOBPCG with a Jacobi preconditioner (the sign-flip
        potential can be huge where |v| is tiny, which stalls plain Lanczos), started from s|v|."""
        from scipy.sparse.linalg import lobpcg
        s = np.where(v >= 0, 1.0, -1.0); g = np.abs(v); nz = g > zero_rel * g.max()
        sel = nz[_R] & nz[_Cc]
        r, c, h = _R[sel], _Cc[sel], _Hh[sel]
        viol = h * s[r] * s[c] > 0
        mp = -np.ones(v.size, np.int64); mp[nz] = np.arange(int(nz.sum())); n = int(nz.sum())
        vsf = np.bincount(r[viol], weights=h[viol] * s[r[viol]] * s[c[viol]] * g[c[viol]] / g[r[viol]], minlength=v.size)
        d = (_HD + vsf)[nz]
        HF = sp.csr_matrix((h[~viol], (mp[r[~viol]], mp[c[~viol]])), shape=(n, n)) + sp.diags(d)
        del r, c, h, viol
        x0 = (s * g)[nz]; x0 /= np.linalg.norm(x0); rq = float(x0 @ (HF @ x0))
        den = np.maximum(d - (rq - 1.0), 0.5)
        Mp = sla.LinearOperator((n, n), matvec=lambda x: (x.ravel() / den).reshape(x.shape), dtype=float)
        e, X = lobpcg(HF, x0[:, None], M=Mp, largest=False, tol=1e-9, maxiter=3000)
        x = X[:, 0] / np.linalg.norm(X[:, 0]); lam = float(x @ (HF @ x)); resid = float(np.linalg.norm(HF @ x - lam * x))
        info = dict(n_removed=int(v.size - n), dE_guide=(rq - E0) / N, resid=resid)
        if tag: FNINFO[tag] = info
        if resid > 1e-5: print('WARNING FN residual', tag, info, flush=True)
        return float((lam - E0) / N)
    arms = ['arm_V'] + [f'arm_R_seed{s}' for s in range(3)]
    kc = min(max(q['k'] for q in src[a]['curve']) for a in arms)
    res['common_rank'] = kc
    res['check_FN_exact_psi0'] = fn_energy(psi0, tag='psi0'); log('FN control psi0 (must be ~0):', res['check_FN_exact_psi0'])
    res['E_FN_marshall'] = fn_energy(sN * np.abs(psi0), tag='marshall'); log('E_FN Marshall', res['E_FN_marshall']); dump()
    kb = tuple(src['best_single']['bc']); v0 = S_states[secs.index(kb)]
    res['span_0'] = dict(bc=kb, dE_var=(energy(v0) - E0) / N, E_FN=fn_energy(v0, tag='0')); log('span 0', res['span_0'])
    rkS, dS, vS = ritz_vec(S_states)
    res['span_S'] = dict(rank=rkS, dE_var=dS, E_FN=fn_energy(vS, tag='S')); log('span S', res['span_S']); dump()
    HvS = H @ vS
    Phi2 = np.array([vS, HvS]).T; G2 = Phi2.T @ Phi2; H2 = Phi2.T @ (H @ Phi2)
    from scipy.linalg import eigh as geigh
    e2, c2 = geigh(H2, G2); wS = Phi2 @ c2[:, 0]
    res['span_S_1aH'] = dict(a=float(c2[1, 0] / c2[0, 0]), dE_var=float((e2[0] - E0) / N), E_FN=fn_energy(wS, tag='S_1aH'))
    log('span S (1+aH)', res['span_S_1aH']); dump()
    for a in arms:
        pre = [q for q in src[a]['curve'] if q['k'] <= kc][-1]
        upto = src[a]['curve'].index(pre)
        sel = [q['added'] for q in src[a]['curve'][1:upto + 1]]
        if a == 'arm_V':
            facs = [1.0 - 2.0 * m for (_, _, _, m) in items]
        else:
            rng = np.random.default_rng(100 + int(a[-1]))
            facs = [np.where(m == 1, rng.uniform(0.5, 1.5, NB), 1.0) for (_, _, _, m) in items]
        vecs = list(S_states)
        for idx in sel:
            bc = items[idx][0]
            v, _ = M.state(best[bc]['params'], bc, fac=facs[idx]); w = symmetrise(v); vecs.append(w / np.linalg.norm(w))
        rk, dE, vg = ritz_vec(vecs)
        res[a] = dict(rank=rk, dE_var=dE, dE_var_stored=pre['dE_site'], E_FN=fn_energy(vg, tag=a))
        log(a, res[a]); dump()
        assert rk == pre['k'] and abs(dE - pre['dE_site']) < 1e-8, ('rebuild mismatch', a, rk, dE, pre)
    res['fn_solver_info'] = FNINFO
    eV = res['arm_V']['E_FN']; eR = [res[f'arm_R_seed{s}']['E_FN'] for s in range(3)]
    if all(eV <= 0.3 * e for e in eR) and eV <= 0.3 * res['E_FN_marshall']: verdict = 'PASS'
    elif sum(eV >= e for e in eR) >= 2: verdict = 'KILL'
    else: verdict = 'INCONCLUSIVE'
    res['verdict'] = dict(E_FN_V=eV, E_FN_R=eR, E_FN_marshall=res['E_FN_marshall'], verdict=verdict,
                          rule='PASS if E_FN(V) <= 0.3 E_FN(R) all seeds and <= 0.3 E_FN(Marshall); KILL if E_FN(V) >= E_FN(R) for >= 2 of 3 seeds')
    dump()
    log('VERDICT C3-FN 4x6', res['verdict'])
