"""C3-T0-2 (PLAN_ansatz.md, "Amendments after the adversarial review"): 4x6, repaired base.
  python3 c3_t0_2_4x6.py OUTDIR NPROC
Converge the AA base (Nelder-Mead on |<psi0|psi_P>|^2, <= 400 evaluations, from the C3-T0-1 point); (S) = the 4 BC
sector states at that point + the 3 symmetrised parameter-derivative states of the AA base (central differences in
D1, D2, mu, h = 1e-3); then V (vison-pair states, 22 strings x 4 sectors) vs R' (same strings, factors in [0.5, 1.5],
seeds 100..102 as in C3-T0-1), greedy by Ritz-energy gain taking only candidates that add a direction (S cut 1e-10),
to the largest common rank. Rule as C3-T0-1. C3-FN (ten Haaf, LOBPCG as in c3_fn_4x6.py) on the final spans.
"""
import sys, os, json
OUT, NPROC = sys.argv[1], int(sys.argv[2])
sys.argv = [sys.argv[0], OUT, str(NPROC)]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.optimize import minimize
from scipy.linalg import eigh as geigh
from multiprocessing import Pool
import c3_t0_4x6 as M
from c3_t0_4x6 import log, H, psi0, E0, N, Dim, LX, LY, BONDS, NB, sN, energy, WM

if __name__ == '__main__':
    M.POOL = Pool(NPROC)
    res = {}
    def dump(): json.dump(res, open(os.path.join(OUT, 'c3_t0_2.json'), 'w'), indent=1, default=str)
    MAPS = M.build_maps(); FLIP = M.LUT[M.basis ^ ((1 << N) - 1)]
    CHI = np.sign([float(psi0 @ psi0[m]) for m in MAPS]); chiF = float(psi0 @ psi0[FLIP])
    def symmetrise(v):
        w = np.zeros(Dim)
        for ch, m in zip(CHI, MAPS): w += ch * v[m]
        if abs(chiF) > 0.999: w = w + np.sign(chiF) * w[FLIP]
        return w
    # ---------------- 1. converge the AA base
    NEV = [0]
    def negov(p):
        v, _ = M.state(p, (1, 1)); NEV[0] += 1
        ov = 0.0 if v is None else float((psi0 @ v) ** 2)
        if NEV[0] % 10 == 0: log('PROGRESS NM eval', NEV[0], 'p', [round(q, 6) for q in p], 'overlap2', ov)
        return -ov
    p_start = (1.1946737126021922, 0.525530157915381, 0.0014595012985779035)
    r = minimize(negov, p_start, method='Nelder-Mead', options=dict(xatol=1e-5, fatol=1e-9, maxfev=400,
                 initial_simplex=np.array(p_start) + np.array([[0, 0, 0], [0.15, 0, 0], [0, 0.1, 0], [0, 0, 0.05]])))
    popt = [float(q) for q in r.x]
    res['AA_base'] = dict(start=p_start, overlap2_start=None, overlap2=float(-r.fun), params=popt, nfev=int(r.nfev),
                          message=str(r.message))
    log('AA base converged', res['AA_base']); dump()
    secs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    S_states, single = [], {}
    for bc in secs:
        v, gap = M.state(popt, bc)
        if v is None: log('sector', bc, 'ill-defined, skipped'); continue
        single[f'{bc[0]}{bc[1]}'] = dict(overlap2=float((psi0 @ v) ** 2), E_site=energy(v) / N, gap=gap, W_M=WM(v))
        S_states.append(v)
    res['single_by_bc'] = single
    kb = min(single, key=lambda k: single[k]['E_site'])
    res['best_single'] = dict(bc=kb, E_site=single[kb]['E_site'], dE_site=single[kb]['E_site'] - E0 / N)
    log('sectors at the base', single)
    vAA, _ = M.state(popt, (1, 1)); h = 1e-3; derivs = []
    for i, nm in enumerate(('D1', 'D2', 'mu')):
        pp = list(popt); pm = list(popt); pp[i] += h; pm[i] -= h
        vp, _ = M.state(pp, (1, 1)); vm, _ = M.state(pm, (1, 1))
        vp *= np.sign(vp @ vAA); vm *= np.sign(vm @ vAA)               # fix the overall sign gauge
        dv = symmetrise((vp - vm) / (2 * h)); derivs.append(dv / np.linalg.norm(dv))
        log('derivative state', nm, 'norm before symmetrisation', float(np.linalg.norm((vp - vm) / (2 * h))))
    S_states = S_states + derivs
    # ---------------- 2. strings (verbatim from c3_t0_4x6.py)
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
    items = [(bc, cl, nm, m) for bc in secs for (cl, nm, m) in strings]
    # ---------------- 3. arms and greedy (new directions only)
    def build(fac_fn, label):
        out = []
        for i, (bc, cl, nm, m) in enumerate(items):
            v, gap = M.state(popt, bc, fac=fac_fn(m))
            if (i + 1) % 11 == 0: log('PROGRESS', label, 'state', i + 1, '/', len(items))
            if v is None: continue
            w = symmetrise(v); n = np.linalg.norm(w)
            if n > 1e-8: out.append(w / n)
        return out
    def greedy(cands, label):
        Phi = S_states + cands; n = len(Phi); nS = len(S_states)
        Gm = np.empty((n, n)); Hm = np.empty((n, n))
        for i, v in enumerate(Phi):
            hv = H @ v
            for j in range(i + 1):
                Gm[i, j] = Gm[j, i] = Phi[j] @ v; Hm[i, j] = Hm[j, i] = Phi[j] @ hv
        def ritz(idx, cut=1e-10):
            s, U = np.linalg.eigh(Gm[np.ix_(idx, idx)]); keep = s > cut * s.max()
            X = U[:, keep] / np.sqrt(s[keep]); e, c = np.linalg.eigh(X.T @ Hm[np.ix_(idx, idx)] @ X)
            return int(keep.sum()), float((e[0] - E0) / N), X @ c[:, 0]
        cur = list(range(nS)); rk, dE, _ = ritz(cur); curve = [(rk, dE, list(cur))]; rest = list(range(nS, n))
        while rest:
            tr = [(ritz(cur + [c])[:2], c) for c in rest]
            tr = [q for q in tr if q[0][0] > rk]
            if not tr: break
            (rk, dE), c = min(tr, key=lambda q: q[0][1]); cur.append(c); rest.remove(c); curve.append((rk, dE, list(cur)))
        log(label, 'candidates', len(cands), 'S rank', curve[0][0], 'final rank', rk, 'curve',
            [(q[0], f'{q[1]:.3e}') for q in curve])
        def vec_at(k):
            sel = [q for q in curve if q[0] <= k][-1]
            rk_, dE_, coef = ritz(sel[2]); v = sum(cf * Phi[i] for cf, i in zip(coef, sel[2]))
            return rk_, dE_, v / np.linalg.norm(v)
        return curve, vec_at
    arms = {}
    arms['V'] = greedy(build(lambda m: 1.0 - 2.0 * m, 'V'), 'V')
    res['curves'] = {'V': [(q[0], q[1]) for q in arms['V'][0]]}; dump()
    for sd in range(3):
        rng = np.random.default_rng(100 + sd)
        arms[f'R{sd}'] = greedy(build(lambda m, rng=rng: np.where(m == 1, rng.uniform(0.5, 1.5, NB), 1.0), f"R' seed{sd}"),
                                f"R' seed{sd}")
        res['curves'][f'R{sd}'] = [(q[0], q[1]) for q in arms[f'R{sd}'][0]]; dump()
    kc = min(arms[a][0][-1][0] for a in arms)
    res['S_rank'] = arms['V'][0][0][0]; res['S_dE'] = arms['V'][0][0][1]; res['common_rank'] = kc
    at = {a: [q for q in arms[a][0] if q[0] <= kc][-1] for a in arms}
    res['dE_at_common_rank'] = {a: (at[a][0], at[a][1]) for a in arms}
    rV = at['V'][1]; rR = [at[f'R{s}'][1] for s in range(3)]; ratios = [rV / x for x in rR]
    if all(x <= 0.3 for x in ratios): verdict = 'PASS'
    elif sum(x >= 0.7 for x in ratios) >= 2: verdict = 'KILL'
    else: verdict = 'INCONCLUSIVE'
    res['verdict'] = dict(k=kc, V=rV, R=rR, ratios=ratios, verdict=verdict,
                          rule='PASS if V <= 0.3 R for all seeds; KILL if V >= 0.7 R for >= 2 of 3 seeds')
    log('C3-T0-2 result', res['verdict']); dump()
    # ---------------- 4. C3-FN on the final spans (solver as c3_fn_4x6.py)
    _C = H.tocoo(); _off = _C.row != _C.col
    _R, _Cc, _Hh = _C.row[_off].astype(np.int64), _C.col[_off].astype(np.int64), _C.data[_off]; _HD = H.diagonal(); del _C, _off
    FNINFO = {}
    def fn_energy(v, zero_rel=1e-14, tag=None):
        from scipy.sparse.linalg import lobpcg
        s = np.where(v >= 0, 1.0, -1.0); g = np.abs(v); nz = g > zero_rel * g.max()
        sel = nz[_R] & nz[_Cc]; r, c, h_ = _R[sel], _Cc[sel], _Hh[sel]
        viol = h_ * s[r] * s[c] > 0
        mp = -np.ones(v.size, np.int64); mp[nz] = np.arange(int(nz.sum())); n = int(nz.sum())
        vsf = np.bincount(r[viol], weights=h_[viol] * s[r[viol]] * s[c[viol]] * g[c[viol]] / g[r[viol]], minlength=v.size)
        d = (_HD + vsf)[nz]
        HF = sp.csr_matrix((h_[~viol], (mp[r[~viol]], mp[c[~viol]])), shape=(n, n)) + sp.diags(d)
        del r, c, h_, viol
        x0 = (s * g)[nz]; x0 /= np.linalg.norm(x0); rq = float(x0 @ (HF @ x0))
        den = np.maximum(d - (rq - 1.0), 0.5)
        Mp = sla.LinearOperator((n, n), matvec=lambda x: (x.ravel() / den).reshape(x.shape), dtype=float)
        e, X = lobpcg(HF, x0[:, None], M=Mp, largest=False, tol=1e-9, maxiter=3000)
        x = X[:, 0] / np.linalg.norm(X[:, 0]); lam = float(x @ (HF @ x)); resid = float(np.linalg.norm(HF @ x - lam * x))
        FNINFO[tag] = dict(n_removed=int(v.size - n), resid=resid)
        return float((lam - E0) / N)
    fn = dict(check_psi0=fn_energy(psi0, tag='psi0'), marshall=fn_energy(sN * np.abs(psi0), tag='marshall'))
    rk, dE, vS = arms['V'][1](res['S_rank']); fn['S'] = dict(rank=rk, dE_var=dE, E_FN=fn_energy(vS, tag='S'))
    HvS = H @ vS; Phi2 = np.array([vS, HvS]).T
    e2, c2 = geigh(Phi2.T @ (H @ Phi2), Phi2.T @ Phi2); wS = Phi2 @ c2[:, 0]
    fn['S_1aH'] = dict(a=float(c2[1, 0] / c2[0, 0]), dE_var=float((e2[0] - E0) / N), E_FN=fn_energy(wS, tag='S_1aH'))
    for a in arms:
        rk, dE, v = arms[a][1](kc); fn[a] = dict(rank=rk, dE_var=dE, E_FN=fn_energy(v, tag=a)); log('FN', a, fn[a])
    eV = fn['V']['E_FN']; eR = [fn[f'R{s}']['E_FN'] for s in range(3)]
    if all(eV <= 0.3 * e for e in eR) and eV <= 0.3 * fn['marshall']: fv = 'PASS'
    elif sum(eV >= e for e in eR) >= 2: fv = 'KILL'
    else: fv = 'INCONCLUSIVE'
    fn['verdict'] = dict(E_FN_V=eV, E_FN_R=eR, verdict=fv); fn['solver'] = FNINFO
    res['C3_FN'] = fn; dump()
    log('VERDICT C3-T0-2', verdict, res['verdict'], 'C3-FN', fv, fn['verdict'])
