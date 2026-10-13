"""Check (a0): global vs local Krylov sign step on M decoupled copies of the periodic 4x4 J1-J2 model (J2/J1 = 0.5).

Guide per copy: (|psi0|, Marshall).  Global rule s' = s sgn(T - E_L(X)), E_L(X) = sum_m e(x_m).  An edge (x, y) of copy m
changes its relative sign iff R = sum of the other copies' e lies in (T - max(e_x, e_y), T - min(e_x, e_y)], so
    g_G(M, T) = (1/16) sum_{ordered edges} (-2) w_xy P_{M-1}(lo < R <= hi),   w_xy = psi(x) H_xy psi(y).
M = 1 exact, M = 2 exact CDF, M >= 3 grid CDF by FFT.  Checks: brute force M = 1, 2 (explicit state), exact nested M = 3,
Monte Carlo at M = 16, 64.  See results/local_krylov/PREREG.md.
  python a0_copies.py OUT.json
"""
import sys, json, time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import numba as nb

L = 4; N1 = 16


def bonds():
    q = lambda x, y: (x % L) + L * (y % L)
    nn_, nnn = [], []
    for y in range(L):
        for x in range(L):
            i = q(x, y); nn_ += [(i, q(x + 1, y)), (i, q(x, y + 1))]
            nnn += [(i, q(x + 1, y + 1)), (i, q(x + 1, y - 1))]
    return [(i, j, 1.0) for i, j in nn_] + [(i, j, 0.5) for i, j in nnn]


def build():
    B = bonds(); J = np.array([b[2] for b in B])
    states = np.array([s for s in range(1 << N1) if bin(s).count('1') == N1 // 2], np.int64)
    pos = -np.ones(1 << N1, np.int64); pos[states] = np.arange(len(states))
    nb_ = len(B); idx = -np.ones((nb_, len(states)), np.int64); diag = np.zeros(len(states))
    for k, (i, j, Jb) in enumerate(B):
        anti = ((states >> i) ^ (states >> j)) & 1
        diag += np.where(anti == 1, -Jb / 4, Jb / 4)
        fl = states ^ ((1 << i) | (1 << j))
        idx[k] = np.where(anti == 1, pos[fl], -1)
    rows, cols, vals = [np.arange(len(states))], [np.arange(len(states))], [diag]
    for k in range(nb_):
        m = idx[k] >= 0
        rows.append(np.nonzero(m)[0]); cols.append(idx[k][m]); vals.append(np.full(m.sum(), J[k] / 2))
    H = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(len(states),) * 2)
    A = sum(1 << (x + L * y) for y in range(L) for x in range(L) if (x + y) % 2 == 0)
    mar = np.where(np.array([bin(s & A).count('1') for s in states]) % 2 == 0, 1.0, -1.0)
    return states, H, idx, J, diag, mar


@nb.njit(cache=True)
def brute2_q(psi, diag, idx, J, T):
    """Explicit 2-copy state Psi = psi x psi; E_L(X) = (H Psi)(X)/Psi(X) by applying H = H1 x 1 + 1 x H2 directly;
    q(X) = +1 if E_L <= T else -1 (int8 matrix)."""
    n = len(psi); nb_ = idx.shape[0]
    q = np.ones((n, n), np.int8)
    for a in range(n):
        for b in range(n):
            p = psi[a] * psi[b]
            if p == 0.0:
                continue
            h = (diag[a] + diag[b]) * p
            for k in range(nb_):
                ia = idx[k, a]
                if ia >= 0: h += 0.5 * J[k] * psi[ia] * psi[b]
                ib = idx[k, b]
                if ib >= 0: h += 0.5 * J[k] * psi[a] * psi[ib]
            if h / p > T: q[a, b] = -1
    return q


@nb.njit(cache=True)
def brute2_energy(psi, diag, idx, J, q):
    """<Psi'|H|Psi'> for Psi'(X) = Psi(X) q(X) (norm of Psi' = 1), H applied explicitly."""
    n = len(psi); nb_ = idx.shape[0]; E = 0.0
    for a in range(n):
        for b in range(n):
            p = psi[a] * psi[b] * q[a, b]
            if p == 0.0:
                continue
            h = (diag[a] + diag[b]) * p
            for k in range(nb_):
                ia = idx[k, a]
                if ia >= 0: h += 0.5 * J[k] * psi[ia] * psi[b] * q[ia, b]
                ib = idx[k, b]
                if ib >= 0: h += 0.5 * J[k] * psi[a] * psi[ib] * q[a, ib]
            E += p * h
    return E


@nb.njit(cache=True)
def mc_delta(xs, e, ex, ey, w_over_a2, ptr, T):
    """Monte Carlo of the global rule: xs (ns, M) copy configurations ~ a^2. Returns per-sample change E1 - E_L,
    sum over copies m and their edges (x_m, y): c (q(X) q(Y) - 1), c = H a(y)s s/a(x) = w_xy / a(x)^2."""
    ns, M = xs.shape; out = np.zeros(ns)
    for s in range(ns):
        EL = 0.0
        for m in range(M): EL += e[xs[s, m]]
        qX = 1.0 if EL <= T else -1.0
        d = 0.0
        for m in range(M):
            x = xs[s, m]
            for p in range(ptr[x], ptr[x + 1]):
                ELy = EL - e[x] + ey[p]
                qY = 1.0 if ELy <= T else -1.0
                d += w_over_a2[p] * (qX * qY - 1.0)
        out[s] = d
    return out


def main(out):
    t0 = time.time(); R = {}
    states, H, idx, J, diag, mar = build()
    vals, vecs = eigsh(H, k=1, which='SA', tol=1e-14, ncv=60)
    E0 = float(vals[0]); psi0 = vecs[:, 0]; psi0 /= np.linalg.norm(psi0)
    R['E0'] = E0; R['E0_check'] = abs(E0 - (-8.45792335139486))
    a = np.abs(psi0); a[a < 1e-12] = 0.0; a /= np.linalg.norm(a)
    psi = a * mar; Hpsi = H @ psi
    E_guide = float(psi @ Hpsi)
    R['n_zero_amp'] = int((a == 0).sum()); R['E_guide'] = E_guide; R['E_guide_minus_E0_site'] = (E_guide - E0) / 16
    good = a > 0; e = np.zeros(len(a)); e[good] = Hpsi[good] / psi[good]
    a2 = a ** 2
    R['e_mean'] = float(a2 @ e); R['e_sd'] = float(np.sqrt(a2 @ (e - E_guide) ** 2))
    R['e_quantiles'] = {str(p): float(v) for p, v in zip([1e-4, 0.01, 0.5, 0.99, 0.9999, 1.0],
                                                           np.quantile(e[good], [1e-4, 0.01, 0.5, 0.99, 0.9999, 1.0]))}
    R['e_min'] = float(e[good].min()); R['e_max'] = float(e[good].max())
    # ordered edges (x, y), w = psi(x) H_xy psi(y), only between non-zero amplitudes
    ex, ey, w, xo = [], [], [], []
    for k in range(idx.shape[0]):
        m = (idx[k] >= 0) & good
        y = idx[k][m]; x = np.nonzero(m)[0]; keep = good[y]; x = x[keep]; y = y[keep]
        ex.append(e[x]); ey.append(e[y]); w.append(psi[x] * 0.5 * J[k] * psi[y]); xo.append(x)
    ex = np.concatenate(ex); ey = np.concatenate(ey); w = np.concatenate(w); xo = np.concatenate(xo)
    lo_e = np.minimum(ex, ey); hi_e = np.maximum(ex, ey)
    R['n_edges'] = len(w); R['violating_weight'] = float(w[w > 0].sum()); R['kept_weight'] = float(w[w < 0].sum())
    # sanity: <H> from edges + diagonal
    assert abs(float(a2 @ diag) + w.sum() - E_guide) < 1e-12

    # single-copy distribution of e
    order = np.argsort(e[good]); es = e[good][order]; cs = np.cumsum(a2[good][order])

    def F_exact(t):  # P(e <= t)
        i = np.searchsorted(es, t, side='right')
        return np.where(i > 0, cs[np.maximum(i - 1, 0)], 0.0)

    def g1(T):
        return float((-2 * w * ((lo_e <= T) & (T < hi_e))).sum() / 16)

    def g_exact2(T):
        return float((-2 * w * (F_exact(T - lo_e) - F_exact(T - hi_e))).sum() / 16)

    # grid CDF of the sum of M-1 copies
    h = float(sys.argv[2]) if len(sys.argv) > 2 else 2e-3
    emin, emax = e[good].min(), e[good].max()
    K = int(np.ceil((emax - emin) / h)) + 1
    p1 = np.bincount(np.rint((e[good] - emin) / h).astype(np.int64), weights=a2[good], minlength=K)
    R['grid_h'] = h; R['grid_K'] = K
    cache = {}

    def F_grid(Mm1):
        if Mm1 not in cache:
            n = Mm1 * (K - 1) + 1; nf = 1 << int(np.ceil(np.log2(n)))
            pm = np.fft.irfft(np.fft.rfft(p1, nf) ** Mm1, nf)[:n]
            pm = np.clip(pm, 0, None); cache[Mm1] = (Mm1 * emin, np.cumsum(pm))
        return cache[Mm1]

    def g_grid(M, T):
        if M == 1: return g1(T)
        off, cdf = F_grid(M - 1)
        xs_ = off + h * (np.arange(len(cdf)) + 0.5)          # rounding error spread over one bin: linear CDF
        def F(t):
            return np.interp(t, xs_, cdf, left=0.0, right=cdf[-1])
        return float((-2 * w * (F(T - lo_e) - F(T - hi_e))).sum() / 16)

    def curve_conv(M):
        """g_M on the T grid T_m = M emin + m h: convolution of g1 (exact, sampled on t_k = emin + k h) with the grid
        pmf of the other M-1 copies (FFT)."""
        tk = emin + h * np.arange(K)
        g1g = g1s_all[np.clip(np.searchsorted(cand, tk, side='right') - 1, 0, None)] * (tk >= cand[0])
        if M == 1: return tk, g1g
        n = M * (K - 1) + 1; nf = 1 << int(np.ceil(np.log2(n)))
        pm = np.fft.irfft(np.fft.rfft(p1, nf) ** (M - 1), nf)
        gM = np.fft.irfft(np.fft.rfft(pm, nf) * np.fft.rfft(g1g, nf), nf)[:n]
        return M * emin + h * np.arange(n), gM

    def opt(M, fn):
        Tc, gc = curve_conv(M); j = int(np.argmin(gc))
        Tf = np.linspace(Tc[j] - 3 * h, Tc[j] + 3 * h, 31)
        gf = np.array([fn(M, T) for T in Tf]); k = int(np.argmin(gf))
        if k in (0, len(Tf) - 1): print('WARNING: refined optimum at the edge of the window, M =', M, flush=True)
        return float(Tf[k]), float(gf[k]), float(gc[j])   # per site; last = convolution-curve value

    # ---- M = 1: exact optimum on all breakpoints
    cand = np.unique(np.r_[lo_e, hi_e])                     # g1 is constant on [cand_k, cand_k+1)
    dif = np.zeros(len(cand) + 1)
    np.add.at(dif, np.searchsorted(cand, lo_e), -2 * w); np.add.at(dif, np.searchsorted(cand, hi_e), 2 * w)
    g1s = np.cumsum(dif)[:len(cand)] / 16; g1s_all = g1s; j = int(np.argmin(g1s))
    T1 = float(cand[j]); G1 = float(g1s[j]); assert abs(g1(T1) - G1) < 1e-14
    R['M1'] = dict(T=T1, T_minus_E=T1 - E_guide, gain_site=G1, flip_frac=float(a2[good][e[good] > T1].sum()))
    print('E0', E0, 'guide', E_guide, 'e_sd', R['e_sd'], 'M1', R['M1'], f'{time.time()-t0:.0f}s', flush=True)

    # ---- brute force M = 1
    chk1 = []
    for T in [T1, T1 - 0.5, T1 + 0.5, E_guide + 1.0, E_guide + 3.0, T1 + 2.0]:
        q = np.where(e <= T, 1.0, -1.0); ps = psi * q
        chk1.append(dict(T=T, brute=(float(ps @ (H @ ps)) - E_guide) / 16, formula=g1(T)))
    R['check_M1'] = chk1; R['check_M1_maxdiff'] = max(abs(c['brute'] - c['formula']) for c in chk1)

    # ---- M = 2: exact formula optimum, grid, brute force
    T2e, G2e, _ = opt(2, lambda M, T: g_exact2(T))
    T2g, G2g, G2c = opt(2, g_grid)
    R['M2_exact'] = dict(T=T2e, gain_site=G2e); R['M2_grid'] = dict(T=T2g, gain_site=G2g, conv_curve_min=G2c)
    chk2 = []
    Ib = idx.astype(np.int64)
    for T in [T2e, T2e - 1.0, T2e + 1.0]:
        tb = time.time()
        q = brute2_q(psi, diag, Ib, J, T)
        Eb = brute2_energy(psi, diag, Ib, J, q)
        chk2.append(dict(T=T, brute=(Eb - 2 * E_guide) / 32, formula=g_exact2(T), grid=g_grid(2, T),
                         sec=time.time() - tb))
        print('brute2', chk2[-1], flush=True)
    # local rule on the explicit 2-copy state: flip iff odd number of copies with e > T1
    qa = np.where(e > T1, -1, 1).astype(np.int8); qL = np.outer(qa, qa).astype(np.int8)
    EbL = brute2_energy(psi, diag, Ib, J, qL)
    R['check_M2'] = chk2; R['check_M2_maxdiff'] = max(abs(c['brute'] - c['formula']) for c in chk2)
    R['check_M2_local'] = dict(brute=(EbL - 2 * E_guide) / 32, M1=G1, diff=(EbL - 2 * E_guide) / 32 - G1)

    # ---- M = 3: exact nested sum g3(T) = sum_x a^2(x) g2(T - e(x)) at T*(3) from the grid
    T3g, G3g, G3c = opt(3, g_grid)
    ga = good.nonzero()[0]
    def g_exact3(T):
        """exact: g3(T) = sum_{x1, x2} a^2(x1) a^2(x2) g1(T - e(x1) - e(x2)), g1 the exact M = 1 step function."""
        eg = e[ga]; ag = a2[ga]; tot = 0.0
        for i in range(len(ga)):
            tt = T - eg[i] - eg
            k = np.searchsorted(cand, tt, side='right') - 1
            tot += ag[i] * float(ag @ np.where(k >= 0, g1s_all[np.maximum(k, 0)], 0.0))
        return tot
    R['M3'] = dict(T_grid=T3g, gain_site_grid=G3g, conv_curve_min=G3c)
    tb = time.time(); R['M3']['gain_site_exact_at_T'] = g_exact3(T3g); R['M3']['sec'] = time.time() - tb
    R['M3']['grid_at_T'] = g_grid(3, T3g)
    print('M3', R['M3'], flush=True)

    # ---- curve M = 1..64
    curve = []
    for M in range(1, 65):
        T, G, Gc = (T1, G1, G1) if M == 1 else opt(M, g_grid)
        curve.append(dict(M=M, N=16 * M, T=T, T_minus_E=T - M * E_guide, gain_site=G, ratio=G / G1, conv_min=Gc))
        if M in (1, 2, 4, 8, 16, 32, 64): print(curve[-1], f'{time.time()-t0:.0f}s', flush=True)
    R['curve'] = curve
    half = [c['M'] for c in curve if abs(c['gain_site']) <= abs(G1) / 2]
    R['M_half'] = half[0] if half else None; R['N_half'] = 16 * R['M_half'] if half else None

    # ---- Monte Carlo at M = 16, 64 (global rule at the grid optimum)
    ptr = np.zeros(len(a) + 1, np.int64)
    o = np.argsort(xo, kind='stable'); np.add.at(ptr, xo + 1, 1); ptr = np.cumsum(ptr)
    wa = (w / a2[xo])[o]; eyo = ey[o]
    rng = np.random.default_rng(1); pa = a2 / a2.sum()
    mc = {}
    for M in (16, 64):
        T = curve[M - 1]['T']; ds = []
        nsamp = 4_000_000 if M == 16 else 1_000_000
        for _ in range(nsamp // 200_000):
            xs = rng.choice(len(a), size=(200_000, M), p=pa)
            ds.append(mc_delta(xs, e, ex, eyo, wa, ptr, T))
        d = np.concatenate(ds) / (16 * M)
        mc[M] = dict(T=T, mc=float(d.mean()), se=float(d.std(ddof=1) / np.sqrt(len(d))), grid=curve[M - 1]['gain_site'])
        mc[M]['z'] = (mc[M]['mc'] - mc[M]['grid']) / mc[M]['se']
        print('MC', M, mc[M], flush=True)
    R['check_MC'] = mc
    R['sec'] = time.time() - t0
    json.dump(R, open(out, 'w'), indent=1)
    print(json.dumps({k: v for k, v in R.items() if k != 'curve'}, indent=1, default=float))


if __name__ == '__main__':
    main(sys.argv[1])
