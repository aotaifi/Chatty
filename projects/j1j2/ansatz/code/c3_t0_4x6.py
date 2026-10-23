"""C3-T0 Amendment C3-T0-1 (PLAN_ansatz.md): 4x6 PBC, rank-matched control.
  python3 c3_t0_4x6.py OUTDIR [NPROC]
J1-J2, J2 = 0.5, Lx = 4, Ly = 6, Sz = 0. Projected Z2 BCS states in the Slater form verified in c3_t0.py:
psi(x) = sgn_slater(x) det[[U[u,:]], [V[u,:]]] over the N negative-energy Bogoliubov vectors of
M = [[T, Delta], [Delta, -T]] (Nambu (c_up, c_dn^+)); t = 1 NN, D1 d_{x2-y2} NN, D2 d_xy on (1,+-1), mu.
Spans: (S) per-sector optima (sectors with a degenerate mean-field GS skipped); (V) S + symmetrised vison-pair
states (11 separation classes x 2 string orientations x usable sectors); (R') S + the same number of states with
t, D on the same bonds times random factors in [0.5, 1.5], 3 seeds. Greedy addition by Ritz-energy gain;
(E_Ritz - E0)/N versus S-rank k; verdict at the largest common rank.
Symmetrisation over translations x C2v (x spin flip if psi0 is even) by permuting the full basis vector.
"""
import sys, os, json, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.optimize import minimize
from multiprocessing import Pool
OUT = sys.argv[1]; NPROC = int(sys.argv[2]) if len(sys.argv) > 2 else 4
os.makedirs(OUT, exist_ok=True)
T0 = time.time()
def log(*a):
    s = f'[{time.time() - T0:8.1f}s] ' + ' '.join(str(x) for x in a)
    print(s, flush=True); open(os.path.join(OUT, 'c3_t0_4x6.log'), 'a').write(s + '\n')
res = {}
def dump(): json.dump(res, open(os.path.join(OUT, 'c3_t0_4x6.json'), 'w'), indent=1, default=str)

LX, LY = 4, 6; N = LX * LY; J2 = 0.5
def site(x, y): return (x % LX) + LX * (y % LY)
NN = [(site(x, y), site(x + 1, y)) for y in range(LY) for x in range(LX)] + \
     [(site(x, y), site(x, y + 1)) for y in range(LY) for x in range(LX)]
NNN = [(site(x, y), site(x + 1, y + 1)) for y in range(LY) for x in range(LX)] + \
      [(site(x, y), site(x + 1, y - 1)) for y in range(LY) for x in range(LX)]
assert len({tuple(sorted(b)) for b in NN + NNN}) == len(NN + NNN)

# ---------------------------------------------------------------- basis (bit 1 = up), lookup table
def popcount(a, nb=64):
    c = np.zeros(a.shape, np.int64)
    for k in range(nb): c += (a >> k) & 1
    return c
allx = np.arange(1 << N, dtype=np.int64)
pc = popcount(allx, N)
basis = allx[pc == N // 2]; del allx, pc
Dim = basis.size
LUT = np.full(1 << N, -1, np.int32); LUT[basis] = np.arange(Dim, dtype=np.int32)
log('Dim', Dim)

# ---------------------------------------------------------------- Hamiltonian (as small_pretests.py), ground state
def ham():
    dg = np.zeros(Dim); rows, cols = [], []
    for J, bonds in ((1.0, NN), (J2, NNN)):
        for (i, j) in bonds:
            anti = ((basis >> i) & 1) != ((basis >> j) & 1)
            dg += J * np.where(anti, -0.25, 0.25)
    mats = []
    for J, bonds in ((1.0, NN), (J2, NNN)):
        r_, c_ = [], []
        for (i, j) in bonds:
            r = np.nonzero(((basis >> i) & 1) != ((basis >> j) & 1))[0].astype(np.int32)
            r_.append(r); c_.append(LUT[basis[r] ^ ((1 << i) | (1 << j))])
        r_ = np.concatenate(r_); c_ = np.concatenate(c_)
        mats.append(sp.csr_matrix((np.full(r_.size, 0.5 * J), (r_, c_)), shape=(Dim, Dim)))
        del r_, c_
    return (mats[0] + mats[1] + sp.diags(dg)).tocsr()
H = ham(); log('H nnz', H.nnz)
E, V = sla.eigsh(H, k=2, which='SA', tol=1e-10); o = np.argsort(E); E, V = E[o], V[:, o]
psi0 = V[:, 0].copy(); E0 = float(E[0]); del V
log('E0/N', E0 / N, 'gap', E[1] - E[0])
assert abs(E0 / N + 0.52252493537) < 1e-8, E0 / N
res.update(LX=LX, LY=LY, Dim=int(Dim), E0_site=E0 / N, gap=float(E[1] - E[0]))
MA = sum(1 << site(x, y) for y in range(LY) for x in range(LX) if (x + y) % 2 == 0)
sN = 1 - 2 * (popcount(basis & MA, N) & 1)
def WM(v, sgn=sN):
    v = v / np.linalg.norm(v); s = np.sign(np.sum(v * np.abs(v) * sgn)) or 1.0
    return float(np.sum(v ** 2 * ((np.sign(v) * s) != sgn) * (v != 0)))
res['W_M_psi0'] = WM(psi0)

# ---------------------------------------------------------------- Slater-form tables
Uidx = np.empty((Dim, N // 2), np.int16)
cnt = np.zeros(Dim, np.int64)
for i in range(N):
    b = ((basis >> i) & 1).astype(bool)
    Uidx[b, cnt[b]] = i; cnt[b] += 1
del cnt
def slater_sign(Us):
    st = np.full(Us.shape[0], sum(1 << (2 * i + 1) for i in range(N)), np.int64); sg = np.ones(Us.shape[0], np.int64)
    ops = [(2 * Us[:, a].astype(np.int64) + 1, False) for a in range(Us.shape[1] - 1, -1, -1)] + \
          [(2 * Us[:, a].astype(np.int64), True) for a in range(Us.shape[1] - 1, -1, -1)]
    for m, dag in ops:
        assert np.all(((st >> m) & 1) == (0 if dag else 1))
        sg *= 1 - 2 * (popcount(st & ((np.int64(1) << m) - 1), 2 * N) & 1)
        st ^= (np.int64(1) << m)
    return st, sg
SSIGN = np.empty(Dim, np.int8)
for a in range(0, Dim, 400000):
    st, sg = slater_sign(Uidx[a:a + 400000])
    tgt = np.zeros(st.size, np.int64)
    for i in range(N):
        tgt |= np.where((basis[a:a + 400000] >> i) & 1, np.int64(1) << (2 * i), np.int64(1) << (2 * i + 1))
    assert np.all(st == tgt)
    SSIGN[a:a + 400000] = sg
res['slater_sign_values'] = [int(q) for q in np.unique(SSIGN)]
log('Slater sign values', res['slater_sign_values'])

CH = 25000
def _det_chunk(args):
    a, Wn = args
    U = Uidx[a:a + CH].astype(np.int64)
    rows = np.concatenate([U, U + N], axis=1)
    return np.linalg.det(Wn[rows])
POOL = None
def slater_amp(T, Dl):
    w, W = np.linalg.eigh(np.block([[T, Dl], [Dl, -T]]))
    gap = float(min(abs(w[N - 1]), abs(w[N])))
    Wn = W[:, :N]
    parts = POOL.map(_det_chunk, [(a, Wn) for a in range(0, Dim, CH)])
    return np.concatenate(parts) * SSIGN, gap

# ---------------------------------------------------------------- bonds and mean field
BONDS = []
for y in range(LY):
    for x in range(LX):
        for kind, (dx, dy) in enumerate(((1, 0), (0, 1), (1, 1), (1, -1))):
            BONDS.append((site(x, y), site(x + dx, y + dy), kind, x, y, dx, dy))
NB = len(BONDS)
def bc_sign(b, bcx, bcy):
    _, _, _, x, y, dx, dy = b; s = 1
    if not (0 <= x + dx < LX) and bcx: s = -s
    if not (0 <= y + dy < LY) and bcy: s = -s
    return s
def mf_mats(p, bcx, bcy, fac=None):
    D1, D2, mu = p
    T = -mu * np.eye(N); Dl = np.zeros((N, N))
    for k, b in enumerate(BONDS):
        i, j, kind = b[:3]; s = bc_sign(b, bcx, bcy) * (1.0 if fac is None else fac[k])
        if kind < 2:
            T[i, j] += s; T[j, i] += s; d = D1 if kind == 0 else -D1
        else:
            d = D2 if kind == 2 else -D2
        Dl[i, j] += s * d; Dl[j, i] += s * d
    return T, Dl
def state(p, bc, fac=None):
    amp, gap = slater_amp(*mf_mats(p, *bc, fac))
    nrm = np.linalg.norm(amp)
    if gap < 1e-9 or not np.isfinite(nrm) or nrm == 0: return None, gap
    return amp / nrm, gap
def energy(v): return float(v @ (H @ v) / (v @ v))

# ---------------------------------------------------------------- symmetry maps
def site_perms():
    ops = []
    for (a, d) in ((1, 1), (-1, 1), (1, -1), (-1, -1)):           # C2v of the rectangle
        for tx in range(LX):
            for ty in range(LY):
                ops.append([site(a * (s % LX) + tx, d * (s // LX) + ty) for s in range(N)])
    return ops
def build_maps():
    maps = []
    for perm in site_perms():
        img = np.zeros(Dim, np.int64)
        for i in range(N): img |= ((basis >> i) & 1) << perm[i]
        maps.append(LUT[img])
    return maps

if __name__ == '__main__':
    POOL = Pool(NPROC)
    t = time.time(); v, g = state((1.37, 0.67, 0.0), (1, 1)); log('one state eval s', time.time() - t, 'gap', g)
    # ---------------- checks (i) and (ii)
    chk = {}
    for bc in ((1, 1), (0, 1), (1, 0)):
        v, gap = state((0.6, 0.0, 0.0), bc)
        if v is None: chk[str(bc)] = 'ill-defined'; continue
        chk[str(bc)] = dict(gap=gap, W_M=WM(v), W_M_broken_times_sN_of_shifted_lattice=WM(v * (1 - 2 * (popcount(basis & (MA << 1), N) & 1))),
                            W_M_broken_flip_sign_one_config_class=WM(v * np.where(Uidx[:, 0] == 0, -1, 1)))
    res['check_i_marshall'] = chk; log('check (i)', chk)
    MAPS = build_maps(); FLIP = LUT[basis ^ ((1 << N) - 1)]
    CHI = np.array([float(psi0 @ psi0[m]) for m in MAPS]); chiF = float(psi0 @ psi0[FLIP])
    res['psi0_chars'] = dict(min_abs=float(np.min(np.abs(CHI))), n_neg=int(np.sum(CHI < 0)), flip=chiF)
    log('psi0 characters', res['psi0_chars'])
    CHI = np.sign(CHI); use_flip = chiF > 0.999
    def symmetrise(v):
        w = np.zeros(Dim)
        for ch, m in zip(CHI, MAPS): w += ch * v[m]
        if abs(chiF) > 0.999: w = w + np.sign(chiF) * w[FLIP]
        return w
    # ---------------- (0) single-state optimum per sector
    NEV = [0]
    def negov(p, bc):
        v, _ = state(p, bc); NEV[0] += 1
        ov = 0.0 if v is None else float((psi0 @ v) ** 2)
        log('PROGRESS NM eval', NEV[0], 'p', [round(q, 5) for q in p], 'overlap2', ov)
        return -ov
    # clarified amendment: optimise only AA from the 4x4 optimum, <= 60 evaluations; other sectors at that point
    r = minimize(negov, (1.370, 0.666, 0.000), args=((1, 1),), method='Nelder-Mead',
                 options=dict(xatol=1e-4, fatol=1e-8, maxfev=60))
    popt = [float(q) for q in r.x]
    res['AA_optimisation'] = dict(overlap2=float(-r.fun), params=popt, nfev=int(r.nfev), message=str(r.message))
    log('AA optimum', res['AA_optimisation'])
    best = {}
    for bc in ((1, 1), (0, 1), (1, 0), (0, 0)):
        v, gap = state(popt, bc)
        ov = 0.0 if v is None else float((psi0 @ v) ** 2)
        if v is None or ov < 1e-6:
            best[bc] = dict(usable=False, overlap2=ov, gap=gap); log('BC', bc, 'not usable', best[bc]); continue
        best[bc] = dict(usable=True, overlap2=ov, params=popt, E_site=energy(v) / N, gap=gap, W_M=WM(v))
        log('BC', bc, best[bc])
    res['single_by_bc'] = {f'{k[0]}{k[1]}': q for k, q in best.items()}; dump()
    secs = [k for k in sorted(best) if best[k]['usable']]
    kb = min(secs, key=lambda k: best[k]['E_site'])
    res['best_single'] = dict(bc=kb, E_site=best[kb]['E_site'], dE_site=best[kb]['E_site'] - E0 / N)
    S_states = [state(best[k]['params'], k)[0] for k in secs]
    res['check_ii'] = None

    # ---------------- strings
    def seg_cross(p1, p2, q1, q2):
        def orient(a, b, c): return np.sign((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        return orient(p1, p2, q1) * orient(p1, p2, q2) < 0 and orient(q1, q2, p1) * orient(q1, q2, p2) < 0
    EPS = (0.1, 0.03)
    def string_mask(dx, dy, xfirst=True):
        a = (0.5 + EPS[0], 0.5 + EPS[1])
        b = (a[0] + dx, a[1]) if xfirst else (a[0], a[1] + dy)
        c = (a[0] + dx, a[1] + dy)
        mask = np.zeros(NB, int)
        for k, (i, j, kind, x, y, ddx, ddy) in enumerate(BONDS):
            n = 0
            for (p, q) in ((a, b), (b, c)):
                for mx in range(-3, 4):
                    for my in range(-3, 4):
                        if seg_cross((x + mx * LX, y + my * LY), (x + ddx + mx * LX, y + ddy + my * LY), p, q): n += 1
            mask[k] = n & 1
        return mask
    bidx = {(b[3], b[4], b[5], b[6]): k for k, b in enumerate(BONDS)}
    def bs(m, x, y, dx, dy):
        if dx < 0 or (dx == 0 and dy < 0): x, y, dx, dy = x + dx, y + dy, -dx, -dy
        return 1 - 2 * m[bidx[(x % LX, y % LY, dx, dy)]]
    def flux(m):
        plaq, tri = set(), set()
        for y in range(LY):
            for x in range(LX):
                if bs(m, x, y, 1, 0) * bs(m, x + 1, y, 0, 1) * bs(m, x, y + 1, 1, 0) * bs(m, x, y, 0, 1) < 0: plaq.add((x, y))
                ts = [bs(m, x, y, 1, 0) * bs(m, x + 1, y, 0, 1) * bs(m, x, y, 1, 1),
                      bs(m, x, y, 0, 1) * bs(m, x, y + 1, 1, 0) * bs(m, x, y, 1, 1),
                      bs(m, x, y, 1, 0) * bs(m, x, y, 0, 1) * bs(m, x, y + 1, 1, -1),
                      bs(m, x + 1, y, 0, 1) * bs(m, x, y + 1, 1, 0) * bs(m, x, y + 1, 1, -1)]
                if min(ts) < 0: tri.add((x, y))
        return plaq, tri
    CLASSES = [(dx, dy) for dx in range(3) for dy in range(4) if (dx, dy) != (0, 0)]
    strings = []
    for (dx, dy) in CLASSES:
        if dx and dy: variants = [('xfirst', dx, dy, True), ('yfirst', dx, dy, False)]
        elif dx: variants = [('short', dx, 0, True), ('long', dx - LX, 0, True)]
        else: variants = [('short', 0, dy, False), ('long', 0, dy - LY, False)]
        for (nm, ddx, ddy, xf) in variants:
            m = string_mask(ddx, ddy, xf); plaq, tri = flux(m)
            ends = {(0, 0), (dx % LX, dy % LY)}
            ok = plaq == ends and tri <= ends
            res.setdefault('flux_check', []).append(dict(cls=(dx, dy), var=nm, nflip=int(m.sum()), ok=bool(ok)))
            assert ok, (dx, dy, nm, plaq, tri)
            strings.append(((dx, dy), nm, m))
    mloop = (string_mask(1, 1, True) + string_mask(1, 1, False)) % 2
    res['flux_check_closed_loop'] = sorted(flux(mloop)[0])
    log('strings', len(strings), 'all flux checks OK; closed loop pi plaquettes', res['flux_check_closed_loop'])

    # ---------------- arms: Gram and H matrices, greedy
    def arm(vecs_fn, label):
        Phi = [v for v in S_states]
        meta = []
        for item in vecs_fn():
            log('PROGRESS', label, 'state', len(meta) + 1)
            if item[0] is None: meta.append(item[1] + ['ill-defined']); continue
            w = symmetrise(item[0]); n = np.linalg.norm(w)
            if n < 1e-8: meta.append(item[1] + ['zero after symmetrisation']); continue
            Phi.append(w / n); meta.append(item[1])
        nS = len(S_states)
        Gm = np.empty((len(Phi), len(Phi))); Hm = np.empty_like(Gm); b = np.empty(len(Phi))
        HPhi = []
        for i, v in enumerate(Phi):
            hv = H @ v; b[i] = psi0 @ v
            for j in range(i + 1):
                Gm[i, j] = Gm[j, i] = Phi[j] @ v; Hm[i, j] = Hm[j, i] = Phi[j] @ hv
        def ritz(idx, cut=1e-10):
            G = Gm[np.ix_(idx, idx)]; Hs = Hm[np.ix_(idx, idx)]
            s, U = np.linalg.eigh(G); keep = s > cut * s.max()
            X = U[:, keep] / np.sqrt(s[keep]); e = np.linalg.eigvalsh(X.T @ Hs @ X)
            bb = X.T @ b[idx]
            return int(keep.sum()), float((e[0] - E0) / N), float(1 - bb @ bb)
        cur = list(range(nS)); rk, dE, inf = ritz(cur)
        curve = [dict(k=rk, dE_site=dE, infid=inf, added=None)]
        rest = list(range(nS, len(Phi)))
        while rest:
            trials = [(ritz(cur + [c]), c) for c in rest]
            (rk2, dE2, inf2), c = min(trials, key=lambda q: q[0][1])
            if rk2 <= rk: break                                   # no candidate adds a direction above the cut
            cur.append(c); rest.remove(c); rk, dE, inf = rk2, dE2, inf2
            curve.append(dict(k=rk, dE_site=dE, infid=inf, added=c - nS))
        log(label, 'n', len(Phi), 'final rank', rk, 'dE/N', dE, 'curve', [(q['k'], f"{q['dE_site']:.3e}") for q in curve])
        return dict(n=len(Phi), meta=meta, curve=curve, full=ritz(list(range(len(Phi)))))
    if True:
        v1 = S_states[secs.index(kb)]
        r1 = np.linalg.eigvalsh(np.array([[v1 @ (H @ v1)]]))[0]
        res['check_ii'] = float(r1 - energy(v1))
    def vis_items():
        for bc in secs:
            for (cl, nm, m) in strings:
                v, gap = state(best[bc]['params'], bc, fac=1.0 - 2.0 * m)
                yield v, [list(bc), list(cl), nm, gap]
    res['arm_V'] = arm(vis_items, 'V'); dump()
    for sd in range(3):
        def ctl_items(sd=sd):
            rng = np.random.default_rng(100 + sd)
            for bc in secs:
                for (cl, nm, m) in strings:
                    fac = np.where(m == 1, rng.uniform(0.5, 1.5, NB), 1.0)
                    v, gap = state(best[bc]['params'], bc, fac=fac)
                    yield v, [list(bc), list(cl), nm, gap]
        res[f'arm_R_seed{sd}'] = arm(ctl_items, f"R' seed{sd}"); dump()

    # ---------------- verdict at the largest common rank
    def at_rank(curve, k):
        c = [q for q in curve if q['k'] <= k]; return c[-1] if c else None
    kmax = min(max(q['k'] for q in res[a]['curve']) for a in ['arm_V'] + [f'arm_R_seed{s}' for s in range(3)])
    table = {}
    for k in range(4, kmax + 1):
        row = {a: (at_rank(res[a]['curve'], k) or {}).get('dE_site') for a in ['arm_V'] + [f'arm_R_seed{s}' for s in range(3)]}
        table[k] = row
    res['dE_vs_rank'] = table
    rV = table[kmax]['arm_V']; ratios = [rV / table[kmax][f'arm_R_seed{s}'] for s in range(3)]
    if all(r <= 0.3 for r in ratios): verdict = 'PASS'
    elif sum(r >= 0.7 for r in ratios) >= 2: verdict = 'KILL'
    else: verdict = 'INCONCLUSIVE'
    res['verdict'] = dict(k=kmax, V=rV, R=[table[kmax][f'arm_R_seed{s}'] for s in range(3)], ratios=ratios,
                          rule='PASS if V <= 0.3 R for all seeds; KILL if V >= 0.7 R for >= 2 of 3 seeds', verdict=verdict)
    dump()
    log('VERDICT C3-T0-1', verdict, res['verdict'])
