"""R2 (PLAN_ansatz.md, 2026-10-26): reconstruction and amplification of the exp-J2 pre-image. Exact, CPU.
   python3 r2_recon.py LX LY OUT_JSON
J1 = 1, J2 = 0.5, PBC, Sz = 0. Phi_t = expm_multiply(t H_J2, psi0) (unnormalised; round trip expm(-t H_J2) Phi_t = psi0, psi0 unit).
Conventions copied from local_undo.py (basis, bond operators, s_N, W_M, psi0 gauge); fn_energy from c2_t1_small.py (ten Haaf, own sign).
Sections: (a) 4x4 reproduction of the Astra numbers (exit 3 if any misses by > 2%), (b) gamma_M and matched-fidelity J1 control,
(c) eta / C in the stripe frame s_X, (d) Jastrow amplification A, (e) oracle-free sign reconstruction psi_rec vs Marshall, checks.
"""
import sys, json, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.optimize import brentq
LX, LY, OUT = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
N = LX * LY; J2 = 0.5; T0 = time.time()
TS = [0.12, 0.14, 0.16, 0.18, 0.20]; EPS = [0.01, 0.03]; SEEDS = range(5)
def log(*a): print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)
def site(x, y): return (x % LX) + LX * (y % LY)
def xy(i): return i % LX, i // LX
NN = [(site(x, y), site(x + 1, y)) for y in range(LY) for x in range(LX)] + [(site(x, y), site(x, y + 1)) for y in range(LY) for x in range(LX)]
NNN = [(site(x, y), site(x + 1, y + 1)) for y in range(LY) for x in range(LX)] + [(site(x, y), site(x + 1, y - 1)) for y in range(LY) for x in range(LX)]
allx = np.arange(1 << N, dtype=np.int64); pc = np.zeros(1 << N, np.int8)
for i in range(N): pc += ((allx >> i) & 1).astype(np.int8)
basis = allx[pc == N // 2]; del allx, pc; D = basis.size; log('D', D)
def build(bonds, J):
    """off-diagonal (csr, no diagonal) and diagonal of sum_b J S_i.S_j, same matrix elements as local_undo.SS."""
    R, C = [], []; dg = np.zeros(D)
    for i, j in bonds:
        anti = ((basis >> i) & 1) != ((basis >> j) & 1)
        nbr = np.searchsorted(basis, basis ^ ((1 << i) | (1 << j))); r = np.nonzero(anti)[0]
        R.append(r.astype(np.int32)); C.append(nbr[r].astype(np.int32)); dg += J * np.where(anti, -0.25, 0.25)
    R = np.concatenate(R); C = np.concatenate(C)
    M = sp.csr_matrix((np.full(R.size, 0.5 * J), (R, C)), shape=(D, D)); M.sum_duplicates(); return M, dg
O1, d1 = build(NN, 1.0); O2, d2 = build(NNN, J2); log('built', O1.nnz, O2.nnz)
HJ1 = (O1 + sp.diags(d1)).tocsr(); HJ2 = (O2 + sp.diags(d2)).tocsr(); Hoff = (O1 + O2).tocsr(); dH = d1 + d2
H = (Hoff + sp.diags(dH)).tocsr(); del O1
E, V = sla.eigsh(H, k=1, which='SA', tol=0); E0 = float(E[0]); psi0 = V[:, 0]
Aset = [site(x, y) for y in range(LY) for x in range(LX) if (x + y) % 2 == 0]; nA = np.zeros(D, np.int64)
for i in Aset: nA += (basis >> i) & 1
sN = (1 - 2 * (nA & 1)).astype(float)
if psi0 @ (np.abs(psi0) * sN) < 0: psi0 = -psi0
nX = np.zeros(D, np.int64)
for i in range(N):
    if xy(i)[0] % 2 == 1: nX += (basis >> i) & 1
sX = (1 - 2 * (nX & 1)).astype(float)
log('E0/N', E0 / N)
def en(v): return float(v @ (H @ v) / (v @ v))
def WM(f):
    f = f / np.linalg.norm(f); return float(np.sum(f ** 2 * (np.sign(f) * np.sign(np.sum(f * np.abs(f) * sN)) != sN)))
def F(phi): return float((psi0 @ phi) ** 2 / (phi @ phi))
def gam(w): return float(-np.log(abs(1 - 2 * w)) / N) if abs(1 - 2 * w) > 0 else float('nan')
def ex(M, t, v): return sla.expm_multiply(t * M, v)
def wmedian(x, w):
    if x.size == 0: return None
    o = np.argsort(x); cw = np.cumsum(w[o]); return float(x[o][np.searchsorted(cw, 0.5 * cw[-1])])
ROW = np.repeat(np.arange(D, dtype=np.int32), np.diff(Hoff.indptr)); COL = Hoff.indices
def fn_energy(v):
    """lattice FN (ten Haaf), sign (v >= 0 -> +) and guide |v| from v itself."""
    s = np.where(v >= 0, 1.0, -1.0); g = np.maximum(np.abs(v), 1e-300)
    viol = Hoff.data * s[ROW] * s[COL] > 0
    vsf = np.bincount(ROW[viol], weights=Hoff.data[viol] * s[ROW[viol]] * s[COL[viol]] * g[COL[viol]] / g[ROW[viol]], minlength=D)
    Mk = sp.csr_matrix((np.where(viol, 0.0, Hoff.data), COL, Hoff.indptr), shape=(D, D)); dd = dH + vsf
    op = sla.LinearOperator((D, D), matvec=lambda x: Mk @ x + (dd[:, None] * x if x.ndim == 2 else dd * x), dtype=float)
    if D < 100000: return float(sla.eigsh(op, k=1, which='SA', tol=1e-11)[0][0])
    # guide amplitudes span ~1e-10..1, so the vsf diagonal is huge and plain Lanczos stalls: LOBPCG with a Jacobi preconditioner, started from the guide
    x0 = (g / np.linalg.norm(g))[:, None]; lam0 = float(x0[:, 0] @ op.matvec(x0[:, 0]))
    pre = np.maximum(np.abs(dd - lam0) + 1.0, 1.0)
    Minv = sla.LinearOperator((D, D), matvec=lambda x: x / (pre[:, None] if x.ndim == 2 else pre), dtype=float)
    lam, X, hist = sla.lobpcg(op, x0, M=Minv, largest=False, tol=1e-9, maxiter=400, retLambdaHistory=True)
    x = X[:, 0]; r = op.matvec(x) - lam[0] * x; log('  lobpcg iters', len(hist), 'E', lam[0], 'rel resid', float(np.linalg.norm(r) / abs(lam[0])))
    return float(lam[0])
def maxoff_frame(sgn):
    """largest off-diagonal of S H_J2 S (should be <= 0 for a stoquastic frame)."""
    return float(np.max(O2.data * sgn[np.repeat(np.arange(D), np.diff(O2.indptr))] * sgn[O2.indices]))
W0 = WM(psi0); res = dict(LX=LX, LY=LY, N=N, D=int(D), E0=E0, E0_site=E0 / N, W_M_psi0=W0, gamma_M_psi0=gam(W0), t={}, checks={})
log('W_M psi0', W0, 'gamma', gam(W0))
def dump(): res['sec'] = time.time() - T0; json.dump(res, open(OUT, 'w'), indent=1)
def u_match(Ft, hint):
    f = lambda u: F(ex(HJ1, -u, psi0)) - Ft
    return float(brentq(f, 0.7 * hint, 1.8 * hint, xtol=1e-9, rtol=1e-10))
# ------------------------------------------------------------------ checks
ck = res['checks']
ck['stoq_maxoff_sX'] = maxoff_frame(sX); ck['stoq_maxoff_sN'] = maxoff_frame(sN)
try:
    assert ck['stoq_maxoff_sX'] <= 1e-14, 'sX frame not stoquastic'; ck['assert_sX'] = 'passed'
except AssertionError as e: ck['assert_sX'] = 'FAILED: ' + str(e)
try:
    assert ck['stoq_maxoff_sN'] <= 1e-14, 's_N frame not stoquastic'; ck['assert_sN'] = 'passed'
except AssertionError as e: ck['assert_sN'] = 'FAILED (expected): ' + str(e)
log('checks stoquastic: sX', ck['assert_sX'], ck['stoq_maxoff_sX'], '| sN', ck['assert_sN'], ck['stoq_maxoff_sN'])
fn0 = fn_energy(psi0); fnM = fn_energy(sN * np.abs(psi0))
ck['fn_psi0_minus_E0'] = fn0 - E0; ck['fn_marshall_minus_E0'] = fnM - E0
log('fn_energy psi0 - E0 =', fn0 - E0, '; Marshall - E0 =', fnM - E0); dump()
# ------------------------------------------------------------------ (a) reproduction, 4x4 only
dev = lambda x, ref: float((x - ref) / ref)
if N == 16:
    a = dict(); r = lambda t: WM(ex(HJ2, t, psi0)) / W0
    ts = brentq(lambda t: np.log(r(t)) - np.log(1e-3), 0.14, 0.16, xtol=1e-9)
    us = brentq(lambda u: np.log(WM(ex(HJ1, -u, psi0)) / W0) - np.log(1e-3), 0.17, 0.21, xtol=1e-9)
    ph = ex(HJ2, 0.16, psi0); F16 = F(ph); u16 = u_match(F16, 0.1849)
    a['t_star'] = (ts, 0.15918); a['u_star'] = (us, 0.19172); a['F_t0.16'] = (F16, 0.907323)
    a['ratio_J2_t0.16'] = (WM(ph) / W0, 8.80e-4); a['ratio_J1_matched'] = (WM(ex(HJ1, -u16, psi0)) / W0, 1.92e-3); a['u_matched'] = (u16, 0.18487)
    Tf = np.abs(ex(HJ2, -0.16, np.abs(ph)))      # placeholder replaced below by the s_X kernel
    Ht = lambda: None
    p0 = psi0 ** 2
    Xs = sp.diags(sX); HT = (Xs @ HJ2 @ Xs).tocsr(); f = np.abs(ph); Tf = sla.expm_multiply(-0.16 * HT, f); eta = np.abs(psi0) / Tf
    a['eta_med_all'] = (wmedian(eta, p0), 0.694); m = p0 < 1e-7; a['eta_med_tail'] = (wmedian(eta[m], p0[m]), 0.0412)
    res['a_reproduction'] = {k: dict(value=float(v[0]), ref=v[1], dev=dev(v[0], v[1])) for k, v in a.items()}
    for k, v in res['a_reproduction'].items(): log('(a)', k, v)
    dump()
    bad = [k for k, v in res['a_reproduction'].items() if abs(v['dev']) > 0.02]
    res['a_misses_gt_2pct'] = bad; dump()
    if bad and not (len(sys.argv) > 4 and sys.argv[4] == 'force'):
        log('STOP: (a) misses > 2% for', bad); sys.exit(3)
# ------------------------------------------------------------------ Jastrow classes
def cls_id(i, j):
    dx = (xy(j)[0] - xy(i)[0]) % LX; dy = (xy(j)[1] - xy(i)[1]) % LY; orb = []
    for sx in (1, -1):
        for sy in (1, -1):
            a_, b_ = (sx * dx) % LX, (sy * dy) % LY; orb.append((a_, b_))
            if LX == LY: orb.append((b_, a_))
    return min(orb)
cl = {(i, j): cls_id(i, j) for i in range(N) for j in range(i + 1, N)}; ucl = sorted(set(cl.values())); log('Jastrow classes', len(ucl), ucl)
res['n_jastrow_classes'] = len(ucl)
def gfun(seed):
    v = np.random.default_rng(seed).standard_normal(len(ucl)); Vm = np.zeros((N, N))
    for (i, j), c in cl.items(): Vm[i, j] = Vm[j, i] = v[ucl.index(c)]
    out = np.empty(D)
    for k0 in range(0, D, 400000):
        b = basis[k0:k0 + 400000]; n = np.stack([2.0 * ((b >> i) & 1) - 1 for i in range(N)], 1); out[k0:k0 + 400000] = 0.5 * np.sum((n @ Vm) * n, 1)
    return out
G = {s: gfun(s) for s in SEEDS}
def norm_g(g, p): m = np.sum(p * g); return (g - m) / np.sqrt(np.sum(p * (g - m) ** 2))
p0 = psi0 ** 2
den = {}
for s in SEEDS:
    gp = norm_g(G[s], p0)
    for e in EPS: den[(s, e)] = en(psi0 * np.exp(e * gp)) - E0
res['jastrow_denominator'] = {f'{s}_{e}': v for (s, e), v in den.items()}; log('denominators', den)
Xs = sp.diags(sX); HT = (Xs @ HJ2 @ Xs).tocsr()
# ------------------------------------------------------------------ per t
for t in TS:
    tt = time.time(); R = {}
    ph = ex(HJ2, t, psi0); nph = np.linalg.norm(ph); phn = ph / nph
    rt = np.abs(ex(HJ2, -t, ph) - psi0).max(); rw = np.abs(ex(HJ2, +t, ph) - psi0).max()
    R['roundtrip_maxabs_correct'] = float(rt); R['roundtrip_maxabs_wrong_sign'] = float(rw); R['norm_Phi'] = float(nph)
    assert rt < 1e-10, ('round trip', rt)
    Ft = F(ph); wj2 = WM(ph); R['fidelity'] = Ft; R['W_M_J2'] = wj2; R['ratio_J2'] = wj2 / W0; R['gamma_M_J2'] = gam(wj2)
    u = u_match(Ft, 1.16 * t); pj = ex(HJ1, -u, psi0); wj1 = WM(pj)
    R['u_matched'] = u; R['F_check_J1'] = F(pj); R['W_M_J1'] = wj1; R['ratio_J1'] = wj1 / W0; R['gamma_M_J1'] = gam(wj1)
    R['J1_over_J2_ratio'] = (wj1 / wj2) if wj2 > 0 else None
    # (c)
    f = np.abs(ph); Tf = sla.expm_multiply(-t * HT, f); eta = np.abs(psi0) / Tf
    C = float(np.sum(Tf ** 2)); R['C_direct'] = C; R['C_sum_p_eta-2'] = float(np.sum(p0[eta > 0] * eta[eta > 0] ** -2))
    R['lnC'] = float(np.log(C)); R['lnC_per_N'] = R['lnC'] / N; R['eta_max'] = float(eta.max()); R['eta_min'] = float(eta.min())
    R['eta_med_all'] = wmedian(eta, p0); m = p0 < 1e-7; R['eta_med_tail_1e-7'] = wmedian(eta[m], p0[m]); R['tail_weight_1e-7'] = float(p0[m].sum())
    dec = np.floor(np.log10(np.maximum(p0, 1e-300))).astype(int); R['eta_med_per_decade'] = {}
    for d_ in sorted(set(dec)):
        mm = dec == d_; R['eta_med_per_decade'][int(d_)] = dict(weight=float(p0[mm].sum()), n=int(mm.sum()), eta_med=wmedian(eta[mm], p0[mm]))
    # (d)
    R['A'] = {}
    for e in EPS:
        As = []
        for s in SEEDS:
            g = norm_g(G[s], phn ** 2); num = en(ex(HJ2, -t, ph * np.exp(e * g))) - E0; As.append(num / den[(s, e)])
        R['A'][str(e)] = dict(per_draw=[float(x) for x in As], mean=float(np.mean(As)), std=float(np.std(As, ddof=1)), min=float(min(As)), max=float(max(As)))
    # (e)
    prec = ex(HJ2, -t, sN * np.abs(ph)); R['E_var_rec_site'] = (en(prec) - E0) / N; R['E_FN_rec_site'] = (fn_energy(prec) - E0) / N
    R['E_var_marshall_site'] = (en(sN * np.abs(psi0)) - E0) / N; R['E_FN_marshall_site'] = (fnM - E0) / N
    R['sec'] = time.time() - tt; res['t'][str(t)] = R; dump()
    log('t', t, {k: v for k, v in R.items() if k not in ('eta_med_per_decade',)})
res['sec_total'] = time.time() - T0; dump(); log('DONE')
