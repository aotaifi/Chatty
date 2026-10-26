"""R3 (PLAN_ansatz.md, 2026-10-26): O(N) local evaluator psi_L(y) = Phi~(y) prod_{b in J2} g_b(y) of psi0 from the stored pre-image. Exact, CPU.
   python3 r3_local.py LX LY OUT_JSON
Setup (basis, bond operators, s_N, psi0 gauge, fn_energy, u_match) copied verbatim from r2_recon.py lines 8-79.
"""
import sys, json, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.optimize import brentq
LX, LY, OUT = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
N = LX * LY; J2 = 0.5; T0 = time.time()

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
# ======================================================================= R3
TS = [0.16, 0.20]; ck = res['checks']
R2J = json.load(open(f'/scratch/a/A.Otaifi/ansatz_runs/r2/r2_{LX}x{LY}.json'))
def bond_op(i, j, J):
    """J S_i.S_j on the sector basis as a sparse matrix (diag + offdiag)."""
    anti = ((basis >> i) & 1) != ((basis >> j) & 1); r = np.nonzero(anti)[0]
    nbr = np.searchsorted(basis, basis ^ ((1 << i) | (1 << j)))
    M = sp.csr_matrix((np.full(r.size, 0.5 * J), (r.astype(np.int32), nbr[r].astype(np.int32))), shape=(D, D))
    return (M + sp.diags(J * np.where(anti, -0.25, 0.25))).tocsr()
def coef(J, s, c2fac=1.0):
    """exp(s h) = c1 I + c2 h for h = J S.S (triplet J/4, singlet -3J/4)."""
    et, es = np.exp(s * J / 4), np.exp(-3 * s * J / 4)
    return 0.75 * et + 0.25 * es, c2fac * (et - es) / J
J2B = [(b, NNN[k]) for k, b in enumerate(NNN)]
def Gmat(i, j, J, s, c2fac=1.0):
    c1, c2 = coef(J, s, c2fac); return (c1 * sp.identity(D, format='csr') + c2 * bond_op(i, j, J)).tocsr()
def L1(Phit, bonds, J, s):
    """psi_L = Phit * prod_b g_b, g_b = (G_b Phit)/Phit, G_b = exp(s h_b). Returns psi, number of Phit == 0 entries."""
    nz = Phit != 0; psi = Phit.copy()
    for (i, j) in bonds:
        GP = Gmat(i, j, J, s) @ Phit; g = np.ones(D); g[nz] = GP[nz] / Phit[nz]; psi = psi * g
    return psi, int((~nz).sum())
def two_site(J, s, c2fac=1.0):
    sz = np.diag([0.5, -0.5]); sp_ = np.array([[0, 1], [0, 0.]]); SS = np.kron(sz, sz) + 0.5 * (np.kron(sp_, sp_.T) + np.kron(sp_.T, sp_))
    from scipy.linalg import expm
    c1, c2 = coef(J, s, c2fac); return float(np.abs(c1 * np.eye(4) + c2 * (J * SS) - expm(s * J * SS)).max())
# ---- checks
ck['two_site_G_vs_expm_correct'] = max(two_site(J2, -t) for t in (0.16, 0.2)) ; ck['two_site_G_vs_expm_correct_J1'] = two_site(1.0, 0.19)
ck['two_site_G_vs_expm_wrong_c2x1.01'] = min(two_site(J2, -0.16, 1.01), two_site(J2, -0.2, 1.01))
hb0 = bond_op(*NNN[5], J2); v = np.random.default_rng(1).standard_normal(D)
ck['sector_G_vs_expm_multiply_t0.16'] = float(np.abs(Gmat(*NNN[5], J2, -0.16) @ v - sla.expm_multiply(-0.16 * hb0, v)).max())
ck['sector_G_vs_expm_multiply_wrong_c2'] = float(np.abs(Gmat(*NNN[5], J2, -0.16, 1.01) @ v - sla.expm_multiply(-0.16 * hb0, v)).max())
log('checks 2-site', {k: v_ for k, v_ in ck.items() if 'site' in k or 'sector_G' in k})
for t in TS:
    Phi1 = sla.expm_multiply(+t * hb0, psi0); psi1, nz1 = L1(Phi1, [NNN[5]], J2, -t)
    Phiw = sla.expm_multiply(-t * hb0, psi0); psiw, nzw = L1(Phiw, [NNN[5]], J2, -t)
    ck[f'single_bond_L1_vs_psi0_maxabs_t{t}'] = float(np.abs(psi1 - psi0).max()); ck[f'single_bond_L1_wrongsign_maxabs_t{t}'] = float(np.abs(psiw - psi0).max())
    ck[f'single_bond_zeros_t{t}'] = [nz1, nzw]
# ---- main
import signal
def fn_trunc(v, cut):
    """same lattice FN as fn_energy, but states whose FN diagonal potential vsf exceeds cut are dropped (they cannot carry the ground state);
    fn_energy itself loses all precision when the guide spans > 1e30. Returns (E, n_dropped, psi0^2 weight dropped, |v|^2 weight dropped, rel residual)."""
    s = np.where(v >= 0, 1.0, -1.0); g = np.maximum(np.abs(v), 1e-300)
    viol = Hoff.data * s[ROW] * s[COL] > 0
    vsf = np.bincount(ROW[viol], weights=Hoff.data[viol] * s[ROW[viol]] * s[COL[viol]] * g[COL[viol]] / g[ROW[viol]], minlength=D)
    keep = np.nonzero(vsf < cut)[0]; Mk = sp.csr_matrix((np.where(viol, 0.0, Hoff.data), COL, Hoff.indptr), shape=(D, D))[keep][:, keep].tocsr(); dd = (dH + vsf)[keep]; n = keep.size
    op = sla.LinearOperator((n, n), matvec=lambda x: Mk @ x + (dd[:, None] * x if x.ndim == 2 else dd * x), dtype=float)
    vn = v / np.linalg.norm(v); dr = np.ones(D, bool); dr[keep] = False
    if n < 100000: lam = float(sla.eigsh(op, k=1, which='SA', tol=1e-11)[0][0]); return lam, int(D - n), float(p0[dr].sum()), float((vn[dr] ** 2).sum()), None
    x0 = (np.abs(psi0[keep]) / np.linalg.norm(psi0[keep]))[:, None]; lam0 = float(x0[:, 0] @ op.matvec(x0[:, 0])); pre = np.maximum(np.abs(dd - lam0) + 1.0, 1.0)
    Minv = sla.LinearOperator((n, n), matvec=lambda x: x / (pre[:, None] if x.ndim == 2 else pre), dtype=float)
    lam, X, hist = sla.lobpcg(op, x0, M=Minv, largest=False, tol=1e-9, maxiter=400, retLambdaHistory=True); x = X[:, 0]
    r = float(np.linalg.norm(op.matvec(x) - lam[0] * x) / abs(lam[0])); log('  lobpcg trunc iters', len(hist), 'rel resid', r)
    return float(lam[0]), int(D - n), float(p0[dr].sum()), float((vn[dr] ** 2).sum()), r
class TO(Exception): pass
def raw_fn(v, sec=300):
    def h(*a): raise TO()
    signal.signal(signal.SIGALRM, h); signal.alarm(sec)
    try: return (fn_energy(v) - E0) / N
    except TO: return 'timeout %ds' % sec
    finally: signal.alarm(0)
p0 = psi0 ** 2
fn0 = fn_energy(psi0); ck['fn_psi0_minus_E0'] = fn0 - E0; ck['fn_trunc_psi0_minus_E0'] = [fn_trunc(psi0, c)[0] - E0 for c in (1e6, 1e8)]; ck['fn_trunc_marshall_minus_E0'] = fn_trunc(sN * np.abs(psi0), 1e6)[0] - E0; log('fn(psi0)-E0', fn0 - E0, 'checks', ck); dump()
def measure(psiL, p0):
    """metrics of a candidate psiL against psi0."""
    ov = float(psi0 @ psiL / np.linalg.norm(psiL)); psiL = psiL / np.linalg.norm(psiL) * (1 if ov >= 0 else -1); ov = abs(ov)
    err = np.sign(psiL) != np.sign(psi0); out = dict(overlap=ov ** 2, sign_err_weight=float(p0[err].sum()), n_zero=int((psiL == 0).sum()))
    dec = np.floor(np.log10(np.maximum(p0, 1e-300))).astype(int); out['by_decade'] = {}
    for d_ in sorted(set(dec)):
        m = dec == d_; out['by_decade'][int(d_)] = dict(weight=float(p0[m].sum()), err_weight=float(p0[m & err].sum()), n=int(m.sum()), n_err=int((m & err).sum()))
    out['E_var_site'] = (en(psiL) - E0) / N; out['E_FN_site_raw_fn_energy'] = raw_fn(psiL)
    a = np.abs(psiL); out['abs_psiL_min_max'] = [float(a.min()), float(a.max())]; out['E_FN_trunc'] = {}
    for cut in (1e6, 1e8):
        e, nd, w0, wl, r = fn_trunc(psiL, cut); out['E_FN_trunc'][str(cut)] = dict(E_FN_site=(e - E0) / N, n_dropped=nd, psi0sq_dropped=w0, psiLsq_dropped=wl, rel_resid=r)
    out['E_FN_site'] = out['E_FN_trunc']['1000000.0']['E_FN_site']; return out
p0 = psi0 ** 2; res['t'] = {}
for t in TS:
    tt = time.time(); R = {}; ph = sla.expm_multiply(t * HJ2, psi0); Pt = sN * np.abs(ph); R['min_abs_Phi_tilde'] = float(np.abs(Pt).min() / np.linalg.norm(ph))
    R['R2_E_FN_rec_site'] = R2J['t'][str(t)]['E_FN_rec_site']; R['R2_E_FN_marshall_site'] = R2J['t'][str(t)]['E_FN_marshall_site']
    psiL, nz = L1(Pt, [b for _, b in J2B], J2, -t); R['n_Phi_tilde_zero'] = nz; R['L1_J2'] = measure(psiL, p0); log('t', t, 'J2', {k: v_ for k, v_ in R['L1_J2'].items() if k != 'by_decade'})
    Ft = F(ph); u = u_match(Ft, 1.16 * t); pj = sla.expm_multiply(-u * HJ1, psi0); R['u_matched'] = u; R['F_J2'] = Ft; R['F_J1'] = F(pj)
    PtJ = sN * np.abs(pj); R['min_abs_PhiJ1_tilde'] = float(np.abs(PtJ).min() / np.linalg.norm(pj))
    psiJ, nzJ = L1(PtJ, NN, 1.0, +u); R['n_PhiJ1_tilde_zero'] = nzJ; R['L1_J1'] = measure(psiJ, p0); log('t', t, 'J1', {k: v_ for k, v_ in R['L1_J1'].items() if k != 'by_decade'})
    R['sec'] = time.time() - tt; res['t'][str(t)] = R; dump()
res['sec_total'] = time.time() - T0; dump(); log('DONE')
