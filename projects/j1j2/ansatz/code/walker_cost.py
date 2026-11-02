#!/usr/bin/env python3
"""walker_cost.py: PLAN_pfaffian.md "Two-PP class (d)", COST paragraph.  Per-walker-step CPU cost of three guides at L x L.
usage: python3 walker_cost.py OUTDIR L {check|time} [NCONF] [NREP]
Extends t2_lemma.py (T2) to arbitrary L with random complex pairing matrices f (N x N); everything else is T2's convention:
  - full symmetry projection: A(x) = sum over S = 2 x 8 x N images g (C4v x translations x spin flip) of par(z_g) det f[Up_g, Dn_g];
  - per walker step the inverse G_g = M_g(x)^-1 (and slogdet) is recomputed for every image (T2 'Prep'), no persistent inverse;
  - k-swap endpoint ratios by the rank-2k lemma ('B-form': bordered 2k x 2k determinant from G, T = f_DD G, S = G f_UU,
    W = T f_UU), one compiled C loop over (image, move), identical kernel for every arm;
  - sign evaluation (external sign, here the Marshall sign) is excluded from the timing, as T2 excluded its orbit lookup.
Arms (same configurations, same walker count, interleaved per configuration):
  P: plain one PP: G, T and all d one-swap ratios (k=1 kernel, W/S entries by length-n dots) = T2 'plain_step'.
  J: one PP + one Jacobi level: B_aug = |P| e^{c f_J[|P|]}; all ratios of P at y and at every unique second-shell endpoint w
     from one shared G(x) (k=1,2 kernel), then f_J(x), f_J(y) for every y and the augmented ratios = T2 'aug_step'.
  D: two PPs + one Jacobi level: B = |P_1 + lam P_2|; each P_j its own Prep (inverse, T, S, W) and its own lemma ratios at all
     y and w; per endpoint B(e)/B(x) = |r1(e) + rho r2(e)| / |1 + rho|, rho = lam P_2(x)/P_1(x); then f_J as in J.
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'FLEXIBLAS_NUM_THREADS'):
    os.environ[_k] = '1'
import sys, json, time, hashlib, ctypes, subprocess
import numpy as np

OUT = sys.argv[1]; L = int(sys.argv[2]); MODE = sys.argv[3]
NCONF = int(sys.argv[4]) if len(sys.argv) > 4 else 20; NREP = int(sys.argv[5]) if len(sys.argv) > 5 else 2
os.makedirs(OUT, exist_ok=True)
C_AUG = 3.92
T0 = time.time()


def log(m):
    s = f'[{time.time() - T0:8.1f}s] L={L} {m}'; print(s, flush=True)
    with open(f'{OUT}/walker_cost_L{L}.log', 'a') as fh: fh.write(s + '\n')


# ------------------------------------------------------------------ lattice (t2_lemma.py conventions, general L)
N = L * L; n = N // 2
site = lambda x, y: (x % L) + L * (y % L)
_nn, _nnn = [], []
for y in range(L):
    for x in range(L):
        i = site(x, y); _nn += [(i, site(x + 1, y)), (i, site(x, y + 1))]; _nnn += [(i, site(x + 1, y + 1)), (i, site(x + 1, y - 1))]
BI = np.array([b[0] for b in _nn + _nnn]); BJ = np.array([b[1] for b in _nn + _nnn])
JB = np.r_[np.ones(2 * N), np.full(2 * N, 0.5)]; HWB = JB / 2
HCONST = 0.25 * JB.sum()                                              # 27 at 6x6, as T2
EP = -1.0 * N                                                         # any E' below the diagonal keeps f_J finite; cost-neutral


def space_group():
    def pm(f):
        p = np.empty(N, np.int64)
        for y in range(L):
            for x in range(L):
                xx, yy = f(x, y); p[site(x, y)] = site(xx, yy)
        return p
    c4 = pm(lambda x, y: (-y, x)); sx = pm(lambda x, y: (-x, y)); r = np.arange(N); pts = []
    for k in range(4):
        for refl in (False, True):
            q = r.copy()
            for _ in range(k): q = c4[q]
            if refl: q = sx[q]
            pts.append(q)
    out = []
    for ty in range(L):
        for tx in range(L):
            t = pm(lambda x, y: (x + tx, y + ty))
            for q in pts: out.append(t[q])
    return np.array(out)


SG = space_group(); NSG = len(SG); NG = 2 * NSG
IMG = np.concatenate([SG, SG]); FL = np.r_[np.ones(NSG), -np.ones(NSG)]
INV = np.ascontiguousarray(np.argsort(IMG, axis=1).astype(np.int64))
GI = np.arange(NG)
assert len({tuple(p) for p in SG}) == NSG == 8 * N
XY = np.array([(i % L, i // L) for i in range(N)])
MARSHALL = ((XY[:, 0] + XY[:, 1]) % 2 == 0)


class Model:
    def __init__(self, seed, name):
        rng = np.random.default_rng(seed)
        self.f = (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))) / np.sqrt(2.0)
        self.name = name


def build_M(md, Z, Uo, Do):
    return md.f[Uo[:, :, None], Do[:, None, :]]


def parity(Z):
    up = (Z > 0).astype(np.int64); dn = 1 - up
    return 1 - 2 * ((up * (np.cumsum(dn, 1) - dn)).sum(1) % 2)


def images(sig):
    return FL[None, :, None] * sig[:, IMG]


def direct_logA(md, sig):
    """complex log P for configurations sig (C,N), direct slogdet of all NG images (reference only)."""
    chunk = max(1, int(1.5e7 // (NG * n * n))); out = []
    for c0 in range(0, len(sig), chunk):
        s_ = sig[c0:c0 + chunk]; C = len(s_)
        Zf = images(s_).reshape(-1, N)
        order = np.argsort(-Zf, axis=1, kind='stable'); Uo = order[:, :n]; Do = order[:, n:]
        sg, ld = np.linalg.slogdet(build_M(md, Zf, Uo, Do))
        ph = (parity(Zf) * sg).reshape(C, NG); ld = ld.reshape(C, NG); m = ld.max(1)
        out.append(m + np.log((ph * np.exp(ld - m[:, None])).sum(1)))
    return np.concatenate(out)


class Prep:
    """per-image quantities of walker x (T2 Prep).  mode 'plain_C': G, T;  'full': G, T, S, W."""
    def __init__(self, md, sig, mode):
        Z = FL[:, None] * sig[IMG]; self.Z = np.ascontiguousarray(Z)
        order = np.argsort(-Z, axis=1, kind='stable'); self.Uo = Uo = np.ascontiguousarray(order[:, :n]); self.Do = Do = np.ascontiguousarray(order[:, n:])
        M = build_M(md, Z, Uo, Do)
        self.G = np.linalg.inv(M); sg, ld = np.linalg.slogdet(M)
        self.par = parity(Z); self.m = ld.max(); self.w = self.par * sg * np.exp(ld - self.m); self.S = self.w.sum()
        self.logA = self.m + np.log(self.S)
        self.posU = np.full((NG, N), -1, np.int64); self.posU[GI[:, None], Uo] = np.arange(n)
        self.posD = np.full((NG, N), -1, np.int64); self.posD[GI[:, None], Do] = np.arange(n)
        up = (Z > 0).astype(np.int64); dn = 1 - up
        self.cntUp = np.cumsum(up, 1) - up; self.cntDn = np.cumsum(dn, 1) - dn
        self.T = md.f[Do[:, :, None], Do[:, None, :]] @ self.G
        if mode == 'full':
            fUU = md.f[Uo[:, :, None], Uo[:, None, :]]
            self.Sm = self.G @ fUU
            self.W = self.T @ fUU


# ------------------------------------------------------------------ compiled kernel (t2_lemma.py CSRC, N passed in, broken flag added)
CSRC = r"""
#include <complex.h>
#include <stdint.h>
#include <math.h>
typedef double complex cplx;
/* k = 1 or 2; wdot: W,S entries by dot products (k = 1, plain step); broken: drop the Jacobi slot sign (control) */
void ratio_b(int NGi, int N, int n, int k, int64_t nmv, const int64_t *mv, const int64_t *INV, const double *Z,
             const int64_t *Uo, const int64_t *Do, const int64_t *posU, const int64_t *posD, const int64_t *cntUp, const int64_t *cntDn,
             const cplx *G, const cplx *T, const cplx *S, const cplx *W, const cplx *f, const cplx *w, int wdot, int broken, cplx *out) {
  const int64_t nn = (int64_t)n * n;
  for (int64_t m = 0; m < nmv; m++) out[m] = 0.0;
  for (int g = 0; g < NGi; g++) {     /* image outer: G_g, T_g, S_g, W_g stay in cache */
    for (int64_t m = 0; m < nmv; m++) {
      int64_t R[2], A[2], P[2], Q[2]; int nr = 0, na = 0;
      for (int c = 0; c < 2 * k; c++) {
        int64_t s = INV[(int64_t)g * N + mv[m * 2 * k + c]];
        if (Z[(int64_t)g * N + s] > 0) P[nr++] = posU[(int64_t)g * N + s]; else Q[na++] = posD[(int64_t)g * N + s];
      }
      if (k == 2) { if (P[0] > P[1]) { int64_t t = P[0]; P[0] = P[1]; P[1] = t; } if (Q[0] > Q[1]) { int64_t t = Q[0]; Q[0] = Q[1]; Q[1] = t; } }
      for (int a = 0; a < k; a++) { R[a] = Uo[(int64_t)g * n + P[a]]; A[a] = Do[(int64_t)g * n + Q[a]]; }
      const cplx *Gg = G + g * nn, *Tg = T + g * nn, *Sg = S + g * nn;
      int64_t e = 0;
      for (int a = 0; a < k; a++) { e += (broken ? 0 : P[a] + Q[a]) + (n - cntUp[(int64_t)g * N + A[a]]) + (n - cntDn[(int64_t)g * N + R[a]]) + A[a] - R[a];
        for (int b = 0; b < k; b++) e -= (R[b] > A[a]) + (A[a] > R[b]); }
      cplx det;
      if (k == 1) {
        cplx Wq, Sq;
        if (wdot) { Wq = 0.0; Sq = 0.0; const cplx *tr = Tg + Q[0] * n, *gr = Gg + Q[0] * n;
          for (int p = 0; p < n; p++) { cplx fc = f[Uo[(int64_t)g * n + p] * N + R[0]]; Wq += tr[p] * fc; Sq += gr[p] * fc; } }
        else { Wq = W[g * nn + Q[0] * n + P[0]]; Sq = Sg[Q[0] * n + P[0]]; }
        det = (f[A[0] * N + R[0]] - Wq) * Gg[Q[0] * n + P[0]] + Tg[Q[0] * n + P[0]] * Sq;
      } else {
        cplx K[4][4]; const cplx *Wg = W + g * nn;
        for (int a = 0; a < 2; a++) for (int b = 0; b < 2; b++) {
          int64_t ix = Q[a] * n + P[b];
          K[a][b] = f[A[a] * N + R[b]] - Wg[ix]; K[a][2 + b] = -Tg[ix]; K[2 + a][b] = Sg[ix]; K[2 + a][2 + b] = Gg[ix];
        }
        cplx t01 = K[0][0]*K[1][1]-K[0][1]*K[1][0], t02 = K[0][0]*K[1][2]-K[0][2]*K[1][0], t03 = K[0][0]*K[1][3]-K[0][3]*K[1][0];
        cplx t12 = K[0][1]*K[1][2]-K[0][2]*K[1][1], t13 = K[0][1]*K[1][3]-K[0][3]*K[1][1], t23 = K[0][2]*K[1][3]-K[0][3]*K[1][2];
        cplx b01 = K[2][0]*K[3][1]-K[2][1]*K[3][0], b02 = K[2][0]*K[3][2]-K[2][2]*K[3][0], b03 = K[2][0]*K[3][3]-K[2][3]*K[3][0];
        cplx b12 = K[2][1]*K[3][2]-K[2][2]*K[3][1], b13 = K[2][1]*K[3][3]-K[2][3]*K[3][1], b23 = K[2][2]*K[3][3]-K[2][3]*K[3][2];
        det = t01*b23 - t02*b13 + t03*b12 + t12*b03 - t13*b02 + t23*b01;
      }
      out[m] += w[g] * ((e & 1) ? -det : det);
    }
  }
}
"""
_LIB = None
CFLAGS = ['-O3', '-march=native', '-fcx-limited-range']                # = T2


def lib():
    global _LIB
    if _LIB is None:
        so = f'{OUT}/wc_kernel_{hashlib.sha256(CSRC.encode()).hexdigest()[:10]}_{os.uname().nodename.split(".")[0]}.so'
        if not os.path.exists(so):
            src = f'{OUT}/wc_kernel.c'; open(src, 'w').write(CSRC)
            subprocess.check_call(['gcc', *CFLAGS, '-shared', '-fPIC', '-o', so + '.tmp', src]); os.replace(so + '.tmp', so)
        _LIB = ctypes.CDLL(so); _LIB._path = so
    return _LIB


def ratio_C(md, pr, ch, wdot=False, broken=False):
    k = ch.shape[1] // 2; out = np.empty(len(ch), complex)
    P_ = lambda a: a.ctypes.data_as(ctypes.c_void_p)
    S_ = pr.Sm if hasattr(pr, 'Sm') else pr.G; W_ = pr.W if hasattr(pr, 'W') else pr.G
    arrs = [np.ascontiguousarray(ch, dtype=np.int64), INV, pr.Z, pr.Uo, pr.Do, pr.posU, pr.posD, pr.cntUp, pr.cntDn,
            pr.G, pr.T, S_, W_, md.f, np.ascontiguousarray(pr.w, dtype=complex)]
    assert not wdot or k == 1
    lib().ratio_b(ctypes.c_int(NG), ctypes.c_int(N), ctypes.c_int(n), ctypes.c_int(k), ctypes.c_int64(len(ch)), *[P_(a) for a in arrs],
                  ctypes.c_int(int(wdot)), ctypes.c_int(int(broken)), P_(out))
    return out / pr.S


# ------------------------------------------------------------------ moves (T2, with byte-packed keys so that N > 64 works)
def one_swaps(sig):
    v = sig[BI] != sig[BJ]
    return np.stack([BI[v], BJ[v]], 1), HWB[v]


def swap(sig, ij):
    s = np.repeat(sig[None], len(ij), 0); a = np.arange(len(ij))
    s[a, ij[:, 0]], s[a, ij[:, 1]] = sig[ij[:, 1]], sig[ij[:, 0]]
    return s


def hxx(sig):
    return HCONST - 0.5 * (JB * (sig[..., BI] != sig[..., BJ])).sum(-1)


def keys(cfg):
    b = np.packbits(cfg > 0, axis=1)
    return np.ascontiguousarray(b).view(np.dtype((np.void, b.shape[1]))).ravel()


def two_swap_structure(sig):
    ij, hw1 = one_swaps(sig); ys = swap(sig, ij)
    rows_y, rows_hw, ws = [], [], []
    for a, y in enumerate(ys):
        ij2, hw2 = one_swaps(y); ws.append(swap(y, ij2)); rows_y.append(np.full(len(ij2), a)); rows_hw.append(hw2)
    ws = np.concatenate(ws); yi = np.concatenate(rows_y); hw = np.concatenate(rows_hw)
    _, first, inv = np.unique(keys(ws), return_index=True, return_inverse=True)
    wu = ws[first]; diff = wu != sig[None]; kk = diff.sum(1) // 2
    groups = {}
    for k in (0, 1, 2):
        sel = np.nonzero(kk == k)[0]
        if len(sel): groups[k] = (sel, np.nonzero(diff[sel])[1].reshape(len(sel), 2 * k))
    return ij, hw1, ys, yi, hw, inv.ravel(), wu, groups


def marshall(cfg):
    return 1.0 - 2.0 * (((cfg[..., MARSHALL] > 0).sum(-1)) % 2)


def sign_lookup(sig):
    """external sign at x, every y, every unique w (excluded from timing, as T2's oracle lookup)."""
    _, _, ys, _, _, _, wu, _ = two_swap_structure(sig)
    return marshall(sig), marshall(ys), marshall(wu)


def fJ_from(rat_abs, hw, s_x, s_nb, Hd):
    kept = s_x * s_nb < 0
    return np.log((hw * rat_abs * kept).sum()) - np.log(Hd + (hw * rat_abs * ~kept).sum() - EP)


def fJ_y(r1abs, rwabs, yi, hw, inv, ys, sgn):
    sx, sy, sw = sgn
    ra = rwabs[inv] / r1abs[yi]
    kept = sy[yi] * sw[inv] < 0
    num = np.bincount(yi, hw * ra * kept, len(ys)); blk = np.bincount(yi, hw * ra * ~kept, len(ys))
    return np.log(num) - np.log(hxx(ys) + blk - EP)


# ------------------------------------------------------------------ the three walker steps
def plain_step(md, sig, broken=False):
    pr = Prep(md, sig, 'plain_C')
    ij, hw = one_swaps(sig)
    return pr.logA, ratio_C(md, pr, ij, wdot=True, broken=broken)


def pp_all_ratios(md, sig, st, broken=False):
    """one PP: Prep (full) and its ratios at all y and all unique w from the shared G(x)."""
    ij, hw1, ys, yi, hw, inv, wu, groups = st
    pr = Prep(md, sig, 'full')
    r1 = ratio_C(md, pr, ij, broken=broken)
    rw = np.empty(len(wu), complex)
    for k, (sel, ch) in groups.items():
        rw[sel] = 1.0 if k == 0 else ratio_C(md, pr, ch, broken=broken)
    return pr.logA, r1, rw


def aug_from(r1abs, rwabs, sig, st, sgn):
    ij, hw1, ys, yi, hw, inv, wu, groups = st
    fx = fJ_from(r1abs, hw1, sgn[0], sgn[1], hxx(sig))
    fy = fJ_y(r1abs, rwabs, yi, hw, inv, ys, sgn)
    return fx, fy, r1abs * np.exp(C_AUG * (fy - fx))


def aug_step(md, sig, sgn, broken=False):
    st = two_swap_structure(sig)
    lA, r1, rw = pp_all_ratios(md, sig, st, broken)
    fx, fy, raug = aug_from(np.abs(r1), np.abs(rw), sig, st, sgn)
    return np.real(lA) + C_AUG * fx, raug, fx, fy


def twopp_step(md1, md2, lam, sig, sgn, broken=False):
    st = two_swap_structure(sig)
    l1, r1a, rwa = pp_all_ratios(md1, sig, st)
    l2, r1b, rwb = pp_all_ratios(md2, sig, st)
    rho = (1.0 if broken else lam * np.exp(l2 - l1))                   # broken: drop the amplitude ratio P_2(x)/P_1(x)
    den = 1.0 + rho
    R1 = np.abs((r1a + rho * r1b) / den); RW = np.abs((rwa + rho * rwb) / den)
    fx, fy, raug = aug_from(R1, RW, sig, st, sgn)
    return np.real(l1 + np.log(den)) + C_AUG * fx, raug, fx, fy, R1, RW


# ------------------------------------------------------------------ direct references
def clogadd(a, b):
    return a + np.log1p(np.exp(b - a))


def logB2_direct(md1, md2, lam, cfg):
    return np.real(clogadd(direct_logA(md1, cfg), np.log(lam) + direct_logA(md2, cfg)))


def random_config(rng):
    s = -np.ones(N); s[rng.choice(N, n, replace=False)] = 1.0
    return s


def relmax(a, b): return float(np.max(np.abs(a - b) / np.abs(b)))


def check(nconf, seed=11):
    rng = np.random.default_rng(seed)
    md = Model(101, 'P1'); md2 = Model(202, 'P2')
    s0 = random_config(np.random.default_rng(5))
    lam = 0.3 * np.exp(0.7j) * np.exp(np.real(direct_logA(md, s0[None])[0] - direct_logA(md2, s0[None])[0]))   # comparable weights
    rows = []
    for c in range(nconf):
        sig = random_config(rng); sgn = sign_lookup(sig); res = {}
        st = two_swap_structure(sig); ij, hw1, ys, yi, hw, inv, wu, groups = st
        lx = direct_logA(md, sig[None])[0]; ly = direct_logA(md, ys); rd1 = np.exp(ly - lx)
        # P
        lA, r = plain_step(md, sig); res['P_rel'] = relmax(r, rd1); res['P_logA'] = float(abs(lA - lx))
        _, rb = plain_step(md, sig, broken=True); res['P_broken_rel'] = relmax(rb, rd1)
        # J: one-swap and a sample of two-swap ratios; f_J(x); f_J(y) on 3 sampled y from direct A(w)
        _, r1, rw = pp_all_ratios(md, sig, st)
        res['J_r1_rel'] = relmax(r1, rd1)
        pick = rng.choice(len(wu), min(150, len(wu)), replace=False)
        rdw = np.exp(direct_logA(md, wu[pick]) - lx); res['J_rw_rel'] = relmax(rw[pick], rdw)
        kk_pick = (wu[pick] != sig).sum(1) // 2; res['J_rw_k_counts'] = {int(k): int((kk_pick == k).sum()) for k in (0, 1, 2)}
        _, rwb_ = pp_all_ratios(md, sig, st, broken=True)[1:]
        res['J_rw_broken_rel'] = relmax(rwb_[pick], rdw)
        laug, raug, fx, fy = aug_step(md, sig, sgn)
        res['J_fx_abs'] = float(abs(fx - fJ_from(np.abs(rd1), hw1, sgn[0], sgn[1], hxx(sig))))
        ysel = rng.choice(len(ys), 3, replace=False); eJ, eD, eDb = [], [], []
        _, raugD, fxD, fyD, R1D, RWD = twopp_step(md, md2, lam, sig, sgn)
        _, raugDb, fxDb, fyDb, _, _ = twopp_step(md, md2, lam, sig, sgn, broken=True)
        lB_x = logB2_direct(md, md2, lam, sig[None])[0]; lB_y = logB2_direct(md, md2, lam, ys)
        fxD_dir = fJ_from(np.exp(lB_y - lB_x), hw1, sgn[0], sgn[1], hxx(sig))
        res['D_r1_rel'] = relmax(R1D, np.exp(lB_y - lB_x))
        lB_w = logB2_direct(md, md2, lam, wu[pick]); res['D_rw_rel'] = relmax(RWD[pick], np.exp(lB_w - lB_x))
        res['D_fx_abs'] = float(abs(fxD - fxD_dir))
        for a in ysel:
            ij2, hw2 = one_swaps(ys[a]); w2 = swap(ys[a], ij2); sw2 = marshall(w2)
            lw = direct_logA(md, w2)
            fyJ = fJ_from(np.exp(np.real(lw - ly[a])), hw2, sgn[1][a], sw2, hxx(ys[a]))
            eJ.append(abs(fy[a] - fyJ))
            lBw2 = np.real(clogadd(lw, np.log(lam) + direct_logA(md2, w2)))
            fyD_dir = fJ_from(np.exp(lBw2 - lB_y[a]), hw2, sgn[1][a], sw2, hxx(ys[a]))
            eD.append(abs(fyD[a] - fyD_dir)); eDb.append(abs(fyDb[a] - fyD_dir))
            # end-to-end augmented ratio at this y (J and D) from direct quantities
            rJ_dir = np.abs(rd1[a]) * np.exp(C_AUG * (fyJ - fJ_from(np.abs(rd1), hw1, sgn[0], sgn[1], hxx(sig))))
            rD_dir = np.exp(lB_y[a] - lB_x) * np.exp(C_AUG * (fyD_dir - fxD_dir))
            res.setdefault('J_aug_rel', []).append(float(abs(raug[a] - rJ_dir) / rJ_dir))
            res.setdefault('D_aug_rel', []).append(float(abs(raugD[a] - rD_dir) / rD_dir))
        res['J_fy_abs'] = float(max(eJ)); res['D_fy_abs'] = float(max(eD)); res['D_broken_fy_abs'] = float(min(eDb))
        res['D_broken_r1_rel'] = relmax(_two_broken_r1(md, md2, sig, st), np.exp(lB_y - lB_x))
        res['J_aug_rel'] = float(max(res['J_aug_rel'])); res['D_aug_rel'] = float(max(res['D_aug_rel']))
        res['d'] = int(len(ij)); res['q2'] = int(len(wu))
        res['rho_abs'] = float(abs(lam * np.exp(direct_logA(md2, sig[None])[0] - lx)))
        log(f'check conf {c}: {res}'); rows.append(res)
    ok = lambda k: max(r[k] for r in rows)
    exact_keys = ['P_rel', 'J_r1_rel', 'J_rw_rel', 'J_aug_rel', 'D_r1_rel', 'D_rw_rel', 'D_aug_rel']
    fj_keys = ['J_fx_abs', 'J_fy_abs', 'D_fx_abs', 'D_fy_abs']
    broken_keys = ['P_broken_rel', 'J_rw_broken_rel', 'D_broken_fy_abs', 'D_broken_r1_rel']
    summ = dict(n_conf=nconf, lam=[float(lam.real), float(lam.imag)], rho_abs=[r['rho_abs'] for r in rows],
                max_rel={k: ok(k) for k in exact_keys}, max_abs_fJ={k: ok(k) for k in fj_keys},
                broken_min={k: min(r[k] for r in rows) for k in broken_keys})
    summ['PASS'] = bool(max(summ['max_rel'].values()) < 1e-9 and max(summ['max_abs_fJ'].values()) < 1e-9)
    summ['BROKEN_FAIL'] = bool(min(summ['broken_min'].values()) > 1e-3)
    return summ, rows


def _two_broken_r1(md, md2, sig, st):
    """broken D control: B ratio without the amplitude ratio rho = lam P_2(x)/P_1(x) (fixed rho = 1)."""
    l1, r1a, _ = pp_all_ratios(md, sig, st); l2, r1b, _ = pp_all_ratios(md2, sig, st)
    return np.abs((r1a + r1b) / 2.0)


# ------------------------------------------------------------------ timing
def timing(nconf, nrep, seed=777):
    rng = np.random.default_rng(seed)
    md = Model(101, 'P1'); md2 = Model(202, 'P2'); lam = 0.1 + 0.0j
    confs = [random_config(rng) for _ in range(nconf)]
    T = {'P': [], 'J': [], 'D': []}; meta = []
    # warm-up (compile, page in, BLAS init) on a configuration outside the timed set
    w0 = random_config(np.random.default_rng(1)); s0 = sign_lookup(w0)
    plain_step(md, w0); aug_step(md, w0, s0); twopp_step(md, md2, lam, w0, s0)
    orders = [('P', 'J', 'D'), ('D', 'P', 'J'), ('J', 'D', 'P')]
    for c, sig in enumerate(confs):
        sgn = sign_lookup(sig)                                            # excluded
        st = two_swap_structure(sig); meta.append(dict(d=int(len(st[0])), q2=int(len(st[6])),
                                                       n_k={int(k): int(len(v[0])) for k, v in st[7].items()}))
        for rep in range(nrep):
            for arm in orders[(c + rep) % 3]:
                t0 = time.perf_counter()
                if arm == 'P': plain_step(md, sig)
                elif arm == 'J': aug_step(md, sig, sgn)
                else: twopp_step(md, md2, lam, sig, sgn)
                T[arm].append(time.perf_counter() - t0)
        log(f'time conf {c}: d {meta[-1]["d"]} q2 {meta[-1]["q2"]} | P {np.median(T["P"][-nrep:]):.4f}s J {np.median(T["J"][-nrep:]):.4f}s D {np.median(T["D"][-nrep:]):.4f}s')
    # component split on the first configuration (where the time goes)
    sig = confs[0]; sgn = sign_lookup(sig); comp = {}
    t0 = time.perf_counter(); st = two_swap_structure(sig); comp['structure'] = time.perf_counter() - t0
    t0 = time.perf_counter(); pr = Prep(md, sig, 'plain_C'); comp['prep_plain'] = time.perf_counter() - t0
    t0 = time.perf_counter(); prf = Prep(md, sig, 'full'); comp['prep_full'] = time.perf_counter() - t0
    t0 = time.perf_counter(); ratio_C(md, pr, st[0], wdot=True); comp['kernel_plain_k1'] = time.perf_counter() - t0
    t0 = time.perf_counter(); ratio_C(md, prf, st[0]); comp['kernel_full_k1_y'] = time.perf_counter() - t0
    for k in (1, 2):
        if k in st[7]:
            t0 = time.perf_counter(); ratio_C(md, prf, st[7][k][1]); comp[f'kernel_w_k{k}'] = time.perf_counter() - t0
    med = lambda a: float(np.median(a))
    tP, tJ, tD = med(T['P']), med(T['J']), med(T['D'])
    # per-configuration ratios (paired)
    per = lambda a, b: [float(np.median(T[a][i * nrep:(i + 1) * nrep]) / np.median(T[b][i * nrep:(i + 1) * nrep])) for i in range(nconf)]
    pJ, pD = per('J', 'P'), per('D', 'P')
    return dict(n_conf=nconf, n_rep=nrep, n_steps_per_arm=len(T['P']), lam=[0.1, 0.0],
                t_P=tP, t_J=tJ, t_D=tD, J_over_P=tJ / tP, D_over_P=tD / tP, D_over_J=tD / tJ,
                J_over_P_perconf_median=med(pJ), J_over_P_perconf_minmax=[min(pJ), max(pJ)],
                D_over_P_perconf_median=med(pD), D_over_P_perconf_minmax=[min(pD), max(pD)],
                t_P_iqr=[float(np.percentile(T['P'], 25)), float(np.percentile(T['P'], 75))],
                t_J_iqr=[float(np.percentile(T['J'], 25)), float(np.percentile(T['J'], 75))],
                t_D_iqr=[float(np.percentile(T['D'], 25)), float(np.percentile(T['D'], 75))],
                d_median=med([m['d'] for m in meta]), q2_median=med([m['q2'] for m in meta]),
                components_conf0=comp, raw=T, meta=meta)


def opmodel(d, q2):
    """Astra eq. (7)-(8) per image, with the measured d, q2 at this L."""
    C0 = 7 / 3 * n ** 3 + d * (2 * n + 8); C1 = 13 / 3 * n ** 3 + 12 * d + 56 * q2
    return dict(C0=C0, C1=C1, J_over_P=C1 / C0, D_over_P=2 * C1 / C0)


if __name__ == '__main__':
    R = dict(host=os.uname().nodename, L=L, N=N, n=n, n_images=NG, script_sha256=hashlib.sha256(open(__file__, 'rb').read()).hexdigest(),
             numpy=np.__version__, cflags=CFLAGS, affinity=sorted(os.sched_getaffinity(0)))
    lib(); R['kernel_so'] = lib()._path; R['kernel_sha256'] = hashlib.sha256(open(lib()._path, 'rb').read()).hexdigest()
    RF = f'{OUT}/walker_cost_L{L}_{MODE}.json'
    if MODE == 'check':
        summ, rows = check(NCONF); R['check'] = summ; R['rows'] = rows; log(f'check summary {summ}')
    elif MODE == 'time':
        T = timing(NCONF, NREP); T['opmodel'] = opmodel(T['d_median'], T['q2_median']); R['time'] = T
        log(f"time summary t_P {T['t_P']:.4f}s t_J {T['t_J']:.4f}s t_D {T['t_D']:.4f}s J/P {T['J_over_P']:.2f} D/P {T['D_over_P']:.2f} "
            f"(per-conf medians {T['J_over_P_perconf_median']:.2f}, {T['D_over_P_perconf_median']:.2f}) opmodel {T['opmodel']}")
    json.dump(R, open(RF, 'w'), indent=1, default=float)
    log(f'DONE {RF}')
