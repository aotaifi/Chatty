#!/usr/bin/env python3
"""t2_lemma.py: PLAN_pfaffian.md "OPEN QUESTION ... Proposal OQ", item T2 (board 20).  numpy, complex128, CPU.
usage: python3 t2_lemma.py OUTDIR {check1|check3|time|all} [NPROC]
Model: complex PP + 3x3 backflow of test C copy s1 (cx_c1fit.py "C1 MODEL" block re-implemented in numpy), and the
plain PP = same f with u1 = u2 = 0.  A(x) = sum over 576 images g of par(z_g) det M_g(z_g).
Low-rank routes (per image, G_g = M_g(x)^-1):
  plain PP  'B-form': a move with k up spins moved (k=1 one swap, k<=2 two swaps) is a rank-2k update; ratio = signs x
            det [[f[A,R] - W[Q,P], -T[Q,P]], [S[Q,P], G[Q,P]]]  (2k x 2k), T = f[Do,Do] G, S = G f[Uo,Uo], W = T f[Uo,Uo]
            (bordered matrix + Jacobi complementary minor).  P/Q = slots of the removed up/down sites.
  generic   (any backflow): Delta M = M(z) - M(x) in slot form is nonzero on the rows/columns of every site whose 3x3 window
            contains a changed site; Delta M = U V^T with r = (#such rows + #such columns); ratio = det(I_r + V^T G U).
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'FLEXIBLAS_NUM_THREADS'):
    os.environ[_k] = '1'
import sys, json, time, hashlib
import numpy as np

OUT = sys.argv[1]; MODE = sys.argv[2] if len(sys.argv) > 2 else 'all'; NPROC = int(sys.argv[3]) if len(sys.argv) > 3 else 1
os.makedirs(OUT, exist_ok=True)
RD = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_c/runs/c1_cplx_s1/round_1'
THP = f'{RD}/theta.npy'; LAP = f'{RD}/la_table.npy'
FTAB = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_l/F.npy'
S2P = '/project/theorie/a/A.Otaifi/Chatty/runs/compress6/runs/refresh_17409419/s2.npy'
REPS = '/project/theorie/a/A.Otaifi/Chatty/runs/ll6/data/psi0_6x6_table.npz'      # = l_class.py REPS (identical to stall6 copy)
EP = -18.13639441554879; C_AUG = 3.92
T0 = time.time()


def log(m):
    s = f'[{time.time() - T0:8.1f}s] {m}'; print(s, flush=True)
    with open(f'{OUT}/t2.log', 'a') as fh: fh.write(s + '\n')


# ------------------------------------------------------------------ lattice (rbmpp_lib.Lattice conventions)
L = 6; N = 36; n = 18; NG = 576
site = lambda x, y: (x % L) + L * (y % L)
_nn, _nnn = [], []
for y in range(L):
    for x in range(L):
        i = site(x, y); _nn += [(i, site(x + 1, y)), (i, site(x, y + 1))]; _nnn += [(i, site(x + 1, y + 1)), (i, site(x + 1, y - 1))]
BI = np.array([b[0] for b in _nn + _nnn]); BJ = np.array([b[1] for b in _nn + _nnn])
JB = np.r_[np.ones(72), np.full(72, 0.5)]; HWB = JB / 2


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


SG = space_group(); IMG = np.concatenate([SG, SG]); FL = np.r_[np.ones(288), -np.ones(288)]
INV = np.argsort(IMG, axis=1)                     # Z_g[INV[g, i]] = FL_g sigma_i
NBR = np.array([[site(x + dx, y + dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1)] for y in range(L) for x in range(L)])
XY = np.array([(i % L, i // L) for i in range(N)])
DISP = np.array([[((XY[j, 0] - XY[i, 0]) % L) + L * ((XY[j, 1] - XY[i, 1]) % L) for j in range(N)] for i in range(N)])
SH = np.arange(N, dtype=np.uint64); POW = (np.uint64(1) << SH)
GI = np.arange(NG)


class Model:
    def __init__(self, bf):
        th = np.load(THP); assert th.shape == (3888,)
        tc = th[:1944] + 1j * th[1944:]
        self.f = tc[:N * N].reshape(N, N).copy(); u = tc[N * N:].reshape(2, 9, N)
        self.bf = bf; self.u1 = u[0].copy(); self.u2 = u[1].copy()
        self.name = 'backflow' if bf else 'plain'


def build_M(md, Z, Uo, Do):
    """Z (B,36) +-1, Uo/Do (B,n) slot lists -> M (B,n,n) complex (cx_c1fit log_one)."""
    M = md.f[Uo[:, :, None], Do[:, None, :]]
    if md.bf:
        zn = Z[:, NBR]                                                   # (B,36,9)
        za = np.take_along_axis(zn, Uo[:, :, None], 1); zb = np.take_along_axis(zn, Do[:, :, None], 1)
        M1 = za @ md.u1; M2 = zb @ md.u2                                 # (B,n,36)
        M = M + np.take_along_axis(M1, DISP[Uo[:, :, None], Do[:, None, :]], 2) \
              + np.take_along_axis(M2, DISP[Do[:, :, None], Uo[:, None, :]], 2).transpose(0, 2, 1)
    return M


def parity(Z):
    up = (Z > 0).astype(np.int64); dn = 1 - up
    return 1 - 2 * ((up * (np.cumsum(dn, 1) - dn)).sum(1) % 2)


def images(sig):                                   # sig (C,36) -> (C,576,36)
    return FL[None, :, None] * sig[:, IMG]


def direct_logA(md, sig, chunk=64):
    """complex log A for configurations sig (C,36), direct slogdet of all 576 images."""
    out = []
    for c0 in range(0, len(sig), chunk):
        s_ = sig[c0:c0 + chunk]; C = len(s_)
        Zf = images(s_).reshape(-1, N)
        order = np.argsort(-Zf, axis=1, kind='stable'); Uo = order[:, :n]; Do = order[:, n:]
        sg, ld = np.linalg.slogdet(build_M(md, Zf, Uo, Do))
        ph = (parity(Zf) * sg).reshape(C, NG); ld = ld.reshape(C, NG); m = ld.max(1)
        out.append(m + np.log((ph * np.exp(ld - m[:, None])).sum(1)))
    return np.concatenate(out)


def canon(sig):                                    # orbit representative code: min over 576 images
    codes = ((images(sig) > 0).astype(np.uint64) * POW).sum(-1)
    return codes.min(1)


class Prep:
    """per-image quantities of walker x."""
    def __init__(self, md, sig, mode='full'):
        Z = FL[:, None] * sig[IMG]; self.Z = Z
        order = np.argsort(-Z, axis=1, kind='stable'); self.Uo = Uo = order[:, :n]; self.Do = Do = order[:, n:]
        M = build_M(md, Z, Uo, Do); self.M = M
        self.G = np.linalg.inv(M); sg, ld = np.linalg.slogdet(M)
        self.par = parity(Z); self.m = ld.max(); self.w = self.par * sg * np.exp(ld - self.m); self.S = self.w.sum()
        self.logA = self.m + np.log(self.S)
        self.posU = np.full((NG, N), -1); self.posU[GI[:, None], Uo] = np.arange(n)
        self.posD = np.full((NG, N), -1); self.posD[GI[:, None], Do] = np.arange(n)
        up = (Z > 0).astype(np.int64); dn = 1 - up
        self.cntUp = np.cumsum(up, 1) - up; self.cntDn = np.cumsum(dn, 1) - dn
        if mode == 'plain_C':                                              # strict plain baseline: G and T only
            assert not md.bf
            self.T = md.f[Do[:, :, None], Do[:, None, :]] @ self.G
        if mode in ('full', 'plain_min'):
            assert not md.bf
            fUU = md.f[Uo[:, :, None], Uo[:, None, :]]; self.fUU = fUU
            self.T = md.f[Do[:, :, None], Do[:, None, :]] @ self.G          # T[q,p] = (f[Do[q],Do] G)[p]
            self.Sm = self.G @ fUU                                         # S[q,p] = (G f[Uo,Uo[p]])[q]
            if mode == 'full': self.W = self.T @ fUU


def split_moves(pr, ch):
    """ch (Mv, 2k) changed sigma sites -> per image R (removed up), A (added up) sites sorted, slots P, Q: (576,Mv,k)."""
    k = ch.shape[1] // 2
    s = INV[:, ch]                                                       # (576,Mv,2k) image sites
    up = pr.Z[GI[:, None, None], s] > 0
    o = np.argsort(~up, axis=-1, kind='stable'); s = np.take_along_axis(s, o, -1)
    P = np.sort(pr.posU[GI[:, None, None], s[..., :k]], -1); Q = np.sort(pr.posD[GI[:, None, None], s[..., k:]], -1)
    assert (P >= 0).all() and (Q >= 0).all()
    R = pr.Uo[GI[:, None, None], P]; A = pr.Do[GI[:, None, None], Q]
    return P, Q, R, A, s


def ratio_B(md, pr, ch, broken=False, wdot=False):
    """plain-PP low-rank ratios A(w)/A(x) for moves ch (Mv,2k), k>=1.  wdot: W entries by length-n dots (k=1 only)."""
    k = ch.shape[1] // 2
    P, Q, R, A, _ = split_moves(pr, ch)
    g4 = GI[:, None, None, None]; Pi = P[..., None, :]; Qi = Q[..., :, None]
    if wdot:
        assert k == 1
        fcol = md.f[pr.Uo[:, None, :], R[..., 0][:, :, None]]           # (576,Mv,n) f[Uo, r]
        Wq = np.einsum('gmp,gmp->gm', pr.T[GI[:, None], Q[..., 0]], fcol)[..., None, None]
    else:
        Wq = pr.W[g4, Qi, Pi]
    Sc = md.f[A[..., :, None], R[..., None, :]] - Wq
    Tq = pr.T[g4, Qi, Pi]; Sq = pr.Sm[g4, Qi, Pi]; Gq = pr.G[g4, Qi, Pi]
    if k == 1:
        det = Sc[..., 0, 0] * Gq[..., 0, 0] + Tq[..., 0, 0] * Sq[..., 0, 0]
    else:
        K = np.concatenate([np.concatenate([Sc, -Tq], -1), np.concatenate([Sq, Gq], -1)], -2)
        det = np.linalg.det(K)
    jac = (P.sum(-1) + Q.sum(-1)) % 2
    rowinv = (n - pr.cntUp[GI[:, None, None], A]).sum(-1) - (R[..., None, :] > A[..., :, None]).sum((-1, -2))
    colinv = (n - pr.cntDn[GI[:, None, None], R]).sum(-1) - (A[..., :, None] > R[..., None, :]).sum((-1, -2))
    par = (A.sum(-1) - R.sum(-1)) % 2
    e = (rowinv + colinv + par + (0 if broken else jac)) % 2
    rg = (1 - 2 * e) * det
    return (pr.w[:, None] * rg).sum(0) / pr.S



# ------------------------------------------------------------------ compiled kernel for the B-form (fair timing: one C loop over (move, image))
CSRC = r"""
#include <complex.h>
#include <stdint.h>
#include <math.h>
typedef double complex cplx;
static cplx det4(cplx a[4][4]) {
  cplx d = 1.0;
  for (int c = 0; c < 4; c++) {
    int p = c; double best = fabs(creal(a[c][c])) + fabs(cimag(a[c][c]));
    for (int r = c + 1; r < 4; r++) { double v = fabs(creal(a[r][c])) + fabs(cimag(a[r][c])); if (v > best) { best = v; p = r; } }
    if (best == 0.0) return 0.0;
    if (p != c) { for (int k = 0; k < 4; k++) { cplx t = a[c][k]; a[c][k] = a[p][k]; a[p][k] = t; } d = -d; }
    d *= a[c][c]; cplx inv = 1.0 / a[c][c];
    for (int r = c + 1; r < 4; r++) { cplx m = a[r][c] * inv; for (int k = c + 1; k < 4; k++) a[r][k] -= m * a[c][k]; }
  }
  return d;
}
/* k = 1 or 2; wdot: W entries by dot products (k = 1, plain step) */
void ratio_b(int NGi, int n, int k, int64_t nmv, const int64_t *mv, const int64_t *INV, const double *Z,
             const int64_t *Uo, const int64_t *Do, const int64_t *posU, const int64_t *posD, const int64_t *cntUp, const int64_t *cntDn,
             const cplx *G, const cplx *T, const cplx *S, const cplx *W, const cplx *f, const cplx *w, int wdot, cplx *out) {
  const int N = 36; const int64_t nn = (int64_t)n * n;
  for (int64_t m = 0; m < nmv; m++) out[m] = 0.0;
  for (int g = 0; g < NGi; g++) {     /* image outer: G_g, T_g, S_g, W_g stay in cache */
    for (int64_t m = 0; m < nmv; m++) {
      int64_t R[2], A[2], P[2], Q[2]; int nr = 0, na = 0;
      for (int c = 0; c < 2 * k; c++) {
        int64_t s = INV[g * N + mv[m * 2 * k + c]];
        if (Z[g * N + s] > 0) P[nr++] = posU[g * N + s]; else Q[na++] = posD[g * N + s];
      }
      if (k == 2) { if (P[0] > P[1]) { int64_t t = P[0]; P[0] = P[1]; P[1] = t; } if (Q[0] > Q[1]) { int64_t t = Q[0]; Q[0] = Q[1]; Q[1] = t; } }
      for (int a = 0; a < k; a++) { R[a] = Uo[g * n + P[a]]; A[a] = Do[g * n + Q[a]]; }
      const cplx *Gg = G + g * nn, *Tg = T + g * nn, *Sg = S + g * nn;
      int64_t e = 0;
      for (int a = 0; a < k; a++) { e += P[a] + Q[a] + (n - cntUp[g * N + A[a]]) + (n - cntDn[g * N + R[a]]) + A[a] - R[a];
        for (int b = 0; b < k; b++) e -= (R[b] > A[a]) + (A[a] > R[b]); }
      cplx det;
      if (k == 1) {
        cplx Wq;
        cplx Sq;
        if (wdot) { Wq = 0.0; Sq = 0.0; const cplx *tr = Tg + Q[0] * n, *gr = Gg + Q[0] * n;
          for (int p = 0; p < n; p++) { cplx fc = f[Uo[g * n + p] * N + R[0]]; Wq += tr[p] * fc; Sq += gr[p] * fc; } }
        else { Wq = W[g * nn + Q[0] * n + P[0]]; Sq = Sg[Q[0] * n + P[0]]; }
        det = (f[A[0] * N + R[0]] - Wq) * Gg[Q[0] * n + P[0]] + Tg[Q[0] * n + P[0]] * Sq;
      } else {
        cplx K[4][4]; const cplx *Wg = W + g * nn;
        for (int a = 0; a < 2; a++) for (int b = 0; b < 2; b++) {
          int64_t ix = Q[a] * n + P[b];
          K[a][b] = f[A[a] * N + R[b]] - Wg[ix]; K[a][2 + b] = -Tg[ix]; K[2 + a][b] = Sg[ix]; K[2 + a][2 + b] = Gg[ix];
        }
        /* Laplace expansion along rows 0,1 (division-free): sum over column pairs of 2x2 minors */
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


def lib():
    global _LIB
    if _LIB is None:
        import ctypes, subprocess
        src = f'{OUT}/t2_kernel.c'; so = f'{OUT}/t2_kernel_{hashlib.sha256(CSRC.encode()).hexdigest()[:10]}_{os.uname().nodename.split(".")[0]}.so'
        if not os.path.exists(so):
            open(src, 'w').write(CSRC)
            subprocess.check_call(['gcc', '-O3', '-march=native', '-fcx-limited-range', '-shared', '-fPIC', '-o', so + '.tmp', src]); os.replace(so + '.tmp', so)
        _LIB = ctypes.CDLL(so)
    return _LIB


def ratio_C(md, pr, ch, wdot=False):
    import ctypes
    k = ch.shape[1] // 2; out = np.empty(len(ch), complex)
    P_ = lambda a: a.ctypes.data_as(ctypes.c_void_p)
    arrs = [np.ascontiguousarray(a) for a in (ch.astype(np.int64), INV.astype(np.int64), pr.Z, pr.Uo.astype(np.int64), pr.Do.astype(np.int64),
                                             pr.posU.astype(np.int64), pr.posD.astype(np.int64), pr.cntUp.astype(np.int64), pr.cntDn.astype(np.int64),
                                             pr.G, pr.T, pr.Sm if hasattr(pr, 'Sm') else pr.G, pr.W if hasattr(pr, 'W') else pr.G, md.f, pr.w.astype(complex))]
    lib().ratio_b(ctypes.c_int(NG), ctypes.c_int(n), ctypes.c_int(k), ctypes.c_int64(len(ch)), *[P_(a) for a in arrs], ctypes.c_int(int(wdot)), P_(out))
    return out / pr.S


def inv_count(Lst):                                 # (B,n) -> inversion count
    iu = np.triu(np.ones((n, n), bool), 1)
    return ((Lst[:, :, None] > Lst[:, None, :]) & iu).sum((1, 2))


def ratio_generic(md, pr, ch, window=True, chunk=8, rank_sample=0, rng=None):
    """generic low-rank ratios for any model.  window=False: treat the update as rows/cols of the changed sites only
    (correct for plain PP; deliberately broken for backflow).  Returns ratios (Mv,), r_struct (Mv,576), numeric ranks."""
    Mv = len(ch); out = np.empty(Mv, complex); rs = np.empty((Mv, NG), np.int64); nranks = []
    for m0 in range(0, Mv, chunk):
        c = ch[m0:m0 + chunk]; mv = len(c); k = c.shape[1] // 2
        P, Q, R, A, s = split_moves(pr, c)                              # (576,mv,k), s (576,mv,2k)
        B = NG * mv
        gidx = np.repeat(GI, mv)
        Zp = np.repeat(pr.Z, mv, 0).copy(); sf = s.reshape(B, 2 * k)
        Zp[np.arange(B)[:, None], sf] *= -1
        Uo = np.repeat(pr.Uo, mv, 0).copy(); Do = np.repeat(pr.Do, mv, 0).copy()
        Pf = P.reshape(B, k); Qf = Q.reshape(B, k)
        Uo[np.arange(B)[:, None], Pf] = A.reshape(B, k); Do[np.arange(B)[:, None], Qf] = R.reshape(B, k)
        dM = build_M(md, Zp, Uo, Do) - pr.M[gidx]
        aff = np.zeros((B, N), bool)
        if window: aff[np.arange(B)[:, None], NBR[sf].reshape(B, -1)] = True
        else: aff[np.arange(B)[:, None], sf] = True
        rowm = np.take_along_axis(aff, Uo, 1); colm = np.take_along_axis(aff, Do, 1)
        allm = np.concatenate([rowm, colm], 1); r = allm.sum(1); rmax = int(r.max())
        pos = np.cumsum(allm, 1) - 1
        Vt = np.zeros((B, rmax, n), complex); U = np.zeros((B, n, rmax), complex)
        bb, ss = np.nonzero(rowm); kk = pos[bb, ss]
        Vt[bb, kk, :] = dM[bb, ss, :]; U[bb, ss, kk] = 1.0
        dMc = dM * (~rowm)[:, :, None]
        bb, qq = np.nonzero(colm); kk = pos[bb, n + qq]
        Vt[bb, kk, qq] = 1.0; U[bb, :, kk] = dMc[bb, :, qq]
        K = np.eye(rmax) + Vt @ pr.G[gidx] @ U
        det = np.linalg.det(K)
        sgn = (parity(Zp) * pr.par[gidx]) * (1 - 2 * ((inv_count(Uo) + inv_count(Do)) % 2))
        rg = (sgn * det).reshape(NG, mv)
        out[m0:m0 + mv] = (pr.w[:, None] * rg).sum(0) / pr.S
        rs[m0:m0 + mv] = r.reshape(NG, mv).T
        if rank_sample and rng is not None:
            pick = rng.choice(B, min(rank_sample, B), replace=False)
            sv = np.linalg.svd(dM[pick], compute_uv=False)
            nranks += [(int(r[b]), int((sv[t] > 1e-10 * max(sv[t].max(), 1e-300)).sum())) for t, b in enumerate(pick)]
    return out, rs, nranks


# ------------------------------------------------------------------ moves
def sig_of(code): return (((np.uint64(code) >> SH) & np.uint64(1)).astype(np.float64) * 2 - 1)


def one_swaps(sig):
    v = sig[BI] != sig[BJ]
    return np.stack([BI[v], BJ[v]], 1), HWB[v], JB[v]


def swap(sig, ij):
    s = np.repeat(sig[None], len(ij), 0); a = np.arange(len(ij))
    s[a, ij[:, 0]], s[a, ij[:, 1]] = sig[ij[:, 1]], sig[ij[:, 0]]
    return s


def hxx(sig):
    return 27.0 - 0.5 * (JB * (sig[..., BI] != sig[..., BJ])).sum(-1)


def two_swap_structure(sig):
    """neighbours y of x and their neighbours w.  Returns ys (ny,36), y moves (ny,2), for each (y, bond) row:
    yi (row -> y index), hw, uniq index into unique w list; unique w configs and their changed-site lists grouped by k."""
    ij, hw1, _ = one_swaps(sig); ys = swap(sig, ij)
    rows_y, rows_hw, ws = [], [], []
    for a, y in enumerate(ys):
        ij2, hw2, _ = one_swaps(y); ws.append(swap(y, ij2)); rows_y.append(np.full(len(ij2), a)); rows_hw.append(hw2)
    ws = np.concatenate(ws); yi = np.concatenate(rows_y); hw = np.concatenate(rows_hw)
    codes = ((ws > 0).astype(np.uint64) * POW).sum(1)
    uc, first, inv = np.unique(codes, return_index=True, return_inverse=True)
    wu = ws[first]; diff = wu != sig[None]; kk = diff.sum(1) // 2
    groups = {}
    for k in (0, 1, 2):
        sel = np.nonzero(kk == k)[0]
        if len(sel): groups[k] = (sel, np.nonzero(diff[sel])[1].reshape(len(sel), 2 * k))
    return ij, hw1, ys, yi, hw, inv, wu, groups


def fJ_from(rat_abs, hw, s_x, s_nb, Hd):
    kept = s_x * s_nb < 0
    return np.log((hw * rat_abs * kept).sum()) - np.log(Hd + (hw * rat_abs * ~kept).sum() - EP)


# ------------------------------------------------------------------ globals for workers
REPS_A = None; S2 = None; FT12 = None; LA = None


def load_tables():
    global REPS_A, S2, FT12, LA
    REPS_A = np.load(REPS)['reps']; S2 = np.load(S2P).astype(np.float64)
    FT12 = np.load(FTAB, mmap_mode='r'); LA = np.load(LAP, mmap_mode='r')


def orbit_index(sig):
    c = canon(sig); ix = np.searchsorted(REPS_A, c); assert (REPS_A[ix] == c).all(); return ix


# ------------------------------------------------------------------ step 1 + 2 (one walker)
def job1(ix):
    rng = np.random.default_rng(int(ix))
    sig = sig_of(REPS_A[ix]); res = dict(ix=int(ix))
    ij, hw, _ = one_swaps(sig); zs = swap(sig, ij); ch = ij
    for bf in (True, False):
        md = Model(bf); pr = Prep(md, sig, mode='lite' if bf else 'full')
        lz = direct_logA(md, zs); lx = direct_logA(md, sig[None])[0]
        rd = np.exp(lz - lx)
        rl, rs, nr = ratio_generic(md, pr, ch, window=bf, rank_sample=20, rng=rng)
        tag = md.name
        res[f'{tag}_dlogA_prep'] = float(abs(lx - pr.logA))
        res[f'{tag}_rel_generic'] = float(np.max(np.abs(rl - rd) / np.abs(rd)))
        isnn = (np.abs(XY[ch[:, 0]] - XY[ch[:, 1]]) % (L - 1) == 0).all(1) | ((np.abs(XY[ch[:, 0]] - XY[ch[:, 1]]).sum(1) % (L - 1)) == 1)
        j1 = np.array([((a, b) in set(_nn)) or ((b, a) in set(_nn)) for a, b in ch])
        res[f'{tag}_r_J1'] = sorted(set(rs[j1].ravel().tolist())); res[f'{tag}_r_J2'] = sorted(set(rs[~j1].ravel().tolist()))
        res[f'{tag}_numrank'] = nr
        if bf:
            rb, _, _ = ratio_generic(md, pr, ch, window=False)
            res['backflow_rel_broken_rank2'] = float(np.max(np.abs(rb - rd) / np.abs(rd)))
            # step 2: f_J of the copy from the low-rank ratios
            s_nb = S2[orbit_index(zs)]
            fl = fJ_from(np.abs(rl), hw, S2[ix], s_nb, hxx(sig)); fd = fJ_from(np.abs(rd), hw, S2[ix], s_nb, hxx(sig))
            res['fJ_lemma'] = float(fl); res['fJ_direct'] = float(fd); res['fJ_table'] = float(FT12[ix, 12])
            res['la_spread'] = float((np.real(lz) - np.asarray(LA[orbit_index(zs)])).std())      # |A| vs la_table, should be ~0 (const offset)
            res['la_off'] = float(np.real(lx) - LA[ix])
        else:
            rB = ratio_B(md, pr, ch); rBb = ratio_B(md, pr, ch, broken=True); rBd = ratio_B(md, pr, ch, wdot=True)
            res['plain_rel_B'] = float(np.max(np.abs(rB - rd) / np.abs(rd)))
            res['plain_rel_Bdot'] = float(np.max(np.abs(rBd - rd) / np.abs(rd)))
            res['plain_rel_B_broken'] = float(np.max(np.abs(rBb - rd) / np.abs(rd)))
            res['plain_rel_C'] = float(max(np.max(np.abs(ratio_C(md, pr, ch) - rd) / np.abs(rd)), np.max(np.abs(ratio_C(md, pr, ch, wdot=True) - rd) / np.abs(rd))))
    return res


# ------------------------------------------------------------------ step 3 (one walker)
def job3(ix, nw=200):
    rng = np.random.default_rng(10 ** 6 + int(ix))
    sig = sig_of(REPS_A[ix]); res = dict(ix=int(ix))
    _, _, _, _, _, _, wu, groups = two_swap_structure(sig)
    res['n_unique_w'] = int(len(wu)); res['n_by_k'] = {int(k): int(len(v[0])) for k, v in groups.items()}
    for bf in (False, True):
        md = Model(bf); pr = Prep(md, sig, mode='lite' if bf else 'full'); tag = md.name
        errs, errs_b = [], []; rk = {}
        for k in (1, 2):
            sel, ch = groups[k]
            pick = rng.choice(len(sel), min(nw, len(sel)), replace=False)
            rd = np.exp(direct_logA(md, wu[sel[pick]]) - pr.logA)
            if not bf:
                rall = ratio_B(md, pr, ch)                               # all unique w
                errs.append(np.max(np.abs(rall[pick] - rd) / np.abs(rd)))
                errs.append(np.max(np.abs(ratio_C(md, pr, ch)[pick] - rd) / np.abs(rd)))
                errs_b.append(np.max(np.abs(ratio_B(md, pr, ch[pick], broken=True) - rd) / np.abs(rd)))
            rg, rs, nr = ratio_generic(md, pr, ch[pick], window=bf, rank_sample=10 if bf else 0, rng=rng)
            errs.append(np.max(np.abs(rg - rd) / np.abs(rd)))
            rk[k] = [int(rs.min()), int(np.median(rs)), int(rs.max())]
            if bf: res[f'numrank_k{k}'] = nr
        res[f'{tag}_rel'] = float(max(errs)); res[f'{tag}_r_minmedmax'] = rk
        if not bf: res['plain_rel_B_broken'] = float(min(errs_b))
    return res


# ------------------------------------------------------------------ step 4 timing (single core)
def plain_step(md, sig, C=False):
    pr = Prep(md, sig, mode='plain_C' if C else 'plain_min')
    ij, hw, _ = one_swaps(sig)
    return pr.logA, (ratio_C(md, pr, ij, wdot=True) if C else ratio_B(md, pr, ij, wdot=True))


def aug_step(md, sig, s2w_cache, C=False):
    """A_aug(x) and A_aug(y)/A_aug(x) at all one-swap neighbours y.  s2 signs (oracle) come from s2w_cache."""
    pr = Prep(md, sig, mode='full')
    ij, hw1, ys, yi, hw, inv, wu, groups = two_swap_structure(sig)
    RB = ratio_C if C else ratio_B
    r1 = RB(md, pr, ij)                                                  # A(y)/A(x)
    rw = np.empty(len(wu), complex)
    for k, (sel, ch) in groups.items():
        rw[sel] = 1.0 if k == 0 else RB(md, pr, ch)                       # A(w)/A(x)
    sx, sy, sw = s2w_cache
    fx = fJ_from(np.abs(r1), hw1, sx, sy, hxx(sig))
    ra = np.abs(rw[inv]) / np.abs(r1[yi])                                # |A(w)/A(y)| per (y, bond) row
    kept = sy[yi] * sw[inv] < 0
    num = np.bincount(yi, hw * ra * kept, len(ys)); blk = np.bincount(yi, hw * ra * ~kept, len(ys))
    fy = np.log(num) - np.log(hxx(ys) + blk - EP)
    return pr.logA + C_AUG * fx, r1 * np.exp(C_AUG * (fy - fx)), fx, fy


def sign_lookup(sig):
    ij, hw1, ys, yi, hw, inv, wu, groups = two_swap_structure(sig)
    return S2[orbit_index(sig[None])[0]], S2[orbit_index(ys)], S2[orbit_index(wu)]


def direct_aug_check(md, sig, s2c, fy_l):
    """f_J(y) from directly evaluated A(w), A(y) (plain PP), on a few y: validates aug_step end to end."""
    ij, hw1, ys, yi, hw, inv, wu, groups = two_swap_structure(sig)
    lw = direct_logA(md, wu); ly = direct_logA(md, ys)
    sx, sy, sw = s2c
    ra = np.exp(np.real(lw[inv] - ly[yi])); kept = sy[yi] * sw[inv] < 0
    num = np.bincount(yi, hw * ra * kept, len(ys)); blk = np.bincount(yi, hw * ra * ~kept, len(ys))
    fy = np.log(num) - np.log(hxx(ys) + blk - EP)
    return float(np.max(np.abs(fy - fy_l)))


def timing(nwalk=20, nbf=2):
    rng = np.random.default_rng(777); ixs = rng.choice(len(REPS_A), nwalk, replace=False)
    mdp = Model(False); mdb = Model(True); res = dict(walkers=[int(i) for i in ixs])
    tp, ta, ts, tdir, nnb, nw, tpc, tac, cdiff = [], [], [], [], [], [], [], [], []
    for t, ix in enumerate(ixs):
        sig = sig_of(REPS_A[ix])
        t0 = time.perf_counter(); s2c = sign_lookup(sig); ts.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); lA, r1 = plain_step(mdp, sig); tp.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); lAa, ra, fx, fy = aug_step(mdp, sig, s2c); ta.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); lAc, r1c = plain_step(mdp, sig, C=True); tpc.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); lAac, rac, fxc, fyc = aug_step(mdp, sig, s2c, C=True); tac.append(time.perf_counter() - t0)
        cdiff.append(float(max(np.max(np.abs(rac - ra) / np.abs(ra)), np.max(np.abs(r1c - r1) / np.abs(r1)), abs(fxc - fx))))
        ij = one_swaps(sig)[0]
        t0 = time.perf_counter(); direct_logA(mdp, swap(sig, ij)); direct_logA(mdp, sig[None]); tdir.append(time.perf_counter() - t0)
        nnb.append(len(ij)); nw.append(len(two_swap_structure(sig)[6]))
        if t == 0:
            res['aug_end_to_end_fy_maxdiff_vs_direct'] = direct_aug_check(mdp, sig, s2c, fy)
            res['plain_step_ratio_vs_aug_ratio_maxrel'] = float(np.max(np.abs(r1 - ra * np.exp(-C_AUG * (fy - fx))) / np.abs(r1)))
        log(f'time walker {t}: C-kernel plain {tpc[-1]:.4f}s aug {tac[-1]:.4f}s | numpy plain {tp[-1]:.4f}s aug {ta[-1]:.4f}s sign-lookup {ts[-1]:.4f}s direct-plain {tdir[-1]:.4f}s nb {nnb[-1]} unique w {nw[-1]}')
    med = lambda a: float(np.median(a))
    res['plain_PP'] = dict(t_plain=med(tpc), t_aug=med(tac), ratio=med(tac) / med(tpc), C_vs_numpy_maxrel=max(cdiff),
                           numpy_t_plain=med(tp), numpy_t_aug=med(ta), numpy_ratio=med(ta) / med(tp), t_sign_lookup_oracle=med(ts),
                           t_plain_direct_dets=med(tdir), n_nb_median=med(nnb), n_unique_w_median=med(nw))
    # backflow PP: one-swap by direct dets and by generic lemma (r=12/14), two-swap (aug) by direct dets (r >= n there)
    bt = dict(t_plain_direct=[], t_plain_generic=[], t_aug_direct_extra=[])
    for ix in ixs[:nbf]:
        sig = sig_of(REPS_A[ix]); ij = one_swaps(sig)[0]
        t0 = time.perf_counter(); direct_logA(mdb, sig[None]); direct_logA(mdb, swap(sig, ij)); bt['t_plain_direct'].append(time.perf_counter() - t0)
        t0 = time.perf_counter(); pr = Prep(mdb, sig, 'lite'); ratio_generic(mdb, pr, ij, chunk=8); bt['t_plain_generic'].append(time.perf_counter() - t0)
        t0 = time.perf_counter(); wu = two_swap_structure(sig)[6]; direct_logA(mdb, wu); bt['t_aug_direct_extra'].append(time.perf_counter() - t0)
        log(f'time backflow walker {ix}: ' + ' '.join(f'{k} {v[-1]:.2f}s' for k, v in bt.items()))
    b = {k: med(v) for k, v in bt.items()}
    b['t_plain'] = min(b['t_plain_direct'], b['t_plain_generic']); b['t_aug'] = b['t_plain'] + b['t_aug_direct_extra']
    b['ratio'] = b['t_aug'] / b['t_plain']; res['backflow_PP'] = b
    return res


def opmodel(nnb, nw, r1=13, r2=26):
    """complex multiply-add counts per image (LU ~ n^3/3, inverse ~ n^3, matmul n^3)."""
    n3 = n ** 3
    plain = dict(LU_inv=4 / 3 * n3, T=n3, per_nb=nnb * (2 * n + 8)); P = sum(plain.values())
    aug = dict(base=P - nnb * (2 * n + 8), S_W=2 * n3, per_nb=nnb * 12, per_w=nw * (30 + 16 + 10)); Aa = sum(aug.values())
    bfp = dict(LU=n3 / 3, per_nb_direct=nnb * (n3 / 3 + 9 * n * n)); BP = sum(bfp.values())
    bfa = dict(base=BP, per_w_direct=nw * (n3 / 3 + 9 * n * n)); BA = sum(bfa.values())
    lemma_nb_bf = r1 * n * n + r1 * r1 * n + r1 ** 3 / 3
    return dict(plain_PP=dict(plain=plain, aug=aug, ratio=Aa / P), backflow_PP=dict(plain=bfp, aug=bfa, ratio=BA / BP,
                lemma_cost_per_one_swap_r13=lemma_nb_bf, direct_cost_per_swap=n3 / 3 + 9 * n * n,
                lemma_cost_two_swap_r26=r2 * n * n + r2 * r2 * n + r2 ** 3 / 3))


if __name__ == '__main__':
    load_tables()
    R = dict(host=os.uname().nodename, sha256=hashlib.sha256(open(__file__, 'rb').read()).hexdigest(),
             theta=THP, theta_sha256=hashlib.sha256(open(THP, 'rb').read()).hexdigest())
    RF = f'{OUT}/t2_result.json'
    if os.path.exists(RF): R.update({k: v for k, v in json.load(open(RF)).items() if k not in R})
    dump = lambda: json.dump(R, open(RF, 'w'), indent=1, default=float)
    rng = np.random.default_rng(2026)
    X = rng.choice(len(REPS_A), 1000, replace=False)
    # sanity: reps are min-code orbit reps
    assert (canon(np.stack([sig_of(REPS_A[i]) for i in X[:50]])) == REPS_A[X[:50]]).all()
    if MODE in ('check1', 'all'):
        from multiprocessing import Pool
        with Pool(NPROC) as pool:
            rows = []
            for t, r in enumerate(pool.imap_unordered(job1, X, chunksize=4)):
                rows.append(r)
                if t % 100 == 0: log(f'check1 {t + 1}/1000 last {r}')
        json.dump(rows, open(f'{OUT}/check1_rows.json', 'w'), default=float)
        g = lambda k: [r[k] for r in rows]
        R['step1'] = dict(n_x=len(rows),
                          backflow_maxrel_lemma=max(g('backflow_rel_generic')), plain_maxrel_lemmaB=max(g('plain_rel_B')),
                          plain_maxrel_lemmaBdot=max(g('plain_rel_Bdot')), plain_maxrel_generic=max(g('plain_rel_generic')),
                          broken_backflow_as_rank2_minrel=min(g('backflow_rel_broken_rank2')),
                          broken_plain_no_jacobi_sign_minrel=min(g('plain_rel_B_broken')),
                          prep_vs_direct_logA=max(g('backflow_dlogA_prep') + g('plain_dlogA_prep')),
                          r_backflow_J1=sorted(set(sum(g('backflow_r_J1'), []))), r_backflow_J2=sorted(set(sum(g('backflow_r_J2'), []))),
                          r_plain_J1=sorted(set(sum(g('plain_r_J1'), []))), r_plain_J2=sorted(set(sum(g('plain_r_J2'), []))),
                          numrank_backflow=dict(zip(*np.unique(np.array([b - a for r in rows for a, b in r['backflow_numrank']]), return_counts=True))),
                          numrank_plain=dict(zip(*np.unique(np.array([b - a for r in rows for a, b in r['plain_numrank']]), return_counts=True))),
                          la_table_spread_max=max(g('la_spread')))
        R['step1']['numrank_backflow'] = {f'num-struct={int(k)}': int(v) for k, v in R['step1']['numrank_backflow'].items()}
        R['step1']['numrank_plain'] = {f'num-struct={int(k)}': int(v) for k, v in R['step1']['numrank_plain'].items()}
        d = np.array([r['fJ_lemma'] - r['fJ_table'] for r in rows]); dd = np.array([r['fJ_direct'] - r['fJ_table'] for r in rows])
        R['step2'] = dict(max_abs_fJ_lemma_minus_table=float(np.abs(d).max()), max_abs_fJ_direct_minus_table=float(np.abs(dd).max()),
                          fJ_range=[float(min(g('fJ_table'))), float(max(g('fJ_table')))],
                          broken_control_shuffled_table=float(np.median(np.abs(np.array(g('fJ_lemma')) - np.roll(g('fJ_table'), 1)))),
                          PASS=bool(np.abs(d).max() < 1e-8))
        R['step1']['PASS'] = bool(max(R['step1']['backflow_maxrel_lemma'], R['step1']['plain_maxrel_lemmaB']) < 1e-8)
        log(f"step1 {R['step1']}"); log(f"step2 {R['step2']}"); dump()
    if MODE in ('check3', 'all'):
        from multiprocessing import Pool
        with Pool(NPROC) as pool:
            rows = list(pool.imap_unordered(job3, X[:100], chunksize=2))
        json.dump(rows, open(f'{OUT}/check3_rows.json', 'w'), default=float)
        g = lambda k: [r[k] for r in rows]
        R['step3'] = dict(n_x=len(rows), plain_maxrel=max(g('plain_rel')), backflow_maxrel=max(g('backflow_rel')),
                          broken_plain_minrel=min(g('plain_rel_B_broken')),
                          n_unique_w_median=float(np.median(g('n_unique_w'))),
                          plain_r_k1=sorted(set(tuple(r['plain_r_minmedmax']['1']) if '1' in r['plain_r_minmedmax'] else tuple(r['plain_r_minmedmax'][1]) for r in rows)),
                          backflow_r_k1_minmedmax=[min(r['backflow_r_minmedmax'][1][0] for r in rows), float(np.median([r['backflow_r_minmedmax'][1][1] for r in rows])), max(r['backflow_r_minmedmax'][1][2] for r in rows)],
                          backflow_r_k2_minmedmax=[min(r['backflow_r_minmedmax'][2][0] for r in rows), float(np.median([r['backflow_r_minmedmax'][2][1] for r in rows])), max(r['backflow_r_minmedmax'][2][2] for r in rows)],
                          numrank_k2_backflow=sorted(set(b for r in rows for a, b in r['numrank_k2'])))
        R['step3']['PASS'] = bool(max(R['step3']['plain_maxrel'], R['step3']['backflow_maxrel']) < 1e-8)
        log(f"step3 {R['step3']}"); dump()
    if MODE in ('time', 'all'):
        T = timing(); p = T['plain_PP']
        T['opmodel'] = opmodel(p['n_nb_median'], p['n_unique_w_median'])
        p['VERDICT'] = 'PREDICTION HOLDS (<=20)' if p['ratio'] <= 20 else ('KILL cheap (>50)' if p['ratio'] > 50 else 'BETWEEN 20 and 50')
        R['step4'] = T; log(f"step4 {T}"); dump()
    log('DONE')
