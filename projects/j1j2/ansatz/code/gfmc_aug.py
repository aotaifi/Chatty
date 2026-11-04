"""README (build item (2) of "8x8 BUILD, route 2", PLAN_pfaffian.md; worker 2026-11-04).
Base: unmodified copy of experiments/addon_10x10/gfmc_guide.py (sha256 36f6fd11...) = LogpsiGuide, guide factories, CLI, below.
ADDED here: AugGuide, the augmented guide provider for the launched M1 walker code (m1_code/gfmc_fn_lib.py, md5 40a1c409, unchanged):
  CONVENTION: G(x) = s(x) |B(x)| e^{add(x)},  add = c f_J[B] with the FIX (pp2_lemma mode 'fix': no-kept rows -> node add -69,
  else add = c sign(f) min(|f|, f0), i.e. c_eff = c min(1, f0/|f_J|)).  NOT gfmc_guide's Re B: at c = 0 with the network's own sign
  this G equals sgn(Re B) |B|, not Re B (the p=0 8x8 equivalence needs |Re B|; a convention check only).
  Ratios G(y_b)/G(x) come from pp2_lemma.aug_ratios (the 8x8 evaluator; f_J at x and at every y_b from the rank-2/4 lemma).
  s is pluggable: any sign_fn (M, N) +-1 -> (M,) +-1.  Provided: OrbitSign6 (exact 6x6 orbit lookup of a per-rep sign table,
  e.g. s_G = pfaffian_lswo2/s_G.npy or the network's own sign table); sign8 CNN later (pass its sign_fn).
  Walker-level relative weight check (AugGuide.weight_check, every --wcheck_every blocks): on the walker population x_i,
  w_base = n(x)|B(x)|^2 (n = orbit size), w_aug = w_base e^{2 add};  ratio = [max_i w_aug / max_i w_base] * mean_i e^{-2 add(x_i)}
  (the second factor estimates Z_aug/Z_base^-1 with walkers taken as G^2-distributed; they are Phi_FN G-distributed, so it is an
  estimate).  ratio > wcheck_max (10) raises WeightCheckError (augmented max single-orbit share > 10x the base guide's).
  Guide rows are evaluated in power-of-2 walker buckets (<= --aug_chunk) and a monotone dmax, so jit compiles a few shapes only.
Walker time convention (gfmc_fn_lib): continuous time (exponential waiting times, no Trotter error); biases are the resampling
interval dtau (population reconfiguration every dtau) and finite M.
6x6 test c3 data: theta_4 table pfaffian_r4/pdb_tabs/tab_t4/round_1/{theta.npy (2740 = rbmpp arm: one PP + outer RBM NA=4),
la_table.npy = Re logB per rep (checked to 4e-15)}, s_G pfaffian_lswo2/s_G.npy, E' = -18.13670889863218, f0 = 2.4942595701520887/4
(fj_audit.json r4_theta_4 cfJ q9999_nweighted / c), exact capped e = 4.367085e-6 per site (one-sided cap), c = 0: 1.218448e-5.
  python gfmc_aug.py OUT --guide aug6 --c 4 --sign sG --M 64 --dtau 0.1 --blocks B --burn b --seed s  [--E0 -0.5038096538908782]
"""
"""Lattice fixed-node GFMC (p = 0) with a pluggable guide: any batched function logpsi(sigma (B, N) +-1) -> complex (B,).
Guide c(x) = Re psi(x) (its sign = sign Re psi; a global phase w is folded into logpsi as + log w).  J1-J2, J2 = 0.5, L x L PBC.

Walker dynamics, branching and estimators are the LAUNCHED M1 code unchanged (m1_code/gfmc_fn_lib.py, md5 40a1c409: GFMC, GuideBase.finish,
ten Haaf FN: kept hop rate -K r for r = c(y)/c(x) < 0, sign-violating K r > 0 moved to the diagonal = V_sf).  Only the guide provider is new:
LogpsiGuide.rows evaluates logpsi at x and at every one-swap neighbour y (1 + n_anti(x) evaluations per hop), ratio
  r = c(y)/c(x) = Re[exp(l_y - Re l_x)] / Re[exp(i Im l_x)].
Optional mixed precision (ViT): logpsi_lo (f32 'highest') for all pairs; pairs with |psi(y)/psi(x)|_lo < thr (1e-3, the A0.3 held-out
trigger) are recomputed with logpsi_hi (f64, x and y).  drop_vsf=True is the deliberately broken control (V_sf omitted).

  python gfmc_guide.py OUT --guide {rbmpp|table6|vit10} [guide options] --L L --M 256 --dtau 0.1 --blocks B --burn b --seed s
OUT/progress.txt (PROGRESS lines, last 'GFMC_GUIDE_DONE ...'), ckpt.npz (atomic, every --ckpt_every blocks, RNG state; --resume),
blocks.npz, E_mix.npy (per-block mixed estimator / N after burn, time order: quotable.py series input), result.json.
"""
import argparse, json, os, sys, time, pickle
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
M1 = os.environ.get('GFMC_M1_CODE', os.path.normpath(os.path.join(HERE, '..', '..', 'experiments', 'addon_10x10', 'm1_code')))
sys.path.insert(0, M1)
from gfmc_fn_lib import GuideBase, GFMC, blocked_error, iat      # launched M1 walker code (unchanged copy)


def bonds(L, J2=0.5):
    """bond list in the order of rbmpp_lib.Lattice / st6_sector (site = x + L y; NN then NNN)."""
    site = lambda x, y: (x % L) + L * (y % L)
    nn_, nnn = [], []
    for y in range(L):
        for x in range(L):
            i = site(x, y)
            nn_ += [(i, site(x + 1, y)), (i, site(x, y + 1))]
            nnn += [(i, site(x + 1, y + 1)), (i, site(x + 1, y - 1))]
    bl = nn_ + nnn
    return (np.array([b[0] for b in bl], np.int32), np.array([b[1] for b in bl], np.int32),
            np.array([1.0] * len(nn_) + [J2] * len(nnn)))


class LogpsiGuide(GuideBase):
    def __init__(self, logpsi, L, J2=0.5, chunk=256, small=None, in_dtype=np.float64, logpsi_hi=None, thr=1e-3,
                 chunk_hi=64, drop_vsf=False):
        self.N = L * L; self.BI, self.BJ, self.JB = bonds(L, J2); self.NB = len(self.BI)
        self.lo = logpsi; self.hi = logpsi_hi; self.thr = thr; self.drop_vsf = drop_vsf
        self.chunk = chunk; self.small = small or max(chunk // 8, 8); self.chunk_hi = chunk_hi; self.in_dtype = in_dtype
        self.n_eval = 0; self.t_eval = 0.0; self.n_eval_hi = 0
        self.n_pairs = 0; self.n_rechk = 0; self.n_rechk_x = 0; self.n_flip_caught = 0; self.n_x = 0
        self.n_xphase_small = 0; self.max_im = 0.0; self.min_abs_cx = np.inf

    def _eval(self, fn, Z, chunk, small, dt):
        K = Z.shape[0]; out = np.empty(K, np.complex128); i = 0
        while i < K:
            rem = K - i; c = chunk if rem > small else small; n = min(c, rem)
            blk = Z[i:i + n]
            if n < c:
                blk = np.concatenate([blk, np.repeat(blk[:1], c - n, 0)])
            out[i:i + n] = np.asarray(fn(blk.astype(dt)))[:n]
            i += n
        return out

    def ev(self, Z):
        t0 = time.time(); l = self._eval(self.lo, Z, self.chunk, self.small, self.in_dtype)
        self.n_eval += Z.shape[0]; self.t_eval += time.time() - t0
        return l

    def ev_hi(self, Z):
        t0 = time.time(); l = self._eval(self.hi, Z, self.chunk_hi, max(self.chunk_hi // 8, 1), np.float64)
        self.n_eval_hi += Z.shape[0]; self.t_eval += time.time() - t0
        return l

    @staticmethod
    def ratio(LX, LY):
        return np.exp(LY - LX.real).real / np.exp(1j * LX.imag).real

    def rows(self, X):
        B = X.shape[0]; s = X.astype(np.float64)
        anti = X[:, self.BI] != X[:, self.BJ]
        wi, bi = np.nonzero(anti)
        Y = s[wi].copy(); r = np.arange(len(wi))
        Y[r, self.BI[bi]] = s[wi, self.BJ[bi]]; Y[r, self.BJ[bi]] = s[wi, self.BI[bi]]
        l = self.ev(np.concatenate([s, Y]))
        lx = l[:B]; LY = l[B:].copy(); LX = lx[wi].copy()
        self.n_pairs += len(wi); self.n_x += B
        if self.hi is not None:
            sel = np.exp((LY - LX).real) < self.thr          # |psi(y)/psi(x)| < thr in the low precision
            if sel.any():
                ux, inv = np.unique(wi[sel], return_inverse=True)
                lxh = self.ev_hi(s[ux]); lyh = self.ev_hi(Y[sel])
                r_lo = self.ratio(LX[sel], LY[sel])
                LX[sel] = lxh[inv]; LY[sel] = lyh
                self.n_flip_caught += int(np.sum(np.sign(r_lo) != np.sign(self.ratio(LX[sel], LY[sel]))))
                self.n_rechk += int(sel.sum()); self.n_rechk_x += len(ux)
        cxp = np.exp(1j * lx.imag).real
        self.min_abs_cx = min(self.min_abs_cx, float(np.min(np.abs(cxp))))
        self.n_xphase_small += int(np.sum(np.abs(cxp) < 1e-3))
        rr = self.ratio(LX, LY)
        if not (np.all(np.isfinite(rr)) and np.all(np.isfinite(lx.real))):
            raise FloatingPointError(f'non-finite guide ratio: {int(np.sum(~np.isfinite(rr)))} pairs, {int(np.sum(~np.isfinite(lx.real)))} walkers')
        return self.finish(X, anti, wi, bi, rr)

    def finish(self, X, anti, wi, bi, rr):
        if not self.drop_vsf:
            return GuideBase.finish(self, X, anti, wi, bi, rr)
        K = 0.5 * self.JB[bi]; keep = rr < 0                      # BROKEN control: V_sf (sign-violating diagonal) omitted
        rate = np.zeros(anti.shape); rate[wi[keep], bi[keep]] = -K[keep] * rr[keep]
        return rate, self.diag0(X)

    def stats(self):
        return dict(n_eval=self.n_eval, n_eval_hi=self.n_eval_hi, n_pairs=self.n_pairs, n_rechk=self.n_rechk,
                    rechk_frac=self.n_rechk / max(self.n_pairs, 1), n_rechk_x=self.n_rechk_x, n_flip_caught=self.n_flip_caught,
                    n_x=self.n_x, n_xphase_small=self.n_xphase_small, min_abs_cos_phase_x=self.min_abs_cx)


# ------------------------------------------------------------------------------------------ augmented guide (build item (2))
R6 = '/scratch/a/A.Otaifi/ansatz_runs'
REPS6 = '/project/theorie/a/A.Otaifi/Chatty/runs/stall6/data/psi0_6x6_table.npz'      # sorted uint64 reps (= la_table / s_G order)
NORB6 = '/project/theorie/a/A.Otaifi/Chatty/runs/stall6/csr/norb.npy'


class WeightCheckError(RuntimeError):
    pass


class OrbitSign6:
    """exact 6x6 orbit lookup: config (M, 36) +-1 -> rep index (st6_sector byte tables, canonical rep = min over the 288 space-group
    images and their spin flips; bit i = site i up, site = x + 6 y) -> per-rep value.  Raises on a configuration with no rep."""
    def __init__(self, table, reps_path=REPS6, name='?'):
        import st6_sector as SS
        self.reps = np.asarray(np.load(reps_path)['reps']).astype(np.uint64)
        self.T = SS.byte_tables(SS.space_group()); self.FULL = np.uint64((1 << 36) - 1)
        self.code = (np.uint64(1) << np.arange(36, dtype=np.uint64))
        self.s = np.asarray(table).astype(np.float64); self.name = name; self.n_calls = 0; self.n_cfg = 0; self.sec = 0.0
        assert self.s.shape == self.reps.shape and set(np.unique(self.s)) <= {-1.0, 1.0}

    def index(self, S):
        S = np.asarray(S); x = ((S > 0).astype(np.uint64) * self.code).sum(1, dtype=np.uint64)
        img = np.zeros((x.shape[0], self.T.shape[0]), np.uint64)
        for b in range(5):
            img |= self.T[:, b, :][:, ((x >> np.uint64(8 * b)) & np.uint64(255)).astype(np.int64)].T
        m = np.minimum(img.min(1), (img ^ self.FULL).min(1))
        i = np.clip(np.searchsorted(self.reps, m), 0, len(self.reps) - 1)
        if not np.all(self.reps[i] == m):
            raise KeyError(f'OrbitSign6: {int(np.sum(self.reps[i] != m))} configurations without an orbit rep (not Sz = 0?)')
        return i

    def __call__(self, S):
        t0 = time.time(); out = self.s[self.index(S)]; self.n_calls += 1; self.n_cfg += len(out); self.sec += time.time() - t0
        return out


def orbit_size(X, L):
    """number of distinct configurations in the orbit of each row of X under the full space group x spin flip."""
    import pp2_lemma as PL
    lat = PL.lattice(L, 'full'); N = L * L
    Z = (np.asarray(X, np.int8)[:, lat.IMG] * lat.FL[None, :, None].astype(np.int8)) > 0           # (B, nI, N) image configs
    w = (np.uint64(1) << np.arange(N, dtype=np.uint64)) if N <= 64 else None
    codes = (Z.astype(np.uint64) * w).sum(2, dtype=np.uint64)
    return np.array([len(np.unique(c)) for c in codes], np.float64)


class AugGuide(GuideBase):
    """G = s |B| e^{add}, add = c f_J[B] (pp2_lemma mode 'fix' by default), on the launched walker code.  rows() from pp2_lemma.aug_ratios."""
    def __init__(self, params, sign_fn, EP, c, f0, mode='fix', node=None, dtype='f64', chunk=32, wcheck_max=10.0):
        import pp2_lemma as PL
        self.PL = PL; self.p = params; self.sign_fn = sign_fn; self.EP = float(EP); self.c = float(c)
        self.f0 = None if f0 is None else float(f0); self.mode = mode; self.node = PL.NODE_ADD if node is None else float(node)
        self.dtype = dtype; self.chunk = int(chunk); self.wcheck_max = float(wcheck_max)
        lat = params.lat; self.L = lat.L; self.N = lat.N; self.NB = lat.NB
        self.BI = np.asarray(lat.BI); self.BJ = np.asarray(lat.BJ); self.JB = np.asarray(lat.JB, float)
        bi, bj, jb = bonds(lat.L)
        assert np.array_equal(bi, self.BI) and np.array_equal(bj, self.BJ) and np.allclose(jb, self.JB), 'bond order mismatch'
        self.dmax = 8; self.n_eval = 0; self.t_eval = 0.0; self.n_pad = 0; self.n_rows = 0; self.n_x = 0
        self.min_abs_rr = np.inf; self.n_node_x = 0; self.n_cap_x = 0; self.wchecks = []

    def _bucket(self, n):
        b = 1
        while b < n: b *= 2
        return min(b, self.chunk)

    def _aug(self, Xc):
        o = self.PL.aug_ratios(self.p, Xc, self.sign_fn, self.EP, self.c, self.f0, self.mode, dtype=self.dtype, dmax=self.dmax,
                               node=self.node, walker_chunk=Xc.shape[0])
        return o

    def rows(self, X):
        X = np.asarray(X); B = X.shape[0]; t0 = time.time(); ratio = np.empty((B, self.NB)); fx = np.empty(B)
        need = self.PL._dmax_of(self.p.lat, X); self.dmax = max(self.dmax, need)
        i = 0
        while i < B:
            n = min(self.chunk, B - i); bk = self._bucket(n); Xc = X[i:i + n]
            if bk > n: Xc = np.concatenate([Xc, np.repeat(Xc[:1], bk - n, 0)]); self.n_pad += bk - n
            o = self._aug(Xc.astype(np.float64))
            ratio[i:i + n] = np.asarray(o['ratio'])[:n]; fx[i:i + n] = np.asarray(o['fx'])[:n]
            i += n
        self.n_eval += B; self.n_rows += B; self.n_x += B; self.t_eval += time.time() - t0
        self.n_node_x += int(np.sum(~np.isfinite(fx)))
        if self.f0 is not None: self.n_cap_x += int(np.sum(np.isfinite(fx) & (np.abs(fx) > self.f0)))
        anti = X[:, self.BI] != X[:, self.BJ]; wi, bi = np.nonzero(anti); rr = ratio[wi, bi]
        if not np.all(np.isfinite(rr)) or np.any(rr == 0):
            raise FloatingPointError(f'AugGuide: {int(np.sum(~np.isfinite(rr)))} non-finite, {int(np.sum(rr == 0))} zero ratios on antiparallel bonds')
        if np.any(ratio[~anti] != 0):
            raise RuntimeError('AugGuide: non-zero ratio on an aligned bond')
        self.min_abs_rr = min(self.min_abs_rr, float(np.min(np.abs(rr))))
        return self.finish(X, anti, wi, bi, rr)

    def weight_check(self, X, raise_=True):
        """walker-level relative weight check (README).  Returns dict; raises WeightCheckError if ratio > wcheck_max."""
        X = np.asarray(X).astype(np.float64)
        o = self.PL.fJ_fixed(self.p, X, self.sign_fn, self.EP, self.c, self.f0, self.mode, at_y=False, dtype=self.dtype, node=self.node,
                             walker_chunk=X.shape[0])
        lB = np.real(np.asarray(o['logB'])); add = np.asarray(o['add_x']); ln = np.log(orbit_size(X, self.L))
        lwb = ln + 2 * lB; lwa = lwb + 2 * add
        lz = np.log(np.mean(np.exp(-2 * (add - add.max())))) - 2 * add.max()          # log mean e^{-2 add}
        lr = float(lwa.max() - lwb.max() + lz)
        d = dict(ratio=float(np.exp(lr)), add_max=float(add.max()), add_min=float(add.min()), n_node=int(np.sum(add <= self.node)),
                 argmax_aug=int(np.argmax(lwa)), argmax_base=int(np.argmax(lwb)))
        self.wchecks.append(d)
        if raise_ and d['ratio'] > self.wcheck_max:
            raise WeightCheckError(f"augmented max single-orbit share {d['ratio']:.3g}x the base guide's (> {self.wcheck_max})")
        return d

    def stats(self):
        return dict(n_eval=self.n_eval, n_rows=self.n_rows, n_pad=self.n_pad, t_eval=self.t_eval, dmax=self.dmax, n_x=self.n_x,
                    n_node_x=self.n_node_x, n_cap_x=self.n_cap_x, min_abs_ratio=self.min_abs_rr,
                    sign_sec=getattr(self.sign_fn, 'sec', None), sign_cfg=getattr(self.sign_fn, 'n_cfg', None),
                    n_pairs=0, n_rechk=0, n_flip_caught=0)


def theta4_params(path):
    """theta_4 (rbmpp arm: [Re f, Im f, RBM (4, 37)], 2740 values) through PP2.from_cx_theta (P_2 slots zero-padded, then dropped)."""
    import pp2_lemma as PL
    th = np.load(path).astype(np.float64); N = 36
    if th.size == 2 * N * N + 4 * (N + 1):
        p0 = PL.PP2.from_cx_theta(np.concatenate([th, np.zeros(2 * N * N + 2)]))
        return PL.PP2(p0.f1, None, 0.0, p0.b, p0.W, L=6, proj='full', rbm='outer')
    return PL.PP2.from_cx_theta(th)


def guide_aug6(a):
    """6x6 c3: AugGuide on theta (default theta_4) with an exact orbit-table sign."""
    import hashlib
    p = theta4_params(a.theta)
    sfile = {'sG': f'{R6}/pfaffian_lswo2/s_G.npy'}.get(a.sign, a.sign)
    sfn = OrbitSign6(np.load(sfile), name=a.sign)
    g = AugGuide(p, sfn, a.EP, a.c, a.f0, mode=a.mode, chunk=a.aug_chunk, wcheck_max=a.wcheck_max)
    sha = lambda f: hashlib.sha256(open(f, 'rb').read()).hexdigest()
    return g, sfn, dict(theta=a.theta, theta_sha256=sha(a.theta), sign_file=sfile, sign_sha256=sha(sfile), c=a.c, EP=a.EP, f0=a.f0, mode=a.mode)


# ------------------------------------------------------------------------------------------ guide factories
def guide_rbmpp(a):
    """RBM+PP psi_P (launched rbmpp_lib copy, brute force over the projection images, f64), times w (from --consts or --w)."""
    import jax, jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    from rbmpp_lib import Lattice, make_fns
    z = np.load(a.params); use_rbm = bool(z['rbm']) if 'rbm' in z.files else True
    lat = Lattice(a.L, 0.5, a.proj)
    p = {k: jnp.asarray(z[k]) for k in ('f', 'bre', 'bim', 'Wre', 'Wim')}
    w = complex(*json.load(open(a.consts))['w']) if a.consts else complex(*a.w)
    f = jax.jit(jax.vmap(make_fns(lat, jnp.float64, use_rbm)['logpsi'], in_axes=(None, 0)))
    lw = np.log(w)
    return (lambda S: np.asarray(f(p, jnp.asarray(S))) + lw), None, np.float64, dict(w=[w.real, w.imag], images=lat.nI, z=z, use_rbm=use_rbm)


def guide_table6(a):
    """6x6 exact table of a symmetric state (sector reps, per-config log amplitude lP and sign sP, e.g. RBM+PP table A):
    logpsi(x) = lP[rep(x)] + i pi [sP < 0].  numpy, CPU (canonicalisation with the st6_sector byte tables)."""
    import st6_sector as SS
    zt = np.load(a.table); reps = np.asarray(zt['reps']).astype(np.uint64)
    if a.la is not None:                                     # gfmc_aug: per-rep la table (.npy) and sign table (.npy)
        lP = np.load(a.la).astype(np.float64); sP = np.load(a.sgn)
    else:
        zs = np.load(a.syma); lP = zs['lP'].astype(np.float64); sP = zs['sP']
    T = SS.byte_tables(SS.space_group())                    # (288, 5, 256) uint64
    FULL = np.uint64((1 << 36) - 1); code = (np.uint64(1) << np.arange(36, dtype=np.uint64))
    lc = lP + 1j * np.pi * (sP < 0)
    miss = [0]

    def f(S):
        x = ((S > 0).astype(np.uint64) * code).sum(1, dtype=np.uint64)
        img = np.zeros((x.shape[0], T.shape[0]), np.uint64)
        for b in range(5):
            img |= T[:, b, :][:, ((x >> np.uint64(8 * b)) & np.uint64(255)).astype(np.int64)].T
        m = np.minimum(img.min(1), (img ^ FULL).min(1))
        idx = np.clip(np.searchsorted(reps, m), 0, len(reps) - 1)
        bad = reps[idx] != m
        if bad.any():
            miss[0] += int(bad.sum())
        out = lc[idx].copy(); out[bad] = -np.inf
        return out
    return f, None, np.float64, dict(reps=len(reps), miss=miss)


def guide_vit10(a):
    """our 10x10 ViT (vit10_jax), symm or plain; prec f64 or f32h (+ f64 recheck of pairs with |r| < --recheck)."""
    import jax, jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    sys.path.insert(0, HERE); import vit10_jax as V, validate_vit10 as VV
    BI, BJ, JB = bonds(a.L)                                   # ViT site = 10 x + y (transposed labels): same bond set and couplings
    assert {frozenset(p): j for p, j in zip(zip(BI.tolist(), BJ.tolist()), JB)} == {frozenset(p): j for p, j in zip(VV.BI.tolist(), VV.BJ)}
    path = V.SYMM if a.kind == 'symm' else V.PLAIN; mk = V.make_symm if a.kind == 'symm' else V.make_plain
    f64 = jax.jit(mk(V.load(path, jnp.float64), dt=jnp.float64))
    hi = lambda S: np.asarray(f64(jnp.asarray(S, jnp.float64))).astype(np.complex128)
    if a.prec == 'f64':
        return hi, None, np.float64, {}
    f32 = jax.jit(mk(V.load(path, jnp.float32), dt=jnp.float32, prec='highest'))
    lo = lambda S: np.asarray(f32(jnp.asarray(S, jnp.float32))).astype(np.complex128)
    return lo, (hi if a.recheck > 0 else None), np.float32, {}


# ------------------------------------------------------------------------------------------ walkers init
def metropolis_init(logpsi, L, M, seed, nsweep, dt=np.float64):
    """exchange Metropolis on |psi|^2 from random Sz = 0 configs (swap a uniform antiparallel bond, Hastings n_a(x)/n_a(y))."""
    BI, BJ, _ = bonds(L); N = L * L; rng = np.random.default_rng(seed + 7919)
    X = np.ones((M, N)); X[:, N // 2:] = -1
    for r in range(M): rng.shuffle(X[r])
    lp = np.asarray(logpsi(X.astype(dt))).astype(np.complex128); acc = 0
    for it in range(nsweep):
        anti = X[:, BI] != X[:, BJ]; na = anti.sum(1)
        b = np.argmax(rng.random(anti.shape) * anti, 1); r = np.arange(M)
        Y = X.copy(); Y[r, BI[b]], Y[r, BJ[b]] = X[r, BJ[b]], X[r, BI[b]]
        lq = np.asarray(logpsi(Y.astype(dt))).astype(np.complex128); nay = (Y[:, BI] != Y[:, BJ]).sum(1)
        ok = np.log(rng.random(M)) < 2 * (lq.real - lp.real) + np.log(na / nay)
        X[ok] = Y[ok]; lp = np.where(ok, lq, lp); acc += ok.mean()
    return X.astype(np.int8), acc / max(nsweep, 1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('--guide', required=True, choices=['rbmpp', 'table6', 'vit10', 'aug6'])
    ap.add_argument('--L', type=int, required=True); ap.add_argument('--M', type=int, default=256); ap.add_argument('--dtau', type=float, default=0.1)
    ap.add_argument('--blocks', type=int, default=200); ap.add_argument('--burn', type=int, default=50); ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--chunk', type=int, default=256); ap.add_argument('--chunk_hi', type=int, default=64)
    ap.add_argument('--ckpt_every', type=int, default=10); ap.add_argument('--resume', action='store_true')
    ap.add_argument('--maxsec', type=float, default=1e18); ap.add_argument('--E0', type=float, default=None, help='reference energy per site printed as d_ref')
    ap.add_argument('--init', default='metropolis', choices=['metropolis', 'rbmpp', 'shipped', 'file']); ap.add_argument('--X0', help='--init file: (M, N) int8 .npy'); ap.add_argument('--nsweep', type=int, default=2000)
    ap.add_argument('--broken', default='none', choices=['none', 'drop_vsf'])
    # rbmpp
    ap.add_argument('--params'); ap.add_argument('--consts'); ap.add_argument('--proj', default='translations'); ap.add_argument('--w', type=float, nargs=2, default=[1.0, 0.0])
    # table6
    ap.add_argument('--table', default=REPS6); ap.add_argument('--syma'); ap.add_argument('--la'); ap.add_argument('--sgn')
    # aug6 (AugGuide, c3)
    ap.add_argument('--theta', default=f'{R6}/pfaffian_r4/pdb_tabs/tab_t4/round_1/theta.npy')
    ap.add_argument('--sign', default='sG', help="'sG' or a per-rep int8 sign table .npy")
    ap.add_argument('--c', type=float, default=4.0); ap.add_argument('--EP', type=float, default=-18.13670889863218)
    ap.add_argument('--f0', type=float, default=2.4942595701520887 / 4); ap.add_argument('--mode', default='fix', choices=['fix', 'a1', 'old'])
    ap.add_argument('--aug_chunk', type=int, default=32); ap.add_argument('--wcheck_max', type=float, default=10.0); ap.add_argument('--wcheck_every', type=int, default=10)
    ap.add_argument('--init_la', default=f'{R6}/pfaffian_r4/pdb_tabs/tab_t4/round_1/la_table.npy', help='aug6 init: Metropolis on exp(2 la[rep])')
    # vit10
    ap.add_argument('--kind', default='symm'); ap.add_argument('--prec', default='f32h', choices=['f64', 'f32h']); ap.add_argument('--recheck', type=float, default=1e-3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); prog = os.path.join(a.out, 'progress.txt')

    def P(msg):
        line = f'{time.strftime("%Y-%m-%dT%H:%M:%S")} PROGRESS {msg}'
        print(line, flush=True); open(prog, 'a').write(line + '\n')

    N = a.L * a.L
    if a.guide == 'aug6':
        assert a.L == 6 and a.broken == 'none'
        g, sfn6, info = guide_aug6(a); fn_hi = None; idt = np.float64; a.prec = 'f64'
        la6 = np.load(a.init_la, mmap_mode='r')
        fn = lambda S: np.asarray(la6[sfn6.index(S)], np.float64).astype(np.complex128)      # init only (Metropolis on exp(2 la))
        json.dump(info, open(os.path.join(a.out, 'aug_info.json'), 'w'), indent=1)
    else:
        fn, fn_hi, idt, info = {'rbmpp': guide_rbmpp, 'table6': guide_table6, 'vit10': guide_vit10}[a.guide](a)
        g = LogpsiGuide(fn, a.L, chunk=a.chunk, in_dtype=idt, logpsi_hi=fn_hi, thr=a.recheck, chunk_hi=a.chunk_hi, drop_vsf=(a.broken == 'drop_vsf'))
    P(f'guide {a.guide} L {a.L} N {N} bonds {g.NB} broken {a.broken} prec {a.prec if a.guide == "vit10" else "f64"} recheck {fn_hi is not None} thr {a.recheck} '
      f'M {a.M} dtau {a.dtau} seed {a.seed} init {a.init} chunk {a.chunk} info {dict((k, v) for k, v in info.items() if k in ("w", "images", "reps", "c", "mode", "f0", "EP", "sign_sha256", "theta_sha256"))}')
    ck = os.path.join(a.out, 'ckpt.npz'); rec = []; b0 = 0
    if a.resume and os.path.exists(ck):
        zz = np.load(ck, allow_pickle=True); X0 = zz['X']; rec = list(zz['rec'].tolist()); b0 = int(zz['nb'])
        gf = GFMC(g, X0, a.dtau, a.seed); gf.rng.bit_generator.state = pickle.loads(zz['rng'].tobytes())
        P(f'RESUMED at block {b0} from {ck}')
    else:
        t0 = time.time()
        if a.init == 'rbmpp':
            from gfmc_run import init_walkers                      # launched M1 init (f32 Metropolis on |psi_P|^2)
            from rbmpp_lib import Lattice
            X0, acc = init_walkers(Lattice(a.L, 0.5, a.proj), info['z'], a.M, a.seed, info['use_rbm'])
        elif a.init == 'file':
            X0 = np.load(a.X0).astype(np.int8); assert X0.shape == (a.M, N); acc = float('nan')
        elif a.init == 'shipped':
            import validate_vit10 as VV
            S = VV.spins(); X0 = S[np.random.default_rng(a.seed).choice(len(S), a.M, replace=False)].astype(np.int8); acc = float('nan')
        else:
            ev0 = fn if a.guide == 'aug6' else (lambda S: g._eval(fn, S, a.chunk, g.small, idt))
            X0, acc = metropolis_init(ev0, a.L, a.M, a.seed, a.nsweep, idt)
        np.save(os.path.join(a.out, 'X0.npy'), X0)                  # exact initial walkers (trajectory-identity runs reuse them)
        P(f'init walkers {a.init} acc {acc:.3f} {time.time() - t0:.0f}s Sz {np.abs(X0.sum(1)).max()}')
        gf = GFMC(g, X0, a.dtau, a.seed)
        P(f'initial rows {a.M} walkers {g.n_eval} amplitude evaluations {g.t_eval:.0f}s')
    t_start = time.time(); tot = a.burn + a.blocks

    def save(nb):
        tmp = ck + '.tmp.npz'
        np.savez(tmp, X=gf.X, rec=np.array(rec, dtype=object), nb=nb, rng=np.frombuffer(pickle.dumps(gf.rng.bit_generator.state), np.uint8))
        os.replace(tmp, ck)
    stopped = False; status = 'OK'
    for b in range(b0, tot):
        tb = time.time(); n0 = g.n_eval; st0 = g.stats()
        try:
            r = gf.block()
        except FloatingPointError as e:
            status = f'NAN {e}'; P(f'ABORT block {b + 1}: {e}'); break
        if a.guide == 'aug6' and (b + 1) % a.wcheck_every == 0:
            try:
                wc = g.weight_check(gf.X); P(f'WCHECK block {b + 1} ratio {wc["ratio"]:.4g} add_max {wc["add_max"]:.3f} add_min {wc["add_min"]:.3f} n_node {wc["n_node"]}')
            except WeightCheckError as e:
                status = f'WEIGHTCHECK {e}'; P(f'ABORT block {b + 1}: {e}'); break
        r['sec'] = time.time() - tb; r['evals'] = g.n_eval - n0; r['block'] = b; st = g.stats()
        r['pairs'] = st['n_pairs'] - st0['n_pairs']; r['rechk'] = st['n_rechk'] - st0['n_rechk']; r['flip_caught'] = st['n_flip_caught'] - st0['n_flip_caught']
        import hashlib as _hl; r['xhash'] = int(_hl.sha256(np.ascontiguousarray(gf.X).tobytes()).hexdigest()[:15], 16)   # walker set after resampling
        r['M'] = gf.M; r['logw_ok'] = bool(np.isfinite(r['emix']) and np.isfinite(r['egrow']))
        rec.append(r)
        dref = f' d_ref {r["emix"] / N - a.E0:+.4e}' if a.E0 is not None else ''
        P(f'block {b + 1}/{tot} E_mix/N {r["emix"] / N:.8f} E_grow/N {r["egrow"] / N:.8f}{dref} ess {r["ess"]:.1f} elvar {r["elvar"]:.4f} '
          f'hops {r["hops"]} iters {r["iters"]} evals {r["evals"]} rechk {r["rechk"]}/{r["pairs"]} flips_caught {r["flip_caught"]} sec {r["sec"]:.1f}')
        if not r['logw_ok']:
            status = 'NAN energy'; P(f'ABORT block {b + 1}: non-finite energy'); break
        if (b + 1) % a.ckpt_every == 0: save(b + 1)
        if time.time() - t_start > a.maxsec:
            save(b + 1); stopped = True; P(f'MAXSEC reached at block {b + 1}'); break
    nb = len(rec); save(nb)
    em = np.array([r['emix'] for r in rec[a.burn:]]); eg = np.array([r['egrow'] for r in rec[a.burn:]])
    np.save(os.path.join(a.out, 'E_mix.npy'), em / N)
    keys = ('emix', 'egrow', 'ess', 'elvar', 'hops', 'iters', 'xhash', 'sec', 'evals', 'pairs', 'rechk', 'flip_caught')
    np.savez(os.path.join(a.out, 'blocks.npz'), burn=a.burn, dtau=a.dtau, M=a.M, N=N, **{k: np.array([r[k] for r in rec]) for k in keys})
    out = dict(args=vars(a), nblocks=nb, completed=(not stopped and status == 'OK' and nb == tot), status=status, N=N, guide_stats=g.stats())
    if a.guide == 'table6': out['table_miss'] = info['miss'][0]
    if a.guide == 'aug6': out['aug_info'] = info; out['wchecks'] = g.wchecks
    if len(em) >= 32:
        for name, x in (('mix', em), ('grow', eg)):
            m, errs, tau = blocked_error(x)
            out[name] = dict(E_site=m / N, err_site=max(errs[-4:]) / N, err_level0_site=errs[0] / N, tau_int_blocks=iat(x), tau_blocking=tau)
            if a.E0 is not None: out[name]['dE_site'] = m / N - a.E0
    if rec:
        out['sec_per_block_median'] = float(np.median([r['sec'] for r in rec])); out['evals_per_block_mean'] = float(np.mean([r['evals'] for r in rec]))
        out['hops_per_block_mean'] = float(np.mean([r['hops'] for r in rec])); out['sec_per_walker_tau'] = out['sec_per_block_median'] / (a.M * a.dtau)
        out['sec_per_hop'] = float(np.sum([r['sec'] for r in rec]) / max(np.sum([r['hops'] for r in rec]), 1))
    json.dump(out, open(os.path.join(a.out, 'result.json'), 'w'), indent=1, default=float)
    P(f'GFMC_GUIDE_DONE status {status} blocks {nb}/{tot} completed {out["completed"]} ' +
      (f'E_mix/N {out["mix"]["E_site"]:.8f} +- {out["mix"]["err_site"]:.1e}' if 'mix' in out else ''))
