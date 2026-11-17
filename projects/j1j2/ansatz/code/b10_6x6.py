#!/usr/bin/env python3
"""b10_6x6.py: the 6x6 B10 check (runs/j1j2/b10_6x6/CHECK_SPEC.md; STORAGE_PROBLEM.md 'STEP-4 CONSENSUS', 'RUNG 2 RESULT', 'RUNG 2b RETRY').

Design B, amplitude only, oracle sign s2, exact 6x6 symmetric-sector tables (orbit reps, multiplicity n).
Round k: A_k (log table lA_k) -> psi'_k = (1 + alpha_k (H - E_k)) s2 A_k with alpha_k from the exact E_FN line search (cstar_lanczos.py /
rangelimit.py 'N'), guide G_k = A_k |1 + alpha_k (E_L - E_k)|, p_k = FN state of s2 G_k (delta0.efn_vec). A head r_k (x only) is trained,
A_{k+1} = A_k e^{r_k}. Head: symmetric real RBM factor of rangelimit_tangent.py, h_theta(x) = mean over the 144 images g of
sum_k ln cosh(b_k + W_k . z_g), evaluated on the orbit representative; r_theta = h_theta - h_theta0 (as the B10 4x4 code
experiments/frozen_hfn_factorized_sr_exact4x4_chunked.py: r = f(z) - f0), so the head starts at r = 0.
Arms (same head, same SR: (S + shift) d = g, step length set by trust = rms_P(O.d), same steps per round):
  E  exact frozen-H_FN[G_k] Rayleigh quotient over all orbits, exact gradient and exact Fisher (B10 4x4 logic), backtracking on the
     exact frozen energy as in the 4x4 code (--ls-arms);
  S  same objective, sampled: NSAMP rows ~ n chi^2 drawn exactly from the table each step, covariance gradient (Astra eq. (1)),
     sampled Fisher; M_req = 16 d^T Sigma d / (d^T g)^2 along the SR direction (population Sigma, g from the tables; also the sampled one);
  C  control: same sampling and head, physical-H Rayleigh quotient (VMC);
  X  must-fail: the literal STEP4.md (a2) residual loss sum_x [r_x - f_J(x) - ln sum_kept w_xy e^{r_y}]^2 on the same rows (zero at ln(p/G)).
Per round and arm: N(chi) = E_FN of chi's own optimal Lanczos guide, rho_k = (N(A_k) - N(chi)) / (N(A_k) - N(s2 p_k)),
certificate E_var[chi] < E_FN[psi'_k] (= N(A_k)), all per site relative to E0.

usage:
  b10_6x6.py run   OUT [--states EARLY,LATE] [--arms E,S,X,C] [--rounds 3] [--steps 50] [--M 16] [--scale 0.3] ...
  b10_6x6.py pilot OUT [same options]      5 timed steps of each arm at EARLY round 1 + one timed FN solve -> OUT/pilot.json (projection)
  b10_6x6.py test  OUT a|b|c|d|ops         CPU tests of the brief (see test_* below)
  b10_6x6.py gate  OUT                     kill line of CHECK_SPEC.md -> 'VERDICT ...' line, OUT/verdict.json
"""
import os, sys, json, time, hashlib, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import bond_split as BS
import delta0 as D0M

E0 = -18.137147540071613; NS = 36; N = 36
STATES = dict(EARLY='/scratch/a/A.Otaifi/ansatz_runs/pfaffian_c/runs/c1_cplx_s1/round_1/la_table.npy',
              LATE=os.environ.get('B10_LATE', ''))           # LATE path: proposed by the staging worker, approved by the lead (launcher sets it)
REPS = '/project/theorie/a/A.Otaifi/Chatty/runs/stall6/data/psi0_6x6_table.npz'
RL = '/scratch/a/A.Otaifi/ansatz_runs/rangelimit_6x6'
T0 = time.time(); LOGF = [None]


def log(m):
    s = f'[{time.time() - T0:8.1f}s] {m}'; print(s, flush=True)
    if LOGF[0]:
        with open(LOGF[0], 'a') as fh: fh.write(s + '\n')


def lhash(la): return hashlib.sha1(np.ascontiguousarray(la, np.float64).tobytes()).hexdigest()[:16]


# ----------------------------------------------------------------------------------------------------------------- operators
class Sector:
    """n, s2, K (kept, rep basis, J/2 sqrt(n_r/n_r')), Vm (violating, amplitude form), D0 (BS.build). backend 'scipy' or 'jax'."""
    def __init__(self, backend='scipy'):
        self.n = np.load(f'{BS.CSRD}/norb.npy').astype(np.float64); self.s = np.load(BS.S2).astype(np.float64); self.D = len(self.n)
        assert self.n.sum() == 9075135300
        self.sn = np.sqrt(self.n); t = time.time(); self.K, self.Vm, self.D0 = BS.build(self.n, self.s); log(f'built K/Vm {time.time() - t:.0f}s')
        self.backend = backend
        if backend == 'jax': self._to_dev()

    def _to_dev(self):
        import jax, jax.numpy as jnp
        D = self.D; self.jnp = jnp
        def dev(M):
            rows = np.repeat(np.arange(D, dtype=np.int32), np.diff(M.indptr).astype(np.int64))
            return jnp.asarray(M.data), jnp.asarray(M.indices.astype(np.int32)), jnp.asarray(rows)
        self.Kd = dev(self.K); self.Vd = dev(self.Vm)
        self._spmv = jax.jit(lambda dat, idx, rows, x: jax.ops.segment_sum(dat * x[idx], rows, num_segments=D, indices_are_sorted=True))
        log('K/Vm on device')

    def kmv(self, x):
        if self.backend == 'scipy': return self.K @ x
        return np.asarray(self._spmv(*self.Kd, self.jnp.asarray(x)))

    def vmv(self, x):
        if self.backend == 'scipy': return self.Vm @ x
        return np.asarray(self._spmv(*self.Vd, self.jnp.asarray(x)))

    def kamp(self, t): return self.kmv(self.sn * t) / self.sn                 # sum_kept h_xy t_y (amplitude form)

    def Hmag(self, b): return self.D0 * b + self.vmv(b) - self.kamp(b)        # rangelimit.py / cstar_lanczos.py Hmag

    def moments(self, lc):                                                    # rangelimit.py moments (E_var and E_L - E)
        a = np.exp(lc - lc.max()); assert (a > 0).all()
        EL = self.Hmag(a) / a; w = self.n * a * a; Z = w.sum(); E = float((w * EL).sum() / Z); assert np.isfinite(EL).all()
        return E, EL - E

    def efn_vec(self, la, tol=1e-6, maxit=60):
        """delta0.efn_vec; scipy backend calls it verbatim, jax backend swaps only the K matvec."""
        if self.backend == 'scipy': return D0M.efn_vec(self.K, self.Vm, self.D0, self.n, la, tol=tol, maxit=maxit)
        from scipy.sparse.linalg import LinearOperator, lobpcg
        a = BS.guide(la, self.n); w = a * self.sn; Dg = self.D0 + self.vmv(a) / a; N_ = len(a)
        def mm(X):
            X = np.asarray(X); X2 = X.reshape(N_, -1)
            return np.stack([Dg * X2[:, j] - self.kmv(X2[:, j]) for j in range(X2.shape[1])], 1).reshape(X.shape)
        A = LinearOperator((N_, N_), matvec=lambda x: Dg * np.ravel(x) - self.kmv(np.ravel(x)), matmat=mm, dtype=np.float64)
        x = (w / np.linalg.norm(w))[:, None]; th = float(x[:, 0] @ (Dg * x[:, 0] - self.kmv(x[:, 0])))
        den = 1.0 / np.maximum(Dg - th, 1.0)
        M = LinearOperator((N_, N_), matvec=lambda r: den * np.ravel(r), matmat=lambda Rr: den[:, None] * Rr, dtype=np.float64)
        val, vec, hist = lobpcg(A, x, M=M, largest=False, tol=tol, maxiter=maxit, retResidualNormsHistory=True)
        e = float(val[0])
        return dict(E=e, dE_site=(e - BS.E0) / BS.NS, E_guide_var=th, iters=len(hist), res=float(np.ravel(hist[-1])[0])), vec[:, 0]


# ----------------------------------------------------------------------------------------------------------------- N(chi)
class Grader:
    """N(chi) = E_FN of chi's own optimal Lanczos guide: rangelimit.py 'N' (grid 0,-0.1,...,-0.6, extend once to -1.2, parabola vertex
    iterations) with solve() as rangelimit.py. Keeps the best alpha's guide la, its FN amplitude p (sum n p^2 = 1) and E_FN.
    Disk cache per sha1(la) in CACHE (json + p + laG), so round-1 references are shared by the arms."""
    def __init__(self, sec, cache):
        self.sec = sec; self.cache = cache; os.makedirs(cache, exist_ok=True); self.nsolve = 0; self.tsolve = 0.0

    def solve(self, lc, d, al):
        x = 1.0 + al * d; la = lc + np.log(np.maximum(np.abs(x), np.exp(-60.0))) if al != 0 else lc.copy()
        t = time.time(); r, v = self.sec.efn_vec(la); dt = time.time() - t; self.nsolve += 1; self.tsolve += dt
        log(f'  FN solve alpha={al:.4f} dE/site {r["dE_site"]:.6e} iters {r["iters"]} res {r["res"]:.1e} {dt:.0f}s')
        return r, v, la

    def N(self, tag, lc, use_cache=True, grid=(0.0, -0.1, -0.2, -0.3, -0.45, -0.6)):
        h = lhash(lc); J = f'{self.cache}/N_{h}.json'
        if use_cache and os.path.exists(J) and os.path.exists(f'{self.cache}/p_{h}.npy'):
            out = json.load(open(J)); log(f'N({tag}) cached {h}: {out["N"]:.9e}'); return out, np.load(f'{self.cache}/p_{h}.npy'), np.load(f'{self.cache}/laG_{h}.npy')
        E, d = self.sec.moments(lc); cache = {}; keep = {}
        def f(al):
            al = round(float(al), 6)
            if al not in cache:
                r, v, la = self.solve(lc, d, al); cache[al] = r['dE_site']; keep[al] = (r, v, la)
                best = min(cache, key=cache.get)
                for k in list(keep):
                    if k != best: del keep[k]
            return cache[al]
        for al in grid: f(al)
        ext = False
        if min(cache, key=cache.get) == min(grid):
            ext = True
            for al in (-0.8, -1.0, -1.2): f(al)
        st = 'DONE'
        for it in range(8):
            cs = sorted(cache); bst = min(cs, key=cache.get); near = sorted(sorted(cs, key=lambda c: abs(c - bst))[:3])
            pp = np.polyfit(near, [cache[c] for c in near], 2)
            if pp[0] <= 0: st = 'NOT CONVEX'; break
            v = float(np.clip(-pp[1] / (2 * pp[0]), min(cs), 0.0))
            if min(abs(v - c) for c in cache) < 0.01: break
            f(v)
        ast = min(cache, key=cache.get); r, vec, laG = keep[ast]
        vec = vec * np.sign(vec.sum()); p = np.abs(vec) / self.sec.sn; p = p / np.sqrt((self.sec.n * p * p).sum())
        out = dict(tag=tag, hash=h, N=cache[ast], alpha_star=ast, E_FN=r['E'], Evar_site=(E - E0) / NS, extended=ext, status=st,
                   nsolves=len(cache), n_neg_vec=int((vec < 0).sum()), grid={str(k): v_ for k, v_ in sorted(cache.items())})
        log(f'N({tag}) = {out["N"]:.9e} alpha* {ast} E_var {out["Evar_site"]:.6e} solves {len(cache)}')
        if use_cache:
            np.save(f'{self.cache}/p_{h}.npy', p); np.save(f'{self.cache}/laG_{h}.npy', laG); json.dump(out, open(J, 'w'), indent=1)
        return out, p, laG


# ----------------------------------------------------------------------------------------------------------------- head
def images():
    """the 144 'simple' images of c2l_rbmpp_lib.Lattice(6, proj='simple') (rangelimit_tangent.py, verbatim)."""
    Lsz = 6
    def site(x, y): return (x % Lsz) + Lsz * (y % Lsz)
    def pm(f):
        q = np.empty(N, np.int64)
        for y in range(Lsz):
            for x in range(Lsz): xx, yy = f(x, y); q[site(x, y)] = site(xx, yy)
        return q
    c4 = pm(lambda x, y: (-y, x)); sx = pm(lambda x, y: (-x, y)); r_ = np.arange(N); pts = []
    for k in range(4):
        for refl in (False, True):
            q = r_.copy()
            for _ in range(k): q = c4[q]
            if refl: q = sx[q]
            pts.append(q)
    G = np.array([pm(lambda x, y, tx=tx, ty=ty: (x + tx, y + ty))[q] for ty in range(Lsz) for tx in range(Lsz) for q in pts])
    P_ = np.concatenate([G[8 * t + k][None] for t in range(N) for k in (0, 2)])
    IMG = np.concatenate([P_, P_]); FL = np.concatenate([np.ones(len(P_)), -np.ones(len(P_))]); assert len(IMG) == 144
    return IMG, FL


class Head:
    """h_theta(x) = mean_g sum_k ln cosh(b_k + W_k . z_g), z_g = FL_g x[IMG_g]; theta = [b (M), W (36 x M, row-major i,k)], uniform in
    [-scale, scale] (rung 2b). r_theta = h_theta - h_theta0 on the orbit reps (table h0 kept per round). dtype: compute dtype of the
    head pass (float64 default; float32 on the A40 with matmul precision 'highest'); accumulations are float64."""
    def __init__(self, M, scale, seed, X_all, dtype='float64', chunk=8192):
        import jax, jax.numpy as jnp
        jax.config.update('jax_enable_x64', True); jax.config.update('jax_default_matmul_precision', 'highest')
        self.jax, self.jnp = jax, jnp; self.M = M; self.P = M + N * M; self.X = X_all; self.D = len(X_all); self.chunk = chunk
        r0 = np.random.default_rng(seed); b = r0.uniform(-scale, scale, M); W = r0.uniform(-scale, scale, (N, M))
        self.theta0 = np.concatenate([b, W.reshape(-1)]); self.dt = jnp.float32 if dtype == 'float32' else jnp.float64
        IMG, FL = images(); IMGj = jnp.asarray(IMG); FLj = jnp.asarray(FL, self.dt); NI = 144; dt = self.dt; Mm = M

        def unpack(th): th = th.astype(dt); return th[:Mm], th[Mm:].reshape(N, Mm)
        def core(X, th):
            b, W = unpack(th); Z = FLj[None, :, None] * X.astype(dt)[:, IMGj]; t_ = b[None, None, :] + jnp.einsum('bgi,ik->bgk', Z, W)
            return Z, t_
        def h_(X, th):
            _, t_ = core(X, th); a = jnp.abs(t_)
            return (a + jnp.log1p(jnp.exp(-2 * a)) - jnp.log(jnp.asarray(2.0, dt))).sum(2).mean(1).astype(jnp.float64)
        def J_(X, th):
            Z, t_ = core(X, th); T = jnp.tanh(t_)
            return jnp.concatenate([T.mean(1), (jnp.einsum('bgi,bgk->bik', Z, T) / NI).reshape(X.shape[0], -1)], 1)
        self._h = jax.jit(h_); self._J = jax.jit(lambda X, th: J_(X, th).astype(jnp.float64))
        def acc(Sacc, gacc, macc, X, th, q, P):            # g += J^T q, S += J^T diag(P) J, m += J^T P   (float64 accumulators)
            J = J_(X, th); Pd = P.astype(J.dtype)
            return (Sacc + (J.T @ (J * Pd[:, None])).astype(jnp.float64), gacc + J.astype(jnp.float64).T @ q, macc + J.astype(jnp.float64).T @ P)
        self._acc = jax.jit(acc, donate_argnums=(0, 1, 2))
        self._jvp = jax.jit(lambda X, th, v: jax.jvp(lambda t_: h_(X, t_), (th,), (v,))[1])   # forward mode: ~2 forward passes, no Jacobian
        self._vjp = jax.jit(lambda gacc, X, th, w: gacc + J_(X, th).astype(jnp.float64).T @ w, donate_argnums=(0,))
        self.h0 = None

    def _chunks(self, idx=None):
        n_ = self.D if idx is None else len(idx)
        for i0 in range(0, n_, self.chunk):
            i1 = min(n_, i0 + self.chunk); sl = slice(i0, i1); rows = np.arange(i0, i1) if idx is None else idx[i0:i1]
            Xc = self.X[i0:i1] if idx is None else self.X[rows]
            pad = self.chunk - (i1 - i0)
            if pad: Xc = np.concatenate([Xc, np.ones((pad, N), Xc.dtype)])
            yield sl, i1 - i0, self.jnp.asarray(Xc)

    def h(self, theta, idx=None):
        th = self.jnp.asarray(theta); out = np.empty(self.D if idx is None else len(idx))
        for sl, m, Xc in self._chunks(idx): out[sl] = np.asarray(self._h(Xc, th))[:m]
        return out

    def J(self, theta, idx):
        th = self.jnp.asarray(theta); out = np.empty((len(idx), self.P))
        for sl, m, Xc in self._chunks(idx): out[sl] = np.asarray(self._J(Xc, th))[:m]
        return out

    def start_round(self):
        self.h0 = self.h(self.theta0)

    def r(self, theta, idx=None):
        return self.h(theta, idx) - (self.h0 if idx is None else self.h0[idx])

    def acc_all(self, theta, q, P):
        """exact passes over all orbits: g = J^T q, S = J^T diag(P) J - m m^T, m = J^T P."""
        jnp = self.jnp; th = jnp.asarray(theta); S = jnp.zeros((self.P, self.P)); g = jnp.zeros(self.P); m = jnp.zeros(self.P)
        for sl, k, Xc in self._chunks():
            qc = np.zeros(self.chunk); Pc = np.zeros(self.chunk); qc[:k] = q[sl]; Pc[:k] = P[sl]
            S, g, m = self._acc(S, g, m, Xc, th, jnp.asarray(qc), jnp.asarray(Pc))
        S, g, m = np.asarray(S), np.asarray(g), np.asarray(m)
        return S - np.outer(m, m), g, m

    def jv_all(self, theta, v):
        th = self.jnp.asarray(theta); vv = self.jnp.asarray(v); out = np.empty(self.D)
        for sl, m, Xc in self._chunks(): out[sl] = np.asarray(self._jvp(Xc, th, vv))[:m]
        return out

    def vjp_rows(self, theta, idx, w):
        th = self.jnp.asarray(theta); g = self.jnp.zeros(self.P)
        for sl, m, Xc in self._chunks(idx):
            wc = np.zeros(self.chunk); wc[:m] = w[sl]; g = self._vjp(g, Xc, th, self.jnp.asarray(wc))
        return np.asarray(g)


HEADS = dict(rbm=Head)   # head registry: any class with Head's interface (theta0, P, start_round, h, r, J, acc_all, jv_all, vjp_rows)
                          # can be registered here (e.g. a feature head) and selected with --head; the arms, SR and grading do not depend on it.


def make_head(args, X, seed, dtype=None):
    return HEADS[args.head](args.M, args.scale, seed, X, dtype or args.head_dtype, args.chunk)


def spins_all(D):
    reps = np.load(REPS)['reps']; assert len(reps) == D; X = np.empty((D, N), np.int8)
    for i0 in range(0, D, 1 << 20):
        x = reps[i0:i0 + (1 << 20)].astype(np.uint64)
        X[i0:i0 + (1 << 20)] = np.where(((x[:, None] >> np.arange(N, dtype=np.uint64)[None, :]) & np.uint64(1)) == 1, 1, -1)
    return X


# ----------------------------------------------------------------------------------------------------------------- objectives
class Round:
    """frozen quantities of one round: lA (log A_k), laG (guide psi'_k), E_FN[psi'_k], p_k."""
    def __init__(self, sec, lA, laG, EFN):
        self.sec = sec; self.lA = lA; self.laG = laG; self.EFN = EFN
        self.G = BS.guide(laG, sec.n); self.Dg = sec.D0 + sec.vmv(self.G) / self.G     # H_FN[s2 G] diagonal (efn_vec convention)
        self._KG = None

    def tvec(self, r):
        la = self.lA + r; t = np.exp(la - la.max()); return t / np.sqrt((self.sec.n * t * t).sum())

    def HF(self, t): return self.Dg * t - self.sec.kamp(t)                       # frozen H_FN[psi'_k] (amplitude form)

    def EF(self, t):
        n = self.sec.n; return float((n * t * self.HF(t)).sum() / (n * t * t).sum())

    def x_loss_parts(self, r):
        """STEP4.md (a2): res_x = r_x - f_J(x) - ln[ sum_kept h G_y e^{r_y} / K_G(x) ], f_J = ln[K_G / ((D - E) G + V_s)], E = E_FN[psi'_k]."""
        sec = self.sec
        if self._KG is None:
            self._KG = sec.kamp(self.G); den = (sec.D0 - self.EFN) * self.G + sec.vmv(self.G); assert (den > 0).all(), 'f_J: (D-E)G+V_s <= 0'
            self._fJ = np.log(self._KG / den)
        Ge = self.G * np.exp(r); KGe = sec.kamp(Ge)
        return r - self._fJ - np.log(KGe / self._KG), Ge, KGe


def draw(rng, P, k):
    c = np.cumsum(P); c /= c[-1]; return np.searchsorted(c, rng.random(k), side='right')


def train_round(arm, rd, head, args, rng, prog):
    """train the head for one round; returns theta, per-step rows."""
    sec = rd.sec; n = sec.n; theta = head.theta0.copy(); rows = []; P_ = head.P
    lsearch = arm in args.ls_arms.split(',')
    for it in range(1, args.steps + 1):
        ts = time.time(); r = head.r(theta); t = rd.tvec(r); w = n * t * t; Pw = w / w.sum(); row = dict(step=it)
        if arm == 'E':
            HFt = rd.HF(t); e = float((n * t * HFt).sum()); q = 2.0 * n * t * (HFt - e * t)   # dE/dr_x (sum n t^2 = 1)
            S, g, _ = head.acc_all(theta, q, Pw); row.update(E_F=e)
        else:
            idx = draw(rng, Pw, args.nsamp); Ok = head.J(theta, idx); Ob = Ok - Ok.mean(0); S = Ob.T @ Ob / len(idx)
            if arm in ('S', 'C'):
                ELx = (rd.HF(t) / t) if arm == 'S' else (sec.Hmag(t) / t)
                El = ELx[idx]; Eh = El.mean(); fS = 2.0 * Ob * (El - Eh)[:, None]; g = fS.mean(0)
                row.update(E_est=float(Eh), E_exact=float((w * ELx).sum() / w.sum()), n_unique=int(len(np.unique(idx))))
            else:                                                                 # X: (a2) loss on the drawn rows
                res, Ge, KGe = rd.x_loss_parts(r); rs = res[idx]; L = float((rs * rs).mean())
                lam = np.zeros(sec.D); np.add.at(lam, idx, 2.0 * rs / (len(idx) * KGe[idx]))
                c = Ge * (sec.sn * sec.kmv(lam / sec.sn)); nz = np.nonzero(c)[0]
                g = (Ok * (2.0 * rs / len(idx))[:, None]).sum(0) - head.vjp_rows(theta, nz, c[nz])
                row.update(loss_X=L, loss_X_all=float((res * res * Pw).sum()), n_unique=int(len(np.unique(idx))))
        shift = args.shift + args.shift_rel * float(np.trace(S)) / P_
        d = -np.linalg.solve(S + shift * np.eye(P_), g)
        rms = float(np.sqrt(max(d @ S @ d, 1e-300))); eta = args.trust / rms; row.update(gnorm=float(np.linalg.norm(g)), rms_per_eta=rms, eta=eta, shift=shift)
        if arm in ('S', 'C', 'X'):
            row['ESS_kish'] = int(len(idx))                                       # iid exact draws from n chi^2: Kish ESS = draws
            if arm in ('S', 'C'):
                fd = fS @ d; row['ESS_dir'] = float(fd.sum() ** 2 / max((fd * fd).sum(), 1e-300)) if fd.sum() != 0 else 0.0
                row['Mreq_sampled'] = float(16 * fd.var() / max(fd.mean() ** 2, 1e-300))
                if args.mreq_exact and (it - 1) % args.mreq_every == 0:          # Astra eq. (1), population moments along this d
                    u = head.jv_all(theta, d); ub = u - (Pw * u).sum(); bt = ELx - row['E_exact']
                    gd = float((Pw * 2.0 * bt * ub).sum()); vd = float((Pw * 4.0 * bt * bt * ub * ub).sum()) - gd * gd
                    row.update(Mreq=16 * vd / max(gd * gd, 1e-300), dTg_exact=gd, dTg_sampled=float(fd.mean()))
        if lsearch:
            e0 = row.get('E_F'); acc = False
            for fac in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625):
                cand = theta + eta * fac * d; ec = rd.EF(rd.tvec(head.r(cand)))
                if np.isfinite(ec) and ec < e0: theta = cand; acc = True; row.update(factor=fac, E_F_after=ec); break
            row['accepted'] = acc
            if not acc: row['sec'] = time.time() - ts; rows.append(row); prog(arm, row); break
        else:
            theta = theta + eta * d; row['accepted'] = True
        row['sec'] = time.time() - ts; rows.append(row); prog(arm, row)
    return theta, rows


# ----------------------------------------------------------------------------------------------------------------- driver
def setup(args, need_head=True):
    sec = Sector(args.backend); X = spins_all(sec.D) if need_head else None
    return sec, X


def progress_writer(path):
    def prog(arm, row):
        with open(path, 'a') as fh: fh.write(f'{time.strftime("%Y-%m-%dT%H:%M:%S")} PROGRESS arm={arm} ' + json.dumps(row, default=float) + '\n')
        log(f'{arm} step {row["step"]}: ' + ' '.join(f'{k}={v:.6g}' if isinstance(v, float) else f'{k}={v}' for k, v in row.items() if k != 'step'))
    return prog


def run(args, pilot=False):
    os.makedirs(args.out, exist_ok=True); LOGF[0] = f'{args.out}/log.txt'
    json.dump(vars(args), open(f'{args.out}/args.json' if not pilot else f'{args.out}/pilot_args.json', 'w'), indent=1)
    sec, X = setup(args); prog = progress_writer(f'{args.out}/progress.txt'); report = dict(args=vars(args), host=os.uname().nodename)
    for st in args.states.split(','):
        lA0 = np.load(STATES[st]).astype(np.float64); assert lA0.shape == (sec.D,) and np.isfinite(lA0).all(), st
        SD = f'{args.out}/{st}'; gr = Grader(sec, f'{SD}/cache'); os.makedirs(SD, exist_ok=True)
        if pilot:                                                             # one timed FN solve at the rung-2 alpha, 5 steps per arm
            E, d = sec.moments(lA0); t = time.time(); r_, v, laG = gr.solve(lA0, d, -0.1); tsolve = time.time() - t
            vec = v * np.sign(v.sum()); p = np.abs(vec) / sec.sn; p /= np.sqrt((sec.n * p * p).sum())
            rd = Round(sec, lA0, laG, r_['E']); pil = dict(state=st, t_fn_solve=tsolve, fn_iters=r_['iters'], arms={})
            for arm in args.arms.split(','):
                head = make_head(args, X, args.seed); t = time.time(); head.start_round(); th0 = time.time() - t
                t = time.time(); _, rows = train_round(arm, rd, head, args, np.random.default_rng(args.seed + 7), prog)
                pil['arms'][arm] = dict(t_h0=th0, t_steps=time.time() - t, steps=len(rows), t_per_step=(time.time() - t) / max(1, len(rows)), last=rows[-1] if rows else None)
                log(f'PILOT {arm}: {pil["arms"][arm]["t_per_step"]:.1f} s/step (h0 {th0:.1f}s)')
            if args.head_dtype == 'float32':                                  # fp32 vs fp64 head on one exact gradient
                h64 = make_head(args, X, args.seed, 'float64'); h32 = make_head(args, X, args.seed, 'float32')
                th = h64.theta0 + 1e-2 * np.random.default_rng(3).standard_normal(h64.P); h64.start_round(); h32.start_round()
                out = []
                for hh in (h64, h32):
                    tt = rd.tvec(hh.r(th)); HFt = rd.HF(tt); e = float((sec.n * tt * HFt).sum()); q = 2 * sec.n * tt * (HFt - e * tt)
                    S, g, _ = hh.acc_all(th, q, sec.n * tt * tt); out.append((e, g, S))
                pil['fp32_check'] = dict(dE=abs(out[0][0] - out[1][0]), g_rel=float(np.linalg.norm(out[0][1] - out[1][1]) / np.linalg.norm(out[0][1])),
                                         S_rel=float(np.linalg.norm(out[0][2] - out[1][2]) / np.linalg.norm(out[0][2])))
                log(f'fp32 check {pil["fp32_check"]}')
            report.setdefault('pilot', []).append(pil); json.dump(report, open(f'{args.out}/pilot.json', 'w'), indent=1, default=float)
            continue
        for arm in args.arms.split(','):
            AJ = f'{SD}/arm_{arm}.json'; res = json.load(open(AJ)) if os.path.exists(AJ) else dict(state=st, arm=arm, A0=STATES[st], rounds=[])
            nr = 1 if arm == 'X' else args.rounds; lA = lA0.copy()
            for rr in res['rounds']: lA = lA + np.load(rr['r_path'])                # resume: rebuild A_k from the stored heads
            for k in range(len(res['rounds']) + 1, nr + 1):
                tr = time.time(); log(f'=== {st} arm {arm} round {k}')
                Nk, pk, laG = gr.N(f'A_{k}[{arm}]', lA); Nph, _, _ = gr.N(f'phi_{k}[{arm}]', np.log(np.maximum(pk, 1e-300)))
                rd = Round(sec, lA, laG, Nk['E_FN'])
                head = make_head(args, X, args.seed + 100 * k); head.start_round()
                if args.table_head: r = np.load(args.table_head); rows = []; theta = None
                else:
                    theta, rows = train_round(arm, rd, head, args, np.random.default_rng(args.seed + 1000 * k + ord(arm)), prog); r = head.r(theta)
                RD = f'{SD}/arm_{arm}/round_{k}'; os.makedirs(RD, exist_ok=True); np.save(f'{RD}/r.npy', r)
                if theta is not None: np.save(f'{RD}/theta.npy', theta); np.save(f'{RD}/theta0.npy', head.theta0)
                lchi = lA + r; Nc, _, _ = gr.N(f'chi_{k}[{arm}]', lchi)
                B = Nk['N'] - Nph['N']; rho = (Nk['N'] - Nc['N']) / B
                EFF = rd.EF(rd.tvec(r)); EF0 = rd.EF(rd.tvec(np.zeros_like(r))); EFp = rd.EF(pk)
                rec = dict(k=k, N_nohead=Nk['N'], N_phi=Nph['N'], N_chi=Nc['N'], B=B, rho=rho, alpha_k=Nk['alpha_star'], E_FN_psi=Nk['N'],
                           Evar_A=Nk['Evar_site'], Evar_chi=Nc['Evar_site'], certificate=bool(Nc['Evar_site'] < Nk['N']),
                           frozen_gain_fraction=(EF0 - EFF) / (EF0 - EFp) if EF0 != EFp else None, EF_head=EFF, EF_nohead=EF0, EF_p=EFp,
                           r_wstd=float(np.sqrt((sec.n * pk ** 2 * (r - (sec.n * pk ** 2 * r).sum()) ** 2).sum())), steps=rows, r_path=f'{RD}/r.npy',
                           Mreq_max=max([x.get('Mreq', 0) for x in rows], default=None), Mreq_last=rows[-1].get('Mreq') if rows else None,
                           Mreq_median=float(np.median([x['Mreq'] for x in rows if 'Mreq' in x])) if any('Mreq' in x for x in rows) else None,
                           ESS_dir_median=float(np.median([x['ESS_dir'] for x in rows if 'ESS_dir' in x])) if any('ESS_dir' in x for x in rows) else None,
                           sec=time.time() - tr, nsolve_total=gr.nsolve, tsolve_total=gr.tsolve)
                res['rounds'].append(rec); json.dump(res, open(AJ, 'w'), indent=1, default=float)
                log(f'ROUND {st} {arm} k={k}: rho {rho:.6f} N {Nc["N"]:.6e} (nohead {Nk["N"]:.6e}, phi {Nph["N"]:.6e}) Evar {Nc["Evar_site"]:.6e} cert {rec["certificate"]}')
                with open(f'{args.out}/progress.txt', 'a') as fh: fh.write(f'{time.strftime("%Y-%m-%dT%H:%M:%S")} ROUND state={st} arm={arm} k={k} rho={rho:.6f}\n')
                lA = lchi
    report['status'] = 'DONE'; json.dump(report, open(f'{args.out}/{"pilot" if pilot else "run"}_done.json', 'w'), indent=1, default=float)
    log('DONE')


# ----------------------------------------------------------------------------------------------------------------- gate
def gate(out):
    """kill line of CHECK_SPEC.md; prints one 'VERDICT' line."""
    A = {}
    for st in ('EARLY', 'LATE'):
        for arm in 'ESCX':
            f = f'{out}/{st}/arm_{arm}.json'
            if os.path.exists(f): A[(st, arm)] = json.load(open(f))['rounds']
    rho = lambda st, arm, k: (A.get((st, arm)) or [{}] * k)[k - 1].get('rho') if len(A.get((st, arm), [])) >= k else None
    kills, notes = [], []
    for st in ('EARLY', 'LATE'):
        R = A.get((st, 'E'), [])
        if len(R) < 3: kills.append(f'{st}: arm E has {len(R)}/3 rounds (INCOMPLETE)'); continue
        r1, r3 = R[0]['rho'], R[2]['rho']
        if r1 < 0.5: kills.append(f'{st} E rho_1 {r1:.3f} < 0.5')
        if r3 < 0.5: kills.append(f'{st} E rho_3 {r3:.3f} < 0.5')
        fall = max(R[i]['rho'] - R[j]['rho'] for i in range(3) for j in range(i + 1, 3))
        if fall > 0.3: kills.append(f'{st} E rho falls {fall:.3f} > 0.3 over the rounds')
        for x in R:
            if not x['certificate']: kills.append(f'{st} E round {x["k"]} certificate fails (E_var {x["Evar_chi"]:.3e} >= E_FN {x["E_FN_psi"]:.3e})')
        s1 = rho(st, 'S', 1)
        if s1 is None: kills.append(f'{st}: arm S round 1 missing (INCOMPLETE)')
        elif s1 < r1 - 0.15: kills.append(f'{st} S rho_1 {s1:.3f} < E rho_1 - 0.15 ({r1 - 0.15:.3f})')
        for x in A.get((st, 'S'), []):
            if x.get('Mreq_max') and x['Mreq_max'] > 4096 * 50: kills.append(f'{st} S round {x["k"]} M_req max {x["Mreq_max"]:.3g} > 4096 x 50')
        x1 = rho(st, 'X', 1)
        if x1 is None: notes.append(f'{st}: arm X missing -> VOID')
        elif x1 > 0.2: notes.append(f'{st}: arm X rho_1 {x1:.3f} > 0.2 -> VOID')
    void = any('VOID' in s for s in notes); incomplete = any('INCOMPLETE' in s for s in kills)
    allE = [x['rho'] for st in ('EARLY', 'LATE') for x in A.get((st, 'E'), [])]
    Sok = all(abs((rho(st, 'S', k) or -9) - (rho(st, 'E', k) or 9)) <= 0.15 for st in ('EARLY', 'LATE') for k in (1, 2, 3))
    if incomplete: v = 'INCOMPLETE'
    elif void: v = 'VOID'
    elif kills: v = 'KILL'
    elif len(allE) == 6 and min(allE) >= 0.7 and Sok: v = 'PASS'
    else: v = 'BETWEEN (stop, GOAL REPORT)'
    ctrl = None
    if all(rho(st, 'C', 1) is not None and rho(st, 'S', 1) is not None for st in ('EARLY', 'LATE')):
        cge = all(rho(st, 'C', k) is not None and rho(st, 'S', k) is not None and rho(st, 'C', k) >= rho(st, 'S', k)
                  for st in ('EARLY', 'LATE') for k in (1, 2, 3) if rho(st, 'S', k) is not None)
        ctrl = 'physical-H head matches B10' if cge else 'B10 (S) beats the physical-H control in at least one round'
    out_ = dict(verdict=v, kills=kills, notes=notes, control=ctrl, rho={f'{st}/{arm}': [x['rho'] for x in A[(st, arm)]] for (st, arm) in A})
    json.dump(out_, open(f'{out}/verdict.json', 'w'), indent=1)
    print(f'VERDICT b10_6x6 {v}; ' + '; '.join(kills + notes) + (f'; control: {ctrl}' if ctrl else ''))
    print(json.dumps(out_['rho']))


# ----------------------------------------------------------------------------------------------------------------- tests
def test(args, which):
    os.makedirs(args.out, exist_ok=True); LOGF[0] = f'{args.out}/test_{which}.log'; res = dict(test=which, host=os.uname().nodename)
    RJ = f'{args.out}/test_{which}.json'
    def dump(): res['sec'] = time.time() - T0; json.dump(res, open(RJ, 'w'), indent=1, default=float)
    lA0 = np.load(STATES['EARLY']).astype(np.float64)
    if which == 'ops':      # jax operator path (the A40 path) against scipy, on CPU
        sec = Sector('scipy'); x = np.random.default_rng(0).standard_normal(sec.D); y1 = sec.kmv(x); z1 = sec.vmv(x)
        sec.backend = 'jax'; sec._to_dev(); y2 = sec.kmv(x); z2 = sec.vmv(x)
        res['K_rel'] = float(np.abs(y1 - y2).max() / np.abs(y1).max()); res['V_rel'] = float(np.abs(z1 - z2).max() / np.abs(z1).max()); dump()
        E, d = sec.moments(lA0); la = lA0 + np.log(np.maximum(np.abs(1 - 0.1 * d), np.exp(-60.0)))
        t = time.time(); rj, _ = sec.efn_vec(la); res['jax_solve'] = dict(rj, sec=time.time() - t); sec.backend = 'scipy'
        t = time.time(); rs, _ = sec.efn_vec(la); res['scipy_solve'] = dict(rs, sec=time.time() - t)
        res['dE_site_diff'] = rj['dE_site'] - rs['dE_site']; res['ref_rung2'] = 8.682592190013673e-06; dump(); log(json.dumps(res)); return
    sec = Sector('scipy'); gr = Grader(sec, f'{args.out}/cache')
    if which in ('a', 'b'):
        Nk, pk, laG = gr.N('A_1', lA0); Nph, _, _ = gr.N('phi_1', np.log(np.maximum(pk, 1e-300)))
        rb = json.load(open(f'{RL}/rangelimit_B.json'))['chi']
        res['N_A0'] = Nk; res['N_phi'] = Nph; res['ref'] = dict(chi_0=rb['chi_0']['N'], phi=rb['phi']['N'], chi_L488=rb['chi_L4.88']['N'], rho_L488=rb['chi_L4.88']['rho'])
        p_rl = np.load(f'{RL}/p.npy'); res['p_vs_rung2_maxrel'] = float(np.abs(pk - p_rl).max() / p_rl.max()); dump()
        if which == 'a':      # arm E pipeline with the head replaced by the clipped ideal table r* (L = 4.88)
            rs = np.load(f'{RL}/rstar.npy'); m = json.load(open(f'{RL}/rangelimit_A.json'))['m']; u = np.clip(rs - m, -4.88, 4.88)
            Nc, _, _ = gr.N('chi_rstar', lA0 + u); rho = (Nk['N'] - Nc['N']) / (Nk['N'] - Nph['N'])
            res.update(N_chi=Nc, rho=rho, rho_ref=1.000005, diff=rho - 1.000005, diff_vs_rung2_exact=rho - rb['chi_L4.88']['rho'], PASS=bool(abs(rho - 1.000005) < 1e-4))
            # same rho through the harness Round object (certificate and frozen energies as the run reports them)
            rd = Round(sec, lA0, laG, Nk['E_FN']); res['certificate'] = bool(Nc['Evar_site'] < Nk['N']); res['EF_head_minus_EF_p'] = rd.EF(rd.tvec(u)) - rd.EF(pk)
        else:                 # zero head: theta = theta0 -> r = h0 - h0 = 0 exactly -> N(chi) bit-identical to N(chi_0)
            X = spins_all(sec.D); head = Head(args.M, args.scale, args.seed, X, 'float64', args.chunk); head.start_round(); r = head.r(head.theta0)
            res['r_all_zero'] = bool((r == 0).all()); res['la_identical'] = bool(np.array_equal(lA0 + r, lA0))
            Nc, _, _ = gr.N('chi_zero_head', lA0 + r, use_cache=False)
            res.update(N_chi=Nc['N'], N_chi0=Nk['N'], bit_identical=bool(Nc['N'] == Nk['N']), N_chi0_rung2=rb['chi_0']['N'],
                       diff_vs_rung2=Nc['N'] - rb['chi_0']['N'], PASS=bool(Nc['N'] == Nk['N'] and res['r_all_zero']))
            # break test: a head at another init (theta0 of seed+1) is not zero
            r_b = head.h(Head(args.M, args.scale, args.seed + 1, X[:10], 'float64').theta0) - head.h0; res['break_nonzero_head_max_abs_r'] = float(np.abs(r_b).max())
        dump(); log(json.dumps({k: v for k, v in res.items() if k not in ('N_A0', 'N_phi', 'N_chi')}, default=float)); return
    if which in ('c', 'd'):
        E, d = sec.moments(lA0); laG = lA0 + np.log(np.maximum(np.abs(1 - 0.1 * d), np.exp(-60.0)))   # rung-2 guide (alpha* = -0.10)
    if which == 'c':        # gradient of the exact frozen-H_FN Rayleigh quotient vs central finite differences
        X = spins_all(sec.D); rd = Round(sec, lA0, laG, 0.0); head = Head(args.M, args.scale, args.seed, X, 'float64', args.chunk); head.start_round()
        rng = np.random.default_rng(5); th = head.theta0 + 0.05 * args.scale * rng.standard_normal(head.P)
        def EFth(tt): return rd.EF(rd.tvec(head.r(tt)))
        t = rd.tvec(head.r(th)); n = sec.n; HFt = rd.HF(t); e = float((n * t * HFt).sum()); q = 2 * n * t * (HFt - e * t)
        tg = time.time(); S, g, _ = head.acc_all(th, q, n * t * t); res['t_grad_fisher_pass'] = time.time() - tg
        res['E_F'] = e; res['gnorm'] = float(np.linalg.norm(g)); res['checks'] = []
        dirs = [('coord_b0', np.eye(head.P)[0]), ('coord_W5', np.eye(head.P)[args.M + 5 * args.M + 3]), ('rand1', rng.standard_normal(head.P)), ('rand2', rng.standard_normal(head.P)), ('grad', g / np.linalg.norm(g))]
        for nm, v in dirs:
            v = v / np.linalg.norm(v); an = float(g @ v); fds = {}
            for eps in (1e-3, 3e-4, 1e-4):
                ep, em = EFth(th + eps * v), EFth(th - eps * v); fds[str(eps)] = (ep - em) / (2 * eps)
            f1, f2 = fds['1e-3'] if '1e-3' in fds else fds['0.001'], fds['0.0003']
            rich = f2 + (f2 - f1) / ((1e-3 / 3e-4) ** 2 - 1)                                      # Richardson on eps 1e-3, 3e-4
            rel = abs(rich - an) / abs(an); res['checks'].append(dict(dir=nm, analytic=an, fd=fds, fd_richardson=rich, rel=rel)); dump()
            log(f'grad check {nm}: analytic {an:.10e} fd {fds} rich {rich:.10e} rel {rel:.2e}')
        # break test: drop the -E t term (the projection) from q -> wrong gradient must fail the same check
        gw = head.acc_all(th, 2 * n * t * HFt, n * t * t)[1]; v = dirs[2][1] / np.linalg.norm(dirs[2][1])
        res['break_rel'] = float(abs(gw @ v - res['checks'][2]['fd_richardson']) / abs(res['checks'][2]['fd_richardson']))
        res['PASS'] = bool(max(c['rel'] for c in res['checks']) < 1e-5 and res['break_rel'] > 1e-3); dump(); log(f"test c PASS {res['PASS']} max rel {max(c['rel'] for c in res['checks']):.2e} break {res['break_rel']:.2e}"); return
    if which == 'd':        # arm X loss: zero at ln(p/G), nonzero at ln(p/A_0)
        r_, v, la_ = Grader(sec, f'{args.out}/cache_d').solve(lA0, d, -0.1); assert np.array_equal(la_, laG)
        vec = v * np.sign(v.sum()); p = np.abs(vec) / sec.sn; p /= np.sqrt((sec.n * p * p).sum())
        rd = Round(sec, lA0, laG, r_['E']); G = rd.G; w = sec.n * p * p
        out = {}
        for nm, r in (('ln_p_over_G', np.log(p / G)), ('ln_p_over_A0', np.log(p) - lA0), ('zero', np.zeros(sec.D))):
            r = r - (w * r).sum(); res_, _, _ = rd.x_loss_parts(r); out[nm] = dict(loss_p2w=float((w * res_ * res_).sum()), loss_max=float(np.abs(res_).max()),
                                                                                  loss_unw_mean=float((res_ * res_).mean()), q99999=float(np.quantile(np.abs(res_), 0.99999)))
            log(f'X loss at {nm}: {out[nm]}')
        res['loss'] = out; res['fn_res'] = r_['res']; res['ratio_wrong_over_right'] = out['ln_p_over_A0']['loss_p2w'] / max(out['ln_p_over_G']['loss_p2w'], 1e-300)
        res['PASS'] = bool(out['ln_p_over_G']['loss_p2w'] < 1e-3 * out['ln_p_over_A0']['loss_p2w']); dump(); log(f"test d PASS {res['PASS']} ratio {res['ratio_wrong_over_right']:.3e}"); return


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['run', 'pilot', 'test', 'gate']); ap.add_argument('out'); ap.add_argument('which', nargs='?')
    ap.add_argument('--states', default='EARLY,LATE'); ap.add_argument('--arms', default='E,S,X,C'); ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--steps', type=int, default=50); ap.add_argument('--nsamp', type=int, default=4096)
    ap.add_argument('--M', type=int, default=16); ap.add_argument('--scale', type=float, default=0.3); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--trust', type=float, default=0.002); ap.add_argument('--shift', type=float, default=0.0); ap.add_argument('--shift-rel', type=float, default=1e-3)
    ap.add_argument('--ls-arms', default='E'); ap.add_argument('--mreq-exact', type=int, default=1); ap.add_argument('--mreq-every', type=int, default=1)
    ap.add_argument('--backend', default='scipy', choices=['scipy', 'jax']); ap.add_argument('--head-dtype', default='float64', choices=['float64', 'float32'])
    ap.add_argument('--chunk', type=int, default=8192); ap.add_argument('--table-head', default='')
    ap.add_argument('--late', default=''); ap.add_argument('--head', default='rbm', choices=sorted(HEADS))
    args = ap.parse_args()
    if args.late: STATES['LATE'] = args.late
    if 'LATE' in args.states.split(',') and args.mode in ('run', 'pilot'): assert STATES['LATE'], 'LATE path not set (--late or B10_LATE)'
    if args.mode == 'gate': return gate(args.out)
    if args.mode == 'test': return test(args, args.which)
    return run(args, pilot=(args.mode == 'pilot'))


if __name__ == '__main__': main()
