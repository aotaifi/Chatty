"""Lattice fixed-node GFMC (continuous time, importance sampled) with an RBM+PP + one signed Lanczos hop guide.

Guide psi_1(z) = Re[ w psi_P(z) (1 + alpha (E_L(z) - E)) ]   (= v + alpha (H - E) v for the real guide v = Re(w psi_P);
  E_L = (H psi_P)/psi_P complex local energy of the projected RBM+PP, E = <v|H|v>); alpha is the p = 1 Lanczos-Ritz
  coefficient, alpha = (E_1 - E)/var(v) (derivation in ANALYSIS notes: in the basis {v, (H-E)v} the Ritz energy shift is
  alpha var).  Sign s = sign psi_1, amplitude a = |psi_1| (the same signed one-hop state as rbmpp_table.py lanczos[1]).

Fixed node (ten Haaf / van Bemmel, the convention of st6_sector.fn_solve): spin-exchange hop x -> y with
  K_xy = H_xy = J_b / 2 > 0 and ratio r = psi_1(y)/psi_1(x):  r < 0 (sign change) -> kept, rate  -K r = |K| a(y)/a(x);
  r > 0 (sign violating)  -> removed and added to the diagonal, + K r.
  d_FN(x) = 1/4 sum_b J_b s_i s_j + sum_{violating} K r ,  R(x) = sum_kept rate,  E_L(x) = d_FN(x) - R(x).
Walker: exponential waiting time with rate R(x) (exact in tau: no time-step error), log-weight -= E_L dt,
population resampled (systematic) every block of length dtau; the mixed estimator sum w E_L / sum w at the block end
(and the growth estimator -ln(mean w)/dtau) are written per block.  Biases: dtau (resampling interval) and 1/M.
"""
import time
import numpy as np


# --------------------------------------------------------------------------------------------- guide providers
class GuideBase:
    """rows(X) -> rate (B, NB) float64, dFN (B,) for walkers X (B, N) int8 (+-1)."""
    N = NB = None
    BI = BJ = JB = None

    def diag0(self, X):
        s = X.astype(np.float64)
        return 0.25 * np.sum(self.JB[None] * s[:, self.BI] * s[:, self.BJ], 1)

    def finish(self, X, anti, wi, bi, rr):
        """rr = psi_1(y_b)/psi_1(x) for the antiparallel pairs (wi, bi) -> rate and d_FN."""
        K = 0.5 * self.JB[bi]
        keep = rr < 0
        rate = np.zeros(anti.shape)
        rate[wi[keep], bi[keep]] = -K[keep] * rr[keep]
        viol = np.bincount(wi[~keep], weights=(K * rr)[~keep], minlength=X.shape[0])
        return rate, self.diag0(X) + viol


class TableGuide(GuideBase):
    """exact small system: psi1 given on the whole Sz=0 basis; rows precomputed for all configurations."""

    def __init__(self, N, BI, BJ, JB, psi1, break_fn=None):
        self.N, self.BI, self.BJ, self.JB, self.NB = N, BI, BJ, JB, len(BI)
        self.psi1 = np.asarray(psi1, float)
        self.code = (1 << np.arange(N)).astype(np.int64)
        basis = np.array([s for s in range(1 << N) if bin(s).count('1') == N // 2], np.int64)
        self.lookup = -np.ones(1 << N, np.int64); self.lookup[basis] = np.arange(len(basis))
        self.basis = basis
        X = ((basis[:, None] >> np.arange(N)) & 1).astype(np.int8) * 2 - 1
        anti = X[:, BI] != X[:, BJ]
        wi, bi = np.nonzero(anti)
        Y = basis[wi] ^ (1 << BI[bi]) ^ (1 << BJ[bi])
        rr = self.psi1[self.lookup[Y]] / self.psi1[wi]
        self.rate_all, self.d_all = self.finish(X, anti, wi, bi, rr) if break_fn is None else break_fn(self, X, anti, wi, bi, rr)

    def index(self, X):
        return self.lookup[((X > 0).astype(np.int64) * self.code).sum(1)]

    def rows(self, X):
        i = self.index(X)
        return self.rate_all[i], self.d_all[i]


class RBMPPGuide(GuideBase):
    """6x6 (or LxL) RBM+PP + one signed hop on the GPU; psi_P from rbmpp_lib (full projection)."""

    def __init__(self, params, lat, alpha, E, chunk=256, w=1.0, use_rbm=True, small_chunk=None):
        import jax, jax.numpy as jnp
        from rbmpp_lib import make_fns
        self.jax, self.jnp = jax, jnp
        self.lat = lat; self.N = lat.N; self.BI = np.asarray(lat.BI); self.BJ = np.asarray(lat.BJ)
        self.JB = np.asarray(lat.JB, float); self.NB = lat.NB
        self.p = {k: jnp.asarray(params[k]) for k in ('f', 'bre', 'bim', 'Wre', 'Wim')}
        fns = make_fns(lat, jnp.float64, use_rbm)
        self.alpha = float(alpha); self.E = float(E); self.w = complex(w)
        self.chunk = chunk; self.small = small_chunk or max(chunk // 8, 8)
        self._f = jax.jit(jax.vmap(fns['eloc'], in_axes=(None, 0)))
        self.max_im = 0.0; self.n_eval = 0; self.t_eval = 0.0

    def amp(self, Z):
        """Z (K, N) float64 -> (E_L complex, log psi_P complex) as numpy, chunked and padded to fixed shapes."""
        jnp = self.jnp
        K = Z.shape[0]; out_E = np.empty(K, np.complex128); out_l = np.empty(K, np.complex128)
        i = 0
        t0 = time.time()
        while i < K:
            rem = K - i
            c = self.chunk if rem > self.small else self.small
            n = min(c, rem)
            blk = Z[i:i + n]
            if n < c:
                blk = np.concatenate([blk, np.repeat(blk[:1], c - n, 0)])
            E, l = self._f(self.p, jnp.asarray(blk))
            out_E[i:i + n] = np.asarray(E)[:n]; out_l[i:i + n] = np.asarray(l)[:n]
            i += n
        self.n_eval += K; self.t_eval += time.time() - t0
        return out_E, out_l

    def rows(self, X):
        B = X.shape[0]
        s = X.astype(np.float64)
        anti = X[:, self.BI] != X[:, self.BJ]
        wi, bi = np.nonzero(anti)
        Y = s[wi].copy()
        r = np.arange(len(wi))
        Y[r, self.BI[bi]] = s[wi, self.BJ[bi]]; Y[r, self.BJ[bi]] = s[wi, self.BI[bi]]
        E, l = self.amp(np.concatenate([s, Y]))
        Ex, lx = E[:B], l[:B]; Ey, ly = E[B:], l[B:]
        fx = 1.0 + self.alpha * (Ex - self.E)
        fy = np.exp(ly - lx[wi]) * (1.0 + self.alpha * (Ey - self.E))
        cx = (self.w * fx).real; cy = (self.w * fy).real
        self.max_im = max(self.max_im, float(np.max(np.abs((self.w * fx).imag))), float(np.max(np.abs((self.w * fy).imag))))
        rr = cy / cx[wi]
        return self.finish(X, anti, wi, bi, rr)


class RBMPPGuideG(GuideBase):
    """General guide provider (any L and projection, complex RBM, global phase w): c(x) = Re[w psi_P(x) (1 + alpha (E_L(x) - E))].

    Unlike RBMPPGuide (which assumes psi_P(x) real: its ratio drops the phase of psi_P(x)), the phase of psi_P(x) is kept:
    ratio c(y)/c(x) = Re[w e^{l_y - Re l_x} f_y] / Re[w e^{i Im l_x} f_x],  f = 1 + alpha (E_L - E).
    alpha == 0 (p = 0, pure RBM+PP guide Re[w psi_P]): ONE fast-update evaluation per walker gives all bond ratios
    psi_P(y_b)/psi_P(x) (rbmpp_lib eloc_full), c(y_b)/c(x) = Re[w e^{i Im l_x} rP_b] / Re[w e^{i Im l_x}] (1 evaluation per hop).
    alpha != 0 (p = 1): 1 + n_anti(x) evaluations per hop (E_L at x and at every antiparallel neighbour)."""

    def __init__(self, params, lat, alpha, E, w=1.0, chunk=256, use_rbm=True, small_chunk=None, drop_phase=False):
        import jax, jax.numpy as jnp
        from rbmpp_lib import make_fns
        self.jax, self.jnp = jax, jnp
        self.lat = lat; self.N = lat.N; self.BI = np.asarray(lat.BI); self.BJ = np.asarray(lat.BJ)
        self.JB = np.asarray(lat.JB, float); self.NB = lat.NB
        self.p = {k: jnp.asarray(params[k]) for k in ('f', 'bre', 'bim', 'Wre', 'Wim')}
        fns = make_fns(lat, jnp.float64, use_rbm)
        self.alpha = float(alpha); self.E = float(E); self.w = complex(w)
        self.chunk = chunk; self.small = small_chunk or max(chunk // 8, 8)
        self._f = jax.jit(jax.vmap(fns['eloc'], in_axes=(None, 0)))
        self._ff = jax.jit(jax.vmap(fns['eloc_full'], in_axes=(None, 0)))
        self.drop_phase = drop_phase          # deliberately broken version for the checks (phase of psi_P(x) dropped)
        self.max_im = 0.0; self.n_eval = 0; self.t_eval = 0.0

    def _chunked(self, fn, Z, nout):
        jnp = self.jnp
        K = Z.shape[0]; outs = None
        i = 0
        t0 = time.time()
        while i < K:
            rem = K - i
            c = self.chunk if rem > self.small else self.small
            n = min(c, rem)
            blk = Z[i:i + n]
            if n < c:
                blk = np.concatenate([blk, np.repeat(blk[:1], c - n, 0)])
            o = fn(self.p, jnp.asarray(blk))
            o = [np.asarray(a)[:n] for a in o]
            if outs is None:
                outs = [np.empty((K,) + a.shape[1:], a.dtype) for a in o]
            for k in range(nout):
                outs[k][i:i + n] = o[k]
            i += n
        self.n_eval += K; self.t_eval += time.time() - t0
        return outs

    def amp(self, Z):
        """Z (K, N) float64 -> (E_L complex, log psi_P complex)."""
        return self._chunked(self._f, Z, 2)

    def rows(self, X):
        B = X.shape[0]
        s = X.astype(np.float64)
        anti = X[:, self.BI] != X[:, self.BJ]
        wi, bi = np.nonzero(anti)
        if self.alpha == 0.0:
            Ex, lx, rP = self._chunked(self._ff, s, 3)
            ph = np.exp(1j * lx.imag * (0.0 if self.drop_phase else 1.0))
            cx = (self.w * ph).real
            cy = (self.w * ph[wi] * rP[wi, bi]).real
            self.max_im = max(self.max_im, float(np.max(np.abs((self.w * ph).imag))))
            return self.finish(X, anti, wi, bi, cy / cx[wi])
        Y = s[wi].copy()
        r = np.arange(len(wi))
        Y[r, self.BI[bi]] = s[wi, self.BJ[bi]]; Y[r, self.BJ[bi]] = s[wi, self.BI[bi]]
        E, l = self.amp(np.concatenate([s, Y]))
        Ex, lx = E[:B], l[:B]; Ey, ly = E[B:], l[B:]
        fx = 1.0 + self.alpha * (Ex - self.E)
        fy = 1.0 + self.alpha * (Ey - self.E)
        ph = np.exp(1j * lx.imag * (0.0 if self.drop_phase else 1.0))
        cx = (self.w * ph * fx).real
        cy = (self.w * np.exp(ly - lx.real[wi]) * fy).real if not self.drop_phase else \
            (self.w * np.exp(ly - lx[wi]) * fy).real
        self.max_im = max(self.max_im, float(np.max(np.abs((self.w * ph * fx).imag))))
        return self.finish(X, anti, wi, bi, cy / cx[wi])


# --------------------------------------------------------------------------------------------- walkers
def sysres(w, rng, M):
    c = np.cumsum(w); c /= c[-1]
    u = (rng.random() + np.arange(M)) / M
    return np.minimum(np.searchsorted(c, u, side='right'), w.size - 1)


class GFMC:
    def __init__(self, guide, X0, dtau, seed):
        self.g = guide; self.dtau = dtau; self.rng = np.random.default_rng(seed)
        self.X = np.ascontiguousarray(X0, dtype=np.int8); self.M = self.X.shape[0]
        self.rate = np.zeros((self.M, guide.NB)); self.dFN = np.zeros(self.M)
        self.refresh(np.arange(self.M))
        self.nrow = 0

    def refresh(self, idx):
        if idx.size == 0:
            return
        r, d = self.g.rows(self.X[idx])
        self.rate[idx] = r; self.dFN[idx] = d
        self.R = getattr(self, 'R', np.zeros(self.M)); self.EL = getattr(self, 'EL', np.zeros(self.M))
        self.R[idx] = r.sum(1); self.EL[idx] = d - self.R[idx]

    def block(self):
        M, g = self.M, self.g
        rem = np.full(M, self.dtau); logw = np.zeros(M)
        act = np.arange(M); hops = 0; iters = 0
        while act.size:
            with np.errstate(divide='ignore'):
                t = self.rng.exponential(1.0, act.size) / self.R[act]
            stay = t >= rem[act]
            fin = act[stay]; logw[fin] -= rem[fin] * self.EL[fin]
            mv = act[~stay]; tm = t[~stay]
            logw[mv] -= tm * self.EL[mv]; rem[mv] -= tm
            if mv.size:
                cum = np.cumsum(self.rate[mv], 1)
                u = self.rng.random(mv.size) * cum[:, -1]
                k = (cum <= u[:, None]).sum(1)
                k = np.minimum(k, g.NB - 1)
                bad = self.rate[mv, k] <= 0
                if bad.any():
                    k[bad] = np.argmax(self.rate[mv[bad]], 1)
                i, j = g.BI[k], g.BJ[k]
                xi = self.X[mv, i].copy(); self.X[mv, i] = self.X[mv, j]; self.X[mv, j] = xi
                self.refresh(mv)
                hops += mv.size
            act = mv; iters += 1
        mx = logw.max(); w = np.exp(logw - mx)
        sw = w.sum()
        emix = float(np.sum(w * self.EL) / sw)
        egrow = float(-(mx + np.log(sw / M)) / self.dtau)
        ess = float(sw * sw / np.sum(w * w))
        elvar = float(np.sum(w * (self.EL - emix) ** 2) / sw)
        idx = sysres(w, self.rng, M)
        self.X = self.X[idx]; self.rate = self.rate[idx]; self.dFN = self.dFN[idx]
        self.R = self.R[idx]; self.EL = self.EL[idx]
        return dict(emix=emix, egrow=egrow, ess=ess, elvar=elvar, hops=hops, iters=iters)


# --------------------------------------------------------------------------------------------- statistics
def blocked_error(x, min_blocks=16):
    """mean, error from successive pair-averaging (flyvbjerg-petersen): returns (mean, [err per level], tau_int est)."""
    x = np.asarray(x, float); n0 = len(x); m = x.mean()
    errs = []; lv = x.copy()
    while len(lv) >= min_blocks:
        errs.append(float(lv.std(ddof=1) / np.sqrt(len(lv))))
        k = len(lv) // 2
        lv = 0.5 * (lv[:2 * k:2] + lv[1:2 * k:2])
    e0 = errs[0]
    plateau = max(errs[-4:]) if len(errs) >= 4 else max(errs)
    return float(m), errs, float(0.5 * (plateau / e0) ** 2)


def iat(x, cmax=None):
    """integrated autocorrelation time (units of one entry), Sokal window c=5."""
    x = np.asarray(x, float) - np.mean(x); n = len(x)
    f = np.fft.rfft(x, 2 * n); ac = np.fft.irfft(f * np.conj(f))[:n]; ac /= ac[0]
    t = 0.5 + np.cumsum(ac[1:])
    for m in range(1, n):
        if m >= 5 * (0.5 + ac[1:m + 1].sum()):
            return float(0.5 + ac[1:m + 1].sum())
    return float(0.5 + ac[1:].sum())
