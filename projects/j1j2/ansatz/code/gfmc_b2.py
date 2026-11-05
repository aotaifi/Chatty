"""gfmc_b2.py: branch-2 grading providers for the launched M1 walker code (m1_code/gfmc_fn_lib.GFMC; rows(X) -> rate (B, NB), d_FN (B,)), for any pp2_lemma.PP2 amplitude B
(B_k of the amplitude rounds: two-PP + complex RBM 'inner', or rb8_prod itself).  New file; gfmc_aug8.py, pp2_lemma.py and the M1 code are only imported.
(PLAN_pfaffian.md "BRANCH 2 = OPTION (ii)", "BRANCH 2 GRADING PLAN".)

BareM1Sign (mechanism test, c = 0): guide G = sgn(M1) |B|,  sgn(M1) = sgn Re[w B_s] (B_s = rb8_prod, frozen; w from M1's consts.json), from the lemma ratios:
    s(x) = sgn Re[w e^{i Im ln B_s(x)}],  s(y_b) = sgn Re[w e^{i Im ln B_s(x)} r^s_b],  r^s_b = B_s(y_b)/B_s(x)   (v >= 0 -> +1)
    ratio G(y_b)/G(x) = s(x) s(y_b) |r_b|,  r_b = B(y_b)/B(x) from ONE one_swap_ratios call of B (and one of B_s when B is not B_s).  No f_J, no two-swap lemma.
  convention 'abs' (default, the branch-2 guide sgn(M1)|B|) or 're' (M1's p = 0 guide Re[w B]: ratio Re[w e^{i Im ln B(x)} r_b] / Re[w e^{i Im ln B(x)}]; only for B = B_s).
  The two differ by |cos phi(y)| / |cos phi(x)|, phi = arg(w B): |Re[w B]| = |B| |cos phi|.  Same signs by construction.
P1Guide (the final guide, M1's p = 1 pipeline with only the network swapped): psi_1 = (1 + alpha (E_L - E)) B with its OWN sign,
    c(z) = Re[w B(z) (1 + alpha (E_L(z) - E))],   E_L(z) = diag(z) + sum_b (J_b / 2) B(z_b)/B(z)   (complex local energy of B; = m1_code RBMPPGuideG at alpha != 0)
    ratio c(y_b)/c(x) = Re[w e^{ln B(y_b) - Re ln B(x)} f(y_b)] / Re[w e^{i Im ln B(x)} f(x)],  f = 1 + alpha (E_L - E).
  E_L at x and at every anti-aligned one-swap endpoint from pp2_lemma._one_j (one fast evaluation per configuration, 1 + n_anti(x) per row).  alpha, E, w: M1's VMC rule
  (consts.json for rb8_prod; for B_k they come from B_k's own VMC moments).  broken (TEST ONLY): alpha_zero.
"""
import os, sys, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gfmc_aug as GA                                  # puts the M1 walker code on sys.path (GFMC_M1_CODE)
from gfmc_fn_lib import GuideBase
import jax, jax.numpy as jnp
import pp2_lemma as PL


def _sgn(v): return np.where(v >= 0, 1.0, -1.0)


class BareM1Sign(GuideBase):
    def __init__(self, params, w, sign_params=None, convention='abs', walker_chunk=256, dtype='f64', broken=None):
        self.p = params; self.sp = params if sign_params is None else sign_params; self.same = self.sp is self.p
        assert convention in ('abs', 're') and (convention == 'abs' or self.same), "'re' is M1's Re[w B] guide: only for B = the sign network"
        lat = params.lat; self.lat = lat; self.N = lat.N; self.NB = lat.NB; self.BI = np.asarray(lat.BI); self.BJ = np.asarray(lat.BJ); self.JB = np.asarray(lat.JB, float)
        self.w = complex(w); self.conv = convention; self.wc = int(walker_chunk); self.dtype = dtype; self.broken = broken
        self.n_rows = 0; self.n_calls = 0; self.min_margin = np.inf; self.max_abs_cos_dev = 0.0

    def ratios(self, X):
        Xf = np.asarray(X, np.float64)
        lB, r = PL.one_swap_ratios(self.p, Xf, self.dtype, walker_chunk=self.wc, return_logB=True); lB = np.asarray(lB); r = np.asarray(r)
        if self.same: return lB, r, lB, r
        lBs, rs = PL.one_swap_ratios(self.sp, Xf, self.dtype, walker_chunk=self.wc, return_logB=True); return lB, r, np.asarray(lBs), np.asarray(rs)

    def rows(self, X):
        self.n_calls += 1; self.n_rows += X.shape[0]
        anti = X[:, self.BI] != X[:, self.BJ]; wi, bi = np.nonzero(anti)
        lB, r, lBs, rs = self.ratios(X)
        ph = self.w * np.exp(1j * np.imag(lBs)); vx = np.real(ph); vy = np.real(ph[wi] * rs[wi, bi])
        self.min_margin = min(self.min_margin, float(np.min(np.abs(vx) / np.abs(ph))), float(np.min(np.abs(vy) / np.maximum(np.abs(ph[wi] * rs[wi, bi]), 1e-300))) if len(wi) else np.inf)
        if self.conv == 're':
            rr = vy / vx[wi]
        else:
            sx = _sgn(vx); sy = _sgn(vy)
            if self.broken == 'nosign': sy = sx[wi]
            rr = sx[wi] * sy * np.abs(r[wi, bi])
        if not (np.all(np.isfinite(rr)) and np.all(rr != 0)): raise FloatingPointError('BareM1Sign: non-finite or zero ratio')
        return self.finish(X, anti, wi, bi, rr)

    def stats(self):
        return dict(n_rows=self.n_rows, n_calls=self.n_calls, min_rel_margin=self.min_margin)


class P1Guide(GuideBase):
    def __init__(self, params, alpha, E, w, chunk=256, dtype='f64', broken=None):
        self.p = params.cast(dtype); lat = params.lat; self.lat = lat; self.N = lat.N; self.NB = lat.NB
        self.BI = np.asarray(lat.BI); self.BJ = np.asarray(lat.BJ); self.JB = np.asarray(lat.JB, float)
        self.alpha = 0.0 if broken == 'alpha_zero' else float(alpha); self.E = float(E); self.w = complex(w); self.chunk = int(chunk)
        self.rd = jnp.float64 if dtype == 'f64' else jnp.float32
        JBj = jnp.asarray(lat.JB); HWj = jnp.asarray(lat.HW); BI = lat.BI; BJ = lat.BJ

        @jax.jit
        def el(p, z):
            lB, r = PL._one_j(p, z)
            return lB, 0.25 * jnp.sum(JBj * z[:, BI] * z[:, BJ], 1) + jnp.sum(HWj * r, 1)
        self._el = el; self.n_rows = 0; self.n_eval = 0; self.n_calls = 0; self.max_im = 0.0

    def amp(self, Z):
        """Z (K, N) -> (ln B complex (K,), E_L complex (K,)), padded chunks of self.chunk (one compile)."""
        K = len(Z); L, E = [], []
        for i in range(0, K, self.chunk):
            blk = Z[i:i + self.chunk]; m = len(blk)
            if m < self.chunk and K > self.chunk: blk = np.concatenate([blk, np.repeat(blk[:1], self.chunk - m, 0)])
            l, e = self._el(self.p, jnp.asarray(blk, self.rd)); L.append(np.asarray(l)[:m]); E.append(np.asarray(e)[:m])
        self.n_eval += K; return np.concatenate(L), np.concatenate(E)

    def rows(self, X):
        self.n_calls += 1; B = X.shape[0]; self.n_rows += B
        s = X.astype(np.float64); anti = X[:, self.BI] != X[:, self.BJ]; wi, bi = np.nonzero(anti)
        Y = s[wi].copy(); q = np.arange(len(wi)); Y[q, self.BI[bi]] = s[wi, self.BJ[bi]]; Y[q, self.BJ[bi]] = s[wi, self.BI[bi]]
        l, E = self.amp(np.concatenate([s, Y])); lx, Ex = l[:B], E[:B]; ly, Ey = l[B:], E[B:]
        fx = 1.0 + self.alpha * (Ex - self.E); fy = 1.0 + self.alpha * (Ey - self.E)
        ph = np.exp(1j * lx.imag); cx = (self.w * ph * fx).real; cy = (self.w * np.exp(ly - lx.real[wi]) * fy).real
        self.max_im = max(self.max_im, float(np.max(np.abs((self.w * ph * fx).imag))))
        rr = cy / cx[wi]
        if not (np.all(np.isfinite(rr)) and np.all(rr != 0)): raise FloatingPointError('P1Guide: non-finite or zero ratio')
        return self.finish(X, anti, wi, bi, rr)

    def stats(self):
        return dict(n_rows=self.n_rows, n_eval=self.n_eval, n_calls=self.n_calls)
