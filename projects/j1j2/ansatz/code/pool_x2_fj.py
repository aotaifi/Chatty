"""pool_x2_fj.py: f_J of the production 8x8 guide under the working sign sgn(M1) = sgn Re[w psi_rb8_prod], read from the lemma ratios (PLAN_pfaffian.md "Pool expansion"),
at x and at K = 2 blocked partners drawn as pool_fit.build_pool draws them, for the draws of decades -7..-9.
 sign:   s(x) = sgn Re[w e^{i Im lnB(x)}]  (v >= 0 -> +1, as decades8.make_sG8_np at alpha = 0),  s(y_b) = sgn Re[w e^{i (Im lnB(x) + arg r_b)}],  r_b = B(y_b)/B(x) from ONE one_swap_ratios call
         (w, E from gfmc_m13b/prod/rb8/V/consts.json; alpha is NOT used: p = 0 sign).  GUARD: an endpoint with |Re[w e^{i phase}]| < GUARD is recomputed with the direct logB of that configuration.
 f_J(x) = ln SK - ln(H_xx + VA - E'),  SK = sum_{kept} HW |r_b|, VA = sum_{not kept} HW |r_b|, kept = s(x) s(y_b) < 0 over the valid (anti-aligned) bonds, E' = -0.4987565 x 64  (= pp2_lemma._fj_parts
         arithmetic; cross-checked against pp2_lemma.fJ_fixed(at_y = False) through M1LemmaSign below).  mode 'old': non-finite -> 0.
 partners: blocked bonds = s(x) s(y_b) > 0, q(b) = HW |r_b| / W^B_x,  W^B_x = sum_blocked HW |r_b| ; K i.i.d. picks; rho_k = W^B_x exp[c (f_J(y_k) - f_J(x))], c = 4.
"""
import sys, os, json, time, hashlib
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import jax, jax.numpy as jnp
import pp2_lemma as PL
import pool8 as P8

CONSTS = '/scratch/a/A.Otaifi/gfmc_m13b/prod/rb8/V/consts.json'
EP8 = -0.4987565 * 64
GUARD = 1e-8
LN10 = np.log(10.0)


def load_w():
    c = json.load(open(CONSTS)); return complex(*c['w'])


class M1Sign:
    """sgn Re[w psi] from complex ln B and ratios; counts guard fallbacks.  broken: test hooks ('w' = w -> +1, 'ratio' = drop arg r, 'abs' = use |r| phase 0)."""
    def __init__(self, params, w=None, guard=GUARD, broken=None):
        self.p, self.w, self.guard, self.broken = params, load_w() if w is None else w, guard, broken; self.n_checked = 0; self.n_fallback = 0
        if broken == 'w': self.w = 1.0 + 0j

    def margin_x(self, lB): return np.real(self.w * np.exp(1j * np.imag(lB)))

    def sx(self, lB):
        m = self.margin_x(lB); return np.where(m >= 0, 1, -1).astype(np.int8)

    def sy(self, lB, r, X, valid):
        """(M, NB) +-1 at the endpoints of every valid bond; margins below the guard are recomputed directly."""
        ph = np.imag(lB)[:, None] + (np.angle(r) if self.broken != 'ratio' else np.zeros(r.shape))
        m = np.real(self.w * np.exp(1j * ph)); small = valid & (np.abs(m) < self.guard); self.n_checked += int(valid.sum())
        if small.any():
            lat = self.p.lat; mi, bi = np.nonzero(small); Y = X[mi].copy(); ar = np.arange(len(mi)); i_, j_ = lat.BI[bi], lat.BJ[bi]
            Y[ar, i_], Y[ar, j_] = X[mi, j_], X[mi, i_]
            lBy = np.asarray(PL.logB(self.p, Y.astype(np.float64))); m[mi, bi] = np.real(self.w * np.exp(1j * np.imag(lBy))); self.n_fallback += len(mi)
        return np.where(m >= 0, 1, -1).astype(np.int8)


class M1LemmaSign:
    """sign_fn for pp2_lemma.fJ_fixed(at_y = False): lemma_phase protocol endpoints(x, d, at_y, dtype, path, walker_chunk) -> (sx, sy, sz) from the amplitude evaluation d (= _one_as_two of x)."""
    lemma_phase = True
    def __init__(self, m1): self.m1 = m1
    def endpoints(self, x, d, at_y=True, dtype='f64', eval_path=None, walker_chunk=256, margins=False):
        assert not at_y
        lB = np.asarray(d['logB']); r = np.asarray(d['r1']); X = np.asarray(x); valid = np.asarray(d['am'])
        sx = self.m1.sx(lB); sy = self.m1.sy(lB, r, X, valid)
        return jnp.asarray(sx), jnp.asarray(sy), None


class FJ8:
    def __init__(self, params_path=None):
        p = PL.PP2.from_rbmpp_npz(params_path or P8.PROD8); self.p = p; lat = p.lat; self.lat = lat
        self.BI = np.asarray(lat.BI); self.BJ = np.asarray(lat.BJ); self.HW = np.asarray(lat.HW, float); self.JB = np.asarray(lat.JB, float); self.NB = lat.NB
        self.m1 = M1Sign(p)

    def ratios(self, X, chunk=256):
        X = np.asarray(X, np.float64); lBs, rs = [], []
        for i in range(0, len(X), chunk):
            lB, r = PL.one_swap_ratios(self.p, X[i:i + chunk], return_logB=True); jax.block_until_ready(r); lBs.append(np.asarray(lB)); rs.append(np.asarray(r))
        return np.concatenate(lBs), np.concatenate(rs)

    def evaluate(self, X, EP=EP8, m1=None):
        """one lemma call at X (M, 64) +-1: dict lB (complex), r, valid, sx, sy, A = HW |r| on valid bonds, SK, VA, WB, nblk, fx (raw, -inf where SK = 0), H."""
        m1 = m1 or self.m1; X = np.asarray(X, np.float64); lB, r = self.ratios(X); valid = X[:, self.BI] != X[:, self.BJ]
        sx = m1.sx(lB); sy = m1.sy(lB, r, X, valid); A = np.where(valid, self.HW[None] * np.abs(r), 0.0)
        kept = valid & (sx[:, None].astype(int) * sy < 0); blocked = valid & ~kept
        SK = np.where(kept, A, 0.0).sum(1); VA = np.where(blocked, A, 0.0).sum(1); H = 0.25 * (self.JB[None] * X[:, self.BI] * X[:, self.BJ]).sum(1)
        with np.errstate(divide='ignore', invalid='ignore'): fx = np.log(SK) - np.log(H + VA - EP)
        return dict(lB=lB, r=r, valid=valid, sx=sx, sy=sy, A=A, SK=SK, VA=VA, WB=VA, nblk=blocked.sum(1), blocked=blocked, fx=fx, H=H)

    def partners(self, X, ev, rng, K=2):
        """K i.i.d. blocked partners per x (pool_fit.build_pool: q = HW |r_b| / W^B_x on blocked bonds); bond index (M, K), -1 if no blocked bond; Y configs (M, K, 64)."""
        w = np.where(ev['blocked'], ev['A'], 0.0); WB = w.sum(1); has = ev['nblk'] > 0
        cdf = np.cumsum(w, 1) / np.where(has, WB, 1.0)[:, None]; picks = np.full((len(X), K), -1, np.int64)
        for k in range(K): u = rng.random(len(X)); picks[:, k] = np.where(has, np.minimum((cdf < u[:, None]).sum(1), self.NB - 1), -1)
        Y = np.repeat(np.asarray(X, np.float64)[:, None, :], K, 1); mi, ki = np.nonzero(picks >= 0); b = picks[mi, ki]
        Y[mi, ki, self.BI[b]] = np.asarray(X)[mi, self.BJ[b]]; Y[mi, ki, self.BJ[b]] = np.asarray(X)[mi, self.BI[b]]
        return picks, Y

    def fj_fixed_x(self, X):
        """cross-check path: pp2_lemma.fJ_fixed(at_y = False) with the lemma-ratio sign."""
        o = PL.fJ_fixed(self.p, np.asarray(X, np.float64), M1LemmaSign(self.m1), EP8, 4.0, mode='old', at_y=False)
        return np.asarray(o['fx'], np.float64), np.asarray(o['sx']), np.asarray(o['sy'])
