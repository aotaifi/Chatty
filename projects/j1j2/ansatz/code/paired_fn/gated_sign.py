"""gated_sign.py (paired_fn, new 2026-11-13): "gated exact one-step sign" guide for the s306 network, for the shared paired FN estimator.

Guide B:  |psi_B| = |psi_s306| (same amplitude as arm A),   s_B(x) = s306(x) * c(x),
    c(x) = sgn[1 + alpha (E_L(x) - E)]   (sgn(0) = +1)  where decade(x) <= d_gate,  c(x) = +1 elsewhere,
    E_L(x) = (H psi_s306)(x) / psi_s306(x) = H_xx + s306(x) sum_{b antiparallel} (J_b/2) s306(y_b) |psi(y_b)/psi(x)|   (s306's own local energy, all bonds),
    decade(x) = floor((2 Re ln psi_s306(x) - MREF) / ln 10), MREF = -78.4616959545097 (stage 0 / grade_bare.MED).
alpha, E are READ from consts_s306_p1.json (sha 54add61c...), never recomputed.  d_gate = -inf: B == A (no state gated); +inf: every state (the stage-0 label K[s306]).
Arm A is the probe8 s306 guide (grade_bare.build_guide('B0s'), BareM1Sign 'abs', c = 0, proj 'simple'): GatedGuide with gate = -inf and the same cache is arm A
bit for bit (ratio rr = s(x) s(y) |r| with factors c == 1.0 exactly).  Both arms share ONE per-configuration cache of one_swap_ratios (key -> (ln psi, r)), so B's rows at A's configuration cost
no GPU call.  B's extra cost: for every antiparallel neighbour y of x with decade(y) <= d_gate one more one_swap_ratios call at y (E_L(y) needs s306 on y's neighbours; cached).
Neighbour candidates are preselected from x's call (ln psi(y) = ln psi(x) + ln|r_b|, with a 1e-6 margin); the authoritative decade of y is the one from y's own call.
Usage: see gated_probe.py (engine run) and test_gated_sign.py (controls).  Convention: rows = GuideBase.finish (unchanged), Delta = E_FN[B] - E_FN[A].
broken (validation only): 'alpha_flip' (alpha -> -alpha), 'hashmask' (extra sign flip on a hashed sparse set of configurations, p = hash_p), 'no_nbr' (neighbour factors c(y) omitted)."""
import hashlib, json, math, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
BASE = os.environ.get('S306_CODE', '/project/theorie/a/A.Otaifi/runs/j1j2/base_s306_p0/code'); sys.path.insert(0, BASE)
import numpy as np

CONSTS = '/project/theorie/a/A.Otaifi/runs/j1j2/headline_s306/inputs/consts_s306_p1.json'
CONSTS_SHA = '54add61c03ac8884f6058d683896e2752faf5c2cf2079a75b80804edc6a3e65a'
MREF = -78.4616959545097
LN10 = math.log(10.0)
DEC0, DEC1 = -16, 5
U = np.uint64
_PW = (U(1) << np.arange(64, dtype=U))


def read_sign_constants():
    raw = open(CONSTS, 'rb').read(); assert hashlib.sha256(raw).hexdigest() == CONSTS_SHA, 'consts_s306_p1.json sha mismatch'
    c = json.loads(raw); assert c['params_sha256'].startswith('45fa54e6') and c['p'] == 1
    return float(c['alpha']), float(c['E'])


def cfg_keys(X):
    return ((np.asarray(X) > 0).astype(U) * _PW).sum(-1, dtype=U)


def _splitmix(k):
    with np.errstate(over='ignore'):
        z = (np.asarray(k, U) + U(0x9E3779B97F4A7C15)); z = (z ^ (z >> U(30))) * U(0xBF58476D1CE4E5B9); z = (z ^ (z >> U(27))) * U(0x94D049BB133111EB)
        return z ^ (z >> U(31))


def _sgn(v):
    return np.where(v >= 0, 1.0, -1.0)


class Cache:
    """per-configuration cache key -> (ln psi complex, r (NB,) complex); shared by every guide that is given it.  Cleared (whole) BEFORE a lookup when above cap."""
    def __init__(self, cap=300000):
        self.d = {}; self.cap = cap; self.hit = 0; self.miss = 0; self.sec = 0.0; self.calls = 0

    def reset_counts(self):
        self.hit = self.miss = self.calls = 0; self.sec = 0.0


def make_gated_class():
    import grade_bare as GB, gfmc_b2 as G2

    class GatedGuide(GB.GuideRec):
        """BareM1Sign (s306) with a shared cache and the gated exact one-step sign.  Set after construction: .cache, .gate, .alpha, .E, .broken, .hash_p."""
        cache = None; gate = -math.inf; alpha = None; E = None; broken = None; hash_p = 1e-3; hash_salt = 0x5EED306

        def setup(self, cache, gate, alpha, E, broken=None, hash_p=1e-3):
            self.cache = cache; self.gate = float(gate); self.alpha = float(alpha); self.E = float(E); self.broken = broken; self.hash_p = hash_p
            if broken == 'alpha_flip': self.alpha = -self.alpha
            self.HW = self.JB / 2.0
            self.st = dict(rows=0, x=0, x_gated=0, x_flip=0, nb=0, nb_cand=0, nb_gated=0, nb_flip=0, nb_calls=0,
                           nb_dec_hist=np.zeros(DEC1 - DEC0 + 1), nb_gated_hist=np.zeros(DEC1 - DEC0 + 1), nb_flip_hist=np.zeros(DEC1 - DEC0 + 1))
            return self

        # ---------------------------------------------------------------- cached amplitude calls
        def ratios(self, X):
            C = self.cache; X = np.asarray(X)
            k = cfg_keys(X); uk, first, inv = np.unique(k, return_index=True, return_inverse=True)
            if len(C.d) > C.cap: C.d.clear()                                  # clear BEFORE the lookup, never mid-call
            need = [i for i, kk in enumerate(uk) if int(kk) not in C.d]
            C.hit += len(k) - len(need); C.miss += len(need)
            if need:
                t0 = time.perf_counter(); Xn = X[first[need]]; q = len(need); pad = (-q) % self.wc
                if pad: Xn = np.concatenate([Xn, np.repeat(Xn[:1], pad, 0)])
                lB, r, _, _ = G2.BareM1Sign.ratios(self, Xn); lB = np.asarray(lB)[:q]; r = np.asarray(r)[:q]
                for j, i in enumerate(need): C.d[int(uk[i])] = (lB[j], r[j])
                C.sec += time.perf_counter() - t0; C.calls += 1
            lBu = np.array([C.d[int(kk)][0] for kk in uk]); ru = np.stack([C.d[int(kk)][1] for kk in uk])
            lB = lBu[inv]; r = ru[inv]; self.last_lb = 2.0 * np.real(lB); return lB, r, lB, r

        # ---------------------------------------------------------------- s306 pieces from cached (lB, r)
        def _s306(self, lB, r, anti):
            ph = self.w * np.exp(1j * np.imag(lB)); sx = _sgn(ph.real); sy = _sgn((ph[:, None] * r).real)
            return ph, sx, sy

        def el_c(self, X, lB, r):
            """s306 sign sx, local energy E_L, factor c = sgn[1 + alpha (E_L - E)], decade, for configurations X with their cached (lB, r)."""
            X = np.asarray(X, np.float64); anti = X[:, self.BI] != X[:, self.BJ]
            ph, sx, sy = self._s306(lB, r, anti)
            EL = 0.25 * (self.JB[None] * X[:, self.BI] * X[:, self.BJ]).sum(1) + sx * np.where(anti, self.HW[None] * sy * np.abs(r), 0.0).sum(1)
            c = _sgn(1.0 + self.alpha * (EL - self.E)); dec = np.floor((2.0 * np.real(lB) - MREF) / LN10)
            return sx, EL, c, dec

        def _hashmask(self, X):
            return np.where(_splitmix(cfg_keys(X) ^ U(self.hash_salt)) < U(min(int(self.hash_p * 2.0 ** 64), 2 ** 64 - 1)), -1.0, 1.0)

        def signs_of_states(self, Z):
            """(s306(z), s_B(z), decade(z), E_L(z)) for configurations Z (k, 64) +-1; uses the same el_c and gate rule as rows()."""
            Z = np.asarray(Z); lB, r, _, _ = self.ratios(Z); sx, EL, c, dec = self.el_c(Z, lB, r)
            cx = np.where(dec <= self.gate, c, 1.0)
            if self.broken == 'hashmask': cx = cx * self._hashmask(Z)
            return sx.astype(np.int8), (sx * cx).astype(np.int8), dec, EL

        def rel_sign(self, X):
            """c(x) as the relative sign s_B / s306 (the flip set F = {c = -1}); provider diagnostics."""
            s, sb, _, _ = self.signs_of_states(X); return (s * sb).astype(float)

        # ---------------------------------------------------------------- rows
        def rows(self, X):
            self.n_calls += 1; self.n_rows += X.shape[0]; st = self.st; X = np.asarray(X)
            anti = X[:, self.BI] != X[:, self.BJ]; wi, bi = np.nonzero(anti)
            lB, r, _, _ = self.ratios(X)
            ph = self.w * np.exp(1j * np.imag(lB)); vx = ph.real; vy = (ph[wi] * r[wi, bi]).real
            self.min_margin = min(self.min_margin, float(np.min(np.abs(vx) / np.abs(ph))),
                                  float(np.min(np.abs(vy) / np.maximum(np.abs(ph[wi] * r[wi, bi]), 1e-300))) if len(wi) else np.inf)
            sx = _sgn(vx); sy = _sgn(vy)
            lbx = 2.0 * np.real(lB); dx = np.floor((lbx - MREF) / LN10)
            cx = np.ones(len(X)); cy = np.ones(len(wi))
            st['rows'] += 1; st['x'] += len(X); st['nb'] += len(wi)
            np.add.at(st['nb_dec_hist'], np.clip(np.floor((lbx[wi] + 2.0 * np.log(np.abs(r[wi, bi])) - MREF) / LN10).astype(np.int64), DEC0, DEC1) - DEC0, 1)
            gx = dx <= self.gate
            if gx.any():
                ig = np.nonzero(gx)[0]; _, _, c, _ = self.el_c(X[ig], lB[ig], r[ig]); cx[ig] = c; st['x_gated'] += len(ig); st['x_flip'] += int((c < 0).sum())
            if self.gate > -math.inf and self.broken != 'no_nbr':
                est = lbx[wi] + 2.0 * np.log(np.abs(r[wi, bi]))
                ci = np.nonzero(np.floor((est - MREF + 1e-6) / LN10) <= self.gate)[0]; st['nb_cand'] += len(ci)
                if len(ci):
                    Y = X[wi[ci]].copy(); q = np.arange(len(ci)); i_, j_ = self.BI[bi[ci]], self.BJ[bi[ci]]; yi = Y[q, i_].copy(); Y[q, i_] = Y[q, j_]; Y[q, j_] = yi
                    lBy, ry, _, _ = self.ratios(Y); _, _, cyy, dy = self.el_c(Y, lBy, ry); g = dy <= self.gate
                    cy[ci] = np.where(g, cyy, 1.0); st['nb_gated'] += int(g.sum()); st['nb_flip'] += int((g & (cyy < 0)).sum()); st['nb_calls'] += 1
                    np.add.at(st['nb_gated_hist'], np.clip(dy[g].astype(np.int64), DEC0, DEC1) - DEC0, 1)
                    np.add.at(st['nb_flip_hist'], np.clip(dy[g & (cyy < 0)].astype(np.int64), DEC0, DEC1) - DEC0, 1)
            if self.broken == 'hashmask':
                cx = cx * self._hashmask(X); Y = X[wi].copy(); q = np.arange(len(wi)); i_, j_ = self.BI[bi], self.BJ[bi]; yi = Y[q, i_].copy(); Y[q, i_] = Y[q, j_]; Y[q, j_] = yi
                cy = cy * self._hashmask(Y)
            rr = (sx * cx)[wi] * (sy * cy) * np.abs(r[wi, bi])
            if not (np.all(np.isfinite(rr)) and np.all(rr != 0)): raise FloatingPointError('GatedGuide: non-finite or zero ratio')
            return self.finish(X, anti, wi, bi, rr)

    return GatedGuide


def build(gate, cache, alpha=None, E=None, broken=None, wchunk=128, hash_p=1e-3):
    """One guide object (s306 amplitude, shared cache).  gate = d_gate (float, +-inf allowed).  Returns (guide, info) as grade_bare.build_guide."""
    import grade_bare as GB
    a0, E0 = read_sign_constants(); alpha = a0 if alpha is None else alpha; E = E0 if E is None else E
    g, info = GB.build_guide('B0s', wchunk); g.__class__ = make_gated_class(); g.setup(cache, gate, alpha, E, broken, hash_p)
    info.update(gate=float(gate), alpha=alpha, E=E, consts_sign=CONSTS, consts_sign_sha256=CONSTS_SHA, broken=broken)
    return g, info
