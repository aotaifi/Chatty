"""providers.py (paired_fn): guide row providers for the paired lattice-FN engine.

Contract (ASTRA_paired.md section 5): a provider owns the walker STATE representation and returns, for a batch of states S,
    rows(S) -> RowBatch(rate (k, NB) >= 0, dFN (k,), ...)
with rate[b] the importance-sampled kept-hop rate along physical bond b (stable bond IDs, same NB and order for both arms of a pair),
dFN the ten Haaf diagonal d_g(x) = H_xx + sum_violating H_xy s s a_y/a_x, and E_L = dFN - sum_b rate (= (H psi)/psi).
    move(S, b) -> states after exchanging bond b (b an int array, one per state)
    key(S)     -> int64/uint64 configuration keys (equal key <=> equal configuration; used by the resampling coupling)
    sign(S), log_abs(S) (optional) -> guide sign / log amplitude, for flip-set diagnostics
    guide_id   -> sha256 of parameters + convention + floor
Conventions: H_xy = J_b/2 > 0 for an antiparallel bond (J1-J2 Heisenberg, S^z basis); kept bond <=> psi(x) psi(y) < 0 (H_xy rr < 0);
this is m1/gfmc_fn_lib.GuideBase.finish, reused unchanged (ConfigGuide) or reproduced per orbit row (OrbitTableGuide = g2lib.rows).

Providers here:
  ConfigGuide      wraps ANY gfmc_fn_lib.GuideBase (TableGuide for small exact systems; BareM1Sign/SignOpt/P1Guide at 8x8). State = +-1 config
                   rows X (k, N) int8; move = spin exchange; key = bit pattern (N <= 64).
  DenseGuide       state = basis index of a small dense real symmetric H with a dest table; rows from eqs (1)-(2) of ASTRA_paired.md built
                   independently of gfmc_fn_lib (stage 1 cross-check); arbitrary real nonzero guide psi = s a.
  OrbitTableGuide  exact 6x6 symmetric sector (15,804,956 orbits): state = orbit index; rows per orbit rep (g2lib.rows convention:
                   rate = HW_b a_t / a_r on kept columns, V_sf = sum_violating HW_b a_t / a_r), neighbour tables nb_J1/nb_J2 (mmap, shared).
Deliberately broken variants (validation only): broken='no_vsf' (V_sf omitted), 'inverse_ratio' (a_x/a_y in place of a_y/a_x in rates and
V_sf), 'walls_from' (V_sf computed with another sign table: omits only the CHANGED walls).
"""
import hashlib, os, sys
from dataclasses import dataclass, field
import numpy as np

M1_CODE = '/scratch/a/A.Otaifi/ansatz_runs/sign_bad/r3/code/m1'          # gfmc_fn_lib.py (GuideBase.finish), read-only import
J1J2_CODE = '/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/ansatz/code'   # bond_split.py (tables, HW, JW, guide floor), read-only


@dataclass
class RowBatch:
    rate: np.ndarray            # (k, NB) kept-hop rates, >= 0
    dFN: np.ndarray             # (k,) FN diagonal d_g
    valid: np.ndarray = None    # (k, NB) antiparallel bond (a move exists)
    keep: np.ndarray = None     # (k, NB) kept (rate > 0 structurally)
    log_abs: np.ndarray = None
    sign: np.ndarray = None
    guide_id: str = None

    @property
    def EL(self):
        return self.dFN - self.rate.sum(1)


def _sha_arrays(*arrs, extra=''):
    h = hashlib.sha256()
    for a in arrs:
        a = np.ascontiguousarray(a); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    h.update(extra.encode()); return h.hexdigest()


# ------------------------------------------------------------------------------------------------ ConfigGuide (GuideBase wrapper)
_PW = (np.uint64(1) << np.arange(64, dtype=np.uint64))


class ConfigGuide:
    """Wrap a gfmc_fn_lib.GuideBase subclass (unchanged): rows(X) -> (rate, dFN) of GuideBase.finish."""

    def __init__(self, base, guide_id=None, sign_fn=None, logabs_fn=None):
        self.g = base; self.NB = base.NB; self.BI = np.asarray(base.BI); self.BJ = np.asarray(base.BJ)
        self.guide_id = guide_id or f'ConfigGuide:{type(base).__name__}:{id(base)}'
        self._sign = sign_fn; self._logabs = logabs_fn
        assert base.N <= 64

    def rows(self, S, diagnostics=False):
        rate, d = self.g.rows(S)
        rb = RowBatch(np.asarray(rate, float), np.asarray(d, float), guide_id=self.guide_id)
        if diagnostics:
            rb.valid = S[:, self.BI] != S[:, self.BJ]; rb.keep = rb.rate > 0
            if self._sign is not None: rb.sign = self._sign(S)
            if self._logabs is not None: rb.log_abs = self._logabs(S)
        return rb

    def move(self, S, b):
        S = S.copy(); r = np.arange(len(b)); i, j = self.BI[b], self.BJ[b]
        xi = S[r, i].copy(); S[r, i] = S[r, j]; S[r, j] = xi; return S

    def key(self, S):
        return ((np.asarray(S) > 0).astype(np.uint64) * _PW[:S.shape[1]]).sum(-1, dtype=np.uint64)

    def sign(self, S):
        return None if self._sign is None else self._sign(S)


def table_guide(N, BI, BJ, JB, psi, break_fn=None):
    """gfmc_fn_lib.TableGuide (production rows on a small exact Sz=0 basis), wrapped; sign/log_abs from psi."""
    sys.path.insert(0, M1_CODE)
    import gfmc_fn_lib as FL
    tg = FL.TableGuide(N, BI, BJ, JB, psi, break_fn=break_fn)
    psi = np.asarray(psi, float)
    return ConfigGuide(tg, guide_id=_sha_arrays(psi, BI, BJ, JB, extra='TableGuide'),
                       sign_fn=lambda X: np.sign(psi[tg.index(X)]), logabs_fn=lambda X: np.log(np.abs(psi[tg.index(X)])))


# ------------------------------------------------------------------------------------------------ DenseGuide (independent small-matrix rows)
class DenseGuide:
    """State = basis index 0..n-1 of a small real symmetric H.  dest (n, NB): basis index after move b (== x if no move), offd (n, NB): H_{x,dest}
    (0 if no move).  Rows from ASTRA_paired.md eqs (1)-(2): t = H_xy s_x s_y; kept if t < 0: rate = -t a_y/a_x; violating t > 0: d += t a_y/a_x."""

    def __init__(self, Hdiag, dest, offd, psi, broken=None, walls_sign=None):
        self.n, self.NB = dest.shape; self.dest = np.asarray(dest); self.offd = np.asarray(offd, float); self.Hd = np.asarray(Hdiag, float)
        psi = np.asarray(psi, float); assert np.all(psi != 0) and np.all(np.isfinite(psi))
        self.s = np.sign(psi); self.a = np.abs(psi); self.broken = broken
        valid = self.offd != 0
        t = self.offd * self.s[:, None] * self.s[self.dest]
        ratio = self.a[self.dest] / self.a[:, None]
        if broken == 'inverse_ratio': ratio = 1.0 / ratio
        self.rate_all = np.where(valid & (t < 0), -t * ratio, 0.0)
        tw = t if walls_sign is None else self.offd * walls_sign[:, None] * walls_sign[self.dest]
        V = np.where(valid & (tw > 0), tw * ratio, 0.0).sum(1)
        self.d_all = self.Hd + (0.0 if broken == 'no_vsf' else V)
        self.valid = valid
        self.guide_id = _sha_arrays(psi, self.dest, self.offd, extra=f'Dense:{broken}')

    def rows(self, S, diagnostics=False):
        rb = RowBatch(self.rate_all[S], self.d_all[S], guide_id=self.guide_id)
        if diagnostics:
            rb.valid = self.valid[S]; rb.keep = rb.rate > 0; rb.sign = self.s[S]; rb.log_abs = np.log(self.a[S])
        return rb

    def move(self, S, b):
        return self.dest[S, b]

    def key(self, S):
        return np.asarray(S, np.int64)

    def sign(self, S):
        return self.s[S]

    # exact operator (for propagator tests): generator L = -D K D^{-1} in the positive frame, column density f evolves as df/dt = L f
    def generator(self):
        n = self.n; L = np.zeros((n, n))
        for x in range(n):
            for b in range(self.NB):
                if self.rate_all[x, b] > 0: L[self.dest[x, b], x] += self.rate_all[x, b]
        L -= np.diag(self.d_all)
        return L

    def EL_all(self):
        return self.d_all - self.rate_all.sum(1)


# ------------------------------------------------------------------------------------------------ OrbitTableGuide (exact 6x6)
_TABLES = {}


def orbit_tables():
    """(n, t1, t2): orbit multiplicities and the two neighbour tables (mmap; the OS page cache is shared by processes on a node)."""
    if 'n' not in _TABLES:
        sys.path.insert(0, J1J2_CODE)
        import bond_split as BS
        n = np.load(f'{BS.CSRD}/norb.npy').astype(np.float64); assert n.sum() == 9075135300
        t1 = np.load(f'{BS.TAB}/nb_J1.npy', mmap_mode='r'); t2 = np.load(f'{BS.TAB}/nb_J2.npy', mmap_mode='r')
        assert t1.shape == (len(n), 72) and t2.shape == (len(n), 72)
        _TABLES.update(n=n, t1=t1, t2=t2, HW=np.asarray(BS.HW, float), JW=np.asarray(BS.JW, float), E0=BS.E0, NS=BS.NS, BS=BS)
    return _TABLES


def floor_amplitude(la, n, floor=1e-15):
    """bond_split.guide: per-config amplitude normalised sum n a^2 = 1, then floored (the exact solver's order)."""
    la = la - la.max(); a = np.exp(la); a = a / np.sqrt((n * a * a).sum()); return np.maximum(a, floor)


class OrbitTableGuide:
    """Exact 6x6 orbit-table guide psi = sig * a (per configuration, orbit symmetric).  rows = g2lib.rows (verified there against the
    exact operator: E_L = (H_FN w)_r / w_r with w = a sqrt(n)).  NB = 144 columns (72 J1 then 72 J2 bonds of the orbit rep)."""

    def __init__(self, sig, a, name='', broken=None, walls_sig=None, floor=1e-15):
        T = orbit_tables(); self.T = T
        self.t1, self.t2, self.HW, self.JW = T['t1'], T['t2'], T['HW'], T['JW']
        self.sig = np.asarray(sig, np.int8); self.a = np.asarray(a, np.float64); self.la = np.log(self.a)
        assert self.sig.shape == self.a.shape == T['n'].shape
        self.NB = 144; self.broken = broken; self.name = name
        self.walls_sig = None if walls_sig is None else np.asarray(walls_sig, np.int8)
        self.n_floor = int((self.a <= floor).sum())
        self.guide_id = _sha_arrays(self.sig, self.a, extra=f'Orbit:{broken}:{floor}:' + ('' if walls_sig is None else _sha_arrays(self.walls_sig)))
        self.n_rows = 0

    def nbr(self, r):
        return np.concatenate([self.t1[r], self.t2[r]], 1).astype(np.int64)

    def rows(self, S, diagnostics=False):
        r = np.asarray(S, np.int64); t = self.nbr(r); valid = t >= 0; tt = np.where(valid, t, 0)
        lr = self.la[tt] - self.la[r][:, None]
        if self.broken == 'inverse_ratio': lr = -lr
        ratio = self.HW[None, :] * np.exp(lr)
        sr = self.sig[r][:, None].astype(np.int16) * self.sig[tt]
        kept = valid & (sr < 0)
        rate = np.where(kept, ratio, 0.0)
        if self.walls_sig is not None:
            viol = valid & ((self.walls_sig[r][:, None].astype(np.int16) * self.walls_sig[tt]) > 0)
        else:
            viol = valid & (sr > 0)
        V = np.where(viol, ratio, 0.0).sum(1)
        D0 = 27.0 - 0.5 * (valid * self.JW[None, :]).sum(1)
        d = D0 + (0.0 if self.broken == 'no_vsf' else V)
        self.n_rows += len(r)
        rb = RowBatch(rate, d, guide_id=self.guide_id)
        if diagnostics:
            rb.valid = valid; rb.keep = kept; rb.sign = self.sig[r]; rb.log_abs = self.la[r]
        return rb

    def move(self, S, b):
        r = np.asarray(S, np.int64); b = np.asarray(b)
        t = np.where(b < 72, self.t1[r, np.minimum(b, 71)], self.t2[r, np.maximum(b - 72, 0)]).astype(np.int64)
        assert (t >= 0).all(), 'move along a parallel bond'
        return t

    def key(self, S):
        return np.asarray(S, np.int64)

    def sign(self, S):
        return self.sig[np.asarray(S, np.int64)]

    def E_var(self):
        """variational energy sum_r n_r a_r^2 E_L(r) / sum n a^2 (= efn_vec's E_guide_var); chunked full pass."""
        n = self.T['n']; D = len(n); num = 0.0; den = 0.0; R = 65536
        for i0 in range(0, D, R):
            r = np.arange(i0, min(D, i0 + R)); rb = self.rows(r)
            w = n[r] * self.a[r] ** 2; num += float(w @ rb.EL); den += float(w.sum())
        self.n_rows -= D
        return num / den
