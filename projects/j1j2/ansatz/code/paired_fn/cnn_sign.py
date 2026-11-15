"""cnn_sign.py (paired_fn, 2026-11-15): paired-FN guide providers for the 8x8 T-BAD loop (T-SEP).  Amplitude fixed = |M1| = |B0| = |rb8_prod| (pp2_lemma lemma path,
as the T-BAD FN runs: sign_bad/r*/code/r1_grade.py, sign_opts.SignOpt 'net').  Both guides are sign_opts.SignOpt subclasses (GuideBase.finish path, bond order J1 then J2),
sharing ONE amplitude cache and one sign cache per net, so arms that share a net and an amplitude cost no extra evaluation and the A = B control is bitwise exact.

 StoredCNNSign : n(z) = Marshall(z) * sgn(logit(z)) (logit D4 x flip averaged, s8_net.SignNet; sgn(0) = +1), rr_b = n(x) n(y_b) |B0(y_b)/B0(x)|.
 ExactStepSign : K[n](z) = n(z) * sgn[1 + alpha (E_L(|B0| n; z) - E)]  (sgn(0) = +1),   E_L(z) = H_zz + n(z) sum_{b anti} (J_b/2) n(z_b) |B0(z_b)/B0(z)|,
                 rr_b = K(x) K(y_b) |B0(y_b)/B0(x)|; evaluated at walker time at x and at each antiparallel neighbour y_b (E_L at y_b needs |B0| ratios at y_b and n on
                 the two-hop set).  alpha, E READ from the T-BAD FINAL constants below (r1_lib.ALPHA, E_REF), never recomputed.
Broken (validation only): ExactStepSign broken='alpha_flip' (alpha -> -alpha); a net with shuffled weights (shuffle_net) given to either provider."""
import hashlib, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
CODE = os.environ.get('SIGNBAD_CODE', '/scratch/a/A.Otaifi/ansatz_runs/sign_bad/r3/code')
os.environ.setdefault('GFMC_M1_CODE', CODE + '/m1'); sys.path.insert(0, CODE)
import numpy as np

ALPHA = -0.1754318714549749          # T-BAD FINAL (PLAN_sign.md), = r1_lib.ALPHA
E_REF = -31.915135844116154          # T-BAD FINAL, = r1_lib.E_REF
_PW = (np.uint64(1) << np.arange(64, dtype=np.uint64))


def cfg_keys(X):
    return ((np.asarray(X) > 0).astype(np.uint64) * _PW).sum(-1, dtype=np.uint64)


def spins_of(K):
    K = np.asarray(K, np.uint64)
    return (((K[:, None] >> np.arange(64, dtype=np.uint64)[None, :]) & np.uint64(1)).astype(np.float64) * 2 - 1)


def marshall_pm(keys):
    import r1_lib as R
    return R.marshall_pm(keys)


def _bucket(n):
    for b in (64, 256, 1024, 4096):
        if n <= b: return b
    return 4096


class NetSign:
    """stored CNN sign n = Marshall * sgn(logit) with a per-configuration cache (one object per net, shared by every guide that uses that net)."""
    def __init__(self, path_or_net, name='', cap=2_000_000):
        import s8_net as SN
        if isinstance(path_or_net, str):
            self.path = path_or_net; self.sha = hashlib.sha256(open(path_or_net, 'rb').read()).hexdigest(); self.net = SN.SignNet.load(path_or_net)
        else:
            self.path = None; self.sha = None; self.net = path_or_net
        self.name = name; self.d = {}; self.cap = cap; self.n_eval = 0; self.sec = 0.0; self.hit = 0; self.miss = 0

    def logit_batch(self, keys):
        import jax.numpy as jnp
        net = self.net; keys = np.asarray(keys, np.uint64); out = np.empty(len(keys), np.float64); i = 0
        while i < len(keys):
            b = _bucket(len(keys) - i); s = keys[i:i + b]; m = len(s)
            if m < b: s = np.concatenate([s, np.repeat(s[:1], b - m)])
            out[i:i + m] = np.asarray(net._lo(net.params, jnp.asarray(s)))[:m]; i += m
        return out

    def sign(self, keys):
        """int8 +-1 for uint64 keys (cached).  The cache is cleared before a lookup when above cap (never mid-call)."""
        keys = np.asarray(keys, np.uint64)
        if len(self.d) > self.cap: self.d.clear()
        uk, inv = np.unique(keys, return_inverse=True)
        need = np.array([k not in self.d for k in uk.tolist()], bool)
        self.hit += len(uk) - int(need.sum()); self.miss += int(need.sum())
        if need.any():
            t0 = time.perf_counter(); nk = uk[need]; lo = self.logit_batch(nk)
            s = (marshall_pm(nk) * np.where(lo >= 0, 1, -1)).astype(np.int8); self.n_eval += len(nk)
            for k, v in zip(nk.tolist(), s.tolist()): self.d[k] = v
            self.sec += time.perf_counter() - t0
        return np.array([self.d[k] for k in uk.tolist()], np.int8)[inv]


class AmpCache:
    """|B0| through pp2_lemma.one_swap_ratios, per-configuration cache key -> (2 Re ln B, |B(z_b)/B(z)| (128,)); shared by every guide that is given it."""
    def __init__(self, chunk=64, dtype='f64', cap=150_000):
        import gfmc_aug8 as G8, pp2_lemma as PL, r1_lib as R
        assert R.sha256(G8.PROD8) == R.PROD8_SHA, 'rb8_prod sha mismatch'
        self.PL = PL; self.p = PL.PP2.from_rbmpp_npz(G8.PROD8); self.chunk = int(chunk); self.dtype = dtype; self.cap = cap
        lat = self.p.lat; self.BI = np.asarray(lat.BI); self.BJ = np.asarray(lat.BJ); self.JB = np.asarray(lat.JB, float); self.HW = self.JB / 2; self.NB = lat.NB
        self.masks = (np.uint64(1) << self.BI.astype(np.uint64)) | (np.uint64(1) << self.BJ.astype(np.uint64))
        self.d = {}; self.hit = 0; self.miss = 0; self.sec = 0.0; self.calls = 0

    def reset_counts(self):
        self.hit = self.miss = self.calls = 0; self.sec = 0.0

    def get(self, keys):
        """keys (K,) uint64 -> lb2 (K,), absr (K, NB) float64 (0 on parallel bonds is NOT assumed: callers use the spin pattern)."""
        keys = np.asarray(keys, np.uint64)
        if len(self.d) > self.cap: self.d.clear()
        uk, inv = np.unique(keys, return_inverse=True)
        need = [i for i, k in enumerate(uk.tolist()) if k not in self.d]
        self.hit += len(uk) - len(need); self.miss += len(need)
        if need:
            t0 = time.perf_counter(); Z = spins_of(uk[need]); q = len(need); c = self.chunk; LB = []; R_ = []
            for i in range(0, q, c):
                blk = Z[i:i + c]; m = len(blk)
                if m < c: blk = np.concatenate([blk, np.repeat(blk[:1], c - m, 0)])
                lB, r = self.PL.one_swap_ratios(self.p, blk, self.dtype, walker_chunk=c, return_logB=True)
                LB.append(2.0 * np.real(np.asarray(lB))[:m]); R_.append(np.abs(np.asarray(r))[:m])
            LB = np.concatenate(LB); R_ = np.concatenate(R_, 0)
            for j, i in enumerate(need): self.d[int(uk[i])] = (LB[j], R_[j])
            self.sec += time.perf_counter() - t0; self.calls += 1
        lbu = np.array([self.d[k][0] for k in uk.tolist()]); ru = np.stack([self.d[k][1] for k in uk.tolist()])
        return lbu[inv], ru[inv]


def make_class():
    import sign_opts as SO

    class CnnSign(SO.SignOpt):
        """SignOpt('net') whose rows come from the shared caches; mode 'stored' or 'exact'."""
        def setup(self, amp, nets, mode, alpha=ALPHA, E=E_REF, broken=None):
            assert mode in ('stored', 'exact') and broken in (None, 'alpha_flip')
            self.amp = amp; self.nets = nets; self.mode = mode; self.alpha = float(alpha); self.E = float(E); self.broken = broken
            if broken == 'alpha_flip': self.alpha = -self.alpha
            self.st = dict(rows=0, x=0, nb=0, sec_rows=0.0, K_states=0, K_flips=0)
            return self

        def _diag0(self, Z):
            return 0.25 * np.sum(self.amp.JB[None] * Z[:, self.amp.BI] * Z[:, self.amp.BJ], 1)

        def K_of(self, keys):
            """(K, EL, n) of the exact step at unique-or-not keys: K = n sgn f, f = 1 + alpha (E_L - E); sgn(0) = +1."""
            A = self.amp; keys = np.asarray(keys, np.uint64); lb, absr = A.get(keys); Z = spins_of(keys)
            anti = Z[:, A.BI] != Z[:, A.BJ]; zi, zb = np.nonzero(anti); nbk = keys[zi] ^ A.masks[zb]
            allk = np.unique(np.concatenate([keys, nbk])); nv = self.nets.sign(allk).astype(np.float64)
            nz = nv[np.searchsorted(allk, keys)]; nn = nv[np.searchsorted(allk, nbk)]
            off = np.bincount(zi, weights=A.HW[zb] * nn * absr[zi, zb], minlength=len(keys)) * nz
            EL = self._diag0(Z) + off; f = 1.0 + self.alpha * (EL - self.E); c = np.where(f >= 0, 1.0, -1.0)
            return nz * c, EL, nz

        def signs_of_states(self, keys):
            """(n(z), K[n](z)) int8 for keys; for 'stored' mode K = n."""
            keys = np.asarray(keys, np.uint64)
            if self.mode == 'stored': n = self.nets.sign(keys); return n, n
            K, EL, nz = self.K_of(keys); return nz.astype(np.int8), K.astype(np.int8)

        def rows(self, X):
            t0 = time.perf_counter(); self.n_calls += 1; self.n_rows += X.shape[0]; X = np.asarray(X)
            A = self.amp; anti = X[:, self.BI] != X[:, self.BJ]; wi, bi = np.nonzero(anti)
            kx = cfg_keys(X); ky = kx[wi] ^ A.masks[bi]
            lb, absr = A.get(kx)
            allk = np.unique(np.concatenate([kx, ky]))
            if self.mode == 'stored': sv = self.nets.sign(allk).astype(np.float64)
            else:
                sv, _, _ = self.K_of(allk); self.st['K_states'] += len(allk)
            sx = sv[np.searchsorted(allk, kx)]; sy = sv[np.searchsorted(allk, ky)]
            self.last_lb = lb; self.last_sx = sx; self.last_sy = sy; self.last_wb = (wi, bi)
            rr = sx[wi] * sy * absr[wi, bi]
            if not (np.all(np.isfinite(rr)) and np.all(rr != 0)): raise FloatingPointError('CnnSign: non-finite or zero ratio')
            self.st['rows'] += 1; self.st['x'] += len(X); self.st['nb'] += len(wi)
            out = self.finish(X, anti, wi, bi, rr); self.st['sec_rows'] += time.perf_counter() - t0
            return out

        def rel_sign(self, X):
            """sign of the arm relative to the n_ref sign of the reference net given as self.ref_nets (diagnostic: the flip set)."""
            keys = cfg_keys(np.asarray(X)); _, K = self.signs_of_states(keys)
            return (K * self.ref_nets.sign(keys)).astype(float)

    return CnnSign


def build(mode, amp, nets, alpha=ALPHA, E=E_REF, broken=None, wchunk=64):
    """One guide (SignOpt subclass).  mode 'stored' (StoredCNNSign) or 'exact' (ExactStepSign).  amp: AmpCache, nets: NetSign (the sign net n).
    Returns (guide, info)."""
    import gfmc_aug8 as G8, pp2_lemma as PL
    al, Ec, w = G8.m1_consts()
    g = make_class()(amp.p, w, 'net', net=nets.net, walker_chunk=wchunk, dtype=amp.dtype, ypad=wchunk)
    g.setup(amp, nets, mode, alpha, E, broken)
    info = dict(mode=mode, alpha=g.alpha, E=g.E, broken=broken, net=nets.path, net_sha256=nets.sha, amplitude=G8.PROD8)
    return g, info


def shuffle_net(netsign, seed=0):
    """BROKEN net for validation: every parameter leaf's entries randomly permuted (same shapes and statistics, weights destroyed)."""
    import jax, jax.numpy as jnp, s8_net as SN
    rg = np.random.default_rng(seed); net = SN.SignNet(**netsign.net.cfg)
    net.params = jax.tree_util.tree_map(lambda a: jnp.asarray(rg.permutation(np.asarray(a).ravel()).reshape(np.shape(a))), netsign.net.params)
    return NetSign(net, name='shuffled')
