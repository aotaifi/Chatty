"""Core machinery for the learned FN/Krylov loop on the periodic 6x6 J1-J2 model (J2/J1 = 0.5).

Conventions (identical to k1_repeat_fixed_amp_6x6.py, fn_guides_compare.py, ed_6x6_sign_errors.py):
  state = uint64, bit i = spin up on site i = x + 6 y; network input 2*bit-1 in site order.
  H = sum_bonds J [S_i.S_j]; off-diagonal element between x and y = x with (i,j) exchanged is +J/2.
  Marshall(x) = (-1)^{popcount(x & A)}, A = sites with (x+y) even.
  Krylov sign step on amplitude a and sign s: r(x) = (H a s)(x)/(a(x) s(x)),
      s'(x) = s(x) * (+1 if r(x) <= T else -1).
  Sign chain: s^(0) = Marshall, s^(j+1) = step(amps[j], s^(j), Ts[j]).
  The task's "s_0" (one Krylov step from Marshall with |ViT|, T=-14.985799779) is s^(1).
  Lattice FN for a guide (a, s): an edge (x,y) is kept ("allowed") iff s(x) s(y) = -1 (s H s < 0);
  violating edges move to the diagonal: d_FN(x) = diag(x) + sum_{viol y} J/2 a(y)/a(x).
  Local energy of trial amplitude b with the guide signs under the FROZEN H_FN[a, s]:
      E_L(x) = d_FN(x) - sum_{allowed y} J/2 b(y)/b(x)       (= <H>_{a s} local energy when b = a).
"""
import math, time, json, os
from functools import partial
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import flax
from jax.flatten_util import ravel_pytree
import vit_dt

L = 6; N = 36; J2 = 0.5
T_FN = -14.985799779143964        # threshold of all existing 6x6 FN runs (s^(1))


def _bonds():
    nn_, nnn = [], []
    q = lambda x, y: (x % L) + L * (y % L)
    for y in range(L):
        for x in range(L):
            i = q(x, y); nn_ += [(i, q(x + 1, y)), (i, q(x, y + 1))]
            nnn += [(i, q(x + 1, y + 1)), (i, q(x + 1, y - 1))]
    return nn_, nnn


NN, NNN = _bonds()
ALL = [(i, j, 1., -1.) for i, j in NN] + [(i, j, J2, 1.) for i, j in NNN]
BI = np.array([b[0] for b in ALL], np.uint64); BJ = np.array([b[1] for b in ALL], np.uint64)
JB = np.array([b[2] for b in ALL], float); MRB = np.array([b[3] for b in ALL], float)
MASKS = (np.uint64(1) << BI) | (np.uint64(1) << BJ)
A_MASK = np.uint64(sum(1 << (x + L * y) for y in range(L) for x in range(L) if (x + y) % 2 == 0))
NB_ = len(ALL)
ONE = np.uint64(1)


def log(*a):
    print(time.strftime('[%H:%M:%S]'), *a, flush=True)


def neighbors(S):
    S = np.asarray(S, np.uint64)[:, None]
    valid = (((S >> BI) ^ (S >> BJ)) & ONE).astype(bool)
    return S ^ MASKS, valid


def diag_vec(valid):
    return 0.25 * JB.sum() - 0.5 * (valid.astype(float) @ JB)


def marshall_vec(S):
    return np.where(np.bitwise_count(np.asarray(S, np.uint64) & A_MASK) % 2 == 0, 1., -1.)


def bits_to_spins_np(S):
    a = np.asarray(S, np.uint64).reshape(-1, 1)
    return 2.0 * ((a >> np.arange(N, dtype=np.uint64)) & ONE).astype(np.float64) - 1.0


def spins_to_bits_np(X):
    X = np.asarray(X).reshape(-1, N)
    return np.sum((X > 0).astype(np.uint64) << np.arange(N, dtype=np.uint64)[None, :], axis=1, dtype=np.uint64)


# ----------------------------------------------------------------------------------- network
class Net:
    """Real log-amplitude log|psi_theta(x)| of the 6x6 ViT, parameters as one flat vector."""

    def __init__(self, ckpt, dtype='float32', batch=16384, jac_chunk=256):
        # dtype: 'float64' | 'float32' with optional '+ln2' (two-pass LayerNorm variance) and '+head64'
        # (output head in float64), e.g. 'float32+ln2+head64'
        # '+tf32' allows TF32 matmuls (jax GPU default); otherwise full float32 matmul precision is forced
        # (TF32 gives log-amplitude errors ~0.03 rms on this ViT, see prec_test.py)
        base = dtype.split('+')[0]
        if '+tf32' not in dtype:
            jax.config.update('jax_default_matmul_precision', 'highest')
        vit_dt.set_dtype(base, ln_fast='+ln2' not in dtype, head64='+head64' in dtype)
        self.dtype = dtype
        self.DT = jnp.float32 if base == 'float32' else jnp.float64
        self.model = vit_dt.make_model_6x6()
        tmpl = self.model.init(jax.random.PRNGKey(1234), jnp.zeros((1, N), self.DT))
        obj = flax.serialization.msgpack_restore(open(ckpt, 'rb').read())
        if 'variables' in obj: obj = obj['variables']
        var = flax.serialization.from_state_dict(tmpl, obj)
        params = jax.tree_util.tree_map(lambda p: jnp.asarray(p, self.DT), var['params'])
        self.flat0, self.unravel = ravel_pytree(params)
        self.npar = int(self.flat0.size)
        self.batch = batch; self.jac_chunk = jac_chunk
        model = self.model; unravel = self.unravel; DT = self.DT
        ar = jnp.arange(N, dtype=jnp.uint64)

        def f_spins(flat, X):
            z = vit_dt.logpsi_transl_2d(model.apply, 2, {'params': unravel(flat)}, X)
            return jnp.real(z)

        def f_bits(flat, S):
            X = (((S[:, None] >> ar[None, :]) & jnp.uint64(1)).astype(DT) * 2 - 1)
            return f_spins(flat, X)

        def fc_bits(flat, S):
            X = (((S[:, None] >> ar[None, :]) & jnp.uint64(1)).astype(DT) * 2 - 1)
            return vit_dt.logpsi_transl_2d(model.apply, 2, {'params': unravel(flat)}, X)

        self.f_spins = f_spins
        self._f_bits = jax.jit(f_bits)
        self._fc_bits = jax.jit(fc_bits)
        self._grad_bits = jax.jit(jax.vmap(jax.grad(lambda fl, s: f_bits(fl, s[None])[0]), in_axes=(None, 0)))
        self.neval = 0

    def logabs(self, flat, S):
        """log|psi| for uint64 states (numpy), fixed padded batches (no recompiles); float64 numpy out."""
        S = np.asarray(S, np.uint64); n = len(S); B = self.batch
        out = np.empty(n, np.float64)
        if n == 0: return out
        res = []
        for i in range(0, n, B):
            s = S[i:i + B]; m = len(s)
            if m < B:
                s = np.concatenate([s, np.repeat(s[:1], B - m)])
            res.append((i, m, self._f_bits(flat, jnp.asarray(s))))
        for i, m, z in res:
            out[i:i + m] = np.asarray(z)[:m]
        self.neval += n
        return out

    def logpsi_c(self, flat, S):
        """complex log psi (own ViT phase) for uint64 states."""
        S = np.asarray(S, np.uint64); n = len(S); B = self.batch
        out = np.empty(n, np.complex128)
        for i in range(0, n, B):
            s = S[i:i + B]; m = len(s)
            if m < B: s = np.concatenate([s, np.repeat(s[:1], B - m)])
            out[i:i + m] = np.asarray(self._fc_bits(flat, jnp.asarray(s)))[:m]
        self.neval += n
        return out

    def jac(self, flat, S):
        """(n, npar) device array of d log|psi| / d theta."""
        S = np.asarray(S, np.uint64); n = len(S); c = self.jac_chunk
        npad = (-n) % c
        Sp = np.concatenate([S, np.repeat(S[:1], npad)]) if npad else S
        parts = [self._grad_bits(flat, jnp.asarray(Sp[i:i + c])) for i in range(0, len(Sp), c)]
        return jnp.concatenate(parts, 0)[:n]


class Amp:
    """Callable amplitude log a(x) = Net(flat)(x) with an optional sorted-array cache."""

    def __init__(self, net, flat, cache=False, name=''):
        self.net = net; self.flat = flat; self.cache = cache; self.name = name
        self.keys = np.zeros(0, np.uint64); self.vals = np.zeros(0)
        self.nnew = 0

    def __call__(self, S):
        S = np.asarray(S, np.uint64)
        if not self.cache:
            self.nnew += len(S)
            return self.net.logabs(self.flat, S)
        u, inv = np.unique(S, return_inverse=True)
        pos = np.searchsorted(self.keys, u); pos_c = np.minimum(pos, max(len(self.keys) - 1, 0))
        hit = (len(self.keys) > 0) & (self.keys[pos_c] == u) if len(self.keys) else np.zeros(len(u), bool)
        val = np.empty(len(u))
        if hit.any(): val[hit] = self.vals[pos_c[hit]]
        miss = u[~hit]
        if len(miss):
            vm = self.net.logabs(self.flat, miss); self.nnew += len(miss)
            val[~hit] = vm
            ins = np.searchsorted(self.keys, miss)
            self.keys = np.insert(self.keys, ins, miss); self.vals = np.insert(self.vals, ins, vm)
        return val[inv]


# ----------------------------------------------------------------------------------- sign chains
def build_levels(X0, D, maxstates=40_000_000):
    levels = [np.unique(np.asarray(X0, np.uint64))]
    nb = []
    for l in range(D):
        NBl, Vl = neighbors(levels[l])
        nxt = np.unique(np.concatenate([levels[l], NBl[Vl]]))
        if len(nxt) > maxstates:
            raise MemoryError(f'level {l + 1} has {len(nxt)} states')
        idx = np.searchsorted(nxt, NBl).astype(np.int64); idx[~Vl] = 0
        selfidx = np.searchsorted(nxt, levels[l])
        nb.append((idx, Vl, selfidx)); levels.append(nxt)
        del NBl
    return levels, nb


def r_on_level(levels, nb, l, la_up, s_up):
    """r(x) for x on level l given log a and s on level l+1."""
    idx, Vl, selfidx = nb[l]
    s_self = s_up[selfidx]; la_l = la_up[selfidx]
    w = np.exp(la_up[idx] - la_l[:, None])
    r = diag_vec(Vl) + np.sum(np.where(Vl, 0.5 * JB[None, :] * s_up[idx] * s_self[:, None] * w, 0.), axis=1)
    return r, s_self


def chain_signs(levels, nb, amps, Ts, K):
    """s^(K) on level D-K (D = len(levels)-1).  Returns dict level->(s) for every created level and r's."""
    D = len(levels) - 1
    s = marshall_vec(levels[D]); out = {D: s}; rs = {}
    for j in range(K):
        l = D - 1 - j
        la_up = amps[j](levels[l + 1])
        r, s_self = r_on_level(levels, nb, l, la_up, s)
        s = s_self * np.where(r <= Ts[j], 1., -1.)
        out[l] = s; rs[l] = r
    return out, rs


def restrict(levels, arr, l_from, l_to):
    return arr[np.searchsorted(levels[l_from], levels[l_to])]


class Guide:
    """Guide (a_g, s^(K)) with sign chain amps[0..K-1], Ts[0..K-1]; a_g = amp_g."""

    def __init__(self, amps, Ts, amp_g):
        assert len(amps) == len(Ts)
        self.amps = list(amps); self.Ts = [float(t) for t in Ts]; self.K = len(Ts); self.amp_g = amp_g

    def signs(self, X):
        """s^(K) on states X (any order, duplicates allowed)."""
        X = np.asarray(X, np.uint64)
        if self.K == 0: return marshall_vec(X)
        lv, nb = build_levels(X, self.K)
        s, _ = chain_signs(lv, nb, self.amps, self.Ts, self.K)
        return s[0][np.searchsorted(lv[0], X)]

    def local(self, X):
        """FN local data at states X (in order): d_FN, allowed-edge (owner, child, J/2 * a(y)/a(x)),
        variational local energy E_H, log a_g(X), s(X), and the allowed mask on the full edge list."""
        X = np.asarray(X, np.uint64)
        D = self.K + 1
        lv, nb = build_levels(X, D)
        s_lv, _ = chain_signs(lv, nb, self.amps, self.Ts, self.K)
        s1 = s_lv[1]
        la1 = self.amp_g(lv[1])
        idx, V, selfidx = nb[0]
        p0 = np.searchsorted(lv[0], X)
        idx = idx[p0]; V = V[p0]; sx = s1[selfidx[p0]]; lax_ = la1[selfidx[p0]]
        sy = s1[idx]; w = 0.5 * JB[None, :] * np.exp(la1[idx] - lax_[:, None])
        allowed = V & (sx[:, None] * sy < 0)
        viol = V & ~allowed
        diag = diag_vec(V)
        dfn = diag + np.sum(np.where(viol, w, 0.), axis=1)
        eh = dfn - np.sum(np.where(allowed, w, 0.), axis=1)
        own, col = np.nonzero(V)
        NBx = neighbors(X)[0]
        return dict(dfn=dfn, eh=eh, diag=diag, own=own.astype(np.int64), child=NBx[own, col], rate=w[own, col],
                    allowed=allowed[own, col], lac=la1[idx[own, col]], lax=lax_, s=sx, n=len(X))


LOCAL_KEYS = ('dfn', 'eh', 'diag', 'own', 'child', 'rate', 'allowed', 'lac', 'lax', 's')


def local_chunked(guide, X, chunk):
    """Guide.local on chunks; concatenates (owners shifted)."""
    X = np.asarray(X, np.uint64); parts = []
    for i in range(0, len(X), chunk):
        d = guide.local(X[i:i + chunk]); d['own'] = d['own'] + i; parts.append(d)
    out = {k: np.concatenate([p[k] for p in parts]) for k in LOCAL_KEYS}
    out['n'] = len(X)
    return out


def trial_elocs(dat, rx, rc):
    """For trial amplitude b = a_g exp(r):  rx = r on X, rc = r on all edge children.
    Returns (E_L frozen-H_FN[a_g,s_g] local energy of b s_g,  E_H variational local energy of b s_g).
    rate = J/2 a_g(y)/a_g(x);  allowed edges: s(x)s(y) = -1."""
    er = dat['rate'] * np.exp(np.clip(rc - rx[dat['own']], -50, 50))
    al = dat['allowed']
    off_al = np.bincount(dat['own'][al], weights=er[al], minlength=dat['n'])
    off_vi = np.bincount(dat['own'][~al], weights=er[~al], minlength=dat['n'])
    return dat['dfn'] - off_al, dat['diag'] + off_vi - off_al


# ----------------------------------------------------------------------------------- threshold step
def threshold_edges(guide_old, amp_new, X):
    """Edges for the next Krylov step s' = s^(K) sgn(T - r), r = r[amp_new, s^(K)], on central states X.
    Needs s^(K) on level 2 (D = K+2), amp_new on level 2."""
    X = np.asarray(X, np.uint64)
    K = guide_old.K; D = K + 2
    lv, nb = build_levels(X, D)
    s_lv, _ = chain_signs(lv, nb, guide_old.amps, guide_old.Ts, K)
    s2 = s_lv[2]
    la2 = amp_new(lv[2])
    r1, s1 = r_on_level(lv, nb, 1, la2, s2)            # r and s^(K) on level 1
    la1 = la2[nb[1][2]]
    idx, V, selfidx = nb[0]
    p0 = np.searchsorted(lv[0], X)
    idx = idx[p0]; V = V[p0]; sp = selfidx[p0]
    rx = r1[sp]; sx = s1[sp]
    own, col = np.nonzero(V)
    y = idx[own, col]
    c = 0.5 * JB[col] * np.exp(la1[y] - la1[sp[own]]) * s1[y] * sx[own]
    return dict(xi=own, c=c, rx=rx[own], ry=r1[y], rx_node=rx, diag=diag_vec(V), n=len(X), sx=sx)


def merge_edges(parts):
    out = dict(xi=[], c=[], rx=[], ry=[], rx_node=[], diag=[], sx=[]); off = 0
    for p in parts:
        out['xi'].append(p['xi'] + off); off += p['n']
        for k in ('c', 'rx', 'ry', 'rx_node', 'diag', 'sx'): out[k].append(p[k])
    o = {k: np.concatenate(v) for k, v in out.items()}; o['n'] = off
    return o


def energy_curve(E, mask_x=None):
    """E(T) = mean_x [diag + sum_y c q(x) q(y)], q = +1 iff r <= T (k1_repeat_fixed_amp_6x6.energy_curve)."""
    nx = E['n']
    keep = np.ones(len(E['xi']), bool) if mask_x is None else mask_x[E['xi']]
    nxe = nx if mask_x is None else int(mask_x.sum())
    c = E['c'][keep] / nxe
    rx = E['rx'][keep]; ry = E['ry'][keep]
    base = (E['diag'].sum() if mask_x is None else E['diag'][mask_x].sum()) / nxe + c.sum()
    rvals = np.unique(np.concatenate([rx, ry]))
    cand = np.r_[rvals[0] - 1.0, 0.5 * (rvals[:-1] + rvals[1:]), rvals[-1] + 1.0]
    lo = np.minimum(rx, ry); hi = np.maximum(rx, ry)
    a = np.searchsorted(cand, lo, 'left'); b = np.searchsorted(cand, hi, 'left')
    diff = np.zeros(len(cand) + 1)
    np.add.at(diff, a, -2 * c); np.add.at(diff, b, +2 * c)
    curve = base + np.cumsum(diff)[:len(cand)]
    return base, cand, curve


def eloc_at_T(E, T):
    """per-x local energy of a*s' with s' = s sgn(T - r) (T = +inf -> no flip)."""
    qx = np.where(E['rx'] <= T, 1., -1.); qy = np.where(E['ry'] <= T, 1., -1.)
    e = E['diag'].copy(); np.add.at(e, E['xi'], E['c'] * qx * qy)
    return e


# ----------------------------------------------------------------------------------- sampler
BI_J = jnp.asarray(BI.astype(np.int32)); BJ_J = jnp.asarray(BJ.astype(np.int32))


def make_sampler(net):
    f = net.f_spins

    @partial(jax.jit, static_argnums=(4,))
    def run(flat, X, la, key, nsteps):
        nc = X.shape[0]; ar = jnp.arange(nc)

        def body(carry, _):
            X, la, key, acc = carry
            key, k1, k2 = jax.random.split(key, 3)
            b = jax.random.randint(k1, (nc,), 0, NB_)
            i = BI_J[b]; j = BJ_J[b]
            xi = X[ar, i]; xj = X[ar, j]
            ok = xi != xj
            cand = X.at[ar, i].set(xj).at[ar, j].set(xi)
            lb = f(flat, cand)
            u = jax.random.uniform(k2, (nc,))
            accept = ok & (jnp.log(u) < 2.0 * (lb - la))
            X = jnp.where(accept[:, None], cand, X); la = jnp.where(accept, lb, la)
            return (X, la, key, acc + accept.sum()), None

        (X, la, key, acc), _ = jax.lax.scan(body, (X, la, key, jnp.asarray(0)), None, length=nsteps)
        return X, la, key, acc

    return run


class Chains:
    """Persistent bond-exchange Metropolis chains for |a_theta|^2 (same move set as fn_guides_compare.vmc)."""

    def __init__(self, net, nchains, seed):
        self.net = net; self.run = make_sampler(net); self.nc = nchains
        rg = np.random.default_rng(seed)
        X = np.zeros((nchains, N))
        for k in range(nchains):
            X[k] = -1; X[k, rg.choice(N, N // 2, replace=False)] = 1
        self.X = jnp.asarray(X, net.DT); self.key = jax.random.PRNGKey(seed); self.la = None
        self.acc = 0; self.props = 0

    def reset_la(self, flat):
        self.la = self.net.f_spins(flat, self.X)

    def advance(self, flat, sweeps):
        if self.la is None: self.reset_la(flat)
        X, la, key, acc = self.run(flat, self.X, self.la, self.key, int(sweeps * N))
        self.X, self.la, self.key = X, la, key
        self.acc += int(acc); self.props += int(sweeps * N) * self.nc
        return spins_to_bits_np(np.asarray(X))
