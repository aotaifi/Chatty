"""Shared machinery for learned FN/Krylov iterations 2-3 on 6x6 (J2/J1 = 0.5) with STORED signs.

Builds on ll6_core (learned_loop_6x6) and ss6_core (stored_signs_6x6), both imported unchanged.

Guide spec (JSON):
  {"nets": [sign-net .pkl paths], "Ts_nets": [...],           # stored sign  s_st = Marshall * prod_j sgn(logit_j)
   "hop": null | {"amp": params.npy | "base", "T": float},    # optional ONE exact Krylov hop on top of s_st:
                                                             #   s = s_st * sgn(T - r[a_hop, s_st])
   "amp_g": params.npy | "base",                             # guide amplitude a_g = |ViT(theta)|
   "name": str}
Signs need level 1 (with hop) or level 0 (stored only) around a state, so FN / SR local data need level 1 + hop.
"""
import json, os, time
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import ll6_core as C
import ss6_core as S

N = 36
CKPT = 'vit_J2=0.50_N=6x6_k=0.mpack'
log = C.log
E0_SITE = -0.5038096538908783
E_VIT_SITE = (-0.5036542608124052, 2.131044877124918e-05)


class GpuClock:
    """wall-clock accounting per stage (the job holds one GPU, so wall time = GPU time)."""

    def __init__(self):
        self.t0 = time.time(); self.last = self.t0; self.stages = {}

    def tick(self, name):
        t = time.time(); self.stages[name] = self.stages.get(name, 0.) + (t - self.last); self.last = t
        return {k: round(v / 3600, 4) for k, v in self.stages.items()}

    def hours(self):
        return {k: round(v / 3600, 4) for k, v in self.stages.items()} | {'total': round((time.time() - self.t0) / 3600, 4)}


# ------------------------------------------------------------------ caches
class FnCache:
    """f(states) -> float64 with a sorted table + pending buffer (f must be deterministic)."""

    def __init__(self, f, cap=60_000_000, name=''):
        self.f = f; self.cap = cap; self.name = name
        self.k = np.zeros(0, np.uint64); self.v = np.zeros(0)
        self.bk = []; self.bv = []; self.nb = 0; self.nnew = 0

    def _merge(self):
        if not self.bk: return
        k = np.concatenate([self.k] + self.bk); v = np.concatenate([self.v] + self.bv)
        o = np.argsort(k, kind='stable'); k = k[o]; v = v[o]
        keep = np.r_[True, k[1:] != k[:-1]] if len(k) else np.zeros(0, bool)
        self.k, self.v = k[keep], v[keep]; self.bk, self.bv, self.nb = [], [], 0

    def __call__(self, X):
        X = np.asarray(X, np.uint64)
        u, inv = np.unique(X, return_inverse=True)
        if self.nb > 2_000_000: self._merge()
        val = np.empty(len(u)); hit = np.zeros(len(u), bool)
        if len(self.k):
            p = np.minimum(np.searchsorted(self.k, u), len(self.k) - 1)
            hit = self.k[p] == u; val[hit] = self.v[p[hit]]
        if self.nb and (~hit).any():
            bk = np.concatenate(self.bk); bv = np.concatenate(self.bv); o = np.argsort(bk); bk = bk[o]; bv = bv[o]
            idx = np.nonzero(~hit)[0]; uu = u[idx]
            p = np.minimum(np.searchsorted(bk, uu), len(bk) - 1); h2 = bk[p] == uu
            val[idx[h2]] = bv[p[h2]]; hit[idx[h2]] = True
        miss = u[~hit]
        if len(miss):
            vm = np.asarray(self.f(miss), np.float64); self.nnew += len(miss); val[~hit] = vm
            if len(self.k) + self.nb < self.cap:
                self.bk.append(miss); self.bv.append(vm); self.nb += len(miss)
        return val[inv]

    def clear(self):
        self.k = np.zeros(0, np.uint64); self.v = np.zeros(0); self.bk, self.bv, self.nb = [], [], 0


def load_flat(net, p):
    return net.flat0 if p == 'base' else jnp.asarray(np.load(p), net.DT)


class AmpBank:
    """one cached log|ViT(theta)| per parameter file."""

    def __init__(self, net, cache=True):
        self.net = net; self.d = {}; self.cache = cache

    def __call__(self, p):
        if p not in self.d:
            fl = load_flat(self.net, p)
            f = lambda S, fl=fl: self.net.logabs(fl, S)
            self.d[p] = FnCache(f, name=p) if self.cache else f
        return self.d[p]


# ------------------------------------------------------------------ sign nets with bucketed padding
_BUCKETS = tuple(b for b in (1024, 4096, 16384, 65536) if b <= int(os.environ.get('IT2_NET_BATCH', '16384')))


def net_logit(snet, X):
    X = np.asarray(X, np.uint64); out = np.empty(len(X), np.float32)
    i = 0
    while i < len(X):
        m = min(len(X) - i, _BUCKETS[-1]); B = next(b for b in _BUCKETS if b >= m)
        s = X[i:i + m]
        if m < B: s = np.concatenate([s, np.repeat(s[:1], B - m)])
        out[i:i + m] = np.asarray(snet._lo(snet.params, jnp.asarray(s)))[:m]
        i += m
    snet.nevals += len(X)
    return out


S.SignNet.logit = lambda self, X, B=None: net_logit(self, X)   # bucketed (11 GB GPUs)


class StoredSign:
    def __init__(self, paths):
        self.paths = list(paths); self.nets = [S.SignNet.load(p) for p in self.paths]
        self.cache = FnCache(self._raw, name='stored')

    def _raw(self, X):
        s = C.marshall_vec(X)
        for n in self.nets: s = s * np.where(net_logit(n, X) >= 0, 1., -1.)
        return s

    def __call__(self, X):
        return self.cache(X)


def resolve(p, base_dir):
    if p == 'base' or os.path.isabs(p): return p
    return os.path.join(base_dir, p)


class HGuide:
    """guide (a_g, s) from a spec dict (see module doc)."""

    def __init__(self, spec, net, amps=None, base_dir='.'):
        self.spec = spec; self.net = net
        self.amps = amps if amps is not None else AmpBank(net)
        self.st = StoredSign([resolve(p, base_dir) for p in spec['nets']])
        self.hop = spec.get('hop')
        if self.hop:
            self.amp_h = self.amps(resolve(self.hop['amp'], base_dir)); self.T = float(self.hop['T'])
        self.amp_g = self.amps(resolve(spec['amp_g'], base_dir))
        self.D = 1 if self.hop else 0      # extra levels needed for the sign

    def signs_on(self, lv, nb, l):
        """signs on level l (needs level l + D)."""
        if not self.hop: return self.st(lv[l])
        la = self.amp_h(lv[l + 1]); s_up = self.st(lv[l + 1])
        r, s_self = C.r_on_level(lv, nb, l, la, s_up)
        return s_self * np.where(r <= self.T, 1., -1.)

    def signs(self, X, chunk=8192):
        X = np.asarray(X, np.uint64)
        if not self.hop: return self.st(X)
        out = np.empty(len(X))
        for i in range(0, len(X), chunk):
            Xc = X[i:i + chunk]; lv, nb = C.build_levels(Xc, 1)
            s0 = self.signs_on(lv, nb, 0); out[i:i + chunk] = s0[np.searchsorted(lv[0], Xc)]
        return out

    def local(self, X):
        """same dict as ll6_core.Guide.local (frozen-H_FN / FN local data at states X)."""
        X = np.asarray(X, np.uint64)
        lv, nb = C.build_levels(X, 1 + self.D)
        s1 = self.signs_on(lv, nb, 1)
        la1 = self.amp_g(lv[1])
        idx, V, selfidx = nb[0]
        p0 = np.searchsorted(lv[0], X)
        idx = idx[p0]; V = V[p0]; sx = s1[selfidx[p0]]; lax_ = la1[selfidx[p0]]
        sy = s1[idx]; w = 0.5 * C.JB[None, :] * np.exp(la1[idx] - lax_[:, None])
        allowed = V & (sx[:, None] * sy < 0)
        viol = V & ~allowed
        diag = C.diag_vec(V)
        dfn = diag + np.sum(np.where(viol, w, 0.), axis=1)
        eh = dfn - np.sum(np.where(allowed, w, 0.), axis=1)
        own, col = np.nonzero(V)
        NBx = C.neighbors(X)[0]
        return dict(dfn=dfn, eh=eh, diag=diag, own=own.astype(np.int64), child=NBx[own, col], rate=w[own, col],
                    allowed=allowed[own, col], lac=la1[idx[own, col]], lax=lax_, s=sx, n=len(X))

    def local_chunked(self, X, chunk):
        X = np.asarray(X, np.uint64); parts = []
        for i in range(0, len(X), chunk):
            d = self.local(X[i:i + chunk]); d['own'] = d['own'] + i; parts.append(d)
        out = {k: np.concatenate([p[k] for p in parts]) for k in C.LOCAL_KEYS}
        out['n'] = len(X)
        return out


def load_spec(path):
    sp = json.load(open(path)); bd = os.path.dirname(os.path.abspath(path))
    return sp, bd


def spec_dump(spec, path):
    json.dump(spec, open(path, 'w'), indent=1)


# ------------------------------------------------------------------ samplers
def make_tempered_sampler(net, beta):
    f = net.f_spins

    @partial(jax.jit, static_argnums=(4,))
    def run(flat, X, la, key, nsteps):
        nc = X.shape[0]; ar = jnp.arange(nc)

        def body(carry, _):
            X, la, key, acc = carry
            key, k1, k2 = jax.random.split(key, 3)
            b = jax.random.randint(k1, (nc,), 0, C.NB_)
            i = C.BI_J[b]; j = C.BJ_J[b]
            xi = X[ar, i]; xj = X[ar, j]
            ok = xi != xj
            cand = X.at[ar, i].set(xj).at[ar, j].set(xi)
            lb = f(flat, cand)
            u = jax.random.uniform(k2, (nc,))
            accept = ok & (jnp.log(u) < 2.0 * beta * (lb - la))
            X = jnp.where(accept[:, None], cand, X); la = jnp.where(accept, lb, la)
            return (X, la, key, acc + accept.sum()), None

        (X, la, key, acc), _ = jax.lax.scan(body, (X, la, key, jnp.asarray(0)), None, length=nsteps)
        return X, la, key, acc

    return run


def sample(net, flat, beta, nchains, nrounds, between, burn, seed):
    """states (nrounds, nchains) uint64 from |a_theta|^(2 beta) (bond-exchange Metropolis); acceptance."""
    run = make_tempered_sampler(net, float(beta))
    rg = np.random.default_rng(seed)
    X = -np.ones((nchains, N))
    for k in range(nchains): X[k, rg.choice(N, N // 2, replace=False)] = 1
    X = jnp.asarray(X, net.DT); key = jax.random.PRNGKey(seed)
    la = net.f_spins(flat, X); acc = 0; props = 0
    X, la, key, a = run(flat, X, la, key, int(burn * N)); acc += int(a); props += burn * N * nchains
    out = []
    for r in range(nrounds):
        X, la, key, a = run(flat, X, la, key, int(between * N)); acc += int(a); props += between * N * nchains
        out.append(C.spins_to_bits_np(np.asarray(X)))
    return np.stack(out), acc / max(props, 1)


# ------------------------------------------------------------------ statistics
def chain_mean_se(v, nch):
    v = np.asarray(v, float); cm = v.reshape(-1, nch).mean(0)
    return [float(v.mean()), float(cm.std(ddof=1) / np.sqrt(nch))]


def jk_ratio_diff(v1, v0, w0, nch):
    """mean(v1) - sum(w0 v0)/sum(w0) with chain jackknife (rows = rounds, cols = chains)."""
    v1 = np.asarray(v1).reshape(-1, nch); v0 = np.asarray(v0).reshape(-1, nch); w0 = np.asarray(w0).reshape(-1, nch)
    full = v1.mean() - np.sum(w0 * v0) / np.sum(w0); reps = []
    for c in range(nch):
        m = np.ones(nch, bool); m[c] = False
        reps.append(v1[:, m].mean() - np.sum(w0[:, m] * v0[:, m]) / np.sum(w0[:, m]))
    reps = np.asarray(reps)
    return [float(full), float(np.sqrt((nch - 1) / nch * np.sum((reps - reps.mean()) ** 2)))]


def wrong_frac(s, strue):
    m = float(np.mean(s != strue)); g = 1 if m <= 0.5 else -1; w = min(m, 1 - m)
    return dict(w=w, err=float(np.sqrt(max(w, 1e-12) * (1 - w) / len(s))), nwrong=int(round(w * len(s))), g=g)


def paired_wrong(sa, sb, strue):
    ga = 1 if np.mean(sa != strue) <= .5 else -1; gb = 1 if np.mean(sb != strue) <= .5 else -1
    d = (sa != ga * strue).astype(float) - (sb != gb * strue).astype(float)
    return dict(mean=float(d.mean()), err=float(d.std(ddof=1) / np.sqrt(len(d))), n_differ=int(np.sum(sa * ga != sb * gb)))


# ------------------------------------------------------------------ threshold selection
def energy_curve_w(E, wx):
    """C.energy_curve with per-state weights wx (bootstrap multiplicities)."""
    wsum = wx.sum(); we = wx[E['xi']]
    keep = we > 0
    c = E['c'][keep] * we[keep] / wsum
    rx = E['rx'][keep]; ry = E['ry'][keep]
    base = float(np.sum(E['diag'] * wx) / wsum + c.sum())
    rvals = np.unique(np.concatenate([rx, ry]))
    cand = np.r_[rvals[0] - 1.0, 0.5 * (rvals[:-1] + rvals[1:]), rvals[-1] + 1.0]
    lo = np.minimum(rx, ry); hi = np.maximum(rx, ry)
    a = np.searchsorted(cand, lo, 'left'); b = np.searchsorted(cand, hi, 'left')
    diff = np.zeros(len(cand) + 1)
    np.add.at(diff, a, -2 * c); np.add.at(diff, b, +2 * c)
    return base, cand, base + np.cumsum(diff)[:len(cand)]


def robust_threshold(E, nch, sigma=1.0, nboot=100, seed=0):
    """E = merged threshold edges on states ordered (rounds, chains) row-major.
    Returns dict with raw argmin, Gaussian-smoothed argmin (width sigma in r units), half-sample and
    bootstrap (over chains) spreads of both.  'no flip' = +inf candidate (T = 1e9)."""
    base, cand, curve = C.energy_curve(E)
    n = E['n']; chain_of = np.arange(n) % nch

    def pick(cand, curve, base):
        j = int(np.argmin(curve)); raw = 1e9 if j == len(cand) - 1 else float(cand[j])
        # smoothed: evaluate curve on a grid, convolve with a Gaussian in T
        grid = np.linspace(-25., 5., 601)
        cg = np.interp(grid, cand, curve, left=curve[0], right=curve[-1])
        dx = grid[1] - grid[0]; kx = np.arange(-int(4 * sigma / dx), int(4 * sigma / dx) + 1) * dx
        ker = np.exp(-0.5 * (kx / sigma) ** 2); ker /= ker.sum()
        pad = len(kx) // 2
        cp = np.r_[np.full(pad, cg[0]), cg, np.full(pad, cg[-1])]
        cs = np.convolve(cp, ker, 'valid')
        js = int(np.argmin(cs)); sm = float(grid[js])
        if cs[js] >= base - 1e-15: sm = 1e9
        return raw, sm, float(curve[j] - base), float(cs[js] - base) if sm < 1e8 else 0.0, grid, cs

    raw, sm, g_raw, g_sm, grid, cs = pick(cand, curve, base)
    out = dict(T_raw=raw, T_smooth=sm, gain_raw=g_raw, gain_smooth=g_sm, sigma=sigma, n=int(n),
               flip_frac_raw=float(np.mean(E['rx_node'] > raw)), flip_frac_smooth=float(np.mean(E['rx_node'] > sm)))
    for h in (0, 1):
        m = (chain_of % 2) == h
        b2, c2, cu2 = C.energy_curve(E, m); r2, s2, *_ = pick(c2, cu2, b2)
        out[f'T_raw_half{h}'] = r2; out[f'T_smooth_half{h}'] = s2
    rg = np.random.default_rng(seed); br, bs = [], []
    for b in range(nboot):
        cw = np.bincount(rg.integers(0, nch, nch), minlength=nch).astype(float)
        b2, c2, cu2 = energy_curve_w(E, cw[chain_of]); r2, s2, *_ = pick(c2, cu2, b2)
        br.append(r2); bs.append(s2)
    br = np.array(br); bs = np.array(bs)
    q = lambda a: [float(x) for x in np.quantile(a, [0.16, 0.5, 0.84])]
    out['boot_raw_q16_50_84'] = q(br); out['boot_smooth_q16_50_84'] = q(bs)
    out['boot_raw_noflip_frac'] = float(np.mean(br > 1e8)); out['boot_smooth_noflip_frac'] = float(np.mean(bs > 1e8))
    out['curve_grid'] = grid.tolist()[::5]; out['curve_smooth_site'] = (cs / 36).tolist()[::5]
    out['E_noflip_site'] = float(base / 36)
    return out


def threshold_edges_guide(X, amp_new, sign_fn, chunk=512):
    parts = [S.threshold_edges_stored(X[i:i + chunk], amp_new, sign_fn) for i in range(0, len(X), chunk)]
    return C.merge_edges(parts)


# ------------------------------------------------------------------ ViT reference guide (|psi|, binarised phase)
class VitGuide:
    """guide a = |psi_ViT|, s = sgn cos(Im log psi - phi) (phi = global phase fixed on a sample set)."""

    def __init__(self, net, phi):
        self.net = net; self.phi = phi; self.D = 0

    def local(self, X):
        X = np.asarray(X, np.uint64)
        lv, nb = C.build_levels(X, 1)
        z = self.net.logpsi_c(self.net.flat0, lv[1])
        la1 = z.real; s1 = np.where(np.cos(z.imag - self.phi) >= 0, 1., -1.)
        idx, V, selfidx = nb[0]
        p0 = np.searchsorted(lv[0], X)
        idx = idx[p0]; V = V[p0]; sx = s1[selfidx[p0]]; lax_ = la1[selfidx[p0]]
        sy = s1[idx]; w = 0.5 * C.JB[None, :] * np.exp(la1[idx] - lax_[:, None])
        allowed = V & (sx[:, None] * sy < 0); viol = V & ~allowed
        diag = C.diag_vec(V)
        dfn = diag + np.sum(np.where(viol, w, 0.), axis=1)
        eh = dfn - np.sum(np.where(allowed, w, 0.), axis=1)
        own, col = np.nonzero(V); NBx = C.neighbors(X)[0]
        return dict(dfn=dfn, eh=eh, diag=diag, own=own.astype(np.int64), child=NBx[own, col], rate=w[own, col],
                    allowed=allowed[own, col], lac=la1[idx[own, col]], lax=lax_, s=sx, n=len(X))


def vit_phase(net, X):
    z = net.logpsi_c(net.flat0, np.asarray(X, np.uint64))
    return 0.5 * float(np.angle(np.mean(np.exp(2j * z.imag))))


# ------------------------------------------------------------------ oracle / mixed guides (diagnostics)
class FuncGuide:
    """guide from a log-amplitude function la(S) and a sign function sg(S) (any state sets)."""

    def __init__(self, la, sg):
        self.la = la; self.sg = sg; self.D = 0

    def local(self, X):
        X = np.asarray(X, np.uint64)
        lv, nb = C.build_levels(X, 1)
        la1 = self.la(lv[1]); s1 = self.sg(lv[1])
        idx, V, selfidx = nb[0]
        p0 = np.searchsorted(lv[0], X)
        idx = idx[p0]; V = V[p0]; sx = s1[selfidx[p0]]; lax_ = la1[selfidx[p0]]
        sy = s1[idx]; w = 0.5 * C.JB[None, :] * np.exp(np.clip(la1[idx] - lax_[:, None], -700, 700))
        allowed = V & (sx[:, None] * sy < 0); viol = V & ~allowed
        diag = C.diag_vec(V)
        dfn = diag + np.sum(np.where(viol, w, 0.), axis=1)
        eh = dfn - np.sum(np.where(allowed, w, 0.), axis=1)
        own, col = np.nonzero(V); NBx = C.neighbors(X)[0]
        return dict(dfn=dfn, eh=eh, diag=diag, own=own.astype(np.int64), child=NBx[own, col], rate=w[own, col],
                    allowed=allowed[own, col], lac=la1[idx[own, col]], lax=lax_, s=sx, n=len(X))


def make_mixed_guide(desc, net, pool):
    """desc = 'AMP:SIGN' with AMP in {vit, psi0, <params.npy>} and SIGN in {vit, psi0, <spec.json>}."""
    amp_d, sign_d = desc.split(':')
    p0 = None
    def psi0():
        nonlocal p0
        if p0 is None:
            from psi0_6x6 import Psi0
            p0 = FnCache(Psi0('psi0_6x6_table.npz'), name='psi0')
        return p0
    if amp_d == 'vit': la = FnCache(lambda S: net.logabs(net.flat0, S), name='vit_a')
    elif amp_d == 'psi0': la = lambda S: np.log(np.maximum(np.abs(psi0()(S)), 1e-300))
    else: la = FnCache(lambda S, fl=load_flat(net, amp_d): net.logabs(fl, S), name=amp_d)
    if sign_d == 'vit':
        phi = vit_phase(net, pool)
        sg = FnCache(lambda S: np.where(np.cos(net.logpsi_c(net.flat0, S).imag - phi) >= 0, 1., -1.), name='vit_s')
    elif sign_d == 'psi0': sg = lambda S: np.where(psi0()(S) >= 0, 1., -1.)
    else:
        sp, bd = load_spec(sign_d); hg = HGuide(sp, net, base_dir=bd); sg = FnCache(hg.signs, name='hg_s')
    return FuncGuide(la, sg)
