"""Stored Krylov signs on 6x6 J1-J2 (J2/J1 = 0.5) at fixed amplitude a = |psi_ViT|.

Reuses ll6_core (experiments/learned_loop_6x6, imported unchanged; copied next to the run on the cluster).
Stored sign after k steps:  s_hat_k(x) = Marshall(x) * prod_{j=1..k} sgn(logit_j(x)),
logit_j = periodic CNN (translation invariant by sum pooling, spin-flip symmetrised, optional D4 average),
trained on labels c_j(x) = sgn[T_{j-1} - r_{j-1}(x)],  r_{j-1}(x) = (H a s_hat_{j-1})(x) / (a(x) s_hat_{j-1}(x))
(one hop: needs a and s_hat_{j-1} on the neighbours of x only).
All CNN arithmetic float32 (jax_enable_x64 is on globally via ll6_core, so dtypes are explicit).
"""
import time, math, json, os, pickle
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import ll6_core as C

N = 36; L = 6
F32 = jnp.float32
CKPT = 'vit_J2=0.50_N=6x6_k=0.mpack'
log = C.log


# ------------------------------------------------------------------ fixed-amplitude cache
class AmpCache:
    """log|psi_ViT| with a sorted main table + pending buffer (amplitude is fixed for the whole study)."""

    def __init__(self, net, cap=40_000_000):
        self.net = net; self.flat = net.flat0; self.cap = cap
        self.k = np.zeros(0, np.uint64); self.v = np.zeros(0)
        self.bk = []; self.bv = []; self.nb = 0
        self.nnew = 0; self.nreq = 0

    def _merge(self):
        if not self.bk: return
        k = np.concatenate([self.k] + self.bk); v = np.concatenate([self.v] + self.bv)
        o = np.argsort(k, kind='stable'); k = k[o]; v = v[o]
        keep = np.r_[True, k[1:] != k[:-1]]
        self.k, self.v = k[keep], v[keep]; self.bk, self.bv, self.nb = [], [], 0

    def _lookup(self, keys, vals, u):
        if len(keys) == 0: return np.zeros(len(u), bool), np.zeros(len(u))
        p = np.minimum(np.searchsorted(keys, u), len(keys) - 1)
        hit = keys[p] == u
        return hit, np.where(hit, vals[p], 0.)

    def __call__(self, S):
        S = np.asarray(S, np.uint64); self.nreq += len(S)
        u, inv = np.unique(S, return_inverse=True)
        hit, val = self._lookup(self.k, self.v, u)
        if self.bk and (~hit).any():
            bk = np.concatenate(self.bk); bv = np.concatenate(self.bv); o = np.argsort(bk)
            h2, v2 = self._lookup(bk[o], bv[o], u[~hit])
            idx = np.nonzero(~hit)[0]; val[idx[h2]] = v2[h2]; hit[idx[h2]] = True
        miss = u[~hit]
        if len(miss):
            vm = self.net.logabs(self.flat, miss); self.nnew += len(miss)
            val[~hit] = vm
            if len(self.k) < self.cap:          # bounded cache: read-only once the table is full
                self.bk.append(miss); self.bv.append(vm); self.nb += len(miss)
                if self.nb > 4_000_000: self._merge()
        return val[inv]


# ------------------------------------------------------------------ tempered sampler |a|^(2 beta)
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


def sample_tempered(net, beta, nchains, nrounds, between, burn, seed):
    """returns states (nrounds, nchains) uint64 and acceptance."""
    run = make_tempered_sampler(net, float(beta))
    rg = np.random.default_rng(seed)
    X = -np.ones((nchains, N))
    for k in range(nchains): X[k, rg.choice(N, N // 2, replace=False)] = 1
    X = jnp.asarray(X, net.DT); key = jax.random.PRNGKey(seed); flat = net.flat0
    la = net.f_spins(flat, X); acc = 0; props = 0
    X, la, key, a = run(flat, X, la, key, int(burn * N)); acc += int(a); props += burn * N * nchains
    out = []
    for r in range(nrounds):
        X, la, key, a = run(flat, X, la, key, int(between * N)); acc += int(a); props += between * N * nchains
        out.append(C.spins_to_bits_np(np.asarray(X)))
    return np.stack(out), acc / props


# ------------------------------------------------------------------ symmetry helpers
def d4_perms():
    q = lambda x, y: (x % L) + L * (y % L)
    def pm(f):
        p = np.empty(N, np.int64)
        for y in range(L):
            for x in range(L):
                p[q(x, y)] = q(*f(x, y))
        return p
    r = np.arange(N); c4 = pm(lambda x, y: (-y, x)); sx = pm(lambda x, y: (-x, y)); out = []
    for k in range(4):
        for refl in (False, True):
            p = r.copy()
            for _ in range(k): p = c4[p]
            if refl: p = sx[p]
            out.append(p)
    return np.array(out)


D4 = d4_perms()


def permute_bits(S, p):
    """bit i of S moves to site p[i]."""
    S = np.asarray(S, np.uint64); out = np.zeros_like(S)
    for i in range(N):
        out |= ((S >> np.uint64(i)) & C.ONE) << np.uint64(p[i])
    return out


# ------------------------------------------------------------------ sign network
class SignCNN(nn.Module):
    ch: int = 32
    depth: int = 4
    k: int = 3

    @nn.compact
    def __call__(self, x):                   # x: (B, 36) float32 +-1, site i = x + 6 y
        h = x.reshape(-1, L, L, 1)           # [b, y, x, c]
        p = self.k // 2
        def conv(h, c, name):
            h = jnp.pad(h, ((0, 0), (p, p), (p, p), (0, 0)), mode='wrap')
            return nn.Conv(c, (self.k, self.k), padding='VALID', dtype=F32, param_dtype=F32, name=name)(h)
        h = nn.gelu(conv(h, self.ch, 'c0'))
        for d in range(1, self.depth):
            h = h + nn.gelu(conv(nn.LayerNorm(dtype=F32, param_dtype=F32, name=f'ln{d}')(h), self.ch, f'c{d}'))
        o = nn.Dense(2, dtype=F32, param_dtype=F32, name='head')(h)      # per-site (logit, aux)
        o = o.sum(axis=(1, 2)) / 6.0
        return o[:, 0], o[:, 1]


def bits_to_f32(S):
    S = jnp.asarray(S, jnp.uint64)
    return (((S[:, None] >> jnp.arange(N, dtype=jnp.uint64)[None, :]) & jnp.uint64(1)).astype(F32) * 2 - 1)


class SignNet:
    """logit(x); translation invariant; optional average over D4 (d4=True) and spin flip (flip=True)."""

    def __init__(self, ch, depth, d4=False, flip=False, seed=0):
        # NOTE: |psi_ViT| is translation invariant but NOT D4 / spin-flip invariant (prep pilot: c1 changes on
        # 1.4-2.3% of beta=0.5 states under D4 or flip), so the default is translation invariance only.
        self.cfg = dict(ch=ch, depth=depth, d4=d4, flip=flip)
        self.m = SignCNN(ch=ch, depth=depth)
        self.params = self.m.init(jax.random.PRNGKey(seed), jnp.zeros((1, N), F32))['params']
        self.npar = int(sum(np.prod(p.shape) for p in jax.tree_util.tree_leaves(self.params)))
        perms = D4 if d4 else D4[:1]
        self.inv = jnp.asarray(np.argsort(perms, axis=1))   # x'[j] = x[inv[j]]  (site i -> p[i])
        m = self.m; inv = self.inv

        def apply(params, X):
            Xs = [X[:, inv[g]] for g in range(inv.shape[0])]
            if flip: Xs = Xs + [-x for x in Xs]
            Xs = jnp.concatenate(Xs, 0) if len(Xs) > 1 else Xs[0]
            lo, au = m.apply({'params': params}, Xs)
            G = (2 if flip else 1) * inv.shape[0]
            return lo.reshape(G, -1).mean(0), au.reshape(G, -1).mean(0)
        self.apply = apply
        self._lo = jax.jit(lambda p, S: apply(p, bits_to_f32(S))[0])
        self.nevals = 0

    def logit(self, S, B=65536):
        S = np.asarray(S, np.uint64); out = np.empty(len(S), np.float32)
        for i in range(0, len(S), B):
            s = S[i:i + B]; m = len(s)
            if m < B: s = np.concatenate([s, np.repeat(s[:1], B - m)]) if m else s
            out[i:i + m] = np.asarray(self._lo(self.params, jnp.asarray(s)))[:m]
        self.nevals += len(S)
        return out

    def save(self, path):
        pickle.dump(dict(cfg=self.cfg, params=jax.device_get(self.params)), open(path, 'wb'))

    @staticmethod
    def load(path):
        d = pickle.load(open(path, 'rb'))
        n = SignNet(**d['cfg']); n.params = jax.tree_util.tree_map(jnp.asarray, d['params']); return n


class StoredChain:
    """s_hat_k = Marshall * prod_j sgn(logit_j); thresholds Ts[j] used to build net j."""

    def __init__(self, nets=(), Ts=()):
        self.nets = list(nets); self.Ts = list(Ts)

    @property
    def K(self): return len(self.nets)

    def signs(self, S, upto=None):
        S = np.asarray(S, np.uint64); s = C.marshall_vec(S)
        for n in self.nets[:self.K if upto is None else upto]:
            s = s * np.where(n.logit(S) >= 0, 1., -1.)
        return s


# ------------------------------------------------------------------ one-hop quantities with a stored sign
def r_onehop(X, amp, signfn):
    """r(x) = (H a s)(x)/(a(x)s(x)) for central X (any order). signfn(S)->+-1. Returns r, s(X)."""
    X = np.asarray(X, np.uint64)
    lv, nb = C.build_levels(X, 1)
    la = amp(lv[1]); s = signfn(lv[1])
    r, s_self = C.r_on_level(lv, nb, 0, la, s)
    p = np.searchsorted(lv[0], X)
    return r[p], s_self[p]


def r_onehop_chunked(X, amp, signfn, chunk=4096):
    r = np.empty(len(X)); s = np.empty(len(X))
    for i in range(0, len(X), chunk):
        r[i:i + chunk], s[i:i + chunk] = r_onehop(X[i:i + chunk], amp, signfn)
    return r, s


def threshold_edges_stored(X, amp, signfn):
    """Edges for the energy curve E(T) of a * s * sgn(T - r), r from (a, s) with s a stored sign.
    Needs a and s on the 2-hop shell of X (s stored: one net evaluation per state)."""
    X = np.asarray(X, np.uint64)
    lv, nb = C.build_levels(X, 2)
    la2 = amp(lv[2]); s2 = signfn(lv[2])
    r1, s1 = C.r_on_level(lv, nb, 1, la2, s2)
    la1 = la2[nb[1][2]]
    idx, V, selfidx = nb[0]
    p0 = np.searchsorted(lv[0], X)
    idx = idx[p0]; V = V[p0]; sp = selfidx[p0]
    own, col = np.nonzero(V); y = idx[own, col]
    c = 0.5 * C.JB[col] * np.exp(la1[y] - la1[sp[own]]) * s1[y] * s1[sp[own]]
    return dict(xi=own, c=c, rx=r1[sp][own], ry=r1[y], rx_node=r1[sp], diag=C.diag_vec(V), n=len(X), sx=s1[sp])


# ------------------------------------------------------------------ defect-attributed energy vs ViT
def defect_energy(X, nch, amp, net, signfns, chunk=2048):
    """Delta E = <H>_{a s} - <H>_{ViT} for several sign functions on x ~ |a|^2 (states X of shape (nr, nch)
    flattened row-major).  Unordered edge (x,y) attributed to the larger-|a| endpoint (guide_vmc_sym.py):
      Delta = E_x[ -2 sum_{y~x, a_y<a_x, d_y != d_x} J (a_y/a_x) h_x h_y ],  d = s*h, h = binarised ViT sign.
    signfns: dict name -> f(S) giving +-1 (evaluated on X and on its smaller neighbours only)."""
    X = np.asarray(X, np.uint64).reshape(-1)
    zx = net.logpsi_c(net.flat0, X)
    phi = 0.5 * float(np.angle(np.mean(np.exp(2j * zx.imag))))
    res = {}
    dE = {k: np.zeros(len(X)) for k in signfns}; Dfrac = {}
    hx_all = np.where(np.cos(zx.imag - phi) >= 0, 1., -1.)
    for i in range(0, len(X), chunk):
        Xc = X[i:i + chunk]; hx = hx_all[i:i + chunk]; lx = zx.real[i:i + chunk]
        NB, V = C.neighbors(Xc)
        own, col = np.nonzero(V); Y = NB[own, col]
        ly = amp(Y)
        sm = ly < lx[own]
        own, col, Y, ly = own[sm], col[sm], Y[sm], ly[sm]
        uy, iy = np.unique(Y, return_inverse=True)
        zy = net.logpsi_c(net.flat0, uy)
        hy = np.where(np.cos(zy.imag - phi) >= 0, 1., -1.)[iy]
        w = -2.0 * C.JB[col] * np.exp(ly - lx[own]) * hx[own] * hy
        for k, f in signfns.items():
            sx = f(Xc); sy = f(uy)[iy]
            dx = sx * hx; dy = sy * hy
            # global orientation of the guide vs ViT is irrelevant for d_x != d_y
            dE[k][i:i + chunk] = np.bincount(own, weights=np.where(dx[own] != dy, w, 0.), minlength=len(Xc))
            Dfrac.setdefault(k, []).append(dx)
    for k in signfns:
        dx = np.concatenate(Dfrac[k]); g = 1. if dx.mean() >= 0 else -1.
        V = dE[k].reshape(-1, nch); cm = V.mean(0)
        res[k] = dict(dE_site=float(V.mean() / N), se_site=float(cm.std(ddof=1) / np.sqrt(nch) / N),
                      defect_frac=float(np.mean(g * dx < 0)))
    res['_dE_arrays'] = dE
    res['_phi'] = phi
    return res


def paired_diff(dEa, dEb, nch):
    d = (dEa - dEb).reshape(-1, nch); cm = d.mean(0)
    return float(d.mean() / N), float(cm.std(ddof=1) / np.sqrt(nch) / N)


# ------------------------------------------------------------------ ED scoring
def load_ed(nmax=400000):
    X = []; P = []
    for f in ('samples_psi0sq_6x6.npz', 'samples_extra_psi0sq_6x6.npz'):
        d = np.load(f); X.append(d['x'].astype(np.uint64)); P.append(d['psi'])
    return np.concatenate(X)[:nmax], np.sign(np.concatenate(P)[:nmax])


def wrong_frac(s, strue):
    m = float(np.mean(s != strue)); g = 1 if m <= 0.5 else -1; w = min(m, 1 - m)
    return dict(w=w, err=float(np.sqrt(max(w, 1e-12) * (1 - w) / len(s))), nwrong=int(round(w * len(s))), g=g)


def paired_wrong(sa, sb, strue, ga, gb):
    wa = (sa != ga * strue).astype(float); wb = (sb != gb * strue).astype(float); d = wa - wb
    return dict(mean=float(d.mean()), err=float(d.std(ddof=1) / np.sqrt(len(d))), n_differ=int(np.sum(sa * ga != sb * gb)))


def disagree(sa, sb):
    """fraction where two sign functions differ (global sign free)."""
    m = float(np.mean(sa != sb)); return min(m, 1 - m)


# ------------------------------------------------------------------ training
def train_net(snet, S, y, aux, Sv, yv, epochs, lr, batch, wd=1e-4, lam=0.1, seed=0, logevery=1):
    """BCE on y in {+1,-1} (logit>0 <-> +1) + lam * MSE(aux_head, aux). Returns history."""
    params = snet.params
    nsteps = epochs * (len(S) // batch)
    sched = optax.warmup_cosine_decay_schedule(0., lr, max(1, nsteps // 50), max(2, nsteps))
    opt = optax.adamw(sched, weight_decay=wd); st = opt.init(params)
    apply = snet.apply

    def loss_fn(p, Sb, yb, ab):
        lo, au = apply(p, bits_to_f32(Sb))
        bce = jnp.mean(optax.sigmoid_binary_cross_entropy(lo, (yb > 0).astype(F32)))
        return bce + lam * jnp.mean((au - ab) ** 2), bce

    @jax.jit
    def step(p, st, Sb, yb, ab):
        (l, b), g = jax.value_and_grad(loss_fn, has_aux=True)(p, Sb, yb, ab)
        u, st = opt.update(g, st, p); return optax.apply_updates(p, u), st, l, b

    rg = np.random.default_rng(seed); hist = []
    yj = jnp.asarray(y.astype(np.float32)); aj = jnp.asarray(aux.astype(np.float32)); Sj = jnp.asarray(S)
    t0 = time.time()
    for ep in range(epochs):
        perm = jnp.asarray(rg.permutation(len(S)))
        ls = []
        for b in range(len(S) // batch):
            ib = perm[b * batch:(b + 1) * batch]
            params, st, l, bc = step(params, st, Sj[ib], yj[ib], aj[ib]); ls.append(bc)
        snet.params = params
        if (ep + 1) % logevery == 0 or ep == epochs - 1:
            lv = snet.logit(Sv); err = float(np.mean(np.where(lv >= 0, 1, -1) != yv))
            h = dict(ep=ep + 1, bce=float(jnp.mean(jnp.stack(ls))), val_err=err,
                     val_err_flip=float(np.mean(np.where(lv[yv < 0] >= 0, 1, -1) != -1)) if (yv < 0).any() else None,
                     sec=time.time() - t0)
            hist.append(h); log('TRAIN', json.dumps(h))
    return hist


# ------------------------------------------------------------------ training states and precomputed K1 label shards
def build_train_states(Xs, nbr=1):
    """samples + all one-hop neighbours, ordered by originating sample (chunked one-hop shells then overlap)."""
    if not nbr: return np.unique(Xs)
    Xu = np.unique(Xs); rg_ = np.random.default_rng(0); Xu = Xu[rg_.permutation(len(Xu))]
    NB, V = C.neighbors(Xu)
    blk = np.concatenate([Xu[:, None], np.where(V, NB, Xu[:, None])], axis=1).reshape(-1)
    _, first = np.unique(blk, return_index=True)
    return blk[np.sort(first)]


def label_shard_name(beta, K, i, n):
    return f'labels_k{K}_b{beta}_s{i}of{n}.npz'


def load_label_shards(beta, K, U):
    import glob
    fs = sorted(glob.glob(f'labels_k{K}_b{beta}_s*of*.npz'))
    if not fs: return None
    n = int(fs[0].split('of')[-1].split('.')[0])
    parts = [np.load(label_shard_name(beta, K, i, n)) for i in range(n)] if len(fs) == n else None
    if parts is None: return None
    Uc = np.concatenate([p['U'] for p in parts]); r = np.concatenate([p['r'] for p in parts])
    if len(Uc) != len(U) or not np.array_equal(Uc, U): return None
    log('LABELS loaded from', n, 'CPU shards')
    return r
