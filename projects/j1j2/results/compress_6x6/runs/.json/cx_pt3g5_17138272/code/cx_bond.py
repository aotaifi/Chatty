"""Per-bond base information (README round 3): a student that sees, besides the spins and psi_P's one-hop scalars
(log a_P, V_P, W_P), psi_P's 144 per-bond log-ratios r_b(x) = log a_P(x^b) - log a_P(x) and the current sign's
kept/violating flag of every bond, through a bond-equivariant layer (site CNN -> bond messages from (h_i, h_j, r_b,
kept_b, valid_b, bond type) -> summed back onto the two sites -> site CNN -> pooled readout). Cost: one hop of psi_P
(the same evaluations as V_P, W_P) plus one hop of the sign.  Exactly D4 x flip symmetric at evaluation (16 images,
bond features permuted with the sites), translation-equivariant by construction.
"""
import numpy as np
from cx_lib import *
import optax

_pairs = [(int(a), int(b)) for a, b in zip(SS.BI, SS.BJ)]
_idx = {frozenset(p): i for i, p in enumerate(_pairs)}
_inv8 = np.argsort(SS.space_group()[:8], axis=1)
BPERM16 = jnp.asarray(np.array([[_idx[frozenset((int(_inv8[k // 2][a]), int(_inv8[k // 2][b])))] for a, b in _pairs]
                                for k in range(16)], np.int32))
INC = np.zeros((NB, N), np.float32)
for b_, (a, c) in enumerate(_pairs): INC[b_, a] += 1; INC[b_, c] += 1
INC = jnp.asarray(INC)


def rfeat(lab, x, s, la_base=None):
    """x (B,) uint64 configurations (any orientation) -> valid (B,NB), r (B,NB) base log-ratios, kept (B,NB) under s."""
    sec = lab.sec; la_base = lab.lP0 if la_base is None else la_base
    v = valid_bonds(x)
    y = jnp.where(v, x[:, None] ^ MASKS[None, :], x[:, None])
    ix = SS.canon(sec.T, sec.reps, x)
    iy = SS.canon(sec.T, sec.reps, y.reshape(-1)).reshape(x.shape[0], NB)
    r = jnp.where(v, jnp.clip(la_base[iy] - la_base[ix][:, None], -30, 30), 0.0).astype(jnp.float32)
    kept = v & (s[ix][:, None] * s[iy] < 0)
    return v, r, kept


class BondNet(nn.Module):
    C: int = 32
    hid: int = 64
    nout: int = 1
    zero_head: bool = True

    @nn.compact
    def __call__(self, X, r, v, kept, t):
        kw = dict(dtype=jnp.float32, param_dtype=jnp.float32)
        Bn = X.shape[0]

        def conv(h):
            g = h[:, NBR, :]
            return nn.Dense(self.C, **kw)(g.reshape(Bn, N, -1))
        h = nn.gelu(conv(X[:, :, None]))
        for _ in range(2):
            h = h + nn.gelu(conv(nn.LayerNorm(**kw)(h)))
        hi, hj = h[:, BI_I, :], h[:, BJ_I, :]
        vf = v.astype(jnp.float32); kf = kept.astype(jnp.float32)
        temb = nn.Embed(4, 8, param_dtype=jnp.float32)(BTYPE)[None].repeat(Bn, 0)
        be = jnp.stack([0.5 * r, jnp.tanh(r), kf, vf], -1)
        z = jnp.concatenate([hi + hj, hi * hj, be, temb], -1)
        z = nn.gelu(nn.Dense(self.C, **kw)(z))
        z = nn.Dense(self.C, **kw)(z) * vf[..., None]
        m = jnp.einsum('bec,es->bsc', z, INC)
        h = h + nn.gelu(nn.Dense(self.C, **kw)(jnp.concatenate([nn.LayerNorm(**kw)(h), m], -1)))
        h = h + nn.gelu(conv(nn.LayerNorm(**kw)(h)))
        hp = h.sum(axis=1) / 6.0
        om = jnp.asarray([0.25, 0.5, 1.0, 2.0, 4.0], jnp.float32)
        emb = jnp.concatenate([t, 0.1 * t * t, jnp.sin(t[:, :, None] * om).reshape(Bn, -1),
                               jnp.cos(t[:, :, None] * om).reshape(Bn, -1)], 1)
        hp = jnp.concatenate([hp, emb], 1)
        hp = nn.gelu(nn.Dense(self.hid, **kw)(hp)); hp = nn.gelu(nn.Dense(self.hid, **kw)(hp))
        init = nn.initializers.zeros if self.zero_head else nn.initializers.lecun_normal()
        return nn.Dense(self.nout, kernel_init=init, **kw)(hp)


class BondFeatNet:
    """f(x) from spins + base bond log-ratios + kept flags (sign `s`) + scalar features `tfeat` (D, nf) by rep index."""

    def __init__(self, lab, tfeat, s, seed=0, nout=1, zero_head=True, C=32):
        self.lab, self.tfeat, self.s = lab, tfeat, s
        self.net = BondNet(C=C, nout=nout, zero_head=zero_head)
        nf = int(tfeat.shape[1])
        p = self.net.init(jax.random.PRNGKey(seed), jnp.zeros((1, N), jnp.float32), jnp.zeros((1, NB), jnp.float32),
                          jnp.zeros((1, NB), bool), jnp.zeros((1, NB), bool), jnp.zeros((1, nf), jnp.float32))['params']
        self.flat0, self.unravel = ravel_pytree(p)
        self.npar = int(self.flat0.size)
        net, unravel = self.net, self.unravel

        def img(X, r, v, kp, k):
            P = BPERM16[k]
            return (X[:, INV8[k // 2]] * SG2[k % 2], r[:, P], v[:, P], kp[:, P])

        def f_aug(fl, X, r, v, kp, t, G):                     # one image per row
            Xg = jnp.take_along_axis(X, INV8[G // 2], axis=1) * SG2[G % 2][:, None]
            P = BPERM16[G]
            return net.apply({'params': unravel(fl)}, Xg, jnp.take_along_axis(r, P, 1), jnp.take_along_axis(v, P, 1),
                             jnp.take_along_axis(kp, P, 1), t)

        def f_sym(fl, X, r, v, kp, t):
            return jax.lax.map(lambda k: net.apply({'params': unravel(fl)}, *img(X, r, v, kp, k), t), jnp.arange(16)).mean(0)
        self.f_aug, self.f_sym = f_aug, f_sym

        @jax.jit
        def _tab(fl, ii):
            x = lab.sec.reps[ii]
            v, r, kp = rfeat(lab, x, s)
            return f_sym(fl, bits(x), r, v, kp, tfeat[ii])
        self._tab = _tab

    def table(self, fl, chunk=8192):
        D = self.lab.sec.D; out = []
        for i in range(0, D, chunk):
            ii = jnp.minimum(jnp.arange(i, i + chunk), D - 1)
            out.append(self._tab(fl, ii)[:min(chunk, D - i)].astype(f64))
        return jnp.concatenate(out)


def fit_edge_bond(lab, model, la, s, CDF, LIW, DLR, steps, seed, B=256, K=8, lr=1e-3, tag=''):
    """fit_edge with the bond-input student (features of x and of the sampled neighbours y computed on the fly)."""
    sec = lab.sec; ts = model.tfeat; sf = model.s
    sched = optax.warmup_cosine_decay_schedule(0.0, lr, min(200, max(1, steps // 2)), steps, lr * 0.02)
    opt = optax.adam(sched); flat = model.flat0; ost = opt.init(flat)

    @jax.jit
    def batch(key):
        k1, k2 = jax.random.split(key)
        idx = jnp.clip(jnp.searchsorted(CDF, jax.random.uniform(k1, (B,), f64) * CDF[-1]), 0, sec.D - 1)
        x = sec.reps[idx]
        valid = valid_bonds(x); nval = valid.sum(1)
        sc = jnp.where(valid, jax.random.uniform(k2, valid.shape), -1.0)
        _, bsel = jax.lax.top_k(sc, K)
        ok = jnp.take_along_axis(valid, bsel, 1)
        y = x[:, None] ^ MASKS[bsel]
        iy = SS.canon(sec.T, sec.reps, y.reshape(-1)).reshape(B, K)
        rP = jnp.exp(jnp.clip(la[iy] - la[idx][:, None], -60, 60))
        wgt = jnp.where(ok & (s[idx][:, None] * s[iy] < 0), 0.5 * JB[bsel] * rP, 0.0) * (nval[:, None] / K)
        lw = LIW[idx]; iw = jnp.exp(lw - jnp.max(lw))
        xa = jnp.concatenate([x, y.reshape(-1)]); ia = jnp.concatenate([idx, iy.reshape(-1)])
        v, r, kp = rfeat(lab, xa, sf)
        return xa, v, r, kp, ts[ia], DLR[idx], DLR[iy], wgt, iw

    def loss(q, xa, v, r, kp, t, dlx, dly, wgt, iw, G):
        iw = iw / jnp.mean(iw)
        fa = model.f_aug(q, bits(xa), r, v, kp, t, jnp.concatenate([G, jnp.repeat(G, K)]))[:, 0].astype(f64)
        fx, fy = fa[:B], fa[B:].reshape(B, K)
        return 0.5 * jnp.mean(iw * jnp.sum(wgt * ((fx + dlx)[:, None] - (fy + dly)) ** 2, 1)) / N

    @jax.jit
    def step(fl, ost, key):
        k1, k2 = jax.random.split(key)
        bd = batch(k1); G = jax.random.randint(k2, (B,), 0, 16)
        l, g = jax.value_and_grad(loss)(fl, *bd, G)
        upd, ost = opt.update(g, ost, fl)
        return optax.apply_updates(fl, upd), ost, l
    vk = [jax.random.PRNGKey(777 + i) for i in range(16)]
    vfn = jax.jit(lambda fl, key: loss(fl, *batch(key), jax.random.randint(key, (B,), 0, 16)))
    curve = []; key = jax.random.PRNGKey(100 + seed)
    for it in range(1, steps + 1):
        key, kk = jax.random.split(key)
        flat, ost, l = step(flat, ost, kk)
        if it % 2000 == 0 or it == 1:
            curve.append((it, float(np.mean([float(vfn(flat, q)) for q in vk]))))
            log(f'  [{tag}] step {it} own val {curve[-1][1]:.4e}')
    return flat, curve
