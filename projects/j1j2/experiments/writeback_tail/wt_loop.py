"""Step 2 loop: FN/Krylov loop with the FEAT residual written back at every iteration, realistic data, exact 6x6 referee.

Iteration k, guide (la_k, s_k) (exact tables):
  guide-only quantities: V_k, W_k (one hop), D = H_xx + V_k, E_k = frozen-FN energy of the guide (scalar),
  semi-implicit FN step T_k = log(1 + W_k) - log(1 + (D - E_k)); proposal 1/2 a^1 + 1/2 guide-metric node weight of T_k;
  fit b = a_k exp(f), f = FEAT CNN on (log a_k, V_k, W_k) of the CURRENT guide, edge-difference least squares vs T_k,
  Adam 20k steps x 256 configurations x 8 bonds (exactly arm F-SI-Adam);
  exact referee: frac_k of the ideal FN iteration from guide k, <H>, Krylov sign s_{k+1}, E_FN(b, s_{k+1}).
No phi_FN enters training. In a real loop the current-guide features need the guide on the one-hop neighbourhood,
i.e. the stored net of iteration k-1 there (recursion; see README).
  python wt_loop.py OUT SPEC.json
"""
import json, os, sys, time
import numpy as np
from wt_common import log, dump, N, f64, SS, CSR, TABLE, SYM
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from jax.flatten_util import ravel_pytree
jax.config.update('jax_default_matmul_precision', 'highest')

OUT = sys.argv[1]; SPEC = json.load(open(sys.argv[2])); os.makedirs(OUT, exist_ok=True)
F = os.path.join(OUT, 'wt_loop.json'); T00 = time.time()
AR = jnp.arange(N, dtype=jnp.uint64)


def bits(Sx):
    return (((Sx[:, None] >> AR[None, :]) & jnp.uint64(1)).astype(jnp.float32) * 2 - 1)


NBR = jnp.asarray(np.array([[((x + dx) % 6) + 6 * ((y + dy) % 6) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
                            for y in range(6) for x in range(6)], np.int32))


class ResCNN(nn.Module):
    """writeback net: periodic residual CNN, site-summed features, zero-initialised head.
    use_a: concatenate a Fourier embedding of the standardised base log-amplitude t and use a 2-layer MLP head."""
    C: int = 32
    layers: int = 4
    use_a: bool = False
    hid: int = 64
    nfeat: int = 1

    @nn.compact
    def __call__(self, x, t):
        kw = dict(dtype=jnp.float32, param_dtype=jnp.float32)

        def conv(h):
            g = h[:, NBR, :]
            return nn.Dense(self.C, **kw)(g.reshape(g.shape[0], N, -1))
        h = nn.gelu(conv(x[:, :, None]))
        for _ in range(1, self.layers):
            h = h + nn.gelu(conv(nn.LayerNorm(**kw)(h)))
        h = h.sum(axis=1) / 6.0
        if self.use_a:
            om = jnp.asarray([0.25, 0.5, 1.0, 2.0, 4.0], jnp.float32)
            tt = t[:, None] if t.ndim == 1 else t
            B_ = tt.shape[0]
            emb = jnp.concatenate([tt, 0.1 * tt * tt, jnp.sin(tt[:, :, None] * om).reshape(B_, -1),
                                   jnp.cos(tt[:, :, None] * om).reshape(B_, -1)], 1)
            h = jnp.concatenate([h, emb], 1)
            h = nn.gelu(nn.Dense(self.hid, **kw)(h))
            h = nn.gelu(nn.Dense(self.hid, **kw)(h))
        return nn.Dense(1, kernel_init=nn.initializers.zeros, **kw)(h)[:, 0]


class Corr:
    def __init__(self, C, layers, use_a=False, seed=0, head_init=0.0, feat=None):
        """feat: optional (D, nf) table of guide-local input features; t then carries the rep index (exact in fp32)."""
        nf = 1 if feat is None else int(feat.shape[1])
        self.net = ResCNN(C, layers, use_a or feat is not None, nfeat=nf)
        params = self.net.init(jax.random.PRNGKey(seed), jnp.zeros((1, N), jnp.float32),
                               jnp.zeros((1,) if feat is None else (1, nf), jnp.float32))['params']
        if head_init:
            last = sorted(params.keys(), key=lambda k: int(k.split('_')[1]) if k.startswith('Dense_') else -1)[-1]
            k = params[last]['kernel']
            params[last]['kernel'] = head_init * jax.random.normal(jax.random.PRNGKey(seed + 7), k.shape, k.dtype)
        self.flat0, self.unravel = ravel_pytree(params)
        self.npar = int(self.flat0.size)
        inv = jnp.asarray(np.argsort(SS.space_group()[:8], axis=1))
        net0, unravel = self.net, self.unravel
        sg = jnp.asarray([1.0, -1.0], jnp.float32)
        if feat is None:
            net = net0
        else:
            class _W:                          # look the guide features up by rep index carried in t
                @staticmethod
                def apply(v, X, t):
                    return net0.apply(v, X, feat[t.astype(jnp.int32)])
            net = _W

        def f(flat, X, t):                    # exactly symmetric (mean over the 16 images; t is invariant)
            p = unravel(flat)
            out = jax.lax.map(lambda k: net.apply({'params': p}, X[:, inv[k // 2]] * sg[k % 2], t), jnp.arange(16))
            return out.mean(0)

        def f_aug(flat, X, t, G):             # one image per row (training)
            p = unravel(flat)
            Xg = jnp.take_along_axis(X, inv[G // 2], axis=1) * sg[G % 2][:, None]
            return net.apply({'params': p}, Xg, t)
        self.f = f; self.f_aug = f_aug
        self._tab = jax.jit(lambda flat, Sx, t: f(flat, bits(Sx), t))
        self._tab1 = jax.jit(lambda flat, Sx, t: net.apply({'params': unravel(flat)}, bits(Sx), t))

    def table(self, flat, reps, T, batch=16384, sym=True):
        D = reps.shape[0]; out = []
        fn = self._tab if sym else self._tab1
        for i in range(0, D, batch):
            s = reps[i:i + batch]; t = T[i:i + batch]; m = s.shape[0]
            if m < batch:
                s = jnp.concatenate([s, jnp.repeat(s[:1], batch - m)]); t = jnp.concatenate([t, jnp.repeat(t[:1], batch - m)])
            out.append(fn(flat, s, t)[:m].astype(f64))
        return jnp.concatenate(out)



sec = SS.Sector(CSR, TABLE); sec.v0 = sec.la0 = sec.s0 = sec.p0 = None
z = np.load(SYM); lP0 = jnp.asarray(z['lP']); sP0 = jnp.asarray(z['sP'].astype(np.float32)); del z
BI = jnp.asarray(SS.BI); BJ = jnp.asarray(SS.BJ); JB = jnp.asarray(SS.JB); MASKS = jnp.asarray(SS.MASKS)
TIDX = jnp.arange(sec.D, dtype=jnp.float32)


def diag(batch=1 << 20):
    @jax.jit
    def f(x):
        valid = (((x[:, None] >> BI[None, :]) ^ (x[:, None] >> BJ[None, :])) & jnp.uint64(1)).astype(f64)
        return 0.25 * JB.sum() - 0.5 * (valid * JB[None, :]).sum(1)
    return jnp.concatenate([f(sec.reps[i:i + batch]) for i in range(0, sec.D, batch)])


D0 = diag()


def Kop(x, s):
    h1, h2 = sec.H2(x, s * x)
    return 0.5 * (h1 - s * h2)


def H_site(la, s):
    return (sec.energy(sec.vec(la, s)) - sec.E0) / N


res = dict(spec=SPEC, iters=[], ideal=dict(E_FN=[1.018e-4, 6.52e-5, 4.87e-5, 3.81e-5], H_kry=[7.84e-5, 5.58e-5, 4.30e-5]),
           rbm_pp=4.47e-5)
la, s = lP0, sP0
for k in range(1, SPEC.get('n_loop', 4) + 1):
    t0 = time.time(); rec = dict(it=k)
    # ---------------- exact referee of the guide (evaluation only)
    Efn, u, info = sec.fn_solve(la, s); rec['guide_E_FN'] = info['dE_FN_site']; rec['guide_H'] = H_site(la, s)
    Eg = sec.fn_rayleigh(la, s, la)
    rec['ideal_gain_site'] = (Eg - Efn) / N
    del u
    # ---------------- guide-only quantities
    lan = la - jnp.max(la)
    w = jnp.maximum(jnp.exp(lan), 1e-300) * sec.sqrt_n
    h1, h2 = sec.H2(w, s * w)
    V = 0.5 * (h1 + s * h2) / w - D0; W = 0.5 * (h1 - s * h2) / w; del h1, h2, w
    T = jnp.log1p(jnp.maximum(W, 0.0)) - jnp.log1p(jnp.maximum(D0 + V - Eg, 1e-12))
    UA = jnp.exp(lan) * sec.sqrt_n; UA = UA / jnp.linalg.norm(UA); KUA = Kop(UA, s); PA = UA * UA
    MUT = float(jnp.sum(PA * T)); e = T - MUT
    K1 = Kop(UA * e, s); CT = e * e * UA * KUA - 2 * e * UA * K1; del K1
    CT = 0.5 * (CT + UA * Kop(UA * e * e, s)); CT = jnp.maximum(CT, 0.0)
    qb = jnp.exp(lan) * sec.n; qb = qb / jnp.sum(qb)
    Q = 0.5 * qb + 0.5 * CT / jnp.sum(CT); del qb
    CDF = jnp.cumsum(Q); LIW = jnp.log(jnp.maximum(PA, 1e-300)) - jnp.log(jnp.maximum(Q, 1e-300))
    DLR = -(T - MUT)
    ef_t = sec.fn_rayleigh(la, s, la + T); rec['SI_target_frac'] = (Eg - ef_t) / (Eg - Efn)
    meas = 0.5 * PA + 0.5 * CT / jnp.sum(CT)
    cols = []
    for v_ in (lan, jnp.log(jnp.maximum(V, 0) + 1e-6), jnp.log(jnp.maximum(W, 0) + 1e-6)):
        m_ = float(jnp.sum(meas * v_)); s_ = float(jnp.sqrt(jnp.sum(meas * (v_ - m_) ** 2))) + 1e-30
        cols.append(((v_ - m_) / s_).astype(jnp.float32))
    FEAT = jnp.stack(cols, 1); del cols, V, W, KUA, CT, meas, e
    log(f'[it {k}] guide <H> {rec["guide_H"]:.4e} E_FN {rec["guide_E_FN"]:.4e} SI-target frac {rec["SI_target_frac"]:.4f}')
    # ---------------- fit (realistic): F-SI-Adam
    sec.offload()
    B, K, steps = SPEC.get('B', 256), SPEC.get('K', 8), SPEC.get('steps', 20000)
    model = Corr(32, 4, seed=k, feat=FEAT); flat = model.flat0
    sched = optax.warmup_cosine_decay_schedule(0.0, SPEC.get('lr', 3e-3), min(200, max(1, steps // 2)), steps, SPEC.get('lr', 3e-3) * 0.02)
    opt = optax.adam(sched); ost = opt.init(flat)

    @jax.jit
    def batch(key):
        k1, k2 = jax.random.split(key)
        idx = jnp.clip(jnp.searchsorted(CDF, jax.random.uniform(k1, (B,), f64) * CDF[-1]), 0, sec.D - 1)
        x = sec.reps[idx]
        valid = (((x[:, None] >> BI[None, :]) ^ (x[:, None] >> BJ[None, :])) & jnp.uint64(1)).astype(bool)
        nval = valid.sum(1)
        sc = jnp.where(valid, jax.random.uniform(k2, valid.shape), -1.0)
        _, bsel = jax.lax.top_k(sc, K)
        ok = jnp.take_along_axis(valid, bsel, 1)
        y = x[:, None] ^ MASKS[bsel]
        iy = SS.canon(sec.T, sec.reps, y.reshape(-1)).reshape(B, K)
        rP = jnp.exp(jnp.clip(la[iy] - la[idx][:, None], -60, 60))
        wgt = jnp.where(ok & (s[idx][:, None] * s[iy] < 0), 0.5 * JB[bsel] * rP, 0.0) * (nval[:, None] / K)
        lw = LIW[idx]; iw = jnp.exp(lw - jnp.max(lw))
        return x, y, DLR[idx], DLR[iy], wgt, iw, TIDX[idx], TIDX[iy]

    def loss(q, x, y, dlx, dly, wgt, iw, tx, ty, G):
        X = bits(jnp.concatenate([x, y.reshape(-1)])); Tt = jnp.concatenate([tx, ty.reshape(-1)])
        fa = model.f_aug(q, X, Tt, jnp.concatenate([G, jnp.repeat(G, K)])).astype(f64)
        fx, fy = fa[:B], fa[B:].reshape(B, K)
        iw = iw / jnp.mean(iw)
        return 0.5 * jnp.mean(iw * jnp.sum(wgt * ((fx + dlx)[:, None] - (fy + dly)) ** 2, 1)) / N

    @jax.jit
    def step(fl, ost, key):
        k1, k2 = jax.random.split(key)
        bd = batch(k1); G = jax.random.randint(k2, (B,), 0, 16)
        l, g = jax.value_and_grad(loss)(fl, *bd, G)
        upd, ost = opt.update(g, ost, fl)
        return optax.apply_updates(fl, upd), ost, l
    vk = [jax.random.PRNGKey(777 + i) for i in range(32)]
    vfn = jax.jit(lambda fl, key: loss(fl, *batch(key), jax.random.randint(key, (B,), 0, 16)))
    curve = []
    key = jax.random.PRNGKey(100 + k)
    for it in range(1, steps + 1):
        key, kk = jax.random.split(key)
        flat, ost, l = step(flat, ost, kk)
        if it % 1000 == 0 or it == 1:
            curve.append((it, float(np.mean([float(vfn(flat, q)) for q in vk])))); log(f'  [it {k}] step {it} own val {curve[-1][1]:.4e}')
    rec['fit_curve'] = curve; rec['fit_sec'] = time.time() - t0
    np.save(os.path.join(OUT, f'params_it{k}.npy'), np.asarray(flat))
    # ---------------- exact referee of the written-back state
    fb = model.table(flat, sec.reps, TIDX)
    la_new = la + fb; del fb
    ef = sec.fn_rayleigh(la, s, la_new)
    rec['frac'] = (Eg - ef) / (Eg - Efn)
    rec['H_oldsign'] = H_site(la_new, s)
    s_new, _, _ = sec.krylov(sec.vec(la_new, jnp.ones_like(la_new)), s)
    rec['H_kry'] = H_site(la_new, s_new)
    _, _, i2 = sec.fn_solve(la_new, s_new); rec['E_FN_next'] = i2['dE_FN_site']
    rec['sec'] = time.time() - t0
    log(f'== it {k}: frac {rec["frac"]:.4f}  <H>(old s) {rec["H_oldsign"]:.4e}  <H>(Krylov) {rec["H_kry"]:.4e}  '
        f'E_FN next {rec["E_FN_next"]:.4e}  ({rec["sec"]:.0f}s)')
    res['iters'].append(rec); dump(F, res)
    la, s = la_new, s_new
    del model, flat, ost, FEAT, T, DLR, CDF, LIW, PA, UA, Q
res['sec'] = time.time() - T00; dump(F, res); log('DONE', res['sec'])
