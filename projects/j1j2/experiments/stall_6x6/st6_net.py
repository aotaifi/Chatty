"""Networks and supervised fitting on exact symmetric-sector targets (6x6 J1-J2, GPU, JAX).

Model variants (real log-amplitude f(x); the ViT part is the nqsmagic 6x6 ViT, Re log psi):
  vit        : the trained ViT (warm start from the checkpoint)
  vit+cnn    : ViT + periodic CNN correction (2 conv layers, C channels, zero-initialised readout)
  vit+rbm    : ViT + translation-invariant RBM correction  sum_f,i log cosh(conv_f(x)_i + b_f), alpha = C features
  vitbig     : ViT with num_layers / d_model / heads from the spec (fresh init, or distilled first)
Training data are exact: reps r ~ p_r^beta with p_r = v_r^2 of the TARGET state (sector probability),
mapped to a uniformly random orbit element (random space-group element x spin flip; this trains the
point-group / flip symmetry the ViT does not have built in), plus K random one-hop neighbours y (bond
exchanges) with exact target values looked up through the canonical representative.
Losses (target t = per-configuration log amplitude, offset free):
  amp  : Var_x [ f(x) - t(x) ]
  edge : E_x sum_k w_xy [ (f(y) - f(x)) - (t(y) - t(x)) ]^2 ,  w_xy = J_b/2 * min(exp(t(y)-t(x)), e^4) * nvalid/K
         (energy weighting of the second-order expansion of <H> around the target at fixed exact signs)
"""
import math, time
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import optax
from jax.flatten_util import ravel_pytree
import vit_dt
import st6_sector as SS

N = 36


def log(*a):
    print(time.strftime('[%H:%M:%S]'), *a, flush=True)


class CNNCorr(nn.Module):
    C: int = 16
    layers: int = 2

    @nn.compact
    def __call__(self, x):                  # x: (B, 36) +-1
        h = x.reshape(x.shape[0], 6, 6, 1)
        for _ in range(self.layers):
            h = nn.Conv(self.C, (3, 3), padding='CIRCULAR', param_dtype=jnp.float32, dtype=jnp.float32)(h)
            h = nn.gelu(h)
        h = h.sum(axis=(1, 2))
        return nn.Dense(1, kernel_init=nn.initializers.zeros, param_dtype=jnp.float32, dtype=jnp.float32)(h)[:, 0]


class RBMCorr(nn.Module):
    C: int = 8

    @nn.compact
    def __call__(self, x):
        h = x.reshape(x.shape[0], 6, 6, 1)
        h = nn.Conv(self.C, (6, 6), padding='CIRCULAR', use_bias=True, param_dtype=jnp.float32, dtype=jnp.float32,
                    kernel_init=nn.initializers.normal(1e-3))(h)
        return vit_dt.log_cosh(h).sum(axis=(1, 2, 3)).real


class Model:
    def __init__(self, spec, ckpt, seed=0):
        jax.config.update('jax_default_matmul_precision', 'highest')
        vit_dt.set_dtype('float32')
        self.spec = spec
        kind = spec['kind']
        if kind == 'vitbig':
            self.vit = vit_dt.ViT(num_layers=spec.get('layers', 4), d_model=spec.get('d_model', 60),
                                  heads=spec.get('heads', 10), L_eff=9, b=2, transl_invariant=True, two_dimensional=True)
        else:
            self.vit = vit_dt.make_model_6x6()
        tmpl = self.vit.init(jax.random.PRNGKey(1234 + seed), jnp.zeros((1, N), jnp.float32))
        if kind != 'vitbig' or spec.get('warm', False):
            obj = flax.serialization.msgpack_restore(open(ckpt, 'rb').read())
            if 'variables' in obj: obj = obj['variables']
            tmpl = flax.serialization.from_state_dict(tmpl, obj)
        params = {'vit': jax.tree_util.tree_map(lambda p: jnp.asarray(p, jnp.float32), tmpl['params'])}
        self.corr = None
        if kind == 'vit+cnn':
            self.corr = CNNCorr(C=spec.get('C', 16), layers=spec.get('layers', 2))
        elif kind == 'vit+rbm':
            self.corr = RBMCorr(C=spec.get('C', 8))
        if self.corr is not None:
            params['corr'] = self.corr.init(jax.random.PRNGKey(99 + seed), jnp.zeros((1, N), jnp.float32))['params']
        self.flat0, self.unravel = ravel_pytree(params)
        self.npar = int(self.flat0.size)
        self.npar_vit = int(ravel_pytree(params['vit'])[0].size)
        vit = self.vit; corr = self.corr; unravel = self.unravel

        def fc(flat, X):                     # complex log psi of the ViT part + real correction
            p = unravel(flat)
            z = vit_dt.logpsi_transl_2d(vit.apply, 2, {'params': p['vit']}, X)
            if corr is not None:
                z = z + corr.apply({'params': p['corr']}, X)
            return z

        self.fc = fc
        self.f = lambda flat, X: jnp.real(fc(flat, X))

    # ------------------------------------------------------------------ evaluation on all reps
    def eval_reps(self, flat, reps, batch=16384, complex_out=False):
        if not hasattr(self, '_ev'):
            ar = jnp.arange(N, dtype=jnp.uint64)

            def ev(flat, S):
                X = (((S[:, None] >> ar[None, :]) & jnp.uint64(1)).astype(jnp.float32) * 2 - 1)
                z = self.fc(flat, X)
                return jnp.real(z).astype(jnp.float64), jnp.imag(z).astype(jnp.float64)
            self._ev = jax.jit(ev)
        D = reps.shape[0]
        outr, outi = [], []
        for i in range(0, D, batch):
            s = reps[i:i + batch]
            m = s.shape[0]
            if m < batch: s = jnp.concatenate([s, jnp.repeat(s[:1], batch - m)])
            a, b = self._ev(flat, s)
            outr.append(a[:m]); outi.append(b[:m])
        la = jnp.concatenate(outr)
        return (la, jnp.concatenate(outi)) if complex_out else la


# ------------------------------------------------------------------ supervised fit
def bits_to_spins(S):
    ar = jnp.arange(N, dtype=jnp.uint64)
    return (((S[:, None] >> ar[None, :]) & jnp.uint64(1)).astype(jnp.float32) * 2 - 1)


def make_step(model, opt, B, K, lam_amp, lam_edge, wcap=4.0):
    MASKS = jnp.asarray(SS.MASKS); BI = jnp.asarray(SS.BI); BJ = jnp.asarray(SS.BJ)
    JB = jnp.asarray(SS.JB, jnp.float32)
    f = model.f

    def batch_data(key, cdf, tlog, T, reps):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        u = jax.random.uniform(k1, (B,), jnp.float64) * cdf[-1]
        idx = jnp.clip(jnp.searchsorted(cdf, u), 0, cdf.shape[0] - 1)
        g = jax.random.randint(k2, (B,), 0, T.shape[0]); fl = jax.random.bernoulli(k3, 0.5, (B,))
        x = SS.image(T, reps[idx], g, fl)
        valid = (((x[:, None] >> BI[None, :]) ^ (x[:, None] >> BJ[None, :])) & jnp.uint64(1)).astype(bool)
        nval = valid.sum(1)
        sc = jnp.where(valid, jax.random.uniform(k4, valid.shape), -1.0)
        _, bsel = jax.lax.top_k(sc, K)                                    # K distinct valid bonds
        y = x[:, None] ^ MASKS[bsel]
        iy = SS.canon(T, reps, y.reshape(-1)).reshape(B, K)
        tx = tlog[idx]; ty = tlog[iy]
        dt = ty - tx[:, None]
        w = 0.5 * JB[bsel] * jnp.exp(jnp.minimum(dt, wcap)).astype(jnp.float32) * (nval[:, None] / K).astype(jnp.float32)
        return x, y, tx.astype(jnp.float32), dt.astype(jnp.float32), w

    def losses(flat, x, y, tx, dt, w):
        X = bits_to_spins(x); Y = bits_to_spins(y.reshape(-1))
        fx = f(flat, X); fy = f(flat, Y).reshape(y.shape)
        d = fx - tx
        l_amp = jnp.mean((d - jnp.mean(d)) ** 2)
        e = (fy - fx[:, None]) - dt
        l_edge = jnp.sum(w * e * e) / B
        return l_amp, l_edge

    def loss_fn(flat, x, y, tx, dt, w):
        la, le = losses(flat, x, y, tx, dt, w)
        return lam_amp * la + lam_edge * le, (la, le)

    @jax.jit
    def step(flat, ost, key, cdf, tlog, T, reps):
        x, y, tx, dt, w = batch_data(key, cdf, tlog, T, reps)
        (l, (la, le)), gr = jax.value_and_grad(loss_fn, has_aux=True)(flat, x, y, tx, dt, w)
        upd, ost = opt.update(gr, ost, flat)
        return optax.apply_updates(flat, upd), ost, la, le

    @jax.jit
    def val(flat, key, cdf, tlog, T, reps):
        x, y, tx, dt, w = batch_data(key, cdf, tlog, T, reps)
        return losses(flat, x, y, tx, dt, w)

    return step, val


def fit(model, flat, sec, tlog, *, beta=1.0, steps=4000, lr=1e-4, B=1024, K=8, lam_amp=1.0, lam_edge=1.0,
        seed=0, eval_every=0, eval_fn=None, warmup=200, end_lr_frac=0.05, nval=8, val_B_mult=8, hist=None,
        opt='adam'):
    """Fit model (flat params) to the per-config log target tlog (D,) on the sector of `sec`.
    Sampling distribution of reps: p_r^beta with p_r = n_r exp(2 tlog_r) (the target's own sector weights)."""
    lp = 2.0 * tlog + jnp.log(sec.n)
    lp = beta * (lp - jnp.max(lp))
    cdf = jnp.cumsum(jnp.exp(lp))
    sched = optax.warmup_cosine_decay_schedule(0.0, lr, warmup, max(steps, warmup + 1), lr * end_lr_frac)
    opt = optax.adam(sched) if opt == 'adam' else optax.chain(optax.clip_by_global_norm(1.0), optax.adam(sched))
    ost = opt.init(flat)
    step, val = make_step(model, opt, B, K, lam_amp, lam_edge)
    _, valb = make_step(model, opt, B * val_B_mult, K, lam_amp, lam_edge)
    key = jax.random.PRNGKey(seed)
    vkeys = [jax.random.PRNGKey(10_000 + seed + i) for i in range(nval)]

    def vloss(fl):
        r = np.array([[float(z) for z in valb(fl, k, cdf, tlog, sec.T, sec.reps)] for k in vkeys])
        return r.mean(0)

    hist = [] if hist is None else hist
    v0 = vloss(flat)
    hist.append(dict(step=0, val_amp=float(v0[0]), val_edge=float(v0[1])))
    log(f'  fit start val amp {v0[0]:.4e} edge {v0[1]:.4e}')
    t0 = time.time()
    la_acc = le_acc = 0.0; nacc = 0
    for it in range(1, steps + 1):
        key, k = jax.random.split(key)
        flat, ost, la, le = step(flat, ost, k, cdf, tlog, sec.T, sec.reps)
        if it % 200 == 0 or it == steps:
            la_acc = float(la); le_acc = float(le)
        if it % 1000 == 0 or it == steps:
            v = vloss(flat)
            rec = dict(step=it, train_amp=la_acc, train_edge=le_acc, val_amp=float(v[0]), val_edge=float(v[1]),
                       sec=time.time() - t0)
            if eval_every and eval_fn is not None and (it % eval_every == 0 or it == steps):
                rec.update(eval_fn(flat))
            hist.append(rec)
            log('  fit', {k_: (round(v_, 7) if isinstance(v_, float) else v_) for k_, v_ in rec.items()})
    return flat, hist
