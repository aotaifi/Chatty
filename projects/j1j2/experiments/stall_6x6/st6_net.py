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
        if kind in ('cnn', 'rbm'):                 # correction factor alone (the base log-amplitude is a fixed table)
            self.corr = CNNCorr(C=spec.get('C', 16), layers=spec.get('layers', 2)) if kind == 'cnn' else RBMCorr(C=spec.get('C', 8))
            params = {'corr': self.corr.init(jax.random.PRNGKey(99 + seed), jnp.zeros((1, N), jnp.float32))['params']}
            self.flat0, self.unravel = ravel_pytree(params)
            self.npar = int(self.flat0.size); self.npar_vit = 0
            corr = self.corr; unravel = self.unravel
            # symmetrised over D4 x spin flip (translations: built in), so the factor is a smooth symmetric function
            pg = SS.space_group()[:8]
            inv = np.argsort(pg, axis=1)                    # X'[:, p[i]] = X[:, i]  <=>  X' = X[:, inv]
            invj = jnp.asarray(inv)

            def fsym(flat, X):
                pr = {'params': unravel(flat)['corr']}
                out = 0.0
                for k in range(8):
                    Xk = X[:, invj[k]]
                    out = out + corr.apply(pr, Xk) + corr.apply(pr, -Xk)
                return out / 16.0
            self.f = fsym
            self.fc = lambda flat, X: fsym(flat, X).astype(jnp.complex64)
            return
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
        if self.corr is not None:
            mk = jax.tree_util.tree_map(jnp.zeros_like, params); mk['corr'] = jax.tree_util.tree_map(jnp.ones_like, params['corr'])
            self.mask_corr = ravel_pytree(mk)[0]

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


def make_step(model, opt, B, K, lam_amp, lam_edge, wcap=4.0, images=True):
    MASKS = jnp.asarray(SS.MASKS); BI = jnp.asarray(SS.BI); BJ = jnp.asarray(SS.BJ)
    JB = jnp.asarray(SS.JB, jnp.float32)
    f = model.f

    def batch_data(key, cdf, tlog, T, reps):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        u = jax.random.uniform(k1, (B,), jnp.float64) * cdf[-1]
        idx = jnp.clip(jnp.searchsorted(cdf, u), 0, cdf.shape[0] - 1)
        g = jax.random.randint(k2, (B,), 0, T.shape[0]); fl = jax.random.bernoulli(k3, 0.5, (B,))
        x = SS.image(T, reps[idx], g, fl) if images else reps[idx]
        valid = (((x[:, None] >> BI[None, :]) ^ (x[:, None] >> BJ[None, :])) & jnp.uint64(1)).astype(bool)
        nval = valid.sum(1)
        sc = jnp.where(valid, jax.random.uniform(k4, valid.shape), -1.0)
        _, bsel = jax.lax.top_k(sc, K)                                    # K distinct valid bonds
        y = x[:, None] ^ MASKS[bsel]
        iy = SS.canon(T, reps, y.reshape(-1)).reshape(B, K)
        if not images: y = reps[iy]                      # rep-evaluated network: neighbours by their reps
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
        opt='adam', images=True, samp_log=None):
    """Fit model (flat params) to the per-config log target tlog (D,) on the sector of `sec`.
    Sampling distribution of reps: p_r^beta with p_r = n_r exp(2 tlog_r) (the target's own sector weights)."""
    if hasattr(sec, 'offload'): sec.offload()      # H not needed while fitting (device memory)
    lp = 2.0 * (tlog if samp_log is None else samp_log) + jnp.log(sec.n)
    lp = beta * (lp - jnp.max(lp))
    cdf = jnp.cumsum(jnp.exp(lp))
    sched = optax.warmup_cosine_decay_schedule(0.0, lr, warmup, max(steps, warmup + 1), lr * end_lr_frac)
    opt = optax.adam(sched) if opt == 'adam' else optax.chain(optax.clip_by_global_norm(1.0), optax.adam(sched))
    ost = opt.init(flat)
    step, val = make_step(model, opt, B, K, lam_amp, lam_edge, images=images)
    _, valb = make_step(model, opt, B * val_B_mult, K, lam_amp, lam_edge, images=images)
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
                if hasattr(sec, 'offload'): sec.offload()
            hist.append(rec)
            log('  fit', {k_: (round(v_, 7) if isinstance(v_, float) else v_) for k_, v_ in rec.items()})
    return flat, hist


# ------------------------------------------------------------------ energy fit (fixed sign table, exact local data)
def make_energy_step(model, opt, B, mode, nchunk=4):
    """mode 'var':  variational <H> of b*s (s = sign table on reps)
       mode 'fn' :  frozen lattice-FN energy <H_FN[a_g, s]> of the trial amplitude b (minimiser = phi_FN[a_g, s]).
    Samples: reps r ~ n_r exp(2 la_ref_r) (la_ref = recent full table of the network), random orbit image,
    reweighted by exp(2 (f(x) - la_ref(x))); all 144 bond neighbours exact (canonical lookup of s, a_g)."""
    MASKS = jnp.asarray(SS.MASKS); BI = jnp.asarray(SS.BI); BJ = jnp.asarray(SS.BJ)
    JB = jnp.asarray(SS.JB)
    f = model.f
    NB = MASKS.shape[0]

    def batch(key, cdf, s_tab, la_g, T, reps):
        k1, k2, k3 = jax.random.split(key, 3)
        u = jax.random.uniform(k1, (B,), jnp.float64) * cdf[-1]
        idx = jnp.clip(jnp.searchsorted(cdf, u), 0, cdf.shape[0] - 1)
        g = jax.random.randint(k2, (B,), 0, T.shape[0]); fl = jax.random.bernoulli(k3, 0.5, (B,))
        x = SS.image(T, reps[idx], g, fl)
        valid = (((x[:, None] >> BI[None, :]) ^ (x[:, None] >> BJ[None, :])) & jnp.uint64(1)).astype(bool)
        y = jnp.where(valid, x[:, None] ^ MASKS[None, :], x[:, None])
        iy = SS.canon(T, reps, y.reshape(-1)).reshape(B, NB)
        diag = 0.25 * JB.sum() - 0.5 * (valid * JB[None, :]).sum(1)
        sx = s_tab[idx].astype(jnp.float64); sy = s_tab[iy].astype(jnp.float64)
        h = 0.5 * JB[None, :] * valid
        if mode == 'fn':
            rg = jnp.exp(jnp.clip(la_g[iy] - la_g[idx][:, None], -60, 60))
            viol = valid & (sx[:, None] * sy > 0)
            dfn = diag + jnp.sum(jnp.where(viol, h * rg, 0.0), 1)
            hh = jnp.where(viol, 0.0, h)                 # kept (allowed) edges, sign -1 (s s' = -1)
            return x, y, idx, dfn, -hh
        return x, y, idx, diag, h * sx[:, None] * sy

    def fy_all(flat, y):
        yc = y.reshape(nchunk, -1)
        out = jax.lax.map(lambda yy: f(flat, bits_to_spins(yy)), yc)
        return out.reshape(y.shape).astype(jnp.float64)

    def eloc(flat, fx, y, d0, hc):
        fy = jax.lax.stop_gradient(fy_all(flat, y))
        return d0 + jnp.sum(hc * jnp.exp(jnp.clip(fy - fx[:, None], -60, 60)), 1)

    @jax.jit
    def step(flat, ost, key, cdf, la_ref, s_tab, la_g, T, reps):
        x, y, idx, d0, hc = batch(key, cdf, s_tab, la_g, T, reps)
        X = bits_to_spins(x)

        def sur(fl):
            fx = f(fl, X).astype(jnp.float64)
            fs = jax.lax.stop_gradient(fx)
            EL = eloc(fl, fs, y, d0, hc)
            w = jnp.exp(2 * (fs - la_ref[idx])); w = w / jnp.sum(w)
            E = jnp.sum(w * EL)
            return 2.0 * jnp.sum(w * (EL - E) * fx), (E, jnp.sum(w * (EL - E) ** 2), 1.0 / jnp.sum(w * w))
        (l, (E, var, ess)), gr = jax.value_and_grad(sur, has_aux=True)(flat)
        upd, ost = opt.update(gr, ost, flat)
        return optax.apply_updates(flat, upd), ost, E, var, ess

    return step


def fit_energy(model, flat, sec, s_tab, *, mode='var', la_g=None, steps=3000, lr=1e-5, B=256, seed=0,
               refresh=500, warmup=100, end_lr_frac=0.1, eval_fn=None, hist=None, keep_best=True):
    """Minimise the exact-local-data energy (mode var / fn) of the network amplitude at fixed signs s_tab.
    Every `refresh` steps: full table of the network on all reps (sampling reference) and eval_fn(flat, la_tab)
    (exact sector energy).  Returns the params with the lowest eval_fn()['obj'] if keep_best."""
    if hasattr(sec, 'offload'): sec.offload()
    sched = optax.warmup_cosine_decay_schedule(0.0, lr, warmup, max(steps, warmup + 1), lr * end_lr_frac)
    opt = optax.adam(sched)
    ost = opt.init(flat)
    step = make_energy_step(model, opt, B, mode)
    la_g = jnp.zeros(1) if la_g is None else la_g
    key = jax.random.PRNGKey(seed)
    hist = [] if hist is None else hist

    def refresh_tab(fl):
        la = model.eval_reps(fl, sec.reps)
        lp = 2.0 * (la - jnp.max(la)) + jnp.log(sec.n)
        return la, jnp.cumsum(jnp.exp(lp))

    la_ref, cdf = refresh_tab(flat)
    best = (np.inf, flat, None)
    ev = eval_fn(flat, la_ref) if eval_fn is not None else {}
    if hasattr(sec, 'offload'): sec.offload()
    hist.append(dict(step=0, **ev))
    log('  efit start', ev)
    if ev: best = (ev['obj'], flat, ev)
    t0 = time.time(); Es = []
    for it in range(1, steps + 1):
        key, k = jax.random.split(key)
        flat, ost, E, var, ess = step(flat, ost, k, cdf, la_ref, s_tab, la_g, sec.T, sec.reps)
        if it % 50 == 0: Es.append((float(E), float(var), float(ess)))
        if it % refresh == 0 or it == steps:
            la_ref, cdf = refresh_tab(flat)
            ev = eval_fn(flat, la_ref) if eval_fn is not None else {}
            if hasattr(sec, 'offload'): sec.offload()
            Ea = np.array(Es); Es = []
            rec = dict(step=it, E_sampled_site=float(Ea[:, 0].mean()) / N, var_site=float(Ea[:, 1].mean()) / N,
                       ess=float(Ea[:, 2].mean()), sec=time.time() - t0, **ev)
            hist.append(rec)
            log('  efit', {k_: (round(v_, 8) if isinstance(v_, float) else v_) for k_, v_ in rec.items()})
            if ev and ev['obj'] < best[0]: best = (ev['obj'], flat, ev)
    if keep_best and best[2] is not None:
        log('  efit best', best[2])
        return best[1], hist
    return flat, hist


# ------------------------------------------------------------------ energy fit on the exact sector table
def fit_table(model, flat, sec, s_tab, *, mode='var', la_g=None, outer=15, inner=40, B=4096, lr=1e-5, seed=0,
              warmup=20, end_lr_frac=0.1, eval_extra=None, hist=None, keep_best=True, log_every=1):
    """Minimise an exact sector energy of the network state (network evaluated on the reps):
         mode 'var': E = <v|H|v>, v_r = s_r b_r sqrt(n_r)           (variational energy at fixed signs s_tab)
         mode 'fn' : E = <v|H_FN[a_g, s_tab]|v>, v_r = b_r sqrt(n_r) (frozen lattice-FN energy; minimiser phi_FN)
    Outer iteration: full table of the network on all D reps, exact (H v) -> exact energy and exact gradient
    weights c_r = v_r ((H v)_r - E v_r)  (dE/dtheta = 2 sum_r c_r d f(rep_r)/dtheta).
    Inner steps: Adam on the importance-sampled gradient (B reps drawn ~ |c_r|; c from the last table, i.e.
    a first-order-stale but sampling-noise-only estimate).  Keeps the parameters with the lowest exact E."""
    if mode == 'fn':
        A = sec.fn_op(la_g, s_tab)
    total = outer * inner
    sched = optax.warmup_cosine_decay_schedule(0.0, lr, warmup, max(total, warmup + 1), lr * end_lr_frac)
    opt = optax.adam(sched); ost = opt.init(flat)
    f = model.f

    @jax.jit
    def gstep(fl, ost, key, cdf, sgn, Z, reps):
        u = jax.random.uniform(key, (B,), jnp.float64) * cdf[-1]
        idx = jnp.clip(jnp.searchsorted(cdf, u), 0, cdf.shape[0] - 1)
        X = bits_to_spins(reps[idx])
        wts = (2.0 * Z / B * sgn[idx]).astype(jnp.float32)
        g = jax.grad(lambda q: jnp.sum(wts * f(q, X)))(fl)
        upd, ost = opt.update(g, ost, fl)
        return optax.apply_updates(fl, upd), ost

    key = jax.random.PRNGKey(seed)
    hist = [] if hist is None else hist
    best = (np.inf, flat, None)
    t0 = time.time()
    for o in range(outer + 1):
        la = model.eval_reps(flat, sec.reps)
        if mode == 'var':
            v = sec.vec(la, s_tab); Hv = sec.H(v)
        else:
            v = sec.vec(la, jnp.ones_like(la)); Hv = A(v)
        E = float(v @ Hv)
        c = v * (Hv - E * v); del Hv, v
        rec = dict(outer=o, step=o * inner, E_site=E / N, dE_site=(E - sec.E0) / N, grad_l1=float(jnp.sum(jnp.abs(c))),
                   sec=time.time() - t0)
        if eval_extra is not None: rec.update(eval_extra(flat, la))
        hist.append(rec)
        if o % log_every == 0 or o == outer:
            log('  tfit', {k_: (round(v_, 9) if isinstance(v_, float) else v_) for k_, v_ in rec.items()})
        if rec['dE_site'] < best[0]: best = (rec['dE_site'], flat, rec)
        if o == outer: break
        ac = jnp.abs(c); Z = jnp.sum(ac); cdf = jnp.cumsum(ac); sgn = jnp.sign(c); del ac, c
        if hasattr(sec, 'offload'): sec.offload()
        for _ in range(inner):
            key, k = jax.random.split(key)
            flat, ost = gstep(flat, ost, k, cdf, sgn, Z, sec.reps)
        del cdf, sgn
    if keep_best:
        log('  tfit best', best[2])
        return best[1], hist
    return flat, hist


# ------------------------------------------------------------------ exact-verified optimiser (SR + gradient candidates)
@partial(jax.jit, donate_argnums=(0,))
def _set_rows(O, rows, i):
    return jax.lax.dynamic_update_slice(O, rows, (i, 0))


def fit_exact(model, flat, sec, s_tab, *, mode='var', la_g=None, outer=12, N_sr=6144, B_g=16384, lam=1e-3,
              eta_sr=0.3, eta_g=1e-5, seed=0, eval_extra=None, hist=None, jac_chunk=128, max_tries=3, mask=None,
              base_tab=None):
    """Minimise an exact sector energy of the network-on-reps state (mode 'var' or 'fn', see fit_table).
    Per outer iteration, from the exact table of the current parameters (energy E, exact local energies
    E_L = (Hv)_r / v_r, exact gradient weights c_r = v_r((Hv)_r - E v_r)):
      SR candidate  : N_sr reps ~ v_r^2, O = d f/d theta (centred), eps = E_L - E,
                      delta = -O^T (O O^T + lam tr(OO^T)/N I)^{-1} eps,  theta + eta_sr * delta
      grad candidate: g = 2 sum_r c_r df_r (B_g reps ~ |c_r|),  theta - eta_g * g / rms(g)
    Both candidates are scored by the EXACT energy (one full table each); the better one is accepted if it
    lowers E (its step size grows x1.5), otherwise both step sizes shrink /3 (up to max_tries).
    Every accepted step is an exact, noise-free improvement."""
    if mode == 'fn':
        A = sec.fn_op(la_g, s_tab)
    f = model.f
    gradf = jax.jit(jax.vmap(jax.grad(lambda fl, x: f(fl, x[None])[0]), in_axes=(None, 0)))

    def table(fl):
        la = model.eval_reps(fl, sec.reps)
        if base_tab is not None: la = la + base_tab          # fixed base log-amplitude (e.g. projected ViT)
        if mode == 'var':
            v = sec.vec(la, s_tab); Hv = sec.H(v)
        else:
            v = sec.vec(la, jnp.ones_like(la)); Hv = A(v)
        E = float(v @ Hv)
        return la, v, Hv, E

    @jax.jit
    def gsum(fl, key, cdf, sgn, Z, reps):
        B = 4096
        u = jax.random.uniform(key, (B,), jnp.float64) * cdf[-1]
        idx = jnp.clip(jnp.searchsorted(cdf, u), 0, cdf.shape[0] - 1)
        X = bits_to_spins(reps[idx])
        wts = (2.0 * Z / B * sgn[idx]).astype(jnp.float32)
        return jax.grad(lambda q: jnp.sum(wts * f(q, X)))(fl)

    key = jax.random.PRNGKey(seed)
    hist = [] if hist is None else hist
    t0 = time.time()
    la, v, Hv, E = table(flat)
    for o in range(outer + 1):
        rec = dict(outer=o, E_site=E / N, dE_site=(E - sec.E0) / N, eta_sr=eta_sr, eta_g=eta_g, sec=time.time() - t0)
        if eval_extra is not None: rec.update(eval_extra(flat, la))
        if o == outer:
            hist.append(rec); log('  xfit', rec); break
        # ---- directions
        c = v * (Hv - E * v)
        p = v * v
        EL = Hv / jnp.where(jnp.abs(v) > 1e-300, v, 1.0)
        del Hv
        ac = jnp.abs(c); Z = jnp.sum(ac); cdf = jnp.cumsum(ac); sgn = jnp.sign(c); del ac, c
        g = jnp.zeros_like(flat)
        for _ in range(max(1, B_g // 4096)):
            key, k = jax.random.split(key)
            g = g + gsum(flat, k, cdf, sgn, Z, sec.reps)
        g = g / max(1, B_g // 4096)
        if mask is not None: g = g * mask
        del cdf, sgn
        cdfp = jnp.cumsum(p); key, k = jax.random.split(key)
        idx = jnp.clip(jnp.searchsorted(cdfp, jax.random.uniform(k, (N_sr,), jnp.float64) * cdfp[-1]), 0, sec.D - 1)
        del cdfp, p
        eps = EL[idx] - E; del EL
        if hasattr(sec, 'offload'): sec.offload()
        X = bits_to_spins(sec.reps[idx])
        O = jnp.zeros((N_sr, flat.shape[0]), jnp.float32)            # built in place (donated updates)
        for i in range(0, N_sr, jac_chunk):
            rows = gradf(flat, X[i:i + jac_chunk])
            if mask is not None: rows = rows * mask[None, :]
            O = _set_rows(O, rows, i)
        m = jnp.mean(O, 0)                                           # centring done implicitly (no copy of O)
        T = jnp.zeros((N_sr, N_sr), jnp.float64)
        for j in range(0, O.shape[1], 16384):                       # chunked O O^T (memory / autotuning)
            Oj = O[:, j:j + 16384]
            T = T + (Oj @ Oj.T).astype(jnp.float64)
            del Oj
        Om = sum(O[:, j:j + 16384] @ m[j:j + 16384] for j in range(0, O.shape[1], 16384)).astype(jnp.float64)
        mm = float(m.astype(jnp.float64) @ m.astype(jnp.float64))
        T = T - Om[:, None] - Om[None, :] + mm
        sh = lam * jnp.trace(T) / N_sr
        alpha = jnp.linalg.solve(T + sh * jnp.eye(N_sr), eps)
        a32 = alpha.astype(jnp.float32)
        dsr = -(jnp.concatenate([O[:, j:j + 16384].T @ a32 for j in range(0, O.shape[1], 16384)]) - m * jnp.sum(a32))
        pred = float(-(eps @ (T @ alpha)) / N_sr * 2)     # rough first-order model decrease (diagnostic)
        del O, T, X, alpha, a32, Om, m
        dg = -g / jnp.sqrt(jnp.sum(g * g) / (flat.shape[0] if mask is None else jnp.sum(mask))); del g
        rec.update(sr_norm=float(jnp.linalg.norm(dsr)), sr_pred=pred)
        accepted = False
        for tr in range(max_tries):
            cand = {}
            for name, d, eta in (('sr', dsr, eta_sr), ('grad', dg, eta_g)):
                fl = flat + eta * d
                cl, cv, cH, cE = table(fl)
                cand[name] = (cE, fl, cl, cv, cH)
                rec[f'try{tr}_{name}_dE'] = (cE - E) / N
            bname = min(cand, key=lambda k_: cand[k_][0])
            if cand[bname][0] < E:
                cE, flat, la, v, Hv = cand[bname]; E = cE
                if bname == 'sr': eta_sr *= 1.5
                else: eta_g *= 1.5
                rec['accepted'] = bname; accepted = True
                del cand
                break
            del cand
            eta_sr /= 3; eta_g /= 3
        if not accepted:
            rec['accepted'] = None
            la, v, Hv, E = table(flat)
        hist.append(rec)
        log('  xfit', {k_: (round(v_, 9) if isinstance(v_, float) else v_) for k_, v_ in rec.items()})
    return flat, hist
