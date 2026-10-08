"""Write-back arms with the energy-metric (tail) proposal, exact 6x6 oracle harness (see README.md).

Model b = a exp(f), a = |psi_P| frozen, f = exactly D4 x flip symmetric residual CNN (writeback harness), optionally
with the base log-amplitude log a(x) as an extra input ('+a').
Proposals over configurations x (rep-level sampling, self-normalised importance weights p/q, p = phi^2):
  'old'  : q ~ phi^(2 beta) (beta = 0.5), K bonds uniform among valid bonds
  'mix'  : q = (1-eps) phi^(2 beta) + eps c_delta/sum c_delta (node weight of the FN quadratic form at f = 0)
  'amix' : as 'mix', tail component refreshed every refresh steps to c_e of the current residual e = delta - f
  any proposal can be restricted to the bulk (phi^2 >= 1e-10 per configuration) with 'bulk_only'.
Bonds: 'uniform' (K of the valid bonds, weight nvalid/K) or 'weighted' (K draws prop. to the FN edge weight
|H_xy| phi_y/phi_x over the allowed bonds, estimator W_x mean_k).
Losses: 'energy' = exact frozen-FN edge identity (nonlinear in g = b/phi), 'quad' = FN-Laplacian least squares of edge
differences 1/2 sum w_xy ((f_x - f_y) - (delta_x - delta_y))^2.

  python wt_run.py OUT SPEC.json
"""
import json, os, sys, time
import numpy as np
from wt_common import Setup, log, dump, N, f64, BULK_CUT, SS
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from jax.flatten_util import ravel_pytree
jax.config.update('jax_default_matmul_precision', 'highest')   # full fp32

OUT = sys.argv[1]
SPEC = json.load(open(sys.argv[2]))
os.makedirs(OUT, exist_ok=True)
T00 = time.time()
AR = jnp.arange(N, dtype=jnp.uint64)
F = os.path.join(OUT, 'wt.json')


def bits(Sx):
    return (((Sx[:, None] >> AR[None, :]) & jnp.uint64(1)).astype(jnp.float32) * 2 - 1)


# ============================================================================================ networks
NBR = jnp.asarray(np.array([[((x + dx) % 6) + 6 * ((y + dy) % 6) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
                            for y in range(6) for x in range(6)], np.int32))


class ResCNN(nn.Module):
    """writeback net: periodic residual CNN, site-summed features, zero-initialised head.
    use_a: concatenate a Fourier embedding of the standardised base log-amplitude t and use a 2-layer MLP head."""
    C: int = 32
    layers: int = 4
    use_a: bool = False
    hid: int = 64

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
            tt = t[:, None]
            emb = jnp.concatenate([tt, 0.1 * tt * tt, jnp.sin(om * tt), jnp.cos(om * tt)], 1)
            h = jnp.concatenate([h, emb], 1)
            h = nn.gelu(nn.Dense(self.hid, **kw)(h))
            h = nn.gelu(nn.Dense(self.hid, **kw)(h))
        return nn.Dense(1, kernel_init=nn.initializers.zeros, **kw)(h)[:, 0]


class Corr:
    def __init__(self, C, layers, use_a=False, seed=0, head_init=0.0):
        self.net = ResCNN(C, layers, use_a)
        params = self.net.init(jax.random.PRNGKey(seed), jnp.zeros((1, N), jnp.float32), jnp.zeros(1, jnp.float32))['params']
        if head_init:
            last = sorted(params.keys(), key=lambda k: int(k.split('_')[1]) if k.startswith('Dense_') else -1)[-1]
            k = params[last]['kernel']
            params[last]['kernel'] = head_init * jax.random.normal(jax.random.PRNGKey(seed + 7), k.shape, k.dtype)
        self.flat0, self.unravel = ravel_pytree(params)
        self.npar = int(self.flat0.size)
        inv = jnp.asarray(np.argsort(SS.space_group()[:8], axis=1))
        net, unravel = self.net, self.unravel
        sg = jnp.asarray([1.0, -1.0], jnp.float32)

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


# ============================================================================================ exact setup
S = Setup(); sec = S.sec
lP, sP, lphi, u, p, lpc = S.lP, S.sP, S.lphi, S.u, S.p, S.lpc
res = dict(spec=SPEC, G0=S.G0, Qdelta=S.Qdelta, E_FN_dE_site=(S.Efn - sec.E0) / N)
dl = lP - lphi + S.mu                                         # log g at f = 0: e = f + dl (up to a constant)
MASKS = jnp.asarray(SS.MASKS); BI = jnp.asarray(SS.BI); BJ = jnp.asarray(SS.BJ); JB = jnp.asarray(SS.JB)
NB = int(MASKS.shape[0])
# standardised base log-amplitude for the '+a' input (moments under the gain/mass mixture measure)
meas = 0.5 * p + 0.5 * S.c_delta / jnp.sum(S.c_delta)
ma = float(jnp.sum(meas * lP)); sa = float(jnp.sqrt(jnp.sum(meas * (lP - ma) ** 2)))
TA = ((lP - ma) / sa).astype(jnp.float32)
res['a_standardisation'] = dict(mean=ma, sd=sa)
log_old = lambda beta: beta * (lpc - jnp.max(lpc)) + jnp.log(sec.n)       # rep-level log weight of phi^(2 beta)


def make_q(prop, beta, eps, ctail, bulk_only):
    """rep-level proposal q (normalised) and log importance weights log(p/q)."""
    q = jnp.exp(log_old(beta)); q = q / jnp.sum(q)
    if prop in ('mix', 'amix'):
        q = (1 - eps) * q + eps * ctail / jnp.sum(ctail)
    if bulk_only:
        q = jnp.where(S.bulk, q, 0.0); q = q / jnp.sum(q)
    liw = jnp.log(jnp.maximum(p, 1e-300)) - jnp.log(jnp.maximum(q, 1e-300))
    liw = jnp.where(q > 0, liw, -jnp.inf)
    return jnp.cumsum(q), liw


def make_batch(B, K, bonds):
    @jax.jit
    def batch(key, cdf, liw):
        k1, k2 = jax.random.split(key)
        uu = jax.random.uniform(k1, (B,), f64) * cdf[-1]
        idx = jnp.clip(jnp.searchsorted(cdf, uu), 0, sec.D - 1)
        lw = liw[idx]; iw = jnp.exp(lw - jnp.max(lw))
        x = sec.reps[idx]
        valid = (((x[:, None] >> BI[None, :]) ^ (x[:, None] >> BJ[None, :])) & jnp.uint64(1)).astype(bool)
        if bonds == 'uniform':
            nval = valid.sum(1)
            sc = jnp.where(valid, jax.random.uniform(k2, valid.shape), -1.0)
            _, bsel = jax.lax.top_k(sc, K)
            ok = jnp.take_along_axis(valid, bsel, 1)
            y = x[:, None] ^ MASKS[bsel]
            iy = SS.canon(sec.T, sec.reps, y.reshape(-1)).reshape(B, K)
            allowed = ok & (sP[idx][:, None] * sP[iy] < 0)
            w = jnp.where(allowed, 0.5 * JB[bsel] * jnp.exp(lphi[iy] - lphi[idx][:, None]), 0.0) * (nval[:, None] / K)
        else:                                                  # draws proportional to the FN edge weight
            yall = jnp.where(valid, x[:, None] ^ MASKS[None, :], x[:, None])
            iya = SS.canon(sec.T, sec.reps, yall.reshape(-1)).reshape(B, NB)
            allowed = valid & (sP[idx][:, None] * sP[iya] < 0)
            wa = jnp.where(allowed, 0.5 * JB[None, :] * jnp.exp(lphi[iya] - lphi[idx][:, None]), 0.0)
            Wx = wa.sum(1)
            lg = jnp.where(allowed, jnp.log(jnp.maximum(wa, 1e-300)), -jnp.inf)
            lg = jnp.where(Wx[:, None] > 0, lg, 0.0)
            bsel = jax.random.categorical(k2, lg[:, None, :], axis=-1, shape=(B, K))
            y = x[:, None] ^ MASKS[bsel]
            iy = jnp.take_along_axis(iya, bsel, 1)
            w = jnp.where(Wx[:, None] > 0, Wx[:, None] / K, 0.0) * jnp.ones((B, K))
        return x, y, dl[idx], dl[iy], w, iw, TA[idx], TA[iy]
    return batch


def losses(model, flat, x, y, dlx, dly, w, iw, tx, ty, G=None):
    B, K = y.shape
    X = bits(jnp.concatenate([x, y.reshape(-1)])); T = jnp.concatenate([tx, ty.reshape(-1)])
    if G is None:
        fa = model.f(flat, X, T).astype(f64)
    else:
        fa = model.f_aug(flat, X, T, jnp.concatenate([G, jnp.repeat(G, K)])).astype(f64)
    fx, fy = fa[:B], fa[B:].reshape(B, K)
    iw = iw / jnp.mean(iw)
    ex = fx + dlx; ey = fy + dly
    gx = jnp.exp(ex); gy = jnp.exp(ey)
    den = jnp.mean(iw * gx * gx)
    LE = 0.5 * jnp.mean(iw * jnp.sum(w * (gx[:, None] - gy) ** 2, 1)) / den / N
    LQ = 0.5 * jnp.mean(iw * jnp.sum(w * (ex[:, None] - ey) ** 2, 1)) / N
    return LE, LQ


def exact_eval(model, flat, final=False):
    fb = model.table(flat, sec.reps, TA)
    la = lP + fb
    ef = sec.fn_rayleigh(lP, sP, la)
    e = fb - S.delta
    ce = S.node_c(e)
    Qe = float(jnp.sum(ce)) / N
    out = dict(frozenFN_minus_EFN_site=(ef - S.Efn) / N, frac_gain=(S.E_f0 - ef) / (S.E_f0 - S.Efn),
               quad_frac=1 - Qe / S.Qdelta,
               rms_f=float(jnp.sqrt(jnp.sum(p * (fb - jnp.sum(p * fb)) ** 2))),
               rms_err=float(jnp.sqrt(jnp.sum(p * (e - jnp.sum(p * e)) ** 2))))
    out['captured_bulk_tail'] = S.bulk_tail(S.c_delta - ce, tot=S.Qdelta * N)
    if final:
        out['captured_decades'] = S.decades(S.c_delta - ce, tot=S.Qdelta * N)
        out['residual_decades'] = S.decades(ce, tot=S.Qdelta * N)
        out['H_ownsign'] = (sec.energy(sec.vec(la, sP)) - sec.E0) / N
    del fb, la, e, ce
    return out


# ============================================================================================ estimator check
if SPEC.get('check', False):
    chk = []
    for hi in (0.0, 0.05):
        mdl = Corr(32, 2, seed=3, head_init=hi)
        ex = exact_eval(mdl, mdl.flat0)
        ex_LE = ex['frozenFN_minus_EFN_site']; ex_LQ = S.Qdelta * (1 - ex['quad_frac'])
        lf = jax.jit(lambda fl, *a: losses(mdl, fl, *a))
        for prop, bonds in (('old', 'uniform'), ('mix', 'uniform'), ('mix', 'weighted'), ('old', 'weighted')):
            cdf, liw = make_q(prop, 0.5, 0.5, S.c_delta, False)
            bt = make_batch(1024, 8, bonds)
            est = np.array([[float(q) for q in lf(mdl.flat0, *bt(jax.random.PRNGKey(500 + i), cdf, liw))] for i in range(64)])
            r = dict(head_init=hi, prop=prop, bonds=bonds, LE_est=est[:, 0].mean(), LE_se=est[:, 0].std() / 8,
                     LE_exact=ex_LE, LQ_est=est[:, 1].mean(), LQ_se=est[:, 1].std() / 8, LQ_exact=ex_LQ)
            chk.append(r); log('CHECK', r)
    res['check'] = chk
    dump(F, res)

# ============================================================================================ arms
res['arms'] = []
for arm in SPEC['arms']:
    tag = arm['tag']; t0 = time.time()
    model = Corr(arm['C'], arm['layers'], arm.get('use_a', False), seed=arm.get('seed', 0))
    B, K, steps = arm.get('B', 1024), arm.get('K', 8), arm['steps']
    prop = arm['prop']; beta = arm.get('beta', 0.5); eps = arm.get('eps', 0.5); bulk_only = arm.get('bulk_only', False)
    lossk = arm['loss']
    sched = optax.warmup_cosine_decay_schedule(0.0, arm['lr'], arm.get('warmup', 200), steps, arm['lr'] * 0.02)
    opt = optax.adam(sched)
    flat = model.flat0; ost = opt.init(flat)
    cdf, liw = make_q(prop, beta, eps, S.c_delta, bulk_only)
    bt = make_batch(B, K, arm.get('bonds', 'uniform'))

    @jax.jit
    def step(fl, ost, key, cdf, liw):
        k1, k2 = jax.random.split(key)
        bd = bt(k1, cdf, liw)
        G = jax.random.randint(k2, (B,), 0, 16)

        def L(q):
            LE, LQ = losses(model, q, *bd, G=G)
            return (LE if lossk == 'energy' else LQ), (LE, LQ)
        (l, (LE, LQ)), g = jax.value_and_grad(L, has_aux=True)(fl)
        upd, ost = opt.update(g, ost, fl)
        return optax.apply_updates(fl, upd), ost, LE, LQ

    # common validation for every arm: static tail mixture, weighted bonds, exact frozen-FN identity (LE)
    vcdf, vliw = make_q('mix', 0.5, 0.5, S.c_delta, False)
    vb = make_batch(1024, 8, 'weighted')
    vkeys = [jax.random.PRNGKey(90_000 + i) for i in range(32)]
    vl = jax.jit(lambda fl, *a: losses(model, fl, *a))

    def val(fl):
        return np.array([[float(q) for q in vl(fl, *vb(k, vcdf, vliw))] for k in vkeys]).mean(0)
    log(f'== arm {tag}: prop {prop} bonds {arm.get("bonds", "uniform")} loss {lossk} C {arm["C"]} '
        f'use_a {arm.get("use_a", False)} bulk_only {bulk_only} npar {model.npar} B {B} K {K} steps {steps}')
    v = val(flat)
    acc = [dict(step=0, val_LE=float(v[0]), val_LQ=float(v[1]), sec=0.0)]
    log('  val', acc[-1])
    best = (v[0], flat, 0); hist = []
    key = jax.random.PRNGKey(1000 + arm.get('seed', 0))
    for it in range(1, steps + 1):
        key, k = jax.random.split(key)
        flat, ost, LE, LQ = step(flat, ost, k, cdf, liw)
        if prop == 'amix' and it % arm.get('refresh', 1000) == 0 and it < steps:
            fb1 = model.table(flat, sec.reps, TA, sym=False)          # one image: proposal only, not an evaluation
            ce = S.node_c(fb1 - S.delta)
            cdf, liw = make_q(prop, beta, eps, ce + 1e-3 * S.c_delta, bulk_only)
            del fb1, ce
        if it % arm.get('val_every', 1000) == 0 or it == steps:
            v = val(flat)
            acc.append(dict(step=it, val_LE=float(v[0]), val_LQ=float(v[1]), sec=time.time() - t0))
            log('  val', acc[-1])
            if v[0] < best[0]: best = (v[0], flat, it)
        if arm.get('eval_every') and it % arm['eval_every'] == 0 and it != steps:
            ev = exact_eval(model, flat); ev.update(step=it, sec=time.time() - t0)
            hist.append(ev); log('  exact', ev)
    t_train = time.time() - t0
    np.save(os.path.join(OUT, f'params_{tag}_last.npy'), np.asarray(flat))
    np.save(os.path.join(OUT, f'params_{tag}_best.npy'), np.asarray(best[1]))
    fin_last = exact_eval(model, flat, final=True)
    fin_best = fin_last if best[2] == steps else exact_eval(model, best[1], final=True)
    hist.append(dict(fin_last, step=steps))
    r = dict(tag=tag, arm=arm, npar=model.npar, hist=hist, val=acc, final_last=fin_last, final_best=fin_best,
             best_step=best[2], train_sec=t_train, sec=time.time() - t0)
    res['arms'].append(r)
    log(f'== done {tag}', json.dumps(dict(last={k: v for k, v in fin_last.items() if 'decades' not in k},
                                          best={k: v for k, v in fin_best.items() if 'decades' not in k})))
    dump(F, res)
    del model, flat, ost, best
res['sec'] = time.time() - T00
dump(F, res)
log('DONE', res['sec'])
