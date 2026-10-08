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


class VitModel:
    """the symmetrised ViT itself, b = |psi_P,theta| (no residual factor): psi_P = sum over the 16 D4 x flip images of
    the complex ViT (with its built-in 2x2 patch translations), exactly as sym_tables.npz (stall_6x6 test 0) was built.
    f = log|psi_P,theta(x)| - log a(x) (a = the frozen table, recovered from the standardised input t)."""

    def __init__(self, ckpt):
        import flax
        for p_ in (os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'learned_loop_6x6'),):
            if p_ not in sys.path: sys.path.append(p_)
        import vit_dt
        vit_dt.set_dtype('float32')
        vit = vit_dt.make_model_6x6()
        tmpl = vit.init(jax.random.PRNGKey(1234), jnp.zeros((1, N), jnp.float32))
        obj = flax.serialization.msgpack_restore(open(ckpt, 'rb').read())
        if 'variables' in obj: obj = obj['variables']
        tmpl = flax.serialization.from_state_dict(tmpl, obj)
        params = jax.tree_util.tree_map(lambda q: jnp.asarray(q, jnp.float32), tmpl['params'])
        self.flat0, self.unravel = ravel_pytree(params)
        self.npar = int(self.flat0.size)
        inv = jnp.asarray(np.argsort(SS.space_group()[:8], axis=1))
        sg = jnp.asarray([1.0, -1.0], jnp.float32)
        unravel = self.unravel

        def f(flat, X, t):
            p = {'params': unravel(flat)}
            zs = jax.lax.map(lambda k: vit_dt.logpsi_transl_2d(vit.apply, 2, p, X[:, inv[k // 2]] * sg[k % 2]),
                             jnp.arange(16))
            m = jnp.max(jnp.real(zs), 0)
            Ssum = jnp.sum(jnp.exp(zs - m[None, :]), 0)
            return m + jnp.log(jnp.abs(Ssum)) - (t * SA + MA)
        self.f = f
        self.f_aug = lambda flat, X, t, G: f(flat, X, t)
        self._tab = jax.jit(lambda flat, Sx, t: f(flat, bits(Sx), t))

    def table(self, flat, reps, T, batch=16384, sym=True):
        D = reps.shape[0]; out = []
        for i in range(0, D, batch):
            s = reps[i:i + batch]; t = T[i:i + batch]; m = s.shape[0]
            if m < batch:
                s = jnp.concatenate([s, jnp.repeat(s[:1], batch - m)]); t = jnp.concatenate([t, jnp.repeat(t[:1], batch - m)])
            out.append(self._tab(flat, s, t)[:m].astype(f64))
        return jnp.concatenate(out)


# ============================================================================================ exact setup
S = Setup(SPEC.get('n_iter', 1)); sec = S.sec
lP, sP, lphi, u, p, lpc = S.lP, S.sg, S.lphi, S.u, S.p, S.lpc     # sP here = sign of the frozen-FN guide
res = dict(spec=SPEC, G0=S.G0, Qdelta=S.Qdelta, E_FN_dE_site=(S.Efn - sec.E0) / N, loop=S.loop)
dl = lP - lphi + S.mu                                         # log g at f = 0: e = f + dl (up to a constant)
MASKS = jnp.asarray(SS.MASKS); BI = jnp.asarray(SS.BI); BJ = jnp.asarray(SS.BJ); JB = jnp.asarray(SS.JB)
NB = int(MASKS.shape[0])
# standardised base log-amplitude for the '+a' input (moments under the gain/mass mixture measure)
meas = 0.5 * p + 0.5 * S.c_delta / jnp.sum(S.c_delta)
ma = float(jnp.sum(meas * lP)); sa = float(jnp.sqrt(jnp.sum(meas * (lP - ma) ** 2)))
MA, SA = ma, sa
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


def exact_eval(model, flat, final=False, fn_final=False):
    fb = model.table(flat, sec.reps, TA)
    la = lP + fb
    ef = sec.fn_rayleigh(S.lg, S.sg, la)
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
        out['H_ownsign'] = (sec.energy(sec.vec(la, S.sP)) - sec.E0) / N        # psi_P sign
        if S.n_iter > 1 or fn_final:
            out['H_targetsign'] = (sec.energy(sec.vec(la, S.sT)) - sec.E0) / N  # Krylov sign of the target
            _, _, i1 = sec.fn_solve(la, S.sT); out['FN_targetsign'] = i1['dE_FN_site']
    del fb, la, e, ce
    return out


def polish(model, flat, pol):
    """fixed-sign VMC with minSR on <H>(b, sT), b = |psi_P| exp(f): no phi_FN anywhere. Samples from the exact b^2 table
    at the polish start (reweighted by exp(2 (f - f_start)) afterwards), full local energies over all valid bonds."""
    Bp = pol.get('B', 512); eta = pol.get('eta', 0.02); lam = pol.get('lam', 1e-3)
    sE = S.sT
    ftab = model.table(flat, sec.reps, TA)
    la0 = lP + ftab; del ftab
    cdf = jnp.cumsum(jnp.exp(2 * (la0 - jnp.max(la0))) * sec.n); del la0
    flat0 = flat

    def fi(q, xi, ti, gi):
        return model.f_aug(q, xi[None], ti[None], gi[None])[0]
    jac = jax.vmap(jax.grad(fi), in_axes=(None, 0, 0, 0))

    @jax.jit
    def pstep(fl, key):
        k1, k2 = jax.random.split(key)
        idx = jnp.clip(jnp.searchsorted(cdf, jax.random.uniform(k1, (Bp,), f64) * cdf[-1]), 0, sec.D - 1)
        x = sec.reps[idx]
        valid = (((x[:, None] >> BI[None, :]) ^ (x[:, None] >> BJ[None, :])) & jnp.uint64(1)).astype(bool)
        yall = jnp.where(valid, x[:, None] ^ MASKS[None, :], x[:, None])
        iya = SS.canon(sec.T, sec.reps, yall.reshape(-1)).reshape(Bp, NB)
        G = jax.random.randint(k2, (Bp,), 0, 16)
        Xa = bits(jnp.concatenate([x, yall.reshape(-1)])); Ta = jnp.concatenate([TA[idx], TA[iya].reshape(-1)])
        GG = jnp.concatenate([G, jnp.repeat(G, NB)])
        fa = model.f_aug(fl, Xa, Ta, GG).astype(f64)
        fx, fy = fa[:Bp], fa[Bp:].reshape(Bp, NB)
        fx0 = model.f_aug(flat0, bits(x), TA[idx], G).astype(f64)
        lx = lP[idx] + fx; ly = lP[iya] + fy
        d0 = 0.25 * JB.sum() - 0.5 * (valid * JB[None, :]).sum(1)
        EL = d0 + jnp.sum(jnp.where(valid, 0.5 * JB[None, :] * sE[idx][:, None] * sE[iya]
                                    * jnp.exp(jnp.clip(ly - lx[:, None], -60, 60)), 0.0), 1)
        lr_ = 2 * (fx - fx0); rho = jnp.exp(lr_ - jnp.max(lr_)); rho = rho / jnp.sum(rho)
        Eb = jnp.sum(rho * EL)
        eps_ = jnp.sqrt(rho) * (EL - Eb)
        O = jac(fl, bits(x), TA[idx], G)
        r32 = rho.astype(jnp.float32)
        Y = jnp.sqrt(r32)[:, None] * (O - (r32 @ O)[None, :])
        Tm = (Y @ Y.T).astype(f64)
        alpha = jnp.linalg.solve(Tm + (lam * jnp.trace(Tm) / Bp + 1e-30) * jnp.eye(Bp), eps_)
        return fl - eta * (Y.T @ alpha.astype(jnp.float32)), Eb
    key = jax.random.PRNGKey(4242); trace = []
    for it in range(1, pol['steps'] + 1):
        key, k = jax.random.split(key)
        flat, Eb = pstep(flat, k)
        trace.append((float(Eb) - sec.E0) / N)
        if it % 50 == 0: log(f'  polish {it}: VMC <H> - E0 (per site, mean of last 50) {np.mean(trace[-50:]):.4e}')
    return flat, trace


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
    if arm.get('model') == 'vit':
        model = VitModel(arm.get('ckpt', '/project/theorie/a/A.Otaifi/chatty_stall6/data/vit_J2=0.50_N=6x6_k=0.mpack'))
        ii = jax.random.randint(jax.random.PRNGKey(5), (512,), 0, sec.D)
        chk = np.asarray(jax.jit(model.f)(model.flat0, bits(sec.reps[ii]), TA[ii]))
        log(f'  ViT start check: f = log|psi_P,theta0| - log a on 512 reps: mean {chk.mean():.2e} rms {chk.std():.2e} max {np.abs(chk).max():.2e}')
        assert np.abs(chk - chk.mean()).max() < 1e-3, 'ViT does not reproduce the psi_P table'
    else:
        model = Corr(arm['C'], arm['layers'], arm.get('use_a', False), seed=arm.get('seed', 0))
    if arm.get('init_params'):                                 # e.g. polish-only runs from a saved fit
        model.flat0 = jnp.asarray(np.load(arm['init_params']), jnp.float32)
    B, K, steps = arm.get('B', 1024), arm.get('K', 8), arm['steps']
    prop = arm['prop']; beta = arm.get('beta', 0.5); eps = arm.get('eps', 0.5); bulk_only = arm.get('bulk_only', False)
    lossk = arm['loss']
    opt_kind = arm.get('opt', 'adam')
    sec.offload()                                             # H (4.2 GB) is reloaded on demand by exact evaluations
    bt = make_batch(B, K, arm.get('bonds', 'uniform'))
    cdf, liw = make_q(prop, beta, eps, S.c_delta, bulk_only)
    flat = model.flat0; ost = None
    if opt_kind == 'adam':
        sched = optax.warmup_cosine_decay_schedule(0.0, arm['lr'], arm.get('warmup', 200), steps, arm['lr'] * 0.02)
        opt = optax.adam(sched)
        ost = opt.init(flat)

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
    elif opt_kind == 'sr':
        # minSR (Chen & Heyl): rows Y_i = sqrt(rho_i) (O_i - <O>), rho = b^2/q (self-normalised), O = df/dtheta at x;
        # eps_i = sqrt(rho_i) (E_L(x_i) - <E_L>), frozen-FN local energy from the exact edge form
        # E_L(x) - E_FN = sum_{y allowed} |H_xy| (phi_y/phi_x) (1 - g_y/g_x)  (estimated from the batch bonds);
        # dtheta = -eta Y^T (Y Y^T + lam tr(YY^T)/B I)^{-1} eps.
        eta_sr = arm.get('eta', 0.02); lam_sr = arm.get('lam', 1e-3); max_df = arm.get('max_df', 0.01)

        def fi(q, xi, ti, gi):
            return model.f_aug(q, xi[None], ti[None], gi[None])[0]
        jac = jax.vmap(jax.grad(fi), in_axes=(None, 0, 0, 0))

        @jax.jit
        def step(fl, ost, key, cdf, liw):
            k1, k2 = jax.random.split(key)
            x, y, dlx, dly, w, iw, tx, ty = bt(k1, cdf, liw)
            G = jax.random.randint(k2, (B,), 0, 16)
            LE, LQ = losses(model, fl, x, y, dlx, dly, w, iw, tx, ty, G=G)
            Xa = bits(jnp.concatenate([x, y.reshape(-1)])); Ta = jnp.concatenate([tx, ty.reshape(-1)])
            fa = model.f_aug(fl, Xa, Ta, jnp.concatenate([G, jnp.repeat(G, K)])).astype(f64)
            fx, fy = fa[:B], fa[B:].reshape(B, K)
            ex = fx + dlx; ey = fy + dly
            EL = jnp.sum(w * (1.0 - jnp.exp(ey - ex[:, None])), 1)
            lr_ = jnp.log(iw) + 2 * ex
            rho = jnp.exp(lr_ - jnp.max(lr_)); rho = rho / jnp.sum(rho)
            eps_ = jnp.sqrt(rho) * (EL - jnp.sum(rho * EL))
            O = jac(fl, bits(x), tx, G)
            r32 = rho.astype(jnp.float32)
            Y = jnp.sqrt(r32)[:, None] * (O - (r32 @ O)[None, :])
            Tm = (Y @ Y.T).astype(f64)
            sh = lam_sr * jnp.trace(Tm) / B + 1e-30
            alpha = jnp.linalg.solve(Tm + sh * jnp.eye(B), eps_)
            d = Y.T @ alpha.astype(jnp.float32)
            dfr = jnp.sqrt(jnp.mean((O @ d) ** 2)) * eta_sr               # rms change of f on the batch
            sc = jnp.minimum(1.0, max_df / (dfr + 1e-30))
            return fl - eta_sr * sc * d, ost, LE, LQ
    elif opt_kind == 'gn':
        # Gauss-Newton / Levenberg-Marquardt in the energy metric: residuals r_xy = sqrt(iw_x w_xy / sum iw)
        # ((f_x - f_y) - (delta_x - delta_y)), (J^T J + lam I) d = -J^T r by n_cg CG steps (matrix-free jvp / vjp),
        # trust ratio on the same batch -> accept / reject and adapt lam.
        n_cg = arm.get('n_cg', 20)

        @jax.jit
        def gn_dir(fl, key, cdf, liw, lam):
            k1, k2 = jax.random.split(key)
            x, y, dlx, dly, w, iw, tx, ty = bt(k1, cdf, liw)
            G = jax.random.randint(k2, (B,), 0, 16)
            Xa = bits(jnp.concatenate([x, y.reshape(-1)])); Ta = jnp.concatenate([tx, ty.reshape(-1)])
            GG = jnp.concatenate([G, jnp.repeat(G, K)])
            c = jnp.sqrt(iw[:, None] / jnp.sum(iw) * w).astype(jnp.float32)
            dx = dlx.astype(jnp.float32); dy = dly.astype(jnp.float32)

            def resid(q):
                fa = model.f_aug(q, Xa, Ta, GG)
                fx, fy = fa[:B], fa[B:].reshape(B, K)
                return c * ((fx + dx)[:, None] - (fy + dy))
            r0, Jv = jax.linearize(resid, fl)
            JT = jax.linear_transpose(Jv, fl)
            b = -JT(r0)[0]

            def A(v):
                return JT(Jv(v))[0] + lam * v

            def body(i, st):
                d, r, p, rs = st
                Ap = A(p); al = rs / (p @ Ap)
                d = d + al * p; r = r - al * Ap; rsn = r @ r
                return d, r, p * (rsn / rs) + r, rsn
            d = jax.lax.fori_loop(0, n_cg, body, (jnp.zeros_like(b), b, b, b @ b))[0]
            L0 = 0.5 * jnp.sum(r0 * r0)
            pred = L0 - 0.5 * jnp.sum((r0 + Jv(d)) ** 2)
            r1 = resid(fl + d)
            act = L0 - 0.5 * jnp.sum(r1 * r1)
            return d, pred, act, L0
        lam_gn = arm.get('lam', 1e-2); n_acc = 0

        @jax.jit
        def gn_hold(f0, f1, key, cdf, liw):                          # reduction of the loss on an independent batch
            k1, k2 = jax.random.split(key)
            bd = bt(k1, cdf, liw); G = jax.random.randint(k2, (B,), 0, 16)
            return N * (losses(model, f0, *bd, G=G)[1] - losses(model, f1, *bd, G=G)[1])
    else:
        raise ValueError(opt_kind)
    max_sec = arm.get('max_sec')

    # common validation for every arm: static tail mixture, weighted bonds, exact frozen-FN identity (LE)
    vcdf, vliw = make_q('mix', 0.5, 0.5, S.c_delta, False)
    vb = make_batch(arm.get('val_B', 1024), 8, 'weighted')
    vkeys = [jax.random.PRNGKey(90_000 + i) for i in range(arm.get('val_n', 32))]
    vl = jax.jit(lambda fl, *a: losses(model, fl, *a))

    def val(fl):
        return np.array([[float(q) for q in vl(fl, *vb(k, vcdf, vliw))] for k in vkeys]).mean(0)
    log(f'== arm {tag}: opt {opt_kind} model {arm.get("model", "cnn")} prop {prop} bonds {arm.get("bonds", "uniform")} loss {lossk} C {arm.get("C")} '
        f'use_a {arm.get("use_a", False)} bulk_only {bulk_only} npar {model.npar} B {B} K {K} steps {steps}')
    v = val(flat)
    acc = [dict(step=0, val_LE=float(v[0]), val_LQ=float(v[1]), sec=0.0)]
    log('  val', acc[-1])
    best = (v[0], flat, 0); hist = []
    key = jax.random.PRNGKey(1000 + arm.get('seed', 0))
    for it in range(1, steps + 1):
        key, k = jax.random.split(key)
        if opt_kind == 'gn':
            k_a, k_b = jax.random.split(k)
            d, pred, act, L0 = gn_dir(flat, k_a, cdf, liw, jnp.float32(lam_gn))
            hv = gn_hold(flat, flat + d, k_b, cdf, liw)               # held-out batch: generalisation, not the fit
            pred, act_h = float(pred), float(hv); ratio = act_h / pred if pred > 0 else -1.0
            if ratio > 0: flat = flat + d; n_acc += 1
            lam_gn = min(lam_gn * 4, 1e4) if ratio < 0.25 else (max(lam_gn / 2, 1e-8) if ratio > 0.75 else lam_gn)
        else:
            flat, ost, LE, LQ = step(flat, ost, k, cdf, liw)
        if prop == 'amix' and it % arm.get('refresh', 1000) == 0 and it < steps:
            fb1 = model.table(flat, sec.reps, TA, sym=False)          # one image: proposal only, not an evaluation
            ce = S.node_c(fb1 - S.delta)
            cdf, liw = make_q(prop, beta, eps, ce + 1e-3 * S.c_delta, bulk_only)
            del fb1, ce
            sec.offload()
        if it % arm.get('val_every', 1000) == 0 or it == steps:
            v = val(flat)
            acc.append(dict(step=it, val_LE=float(v[0]), val_LQ=float(v[1]), sec=time.time() - t0))
            if opt_kind == 'gn': acc[-1].update(lam=lam_gn, n_acc=n_acc)
            log('  val', acc[-1])
            if v[0] < best[0]: best = (v[0], flat, it)
            if not np.isfinite(v[0]): log('  non-finite validation, stopping arm'); break
            if max_sec and time.time() - t0 > max_sec:
                log('  time limit for arm reached'); break
        if arm.get('eval_every') and it % arm['eval_every'] == 0 and it != steps:
            ev = exact_eval(model, flat); ev.update(step=it, sec=time.time() - t0)
            hist.append(ev); log('  exact', ev)
    t_train = time.time() - t0
    np.save(os.path.join(OUT, f'params_{tag}_last.npy'), np.asarray(flat))
    np.save(os.path.join(OUT, f'params_{tag}_best.npy'), np.asarray(best[1]))
    if not arm.get('final', True):                            # smoke / hyper-parameter scan: validation only
        res['arms'].append(dict(tag=tag, arm=arm, val=acc, best_step=best[2], train_sec=t_train)); dump(F, res)
        log(f'== done {tag} (no exact eval) best val_LE {best[0]:.6e} at {best[2]}')
        continue
    fnf = bool(arm.get('polish'))
    if arm.get('eval_best_only') and best[2] != it:
        fin_last = None
    else:
        fin_last = exact_eval(model, flat, final=True, fn_final=fnf) if np.all(np.isfinite(np.asarray(flat))) else dict(frac_gain=float('nan'))
    if arm.get('eval_best_only') and best[2] != it:
        fin_last = dict(skipped=True, val_LE=acc[-1]['val_LE'])
    if best[2] == 0 and not arm.get('polish'):                # validation never improved: the start (frac = 0 exactly)
        fin_best = dict(frac_gain=0.0, quad_frac=0.0, note='best = start parameters (no validated improvement)')
    else:
        fin_best = fin_last if best[2] == it else exact_eval(model, best[1], final=True, fn_final=fnf)
    hist.append(dict(fin_best, step=best[2]))
    pol_res = None
    if arm.get('polish'):
        tp = time.time()
        fl_p, trace = polish(model, best[1], arm['polish'])
        np.save(os.path.join(OUT, f'params_{tag}_polished.npy'), np.asarray(fl_p))
        pol_res = dict(trace=trace, exact=exact_eval(model, fl_p, final=True, fn_final=True), sec=time.time() - tp)
        log(f'== polished {tag}', json.dumps({k: v for k, v in pol_res['exact'].items() if 'decades' not in k}))
    r = dict(tag=tag, arm=arm, npar=model.npar, hist=hist, val=acc, final_last=fin_last, final_best=fin_best,
             best_step=best[2], steps_done=it, train_sec=t_train, polish=pol_res, sec=time.time() - t0)
    res['arms'].append(r)
    log(f'== done {tag}', json.dumps(dict(last={k: v for k, v in (fin_last or {}).items() if 'decades' not in k},
                                          best={k: v for k, v in fin_best.items() if 'decades' not in k})))
    dump(F, res)
    del model, flat, ost, best
    if opt_kind == 'gn': del gn_dir, gn_hold
res['sec'] = time.time() - T00
dump(F, res)
log('DONE', res['sec'])
