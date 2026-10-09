"""Test 3: AMORTISED V/W. For the table-loop guides la_k (k = 3, 6) train small nets that predict the guide's own
one-hop quantities u_V = log(V_k + 1e-6), u_W = log(W_k + 1e-6) (exactly the FEAT input transform) from
(spins, log a_P, V_P, W_P of the frozen psi_P), i.e. from ONE hop of the base, then replace the exact V_k, W_k by the
predictions in the next write-back step (stored wtLOOP8 net k+1, net input AND semi-implicit target) and measure the
capture.  Net: the FEAT residual CNN (C = 32, 4 layers, 2-layer MLP head with the base-feature embedding), 2 outputs,
exactly D4 x flip symmetric at evaluation, one random image per sample in training.  Loss: standardised squared error
of (u_V, u_W) weighted by the FEAT standardisation measure (1/2 a_k^2 + 1/2 Dirichlet node weight of T_k), samples
from the guide-metric tail proposal of T_k, 20% of orbits held out.  Adam 3e-3, 20k steps x 512.
  python cx_vw.py OUT SPEC.json        SPEC: ks, seed, steps
"""
import os, sys, time
import numpy as np
from cx_lib import *
import optax

OUT = sys.argv[1]; SPEC = json.load(open(sys.argv[2])); os.makedirs(OUT, exist_ok=True)
F = os.path.join(OUT, 'cx_vw.json'); T00 = time.time()
lab = Lab(); sec = lab.sec
FEAT_BASE = base_features(lab)
REFS = json.load(open(os.path.join(TABDIR, 'cx_replay.json')))['refs']
res = dict(spec=SPEC, ks=[])


class ResCNN2(nn.Module):
    C: int = 32
    layers: int = 4
    hid: int = 64
    nout: int = 2

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
        om = jnp.asarray([0.25, 0.5, 1.0, 2.0, 4.0], jnp.float32)
        B_ = t.shape[0]
        emb = jnp.concatenate([t, 0.1 * t * t, jnp.sin(t[:, :, None] * om).reshape(B_, -1),
                               jnp.cos(t[:, :, None] * om).reshape(B_, -1)], 1)
        h = jnp.concatenate([h, emb], 1)
        h = nn.gelu(nn.Dense(self.hid, **kw)(h))
        h = nn.gelu(nn.Dense(self.hid, **kw)(h))
        return nn.Dense(self.nout, **kw)(h) + t[:, 1:3]               # start from the base's own (u_V, u_W)


SEED = SPEC.get('seed', 0); STEPS = SPEC.get('steps', 20000); B = SPEC.get('B', 512)
for k in SPEC.get('ks', [3, 6]):
    t0 = time.time(); rec = dict(k=k)
    la, s = load_guide(k)
    ctx = GuideCtx(lab, la, s, load_loop_params(k + 1), sec.E0 + N * REFS[str(k)]['E_FN_stack'])
    rec['frac_exact'], rec['SI_exact'] = ctx.step()
    # zero-training baseline: the base's own V_P, W_P in place of V_k, W_k
    _, VP, WP = lab.VW(lab.lP0, lab.sP0)
    rec['frac_baseVW'], rec['SI_baseVW'] = ctx.step((ctx.lan, VP, WP))
    rec['err_baseVW'] = ctx.vw_err(VP, WP); del VP, WP
    log(f'[k={k}] exact frac {rec["frac_exact"]:.4f}  with V_P, W_P: {rec["frac_baseVW"]:.4f} (err {rec["err_baseVW"]})')
    uV = jnp.log(jnp.maximum(ctx.V, 0) + 1e-6); uW = jnp.log(jnp.maximum(ctx.W, 0) + 1e-6)
    # FEAT_BASE columns are standardised (log a_P, log V_P, log W_P); standardise the targets the same way
    m = ctx.meas
    st = [(float(jnp.sum(m * q)), float(jnp.sqrt(jnp.sum(m * (q - jnp.sum(m * q)) ** 2)))) for q in (uV, uW)]
    TGT = jnp.stack([(uV - st[0][0]) / st[0][1], (uW - st[1][0]) / st[1][1]], 1).astype(jnp.float32)
    # residual start: the head adds the base's standardised (log V_P, log W_P); re-express the target relative to it
    T = lab.T_of(ctx.V, ctx.W, ctx.Eg)
    Q, PA, _ = lab.proposal_q(T, la, s); del T
    Q = jnp.where(lab.HO, 0.0, Q); Q = Q / jnp.sum(Q); CDF = jnp.cumsum(Q)
    LW = jnp.where(Q > 0, jnp.log(jnp.maximum(m, 1e-300)) - jnp.log(jnp.maximum(Q, 1e-300)), -jnp.inf); del Q
    net = ResCNN2()
    p0 = net.init(jax.random.PRNGKey(SEED + 10 * k), jnp.zeros((1, N), jnp.float32), jnp.zeros((1, 3), jnp.float32))['params']
    flat0, unravel = ravel_pytree(p0)
    fa = lambda fl, X, ft, G: net.apply({'params': unravel(fl)}, jnp.take_along_axis(X, INV8[G // 2], axis=1) * SG2[G % 2][:, None], ft)

    def fsym(fl, X, ft):
        return jax.lax.map(lambda q: net.apply({'params': unravel(fl)}, X[:, INV8[q // 2]] * SG2[q % 2], ft), jnp.arange(16)).mean(0)
    sched = optax.warmup_cosine_decay_schedule(0.0, 3e-3, 200, STEPS, 6e-5)
    opt = optax.adam(sched); flat = flat0; ost = opt.init(flat)

    @jax.jit
    def batch(key, cdf):
        idx = jnp.clip(jnp.searchsorted(cdf, jax.random.uniform(key, (B,), f64) * cdf[-1]), 0, sec.D - 1)
        lw = LW[idx]
        return sec.reps[idx], FEAT_BASE[idx], TGT[idx], jnp.exp(lw - jnp.max(lw))

    def loss(fl, x, ft, tg, iw, G):
        pr = fa(fl, bits(x), ft, G)
        iw = iw / jnp.mean(iw)
        return jnp.mean(iw[:, None] * (pr - tg) ** 2)

    @jax.jit
    def step(fl, ost, key):
        k1, k2 = jax.random.split(key)
        bd = batch(k1, CDF); G = jax.random.randint(k2, (B,), 0, 16)
        l, g = jax.value_and_grad(loss)(fl, *bd, G)
        upd, ost = opt.update(g, ost, fl)
        return optax.apply_updates(fl, upd), ost, l
    vb = [batch(jax.random.PRNGKey(5000 + i), CDF) for i in range(8)]
    vloss = jax.jit(lambda fl, x, ft, tg, iw: loss(fl, x, ft, tg, iw, jnp.zeros(B, jnp.int32)))
    curve = []; key = jax.random.PRNGKey(200 + SEED); sec.offload()
    for it in range(0, STEPS + 1):
        if it % 2000 == 0:
            curve.append((it, float(np.mean([float(vloss(flat, *b)) for b in vb])))); log(f'  [vw k={k}] step {it} val {curve[-1][1]:.4e}')
        if it == STEPS: break
        key, kk = jax.random.split(key)
        flat, ost, l = step(flat, ost, kk)
    sec.reload(); rec['curve'] = curve
    tab = jax.jit(lambda fl, Sx, ft: fsym(fl, bits(Sx), ft))
    out = []
    for i in range(0, sec.D, 16384):
        ii = jnp.minimum(jnp.arange(i, i + 16384), sec.D - 1)
        out.append(tab(flat, sec.reps[ii], FEAT_BASE[ii])[:min(16384, sec.D - i)])
    P = jnp.concatenate(out).astype(f64); del out
    Vh = jnp.maximum(jnp.exp(P[:, 0] * st[0][1] + st[0][0]) - 1e-6, 0.0)
    Wh = jnp.maximum(jnp.exp(P[:, 1] * st[1][1] + st[1][0]) - 1e-6, 0.0); del P
    rec['err_hat'] = ctx.vw_err(Vh, Wh)
    for split, msk in (('train', ~lab.HO), ('test', lab.HO)):         # held-out split of the PA-weighted errors
        pa = jnp.where(msk, ctx.PA, 0.0)
        ev = jnp.log(jnp.maximum(Vh, 0) + 1e-6) - uV; ew = jnp.log(jnp.maximum(Wh, 0) + 1e-6) - uW
        rec[f'u_rms_{split}'] = (float(jnp.sqrt(jnp.sum(pa * ev * ev) / jnp.sum(pa))), float(jnp.sqrt(jnp.sum(pa * ew * ew) / jnp.sum(pa))))
    rec['frac_hat'], rec['SI_hat'] = ctx.step((ctx.lan, Vh, Wh))
    rec['capture_rel'] = rec['frac_hat'] / rec['frac_exact']
    np.save(os.path.join(OUT, f'params_vw_k{k}.npy'), np.asarray(flat))
    rec['sec'] = time.time() - t0
    log(f'== k={k}: frac exact {rec["frac_exact"]:.4f}  V_P,W_P {rec["frac_baseVW"]:.4f}  amortised {rec["frac_hat"]:.4f} '
        f'(rel {rec["capture_rel"]:.3f})  err logV/logW {rec["err_hat"]}  u-rms train/test {rec["u_rms_train"]} {rec["u_rms_test"]}')
    res['ks'].append(rec); dump(F, res)
    del ctx, la, s, Vh, Wh, uV, uW, TGT, CDF, LW
res['sec'] = time.time() - T00; dump(F, res); log('DONE', res['sec'])
