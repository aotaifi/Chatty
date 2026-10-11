"""P3: continue the FN/Krylov loop for n iterations from a DISTILLED guide (student table la_S, sign s_k).

Each iteration is exactly wt_loop.py (arm F-SI-Adam): guide-only V, W, semi-implicit target T, guide-metric tail
proposal, fresh FEAT CNN on the CURRENT guide's (log a, V, W), edge least squares, Adam 20k steps x 256 x 8 bonds,
written back as la + f, Krylov sign step, exact referee (frac, <H>, E_FN of the next guide).
  python cx_loop.py OUT SPEC.json      SPEC: student_dir, k, n_loop, steps, seed
"""
import os, sys, time
import numpy as np
from cx_lib import *
import optax
from lanczos_lib import lanczos_ritz

OUT = sys.argv[1]; SPEC = json.load(open(sys.argv[2])); os.makedirs(OUT, exist_ok=True)
F = os.path.join(OUT, 'cx_loop.json'); T00 = time.time()
lab = Lab(); sec = lab.sec
K_IT = SPEC['k']
_, s = load_guide(K_IT)
la = jnp.asarray(np.load(os.path.join(SPEC['student_dir'], 'la_student.npy')))
res = dict(spec=SPEC, iters=[])


def fit(model, la, s, CDF, LIW, DLR, steps, seed, B=256, K=8, lr=3e-3):
    """wt_loop.fit, edge loss: sum_kept |H_xy| a_y/a_x ((f_x + DLR_x) - (f_y + DLR_y))^2, guide-metric weights."""
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
        return x, y, DLR[idx], DLR[iy], wgt, iw, lab.TIDX[idx], lab.TIDX[iy]

    def loss(q, x, y, dlx, dly, wgt, iw, tx, ty, G):
        iw = iw / jnp.mean(iw)
        X = bits(jnp.concatenate([x, y.reshape(-1)])); Tt = jnp.concatenate([tx, ty.reshape(-1)])
        fa = model.f_aug(q, X, Tt, jnp.concatenate([G, jnp.repeat(G, K)])).astype(f64)
        fx, fy = fa[:B], fa[B:].reshape(B, K)
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
    curve = []; key = jax.random.PRNGKey(100 + seed)
    for it in range(1, steps + 1):
        key, kk = jax.random.split(key)
        flat, ost, l = step(flat, ost, kk)
        if it % 2000 == 0 or it == 1:
            curve.append((it, float(np.mean([float(vfn(flat, q)) for q in vk])))); log(f'  step {it} own val {curve[-1][1]:.4e}')
    return flat, curve


for j in range(1, SPEC.get('n_loop', 2) + 1):
    t0 = time.time(); rec = dict(it=K_IT + j)
    Efn, u, info = sec.fn_solve(la, s); del u
    rec['guide_E_FN'] = info['dE_FN_site']; rec['guide_H'] = lab.H_site(la, s)
    if j == 1:
        rl = lanczos_ritz(lambda x: sec.Hm(x), sec.vec(la, s), 1)[1]
        rec['guide_lanczos_H'] = (rl['E'] - sec.E0) / N; del rl
    lan, V, W = lab.VW(la, s)
    Eg = lab.frozen(la, s, la)
    T = lab.T_of(V, W, Eg)
    Q, PA, meas = lab.proposal_q(T, la, s)
    mu = float(jnp.sum(PA * T)); DLR = -(T - mu)
    CDF = jnp.cumsum(Q); LIW = jnp.log(jnp.maximum(PA, 1e-300)) - jnp.log(jnp.maximum(Q, 1e-300))
    rec['ideal_gain_site'] = (Eg - Efn) / N
    rec['SI_target_frac'] = (Eg - lab.frozen(la, s, la + T)) / (Eg - Efn)
    FEAT = lab.feat_table(lan, V, W, meas); del lan, V, W, meas, Q, PA
    model = FeatNet(FEAT, seed=K_IT + j + 1000 * SPEC.get('seed', 0))
    sec.offload()
    flat, rec['fit_curve'] = fit(model, la, s, CDF, LIW, DLR, SPEC.get('steps', 20000), K_IT + j + 1000 * SPEC.get('seed', 0))
    sec.reload()
    np.save(os.path.join(OUT, f'params_it{K_IT + j}.npy'), np.asarray(flat))
    fb = model.table(flat, sec.reps, lab.TIDX)
    la_new = la + fb; del fb
    rec['frac'] = (Eg - lab.frozen(la, s, la_new)) / (Eg - Efn)
    s_new, _, _ = sec.krylov(sec.vec(la_new, jnp.ones_like(la_new)), s)
    rec['H_kry'] = lab.H_site(la_new, s_new)
    _, u, i2 = sec.fn_solve(la_new, s_new); del u
    rec['E_FN_next'] = i2['dE_FN_site']
    rl = lanczos_ritz(lambda x: sec.Hm(x), sec.vec(la_new, s_new), 1)[1]
    rec['lanczos_H_next'] = (rl['E'] - sec.E0) / N; del rl
    rec['sec'] = time.time() - t0
    log(f'== it {K_IT + j}: frac {rec["frac"]:.4f}  <H> {rec["H_kry"]:.4e}  E_FN {rec["E_FN_next"]:.4e}  '
        f'+Lanczos {rec["lanczos_H_next"]:.4e}  ({rec["sec"]:.0f}s)')
    res['iters'].append(rec); dump(F, res)
    la, s = la_new, s_new
    del model, flat, FEAT, T, DLR, CDF, LIW
res['sec'] = time.time() - T00; dump(F, res); log('DONE', res['sec'])
