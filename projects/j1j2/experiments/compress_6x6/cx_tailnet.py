"""Route B: spin-only tail representation (README "Route B", pre-registered before any run).
  mode 'G0': linearised (input-gradient) log-ratios / E_L of ViT_1 vs exact, on target-state and tail-proposal samples.
  mode 'T1': additive tail net T on top of ViT_1 (table of ra_ms_s1_k1), fit to la_t1 - la_A; arms TS (spins + la_A,
             C = 64), TS32 (C = 32), TG (+ linearised log V^, log W^, asinh E^_L; needs the input gradient on all orbits).
  python cx_tailnet.py OUT SPEC.json    SPEC: {"mode": "G0" | "T1", "arms": [...]}
"""
import os, sys, time
import numpy as np
from scipy.optimize import minimize_scalar
from cx_lib import *
import flax, vit_dt
OUT = sys.argv[1]; SPEC = json.load(open(sys.argv[2])); os.makedirs(OUT, exist_ok=True)
MODE = SPEC['mode']; EPS = 0.3
VDIR = '/project/theorie/a/A.Otaifi/Chatty/runs/compress6/runs/ra_ms_s1_k1_17174477'
F = os.path.join(OUT, f'cx_tailnet_{MODE}.json'); T00 = time.time()
lab = Lab(); sec = lab.sec
res = dict(spec=SPEC)

# ---------------------------------------------------------------- k = 1 target and ViT_1
la_P, s_P = lab.lP0, lab.sP0
_, V, W = lab.VW(la_P, s_P); v = sec.vec(la_P, s_P); E = float(v @ sec.H(v)); del v
EL = lab.D0 + V - W; del V, W
def build(a):
    f = 1.0 + a * (EL - E); d = jnp.log(jnp.maximum(jnp.abs(f), EPS))
    return la_P + d, s_P * jnp.where(f >= 0, 1.0, -1.0).astype(s_P.dtype)
sd = float(jnp.sqrt(jnp.sum(sec.vec(la_P, s_P) ** 2 * (EL - E) ** 2)))
r_ = minimize_scalar(lambda a: lab.H_site(*build(a)), bounds=(-3.0 / sd, 0.0), method='bounded', options=dict(xatol=1e-4 / sd))
la_t, s_t = build(float(r_.x)); del EL
H_G, H_t = lab.H_site(la_P, s_P), lab.H_site(la_t, s_t)
la_A = jnp.asarray(np.load(os.path.join(VDIR, 'la_table.npy')))
assert np.array_equal(np.load(os.path.join(VDIR, 's_next.npy')).astype(np.float32), np.asarray(s_t))
PT = sec.vec(la_t, s_t) ** 2
res['ref'] = dict(H_G=H_G, H_t=H_t, H_vit=lab.H_site(la_A, s_t))

vit_dt.set_dtype('float32')
vit = vit_dt.make_model_6x6()
tmpl = vit.init(jax.random.PRNGKey(1234), jnp.zeros((1, N), jnp.float32))
obj = flax.serialization.msgpack_restore(open(VIT_CKPT, 'rb').read())
if 'variables' in obj: obj = obj['variables']
tmpl = flax.serialization.from_state_dict(tmpl, obj)
_, unravel = ravel_pytree(jax.tree_util.tree_map(lambda q: jnp.asarray(q, jnp.float32), tmpl['params']))
flatA = jnp.asarray(np.load(os.path.join(VDIR, 'params.npy')), jnp.float32)


def logA(X):
    p = {'params': unravel(flatA)}; B_ = X.shape[0]
    Xg = jnp.stack([X[:, INV8[k // 2]] * SG2[k % 2] for k in range(16)], 0)
    zs = vit_dt.logpsi_transl_2d(vit.apply, 2, p, Xg.reshape(16 * B_, N)).reshape(16, B_)
    m = jnp.max(jnp.real(zs), 0)
    return m + jnp.log(jnp.abs(jnp.sum(jnp.exp(zs - m[None, :]), 0)))


grad_j = jax.jit(jax.grad(lambda X: jnp.sum(logA(X))))


@jax.jit
def lin_quantities(idx, G):
    """for reps idx with input gradients G (n, N): linearised and exact per-bond log-ratios, signs, D0."""
    x = sec.reps[idx]; X = bits(x)
    valid = valid_bonds(x)
    rhat = (G[:, BI_I] * (-2 * X[:, BI_I]) + G[:, BJ_I] * (-2 * X[:, BJ_I])).astype(f64)
    y = jnp.where(valid, x[:, None] ^ MASKS[None, :], x[:, None])
    iy = SS.canon(sec.T, sec.reps, y.reshape(-1)).reshape(x.shape[0], NB)
    rex = la_A[iy] - la_A[idx][:, None]
    ss = s_t[idx][:, None] * s_t[iy]
    return rhat, rex, ss, valid


def EL_from(r, ss, valid, idx):
    return lab.D0[idx] + jnp.sum(jnp.where(valid, 0.5 * JB[None, :] * ss * jnp.exp(jnp.clip(r, -60, 60)), 0.0), 1)


def grads(idx, batch=512):
    out = []
    for i in range(0, len(idx), batch):
        ii = idx[i:i + batch]; m = len(ii)
        if m < batch: ii = jnp.concatenate([ii, jnp.repeat(ii[:1], batch - m)])
        out.append(grad_j(bits(sec.reps[ii]))[:m])
    return jnp.concatenate(out)


if MODE == 'G0':
    Q = lab.proposal_q(la_t - la_A, la_t, s_t)[0]
    dec, _ = lab.decade_of(la_A)
    for name, prob, seed in (('target', PT, 5), ('tail', Q, 6)):
        p = np.array(prob, np.float64); p /= p.sum()
        idx = jnp.asarray(np.random.default_rng(seed).choice(sec.D, size=2048, p=p))
        G = grads(idx)
        rhat, rex, ss, valid = lin_quantities(idx, G)
        e = jnp.where(valid, rhat - rex, 0.0)
        ELx = EL_from(rex, ss, valid, idx); ELh = EL_from(rhat, ss, valid, idx)
        sdE = float(jnp.std(ELx))
        ua, ub = jnp.arcsinh((ELx - jnp.mean(ELx)) / sdE), jnp.arcsinh((ELh - jnp.mean(ELx)) / sdE)
        dx = dec[idx]
        out = dict(ratio_rms=float(jnp.sqrt(jnp.sum(e ** 2) / jnp.sum(valid))),
                   ratio_rms_by_decade={int(q): float(jnp.sqrt(jnp.sum(jnp.where((dx == q)[:, None], e ** 2, 0)) /
                                                               max(float(jnp.sum(jnp.where((dx == q)[:, None], valid, 0))), 1)))
                                        for q in range(-5, -16, -1)},
                   corr_asinh_EL=float(jnp.corrcoef(ua, ub)[0, 1]), EL_rms_err_over_sd=float(jnp.sqrt(jnp.mean((ELh - ELx) ** 2)) / sdE),
                   frac_below_1e9=float(jnp.mean(dx < -9)))
        lowm = dx < -9
        if int(jnp.sum(lowm)) > 20:
            out['corr_asinh_EL_below_1e9'] = float(jnp.corrcoef(ua[lowm], ub[lowm])[0, 1])
        res[name] = out; log(name, json.dumps(out)); dump(F, res)

if MODE == 'T1':
    resid = la_t - la_A
    _, _, meas = lab.proposal_q(resid, la_t, s_t)
    m_ = float(jnp.sum(meas * la_A)); s_ = float(jnp.sqrt(jnp.sum(meas * (la_A - m_) ** 2)))
    F_A = ((la_A - m_) / s_).astype(jnp.float32)[:, None]
    res['arms'] = {}
    for arm in SPEC.get('arms', ['TS', 'TS32']):
        t0 = time.time()
        if arm in ('TS', 'TS32'):
            FEAT = F_A; C = 64 if arm == 'TS' else 32
        else:                                                     # TG: linearised one-hop features from the input gradient
            cols_V, cols_W, cols_E = [], [], []
            for i in range(0, sec.D, 16384):
                idx = jnp.arange(i, min(i + 16384, sec.D))
                G = grads(idx)
                rhat, _, ss, valid = lin_quantities(idx, G)
                kept = ss < 0; wr = jnp.where(valid, 0.5 * JB[None, :] * jnp.exp(jnp.clip(rhat, -60, 60)), 0.0)
                Vh = jnp.sum(jnp.where(kept, 0.0, wr), 1); Wh = jnp.sum(jnp.where(kept, wr, 0.0), 1)
                cols_V.append(Vh); cols_W.append(Wh); cols_E.append(lab.D0[idx] + Vh - Wh)
            Vh, Wh, Eh = jnp.concatenate(cols_V), jnp.concatenate(cols_W), jnp.concatenate(cols_E)
            res['tg_table_sec'] = time.time() - t0
            lan = la_A - jnp.max(la_A)
            FEAT = lab.feat_table(lan, Vh, Wh, meas)
            u_ = jnp.arcsinh((Eh - jnp.sum(PT * Eh)) / float(jnp.sqrt(jnp.sum(PT * (Eh - jnp.sum(PT * Eh)) ** 2))))
            mu = float(jnp.sum(meas * u_)); su = float(jnp.sqrt(jnp.sum(meas * (u_ - mu) ** 2))) + 1e-30
            FEAT = jnp.concatenate([FEAT, ((u_ - mu) / su).astype(jnp.float32)[:, None]], 1); C = 32
            del Vh, Wh, Eh, u_
        mt, flt, fit = robust_fit(lab, lambda sd_: FeatNet(FEAT, seed=sd_, C=C), la_t, s_t, resid, 40000, 901, 3e-3, arm,
                                  heldout_curve=True)
        T = mt.table(flt, sec.reps, lab.TIDX) if flt is not None else jnp.zeros_like(la_A)
        Gn = la_A + T
        ex = Gn - la_t; ex = ex - jnp.sum(PT * ex)
        gw = float(jnp.sum(jnp.where(ex > 1.0, sec.vec(Gn, s_t) ** 2, 0.0)))
        if gw > 1e-3: Gn = la_A
        dec, _ = lab.decade_of(la_t)
        rec = dict(C=C, H=lab.H_site(Gn, s_t), guard_weight=gw, guard_rejected=gw > 1e-3, fit=fit,
                   err_by_decade={int(q): float(jnp.sqrt(jnp.sum(jnp.where(dec == q, PT * ex ** 2, 0)) /
                                                         max(float(jnp.sum(jnp.where(dec == q, PT, 0))), 1e-300)))
                                  for q in range(-5, -16, -1)})
        rec['share'] = (H_G - rec['H']) / (H_G - H_t)
        _, u, info = sec.fn_solve(Gn, s_t); del u; rec['E_FN'] = info['dE_FN_site']
        rec['sec'] = time.time() - t0
        log(f'== {arm}: share {rec["share"]:.3f}  <H> {rec["H"]:.4e}  E_FN {rec["E_FN"]:.4e}  held-out '
            f'{fit["tries"][-1]["L0_heldout"]:.3e} -> {fit["tries"][-1]["L_heldout"]:.3e}  ({rec["sec"]:.0f}s)')
        res['arms'][arm] = rec; dump(F, res)
        if flt is not None: np.save(os.path.join(OUT, f'params_{arm}.npy'), np.asarray(flt))
        del FEAT, T, Gn, ex
res['sec'] = time.time() - T00; dump(F, res); log('DONE', res['sec'])
