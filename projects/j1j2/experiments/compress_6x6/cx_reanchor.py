"""Rolling re-anchor test (README "Rolling re-anchor test", pre-registered before any run). ONE iteration per job.
Iteration k: state (la_k, s_k) [k = 1: (log|psi_P|, s_P); k >= 2: the previous job's full-sector table of its
fine-tuned network], exact clipped-Lanczos target la_t = la_k + d, s_{k+1} (as cx_chop3); fine-tune the warm
symmetrised ViT (log A = log|sum_{g in D4 x flip} psi_theta(g x)|) to la_t with the stage-2 edge loss (kept bonds of
the target state), increment-weighted sampling, 20% held-out orbits; opt 'minsr' (damped Gauss-Newton in sample
space, trust region, acceptance on a fixed paired validation set) or 'adam' (lr 1e-6, control); held-out acceptance;
full-sector table; blow-up guard; exact evaluation; fp32-vs-fp64 precision floor at the start.
  python cx_reanchor.py OUT SPEC.json
SPEC: k, seed, opt, prev (run dir of iteration k-1), fit_sec (720), B (384 minsr / 256 adam), K (8), remat (4)
"""
import os, sys, time
import numpy as np
from scipy.optimize import minimize_scalar
from cx_lib import *
import optax
import flax
import vit_dt
from lanczos_lib import lanczos_ritz

OUT = sys.argv[1]; SPEC = json.load(open(sys.argv[2])); os.makedirs(OUT, exist_ok=True)
F = os.path.join(OUT, 'cx_reanchor.json'); T00 = time.time()
K_IT = int(SPEC['k']); SEED = int(SPEC.get('seed', 0)); OPT = SPEC.get('opt', 'minsr')
EPS = 0.3; REMAT = SPEC.get('remat', 4); KB = SPEC.get('K', 8)
B_FIT = SPEC.get('B', 384 if OPT == 'minsr' else 256); B_VAL = 256; NVAL = 16
FIT_SEC = SPEC.get('fit_sec', 720)
res = dict(spec=SPEC)

# ---------------------------------------------------------------- previous state / continuation rule
flat_prev = None
if K_IT >= 2:
    P = SPEC['prev']; rp = json.load(open(os.path.join(P, 'cx_reanchor.json')))
    res['prev_share'] = rp['iter']['share']
    if not (rp['iter']['share'] >= 0.30):
        res['stopped'] = f'continuation rule: previous share {rp["iter"]["share"]:.3f} < 0.30'
        dump(F, res); log(res['stopped']); sys.exit(0)
lab = Lab(); sec = lab.sec
if K_IT == 1:
    la_G, s_G = lab.lP0, lab.sP0
else:
    la_G = jnp.asarray(np.load(os.path.join(P, 'la_table.npy')))
    s_G = jnp.asarray(np.load(os.path.join(P, 's_next.npy')).astype(np.float32))
    flat_prev = np.load(os.path.join(P, 'params.npy'))

_z = np.load(TABLE); _v0 = np.asarray(_z['amp'], np.float64) * np.sqrt(np.asarray(sec.n)); del _z
_v0 /= np.linalg.norm(_v0); Q0 = _v0 * np.abs(_v0); del _v0


def w_s(s):
    return float(max(0.0, (1.0 - abs(float(np.dot(Q0, np.asarray(s, np.float64))))) / 2))


def clipped_lanczos(la, s, EL, E):
    def build(alpha):
        f = 1.0 + alpha * (EL - E)
        d = jnp.log(jnp.maximum(jnp.abs(f), EPS))
        return la + d, s * jnp.where(f >= 0, 1.0, -1.0).astype(s.dtype), d
    sd = float(jnp.sqrt(jnp.sum(sec.vec(la, s) ** 2 * (EL - E) ** 2)))
    r = minimize_scalar(lambda a: lab.H_site(*build(a)[:2]), bounds=(-3.0 / sd, 0.0), method='bounded',
                        options=dict(xatol=1e-4 / sd))
    la2, s2, d = build(float(r.x))
    return la2, s2, d, float(r.x)


# ---------------------------------------------------------------- target (exact, as cx_chop3)
H_G = lab.H_site(la_G, s_G)
_, V, W = lab.VW(la_G, s_G)
_v = sec.vec(la_G, s_G); E = float(_v @ sec.H(_v)); del _v
EL = lab.D0 + V - W; del V, W
la_t, s_t, d, alpha = clipped_lanczos(la_G, s_G, EL, E); del EL
H_t = lab.H_site(la_t, s_t)
PT = sec.vec(la_t, s_t) ** 2
mu_d = float(jnp.sum(PT * d)); d_ptrms = float(jnp.sqrt(jnp.sum(PT * (d - mu_d) ** 2)))
it = dict(k=K_IT, H_G=H_G, H_target=H_t, alpha=alpha, exact_gain=H_G - H_t, d_ptrms=d_ptrms,
          d_bond_rms=bond_rms(lab, la_G, d), flips=float(jnp.sum(sec.n * (s_t != s_G)) / jnp.sum(sec.n)))
log(f'[k={K_IT} {OPT} seed {SEED}] <H> {H_G:.4e} -> target {H_t:.4e}  alpha {alpha:.4f}  d ptrms {d_ptrms:.4f} '
    f'bond-rms {it["d_bond_rms"]:.4f}  flips {it["flips"]:.3f}')
res['iter'] = it; dump(F, res)

# ---------------------------------------------------------------- the symmetrised ViT
vit_dt.set_dtype('float32')
vit = vit_dt.make_model_6x6()
tmpl = vit.init(jax.random.PRNGKey(1234), jnp.zeros((1, N), jnp.float32))
obj = flax.serialization.msgpack_restore(open(VIT_CKPT, 'rb').read())
if 'variables' in obj: obj = obj['variables']
tmpl = flax.serialization.from_state_dict(tmpl, obj)
params = jax.tree_util.tree_map(lambda q: jnp.asarray(q, jnp.float32), tmpl['params'])
flat0, unravel = ravel_pytree(params)
if flat_prev is not None: flat0 = jnp.asarray(flat_prev, jnp.float32)
NPAR = int(flat0.size)


def sym_logA(apply, p, X, remat):
    B_ = X.shape[0]
    Xg = jnp.stack([X[:, INV8[k // 2]] * SG2[k % 2] for k in range(16)], 0)
    if remat > 1:
        body = jax.checkpoint(lambda Xi: vit_dt.logpsi_transl_2d(apply, 2, p, Xi))
        zs = jax.lax.map(body, Xg.reshape(remat, (16 // remat) * B_, N)).reshape(16, B_)
    else:
        zs = vit_dt.logpsi_transl_2d(apply, 2, p, Xg.reshape(16 * B_, N)).reshape(16, B_)
    m = jnp.max(jnp.real(zs), 0)
    return m + jnp.log(jnp.abs(jnp.sum(jnp.exp(zs - m[None, :]), 0)))


def logA(flat, X):
    return sym_logA(vit.apply, {'params': unravel(flat)}, X, REMAT)


fwd = jax.jit(lambda fl, X: logA(fl, X).astype(f64))
tab_j = jax.jit(lambda fl, Sx: logA(fl, bits(Sx)).astype(f64))


def table(flat, batch=4096):
    out = []
    for i in range(0, sec.D, batch):
        s_ = sec.reps[i:i + batch]; m = s_.shape[0]
        if m < batch: s_ = jnp.concatenate([s_, jnp.repeat(s_[:1], batch - m)])
        out.append(tab_j(flat, s_)[:m])
    return jnp.concatenate(out)


# ---------------------------------------------------------------- sampling (increment-weighted, held-out orbits)
Q, PA, _ = lab.proposal_q(d, la_t, s_t)
Qtr = jnp.where(lab.HO, 0.0, Q); Qtr = Qtr / jnp.sum(Qtr)
Qte = jnp.where(lab.HO, Q, 0.0); Qte = Qte / jnp.sum(Qte)
CDF_TR, CDF_TE = jnp.cumsum(Qtr), jnp.cumsum(Qte)
LIW_TR = jnp.where(Qtr > 0, jnp.log(jnp.maximum(PA, 1e-300)) - jnp.log(jnp.maximum(Qtr, 1e-300)), -jnp.inf)
LIW_TE = jnp.where(Qte > 0, jnp.log(jnp.maximum(PA, 1e-300)) - jnp.log(jnp.maximum(Qte, 1e-300)), -jnp.inf)
del Q, Qtr, Qte, PA
HOJ = lab.HO


def make_batch(cdf, liw, withhold, B_):
    @jax.jit
    def batch(key):
        k1, k2 = jax.random.split(key)
        idx = jnp.clip(jnp.searchsorted(cdf, jax.random.uniform(k1, (B_,), f64) * cdf[-1]), 0, sec.D - 1)
        x = sec.reps[idx]
        valid = valid_bonds(x); nval = valid.sum(1)
        sc = jnp.where(valid, jax.random.uniform(k2, valid.shape), -1.0)
        _, bsel = jax.lax.top_k(sc, KB)
        ok = jnp.take_along_axis(valid, bsel, 1)
        y = x[:, None] ^ MASKS[bsel]
        iy = SS.canon(sec.T, sec.reps, y.reshape(-1)).reshape(B_, KB)
        r = jnp.clip(la_t[iy] - la_t[idx][:, None], -60, 60)
        w = jnp.where(ok & (s_t[idx][:, None] * s_t[iy] < 0), 0.5 * JB[bsel] * jnp.exp(r), 0.0) * (nval[:, None] / KB)
        if withhold: w = jnp.where(HOJ[iy], 0.0, w)
        lw = liw[idx]; iw = jnp.exp(lw - jnp.max(lw))
        return x, y, la_t[idx], la_t[iy], w, iw
    return batch


def edge_loss(S, tx, ty, w, iw, B_):
    ex = S[:B_] - tx; ey = S[B_:].reshape(B_, KB) - ty
    iwn = iw / jnp.mean(iw)
    return 0.5 * jnp.mean(iwn * jnp.sum(w * (ex[:, None] - ey) ** 2, 1)) / N


def loss_of(flat, bd, B_):
    x, y, tx, ty, w, iw = bd
    S = logA(flat, bits(jnp.concatenate([x, y.reshape(-1)]))).astype(f64)
    return edge_loss(S, tx, ty, w, iw, B_)


lval_j = jax.jit(lambda fl, bd: loss_of(fl, bd, B_VAL))
bv_tr = make_batch(CDF_TR, LIW_TR, True, B_VAL); bv_te = make_batch(CDF_TE, LIW_TE, False, B_VAL)
VB_TR = [bv_tr(jax.random.PRNGKey(9000 + i)) for i in range(NVAL)]
VB_TE = [bv_te(jax.random.PRNGKey(9500 + i)) for i in range(NVAL)]


def lval(fl, VB):
    return float(np.mean([float(lval_j(fl, bd)) for bd in VB]))


# ---------------------------------------------------------------- fp32 precision floor (lead's request)
def floor_check(flat, n=1024):
    rng = np.random.default_rng(77)
    p = np.array(PT, np.float64); p = p / p.sum()
    idx = jnp.asarray(rng.choice(sec.D, size=n, p=p))
    x = sec.reps[idx]; valid = valid_bonds(x)
    sc = jnp.where(valid, jax.random.uniform(jax.random.PRNGKey(5), valid.shape), -1.0)
    _, bsel = jax.lax.top_k(sc, KB); ok = jnp.take_along_axis(valid, bsel, 1)
    y = x[:, None] ^ MASKS[bsel]
    iy = SS.canon(sec.T, sec.reps, y.reshape(-1)).reshape(n, KB)
    w = jnp.where(ok & (s_t[idx][:, None] * s_t[iy] < 0), 0.5 * JB[bsel] * jnp.exp(jnp.clip(la_t[iy] - la_t[idx][:, None], -60, 60)), 0.0)
    X = bits(jnp.concatenate([x, y.reshape(-1)]))
    ch = 1024
    l32 = jnp.concatenate([fwd(flat, X[i:i + ch]) for i in range(0, X.shape[0], ch)])
    vit_dt.set_dtype('float64')
    vit64 = vit_dt.make_model_6x6()
    p64 = jax.tree_util.tree_map(lambda q: jnp.asarray(q, jnp.float64), unravel(flat))
    f64fn = jax.jit(lambda pp, XX: sym_logA(vit64.apply, {'params': pp}, XX.astype(jnp.float64), 1).astype(f64))
    l64 = jnp.concatenate([f64fn(p64, X[i:i + 256]) for i in range(0, X.shape[0], 256)])
    vit_dt.set_dtype('float32')
    e = l32 - l64; e = e - jnp.mean(e)
    ex, ey = e[:n], e[n:].reshape(n, KB)
    dx, dy = d[idx], d[iy]
    sw = float(jnp.sum(w))
    out = dict(floor_pt_rms=float(jnp.sqrt(jnp.mean(e ** 2))), floor_max=float(jnp.max(jnp.abs(e))),
               floor_edge_rms=float(jnp.sqrt(jnp.sum(w * (ex[:, None] - ey) ** 2) / sw)),
               incr_edge_rms=float(jnp.sqrt(jnp.sum(w * (dx[:, None] - dy) ** 2) / sw)))
    out['ratio'] = out['floor_edge_rms'] / out['incr_edge_rms']
    out['meaningful'] = bool(out['ratio'] <= 0.1)
    return out


it['fp32_floor'] = floor_check(flat0)
log('[floor]', it['fp32_floor']); dump(F, res)

# ---------------------------------------------------------------- fine-tune
sec.offload()
L0_tr, L0_te = lval(flat0, VB_TR), lval(flat0, VB_TE)
curve = [dict(t=0.0, step=0, own=L0_tr, heldout=L0_te)]
log(f'[fit] npar {NPAR}  start own {L0_tr:.4e}  held-out {L0_te:.4e}')
flat = flat0; key = jax.random.PRNGKey(100 + SEED + 1000 * K_IT); t0 = time.time(); hist = []
bfit = make_batch(CDF_TR, LIW_TR, True, B_FIT)
if OPT == 'minsr':
    tau = 0.5 * d_ptrms
    JCH = SPEC.get('jac_chunk', 128)
    jac_rows = jax.jit(jax.vmap(jax.grad(lambda fl, X1: logA(fl, X1[None])[0]), in_axes=(None, 0)))

    def gn_direction(fl, bd, lam_rel):
        x, y, tx, ty, w, iw = bd
        X = bits(jnp.concatenate([x, y.reshape(-1)]))
        S = fwd(fl, X)
        ex = S[:B_FIT] - tx; ey = S[B_FIT:].reshape(B_FIT, KB) - ty
        iwn = iw / jnp.mean(iw)
        ce = jnp.sqrt(0.5 * iwn[:, None] * w / (B_FIT * N))
        r = (ce * (ex[:, None] - ey)).reshape(-1)
        G = jnp.concatenate([jac_rows(fl, X[i:i + JCH]) for i in range(0, X.shape[0], JCH)])
        Je = (ce[:, :, None].astype(jnp.float32) * (G[:B_FIT][:, None, :] - G[B_FIT:].reshape(B_FIT, KB, -1))).reshape(B_FIT * KB, -1)
        Tm = (Je @ Je.T).astype(f64)
        lam = lam_rel * jnp.trace(Tm) / Tm.shape[0]
        a = jnp.linalg.solve(Tm + lam * jnp.eye(Tm.shape[0]), r)
        dlt = -(Je.T @ a.astype(jnp.float32)); del Je, Tm
        pc = (G @ dlt).astype(f64); pc = pc - jnp.mean(pc)
        prms = float(jnp.sqrt(jnp.mean(pc ** 2))); del G
        sc_ = min(1.0, tau / max(prms, 1e-30))
        return dlt * sc_, float(jnp.sum(r * r)), prms, sc_

    lam_rel = SPEC.get('lam0', 1e-2); L_cur = L0_tr; n = 0
    while time.time() - t0 < FIT_SEC:
        n += 1; key, kk = jax.random.split(key)
        dlt, rr, prms, sc_ = gn_direction(flat, bfit(kk), lam_rel)
        cand = flat + dlt
        L_new = lval(cand, VB_TR)
        ok = bool(np.isfinite(L_new) and L_new < L_cur)
        if ok:
            flat = cand; L_cur = L_new; lam_rel = max(lam_rel / 2, 1e-6)
        else:
            lam_rel = min(lam_rel * 4, 1e2)
        hist.append(dict(step=n, ok=ok, lam_rel=lam_rel, L=L_cur, L_try=L_new, pred_rms=prms, trust_scale=sc_,
                         dnorm=float(jnp.linalg.norm(dlt))))
        if n % 10 == 0:
            curve.append(dict(t=time.time() - t0, step=n, own=L_cur, heldout=lval(flat, VB_TE)))
            log(f'  minsr {n}: acc {np.mean([q["ok"] for q in hist[-10:]]):.1f} lam_rel {lam_rel:.1e} own {L_cur:.4e} '
                f'held-out {curve[-1]["heldout"]:.4e} pred-rms {prms:.2e} trust {sc_:.2f}')
            res['fit'] = dict(curve=curve, hist=hist); dump(F, res)
else:
    lr = SPEC.get('lr', 1e-6)
    gstep = jax.jit(lambda fl, bd: jax.value_and_grad(lambda q: loss_of(q, bd, B_FIT))(fl))
    for _ in range(3):                                   # compile + time
        key, kk = jax.random.split(key); tt = time.time(); l_, g_ = gstep(flat, bfit(kk)); jax.block_until_ready(g_)
    t_step = time.time() - tt; nst = max(200, int((FIT_SEC - (time.time() - t0)) / (t_step * 1.15)))
    sched = optax.warmup_cosine_decay_schedule(0.0, lr, 100, nst, lr * 0.02)
    opt = optax.adam(sched); ost = opt.init(flat)

    @jax.jit
    def astep(fl, ost, bd):
        l_, g_ = jax.value_and_grad(lambda q: loss_of(q, bd, B_FIT))(fl)
        upd, ost = opt.update(g_, ost, fl)
        return optax.apply_updates(fl, upd), ost, l_
    for n in range(1, nst + 1):
        key, kk = jax.random.split(key)
        flat, ost, l_ = astep(flat, ost, bfit(kk))
        if n % 200 == 0 or n == nst:
            curve.append(dict(t=time.time() - t0, step=n, own=lval(flat, VB_TR), heldout=lval(flat, VB_TE)))
            log(f'  adam {n}/{nst}: own {curve[-1]["own"]:.4e} held-out {curve[-1]["heldout"]:.4e}')
            res['fit'] = dict(curve=curve); dump(F, res)
L1_tr, L1_te = lval(flat, VB_TR), lval(flat, VB_TE)
accepted = bool(L1_te < L0_te)
it['fit'] = dict(opt=OPT, steps=len(curve) and curve[-1]['step'], sec=time.time() - t0, own_start=L0_tr, own_end=L1_tr,
                 heldout_start=L0_te, heldout_end=L1_te, heldout_ratio=L1_te / L0_te, accepted_heldout=accepted,
                 accept_rate=(float(np.mean([q['ok'] for q in hist])) if hist else None))
res['fit'] = dict(curve=curve, hist=hist)
log('[fit]', it['fit']); dump(F, res)
if not accepted: flat = flat0
del CDF_TR, CDF_TE, LIW_TR, LIW_TE, VB_TR, VB_TE

# ---------------------------------------------------------------- exact evaluation
t1 = time.time()
la_new = table(flat) if accepted else la_G                      # la_G is the table of flat0 (k >= 2) / psi_P (k = 1)
it['table_sec'] = time.time() - t1
sec.reload()
PTn = sec.vec(la_new, s_t) ** 2
ex = la_new - la_t; ex = ex - jnp.sum(PT * ex)
it['guard_weight_excess_gt1'] = float(jnp.sum(jnp.where(ex > 1.0, PTn, 0.0)))
it['guard_rejected'] = bool(it['guard_weight_excess_gt1'] > 1e-3)
if it['guard_rejected']:
    la_new = la_G; flat = flat0
it['ptwise_rms_err'] = float(jnp.sqrt(jnp.sum(PT * ex ** 2)))
dec, _ = lab.decade_of(la_t)
it['ptwise_rms_err_by_decade'] = {int(q): float(jnp.sqrt(jnp.sum(jnp.where(dec == q, PT * ex ** 2, 0)) /
                                                         max(float(jnp.sum(jnp.where(dec == q, PT, 0))), 1e-300)))
                                  for q in range(-4, -16, -1)}
del ex, PTn
it['H'] = lab.H_site(la_new, s_t)
it['share'] = (H_G - it['H']) / (H_G - H_t)
it['bond_rms_err'] = bond_rms(lab, la_t, la_new - la_t)
Efn, u, info = sec.fn_solve(la_new, s_t); del u
it['E_FN'] = info['dE_FN_site']
rl = lanczos_ritz(lambda x: sec.Hm(x), sec.vec(la_new, s_t), 2)
it['lanczos_p1'] = (rl[1]['E'] - sec.E0) / N; it['lanczos_p2'] = (rl[2]['E'] - sec.E0) / N
it['w_s'] = w_s(s_t)
log(f'== k={K_IT} {OPT} seed {SEED}: share {it["share"]:.3f}  <H> {H_G:.4e} -> {it["H"]:.4e} (target {H_t:.4e})  '
    f'bond-rms err {it["bond_rms_err"]:.4f}  E_FN {it["E_FN"]:.4e}  +L1 {it["lanczos_p1"]:.4e}  +L2 {it["lanczos_p2"]:.4e}')
np.save(os.path.join(OUT, 'params.npy'), np.asarray(flat))
np.save(os.path.join(OUT, 'la_table.npy'), np.asarray(la_new, np.float64))
np.save(os.path.join(OUT, 's_next.npy'), np.asarray(s_t, np.int8))
res['sec'] = time.time() - T00; dump(F, res); log('DONE', res['sec'])
