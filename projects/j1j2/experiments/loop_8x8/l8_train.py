"""Stage 2: write-back arms on the stored pool (no ViT evaluations here).
Model b = a exp(f), f = FEAT residual CNN (writeback_tail wt_loop.Corr adapted to 8x8: periodic 3x3 residual CNN, C=32,
4 layers, Fourier-embedded guide features (log a, log V, log W) standardised by the pool, 2-layer MLP head, zero head
init; exactly D4 x flip symmetric at evaluation, one random image per row in training).
Arms (spec 'arms'):
  kind 'fn'  : edge least squares vs the semi-implicit FN target T on kept bonds, weight J/2 a_y/a_x (guide metric)
  kind 'vmc' : fixed-sign VMC on <H_ph>(b, s): E_L^b(x) = E_L^a(x) + nval/K sum_k J/2 s_x s_y a_y/a_x (e^{f_y-f_x} - 1)
               (exact guide local energy + K-bond estimate of the correction; same samples, same guide evaluations)
  opt 'adam' (warmup-cosine, as F-SI-Adam) | 'sr' (minSR, sample space, trust radius max_df on rms change of f).
Checkpoints every 'ckpt_every' steps with the held-out own objective (symmetric f).
  python l8_train.py OUT SPEC.json   (SPEC: pool path, arms, steps, B, ...)
"""
import sys, os, time, json
import numpy as np
import l8_core as C
import jax, jax.numpy as jnp
import optax

OUT = sys.argv[1]; SPEC = json.load(open(sys.argv[2])); os.makedirs(OUT, exist_ok=True)
F = os.path.join(OUT, 'train.json'); T00 = time.time(); N = C.N; L = C.L
f64 = jnp.float64

from l8_net import ResCNN, Corr, feats


# ============================================================================================ pool
Z = np.load(SPEC['pool'])
X = Z['X']; Y = Z['Y']; K = Y.shape[1]; P = len(X); ix = Z['ix']; iy = Z['iy']
U = {k: Z['U_' + k] for k in ('la', 's', 'V', 'W', 'D', 'ELa', 'T')}
lw = Z['lw']; nval = Z['nval']; bond = Z['bond'].astype(np.int64)
# standardisation constants: unweighted pool (guide-only measure)
stats = []
for c in (U['la'][ix], np.log(np.maximum(U['V'][ix], 0) + 1e-6), np.log(np.maximum(U['W'][ix], 0) + 1e-6)):
    stats.append((float(c.mean()), float(c.std()) + 1e-30))
FX = feats(U['la'][ix], U['V'][ix], U['W'][ix], stats)
FY = feats(U['la'][iy], U['V'][iy], U['W'][iy], stats)
r = 0.5 * C.JB[bond] * np.exp(np.clip(U['la'][iy] - U['la'][ix][:, None], -60, 60))
sgn = U['s'][ix][:, None] * U['s'][iy]
scale = nval[:, None] / K
WGT = np.where(sgn < 0, r, 0.0) * scale                              # FN: kept-bond guide-metric weight
HV = sgn * r * scale                                                  # VMC: H_xy psi_y / psi_x, K-bond estimator
TX = U['T'][ix]; TY = U['T'][iy]; ELA = U['ELa'][ix]
rg = np.random.default_rng(SPEC.get('split_seed', 3))
# held-out split by chain (no chain in both sets)
chains = Z['chain']; uc = np.unique(chains); ho_c = rg.choice(uc, int(round(SPEC.get('holdout', 0.2) * len(uc))), replace=False)
HO = np.isin(chains, ho_c); TR = ~HO
res = dict(spec=SPEC, P=P, K=K, n_train=int(TR.sum()), n_hold=int(HO.sum()), stats=stats,
           pool_meta=dict(Ea=float(Z['Ea']), phi=float(Z['phi'])), arms=[])
C.log('pool', P, 'train', TR.sum(), 'held-out', HO.sum())

def dev(a, dt=jnp.float32):
    return jnp.asarray(a, dt)

SP = jnp.asarray(C.bits_to_spins_np(X)); SPY = jnp.asarray(C.bits_to_spins_np(Y.reshape(-1))).reshape(P, K, N)
D = dict(FX=dev(FX), FY=dev(FY), WGT=dev(WGT, f64), HV=dev(HV, f64), TX=dev(TX, f64), TY=dev(TY, f64), ELA=dev(ELA, f64),
         LW=dev(lw, f64))
idx_tr = jnp.asarray(np.nonzero(TR)[0]); idx_ho = np.nonzero(HO)[0]


def gather(ii):
    return (SP[ii], SPY[ii], D['FX'][ii], D['FY'][ii], D['WGT'][ii], D['HV'][ii], D['TX'][ii], D['TY'][ii], D['ELA'][ii],
            D['LW'][ii])


def fxy(model, q, sx, sy, fx_, fy_, G=None):
    B_, K_ = sy.shape[:2]
    Xa = jnp.concatenate([sx, sy.reshape(-1, N)]); Ta = jnp.concatenate([fx_, fy_.reshape(-1, fx_.shape[1])])
    if G is None:
        fa = model.f(q, Xa, Ta).astype(f64)
    else:
        fa = model.f_aug(q, Xa, Ta, jnp.concatenate([G, jnp.repeat(G, K_)])).astype(f64)
    return fa[:B_], fa[B_:].reshape(B_, K_)


def fn_loss(model, q, bd, G=None):
    sx, sy, fx_, fy_, wgt, hv, tx, ty, ela, lw_ = bd
    fx, fy = fxy(model, q, sx, sy, fx_, fy_, G)
    iw = jnp.exp(lw_ - jnp.max(lw_)); iw = iw / jnp.sum(iw)
    return 0.5 * jnp.sum(iw * jnp.sum(wgt * ((fx - tx)[:, None] - (fy - ty)) ** 2, 1)) / N


def vmc_terms(model, q, bd, G=None):
    sx, sy, fx_, fy_, wgt, hv, tx, ty, ela, lw_ = bd
    fx, fy = fxy(model, q, sx, sy, fx_, fy_, G)
    EL = ela + jnp.sum(hv * (jnp.exp(jnp.clip(fy - fx[:, None], -60, 60)) - 1.0), 1)
    l_ = lw_ + 2 * fx
    w = jnp.exp(l_ - jnp.max(l_)); w = w / jnp.sum(w)
    E = jnp.sum(w * EL)
    ws, ELs, Es = jax.lax.stop_gradient((w, EL, E))
    return E, 2.0 * jnp.sum(ws * (ELs - Es) * fx), ELs, ws, fx


def heldout(model, q, kind, bs=1024):
    """own objective on the held-out pool with the symmetric f: FN = edge loss; VMC = reweighted energy per site."""
    num = 0.0; den = 0.0; acc = []
    for i in range(0, len(idx_ho), bs):
        ii = jnp.asarray(idx_ho[i:i + bs])
        acc.append(_ho_part(q, ii, kind))
    if kind == 'fn':
        tot_w = sum(float(a[1]) for a in acc)
        return sum(float(a[0]) for a in acc) / tot_w / N
    lmax = max(float(a[2]) for a in acc)
    num = sum(float(a[0]) * np.exp(float(a[2]) - lmax) for a in acc); den = sum(float(a[1]) * np.exp(float(a[2]) - lmax) for a in acc)
    return num / den / N


def make_ho(model):
    @jax.jit
    def part_fn(q, ii):
        sx, sy, fx_, fy_, wgt, hv, tx, ty, ela, lw_ = gather(ii)
        fx, fy = fxy(model, q, sx, sy, fx_, fy_)
        iw = jnp.exp(lw_)
        return (0.5 * jnp.sum(iw * jnp.sum(wgt * ((fx - tx)[:, None] - (fy - ty)) ** 2, 1)), jnp.sum(iw), 0.0)

    @jax.jit
    def part_vmc(q, ii):
        sx, sy, fx_, fy_, wgt, hv, tx, ty, ela, lw_ = gather(ii)
        fx, fy = fxy(model, q, sx, sy, fx_, fy_)
        EL = ela + jnp.sum(hv * (jnp.exp(jnp.clip(fy - fx[:, None], -60, 60)) - 1.0), 1)
        l_ = lw_ + 2 * fx; m = jnp.max(l_); w = jnp.exp(l_ - m)
        return (jnp.sum(w * EL), jnp.sum(w), m)
    return part_fn, part_vmc


# ============================================================================================ arms
for arm in SPEC['arms']:
    tag = arm['tag']; kind = arm['kind']; opt_kind = arm.get('opt', 'adam'); t0 = time.time()
    B = arm.get('B', 256); steps = arm.get('steps', 20000); every = arm.get('ckpt_every', 1000)
    model = Corr(32, 4, seed=arm.get('seed', 0))
    pf, pv = make_ho(model)
    _ho_part = lambda q, ii, k: (pf if k == 'fn' else pv)(q, ii)
    flat = model.flat0
    if opt_kind == 'adam':
        lr = arm['lr']
        sched = optax.warmup_cosine_decay_schedule(0.0, lr, min(200, max(1, steps // 2)), steps, lr * 0.02)
        opt = optax.adam(sched); ost = opt.init(flat)

        @jax.jit
        def step(fl, ost, key):
            k1, k2 = jax.random.split(key)
            ii = idx_tr[jax.random.randint(k1, (B,), 0, idx_tr.shape[0])]
            bd = gather(ii); G = jax.random.randint(k2, (B,), 0, 16)
            if kind == 'fn':
                l, g = jax.value_and_grad(lambda q: fn_loss(model, q, bd, G))(fl)
            else:
                g = jax.grad(lambda q: vmc_terms(model, q, bd, G)[1])(fl)
                l = vmc_terms(model, fl, bd, G)[0] / N
            upd, ost = opt.update(g, ost, fl)
            return optax.apply_updates(fl, upd), ost, l
    else:
        eta = arm.get('eta', 0.02); lam = arm.get('lam', 1e-3); max_df = arm.get('max_df', 0.01); ost = None

        def fi(q, xi, ti, gi):
            return model.f_aug(q, xi[None], ti[None], gi[None])[0]
        jac = jax.vmap(jax.grad(fi), in_axes=(None, 0, 0, 0))

        @jax.jit
        def step(fl, ost, key):
            k1, k2 = jax.random.split(key)
            ii = idx_tr[jax.random.randint(k1, (B,), 0, idx_tr.shape[0])]
            bd = gather(ii); G = jax.random.randint(k2, (B,), 0, 16)
            sx, sy, fx_, fy_, wgt, hv, tx, ty, ela, lw_ = bd
            if kind == 'fn':
                Xa = jnp.concatenate([sx, sy.reshape(-1, N)]); Ta = jnp.concatenate([fx_, fy_.reshape(-1, 3)])
                GG = jnp.concatenate([G, jnp.repeat(G, K)])
                Ga = jac(fl, Xa, Ta, GG)
                fa = model.f_aug(fl, Xa, Ta, GG).astype(f64)
                fx, fy = fa[:B], fa[B:].reshape(B, K)
                iw = jnp.exp(lw_ - jnp.max(lw_)); iw = iw / jnp.sum(iw)
                c = jnp.sqrt(iw[:, None] * wgt)
                rr = (c * ((fx - tx)[:, None] - (fy - ty))).reshape(-1)
                Jm = (c.astype(jnp.float32)[:, :, None] * (Ga[:B][:, None, :] - Ga[B:].reshape(B, K, -1))).reshape(B * K, -1)
                Tm = (Jm @ Jm.T).astype(f64)
                alpha = jnp.linalg.solve(Tm + (lam * jnp.trace(Tm) / (B * K) + 1e-30) * jnp.eye(B * K), rr)
                d = Jm.T @ alpha.astype(jnp.float32)
                dfr = jnp.sqrt(jnp.mean((Ga[:B] @ d) ** 2)) * eta
                l = 0.5 * jnp.sum(rr * rr) / N
            else:
                E, _, EL, rho, _ = vmc_terms(model, fl, bd, G)
                eps_ = jnp.sqrt(rho) * (EL - E)
                O = jac(fl, sx, fx_, G)
                r32 = rho.astype(jnp.float32)
                Yo = jnp.sqrt(r32)[:, None] * (O - (r32 @ O)[None, :])
                Tm = (Yo @ Yo.T).astype(f64)
                alpha = jnp.linalg.solve(Tm + (lam * jnp.trace(Tm) / B + 1e-30) * jnp.eye(B), eps_)
                d = Yo.T @ alpha.astype(jnp.float32)
                dfr = jnp.sqrt(jnp.mean((O @ d) ** 2)) * eta
                l = E / N
            sc = jnp.minimum(1.0, max_df / (dfr + 1e-30))
            return fl - eta * sc * d, ost, l

    key = jax.random.PRNGKey(100 + arm.get('seed', 0))
    curve = []; ck = {}; trl = []
    h0 = heldout(model, flat, kind); curve.append((0, h0, float('nan'))); ck[0] = np.asarray(flat)
    C.log(f'[{tag}] step 0 held-out own {h0:.6e}')
    for it in range(1, steps + 1):
        key, kk = jax.random.split(key)
        flat, ost, l = step(flat, ost, kk); trl.append(float(l)) if it % 50 == 0 else None
        stop = arm.get('max_sec') and time.time() - t0 > arm['max_sec']
        if it % every == 0 or it == steps or stop:
            if not np.all(np.isfinite(np.asarray(flat))):
                C.log(f'[{tag}] non-finite parameters at step {it}'); curve.append((it, float('nan'), float('nan'))); break
            h = heldout(model, flat, kind)
            curve.append((it, h, float(np.mean(trl[-20:])) if trl else float('nan'))); ck[it] = np.asarray(flat)
            C.log(f'[{tag}] step {it} held-out own {h:.6e}  train(last 1k) {curve[-1][2]:.6e}  {time.time() - t0:.0f}s')
            if stop:
                C.log(f'[{tag}] wall-time cap reached at step {it}'); break
    hs = np.array([c[1] for c in curve]); ok = np.isfinite(hs)
    best = int(np.array([c[0] for c in curve])[ok][np.argmin(hs[ok])])
    np.savez(os.path.join(OUT, f'params_{tag}.npz'), **{f's{k}': v for k, v in ck.items()}, best=best,
             stats=np.asarray(stats))
    rec = dict(tag=tag, kind=kind, opt=opt_kind, arm=arm, npar=model.npar, curve=curve, best_step=best,
               final_step=int(curve[-1][0]), sec=time.time() - t0)
    res['arms'].append(rec); C.dump(F, res)
    C.log(f'== {tag}: best held-out step {best}  ({rec["sec"]:.0f}s)')
res['sec'] = time.time() - T00; C.dump(F, res); C.log('DONE', res['sec'])
