"""P2: distil the replayed wtLOOP8 accumulated amplitude la_k (k = 3, 6) into ONE directly callable student.

Arms (spec 'arm'):
  'vit_warm'    (A')  log A(x) = log |sum_{g in D4 x flip} psi_theta(g x)|, psi_theta = the 6x6 ViT (4 layers, d 60, 2x2
                      patches, 4 patch-offset translations inside), warm-started from the checkpoint: at step 0 A = psi_P.
  'vit_scratch' (A)   same ViT, random init, amplitude-only symmetrisation log A(x) = mean_g Re log psi_theta(g x).
  'bond'        (B)   bond-output ratio net: periodic residual CNN -> per-site embedding -> per-bond-type head, outputs
                      R_b(x) ~ log a_k(x^b) - log a_k(x) for all 144 bonds in one pass (V, W follow from R).
Loss (scalar students): L = L_ratio + lam * L_value, in the metric of the target guide a_k:
  L_ratio = E_{x ~ a_k^2} sum_{valid bonds b} (J_b/2) a_k(x^b)/a_k(x) (e(x) - e(x^b))^2 / N   (K bonds sampled per x)
  L_value = Var_{a_k^2}(e),   e = log A - la_k.
  (the |H|-Dirichlet form of the target guide: kept bonds = frozen-FN kinetic metric, violating bonds = wall term)
B: L = E_{x ~ a_k^2} sum_{valid b} (J_b/2) a_k(x^b)/a_k(x) (R_b(x) - r_b(x))^2 / N, all bonds, random D4 x T x flip image.
Sampling: guide-metric tail proposal 1/2 n a_k + 1/2 Dirichlet node weight of D_k = la_k - log|psi_P| (as writeback
pre-test (a)), restricted to training orbits (80%); every bond into a held-out orbit has weight 0.
Optimizer: Adam, warmup + cosine; final parameters are evaluated (no selection).
  python cx_distill.py OUT SPEC.json
"""
import os, sys, time
import numpy as np
from cx_lib import *
import optax
from lanczos_lib import lanczos_ritz

OUT = sys.argv[1]; SPEC = json.load(open(sys.argv[2])); os.makedirs(OUT, exist_ok=True)
F = os.path.join(OUT, 'cx_distill.json'); T00 = time.time()
ARM = SPEC['arm']; K_IT = SPEC['k']; SEED = SPEC.get('seed', 0)
SMOKE = SPEC.get('smoke', False)
lab = Lab(); sec = lab.sec
if SMOKE:                                    # throughput / code-path check only: target = psi_P itself
    REFS = {}; la_k, s_k = lab.lP0, lab.sP0
else:
    REFS = json.load(open(SPEC.get('refs', os.path.join(TABDIR, 'cx_replay.json'))))['refs'][str(K_IT)]
    la_k, s_k = load_guide(K_IT)
res = dict(spec=SPEC, refs={q: v for q, v in REFS.items() if q != 'base_fidelity'})


# ============================================================================================ students
class VitStudent:
    def __init__(self, mode, seed):
        import flax, vit_dt
        vit_dt.set_dtype('float32')
        vit = vit_dt.make_model_6x6()
        tmpl = vit.init(jax.random.PRNGKey(1234 + seed), jnp.zeros((1, N), jnp.float32))
        if mode == 'warm':
            obj = flax.serialization.msgpack_restore(open(VIT_CKPT, 'rb').read())
            if 'variables' in obj: obj = obj['variables']
            tmpl = flax.serialization.from_state_dict(tmpl, obj)
        params = jax.tree_util.tree_map(lambda q: jnp.asarray(q, jnp.float32), tmpl['params'])
        self.flat0, unravel = ravel_pytree(params)
        self.npar = int(self.flat0.size)

        def f(flat, X):                                        # X (B, N) +-1 -> (B,) log A, exactly symmetric
            p = {'params': unravel(flat)}
            B_ = X.shape[0]
            Xg = jnp.stack([X[:, INV8[k // 2]] * SG2[k % 2] for k in range(16)], 0).reshape(16 * B_, N)
            zs = vit_dt.logpsi_transl_2d(vit.apply, 2, p, Xg).reshape(16, B_)
            if mode == 'warm':
                m = jnp.max(jnp.real(zs), 0)
                return m + jnp.log(jnp.abs(jnp.sum(jnp.exp(zs - m[None, :]), 0)))
            return jnp.mean(jnp.real(zs), 0)
        self.f = f
        self._tab = jax.jit(lambda flat, Sx: f(flat, bits(Sx)))

    def table(self, flat, reps, batch=4096):
        D = reps.shape[0]; out = []
        for i in range(0, D, batch):
            s = reps[i:i + batch]; m = s.shape[0]
            if m < batch: s = jnp.concatenate([s, jnp.repeat(s[:1], batch - m)])
            out.append(self._tab(flat, s)[:m].astype(f64))
        return jnp.concatenate(out)


class BondCNN(nn.Module):
    C: int = 64
    layers: int = 6
    hid: int = 64

    @nn.compact
    def __call__(self, x):
        kw = dict(dtype=jnp.float32, param_dtype=jnp.float32)

        def conv(h):
            g = h[:, NBR, :]
            return nn.Dense(self.C, **kw)(g.reshape(g.shape[0], N, -1))
        h = nn.gelu(conv(x[:, :, None]))
        for _ in range(1, self.layers):
            h = h + nn.gelu(conv(nn.LayerNorm(**kw)(h)))
        h = nn.LayerNorm(**kw)(h)
        hi, hj = h[:, BI_I, :], h[:, BJ_I, :]
        si = x[:, BI_I][:, :, None]
        z = jnp.concatenate([hi + hj, hi * hj, si * (hi - hj)], -1)          # swap-symmetric bond features
        Fz = z.shape[-1]
        W1 = self.param('W1', nn.initializers.lecun_normal(), (4, Fz, self.hid), jnp.float32)
        b1 = self.param('b1', nn.initializers.zeros, (4, self.hid), jnp.float32)
        W2 = self.param('W2', nn.initializers.lecun_normal(), (4, self.hid, self.hid), jnp.float32)
        b2 = self.param('b2', nn.initializers.zeros, (4, self.hid), jnp.float32)
        w3 = self.param('w3', nn.initializers.lecun_normal(), (4, self.hid, 1), jnp.float32)
        b3 = self.param('b3', nn.initializers.zeros, (4,), jnp.float32)
        u = nn.gelu(jnp.einsum('bnf,nfh->bnh', z, W1[BTYPE]) + b1[BTYPE][None])
        u = nn.gelu(jnp.einsum('bnf,nfh->bnh', u, W2[BTYPE]) + b2[BTYPE][None])
        return jnp.einsum('bnf,nf->bn', u, w3[BTYPE][:, :, 0]) + b3[BTYPE][None]


class BondStudent:
    def __init__(self, seed, C=64, layers=6):
        self.net = BondCNN(C, layers)
        params = self.net.init(jax.random.PRNGKey(seed), jnp.zeros((1, N), jnp.float32))['params']
        self.flat0, unravel = ravel_pytree(params)
        self.npar = int(self.flat0.size)
        net = self.net
        self.f = lambda flat, X: net.apply({'params': unravel(flat)}, X)
        self._R = jax.jit(lambda flat, Sx: self.f(flat, bits(Sx)))


# ============================================================================================ sampling
Dk = la_k - lab.lP0
Q, PA, _ = lab.proposal_q(Dk, la_k, s_k); del Dk
Qtr = jnp.where(lab.HO, 0.0, Q); Qtr = Qtr / jnp.sum(Qtr)
Qte = jnp.where(lab.HO, Q, 0.0); Qte = Qte / jnp.sum(Qte)
CDF_TR, CDF_TE = jnp.cumsum(Qtr), jnp.cumsum(Qte)
LIW_TR = jnp.where(Qtr > 0, jnp.log(jnp.maximum(PA, 1e-300)) - jnp.log(jnp.maximum(Qtr, 1e-300)), -jnp.inf)
LIW_TE = jnp.where(Qte > 0, jnp.log(jnp.maximum(PA, 1e-300)) - jnp.log(jnp.maximum(Qte, 1e-300)), -jnp.inf)
del Q, Qtr, Qte, PA
B_, K_ = SPEC.get('B', 128), SPEC.get('K', 8)
LAM = SPEC.get('lam_v', 0.1)
HOJ = lab.HO


def draw(key, cdf):
    return jnp.clip(jnp.searchsorted(cdf, jax.random.uniform(key, (B_,), f64) * cdf[-1]), 0, sec.D - 1)


def make_batch_scalar(cdf, liw, withhold):
    @jax.jit
    def batch(key):
        k1, k2 = jax.random.split(key)
        idx = draw(k1, cdf)
        x = sec.reps[idx]
        valid = valid_bonds(x); nval = valid.sum(1)
        sc = jnp.where(valid, jax.random.uniform(k2, valid.shape), -1.0)
        _, bsel = jax.lax.top_k(sc, K_)
        ok = jnp.take_along_axis(valid, bsel, 1)
        y = x[:, None] ^ MASKS[bsel]
        iy = SS.canon(sec.T, sec.reps, y.reshape(-1)).reshape(B_, K_)
        r = jnp.clip(la_k[iy] - la_k[idx][:, None], -60, 60)
        w = jnp.where(ok, 0.5 * JB[bsel] * jnp.exp(r), 0.0) * (nval[:, None] / K_)
        if withhold: w = jnp.where(HOJ[iy], 0.0, w)
        lw = liw[idx]; iw = jnp.exp(lw - jnp.max(lw))
        return x, y, la_k[idx], la_k[iy], w, iw
    return batch


def make_batch_bond(cdf, liw, withhold):
    @jax.jit
    def batch(key):
        k1, k2, k3 = jax.random.split(key, 3)
        idx = draw(k1, cdf)
        g = jax.random.randint(k2, (B_,), 0, sec.T.shape[0]); fl = jax.random.bernoulli(k3, 0.5, (B_,))
        gx = SS.image(sec.T, sec.reps[idx], g, fl)                     # random D4 x T x flip image (augmentation)
        valid = valid_bonds(gx)
        y = jnp.where(valid, gx[:, None] ^ MASKS[None, :], gx[:, None])
        iy = SS.canon(sec.T, sec.reps, y.reshape(-1)).reshape(B_, NB)
        r = jnp.clip(la_k[iy] - la_k[idx][:, None], -60, 60)
        w = jnp.where(valid, 0.5 * JB[None, :] * jnp.exp(r), 0.0)
        if withhold: w = jnp.where(HOJ[iy], 0.0, w)
        lw = liw[idx]; iw = jnp.exp(lw - jnp.max(lw))
        return gx, r, w, iw
    return batch


def loss_scalar(model, flat, x, y, tx, ty, w, iw):
    S = model.f(flat, bits(jnp.concatenate([x, y.reshape(-1)]))).astype(f64)
    ex = S[:B_] - tx; ey = S[B_:].reshape(B_, K_) - ty
    iw = iw / jnp.mean(iw)
    Lr = jnp.mean(iw * jnp.sum(w * (ex[:, None] - ey) ** 2, 1)) / N
    m = jnp.mean(iw * ex); Lv = jnp.mean(iw * ex * ex) - m * m
    return Lr + LAM * Lv, (Lr, Lv)


def loss_bond(model, flat, gx, r, w, iw):
    R = model.f(flat, bits(gx)).astype(f64)
    iw = iw / jnp.mean(iw)
    L = jnp.mean(iw * jnp.sum(w * (R - r) ** 2, 1)) / N
    return L, (L, 0.0 * L)


# ============================================================================================ train
if ARM in ('vit_warm', 'vit_scratch'):
    model = VitStudent('warm' if ARM == 'vit_warm' else 'scratch', SEED)
    mb = make_batch_scalar; lossf = loss_scalar
else:
    model = BondStudent(SEED, SPEC.get('C', 64), SPEC.get('layers', 6))
    mb = make_batch_bond; lossf = loss_bond
res['npar'] = model.npar
if ARM == 'vit_warm':
    chk = np.asarray(model._tab(model.flat0, sec.reps[:4096]).astype(f64) - lab.lP0[:4096])
    res['warm_start_check'] = dict(mean=float(chk.mean()), rms=float(chk.std()), max=float(np.abs(chk - chk.mean()).max()))
    log(f'[warm start] log A - log|psi_P| on 4096 reps: {res["warm_start_check"]}')
batch_tr = mb(CDF_TR, LIW_TR, True)
batch_va = mb(CDF_TR, LIW_TR, True)
batch_te = mb(CDF_TE, LIW_TE, False)                                   # held-out x, all bonds
steps = SPEC.get('steps', 8000); lr = SPEC.get('lr', 1e-4)
sched = optax.warmup_cosine_decay_schedule(0.0, lr, SPEC.get('warmup', 300), steps, lr * 0.02)
opt = optax.adam(sched)
flat = model.flat0; ost = opt.init(flat)


@jax.jit
def step(fl, ost, key):
    bd = batch_tr(key)
    (l, aux), g = jax.value_and_grad(lambda q: lossf(model, q, *bd), has_aux=True)(fl)
    upd, ost = opt.update(g, ost, fl)
    return optax.apply_updates(fl, upd), ost, l, aux


lval = jax.jit(lambda fl, bd: lossf(model, fl, *bd))
VB = [batch_va(jax.random.PRNGKey(9000 + i)) for i in range(8)]
TB = [batch_te(jax.random.PRNGKey(9500 + i)) for i in range(8)]


def val(fl):
    a = np.array([[float(q) for q in (lambda o: (o[0], o[1][0], o[1][1]))(lval(fl, bd))] for bd in VB])
    b = np.array([[float(q) for q in (lambda o: (o[0], o[1][0], o[1][1]))(lval(fl, bd))] for bd in TB])
    return dict(train=a.mean(0).tolist(), test=b.mean(0).tolist())


sec.offload()
curve = [dict(step=0, sec=0.0, **val(flat))]
log(f'[{ARM} k={K_IT} seed {SEED}] npar {model.npar}  step 0 val {curve[-1]}')
key = jax.random.PRNGKey(100 + SEED); tt = time.time(); run_loss = []
every = SPEC.get('val_every', 500)
for it in range(1, steps + 1):
    key, kk = jax.random.split(key)
    flat, ost, l, aux = step(flat, ost, kk)
    run_loss.append(float(l)) if it % 50 == 0 else None
    if it % every == 0 or it == steps:
        curve.append(dict(step=it, sec=time.time() - tt, run_loss=float(np.mean(run_loss[-10:])) if run_loss else None,
                          **val(flat)))
        log(f'  step {it} {curve[-1]}')
        res['curve'] = curve; dump(F, res)
res['train_sec'] = time.time() - tt
np.save(os.path.join(OUT, 'params_final.npy'), np.asarray(flat))
sec.reload()


# ============================================================================================ cost
def timeit(fn, *a, n=5):
    jax.block_until_ready(fn(*a)); ts = []
    for _ in range(n):
        t0 = time.time(); jax.block_until_ready(fn(*a)); ts.append(time.time() - t0)
    return float(np.median(ts))


xs = sec.reps[:4096]
base = VitStudent('warm', 0)
t_base = timeit(base._tab, base.flat0, xs)
t_st = timeit(model._tab, flat, xs) if hasattr(model, '_tab') else timeit(model._R, flat, xs)
res['cost'] = dict(sec_per_4096_base=t_base, sec_per_4096_student=t_st, base_equiv=t_st / t_base,
                   note='base = symmetrised ViT psi_P (16 images x 4 patch translations), fp32, same batch')
log(f'[cost] base {t_base:.4f}s student {t_st:.4f}s per 4096 -> {t_st / t_base:.3f} base passes')
dump(F, res)

# ============================================================================================ exact evaluation
dec, _ = lab.decade_of(la_k)
smp = sample_by_decade(lab, dec, SPEC.get('fid_M', 2048), seed=7)
te = time.time()
if SMOKE:
    nt = 262144
    if ARM in ('vit_warm', 'vit_scratch'):
        t0 = time.time(); _ = model.table(flat, sec.reps[:nt]); jax.block_until_ready(_)
        res['smoke_table_sec_full_est'] = (time.time() - t0) * sec.D / nt
    else:
        ii = jnp.arange(nt)
        v, iy = neighbours(lab, ii[:16384]); jax.block_until_ready(iy)
        t0 = time.time()
        for i in range(0, nt, 16384):
            v, iy = neighbours(lab, ii[i:i + 16384]); jax.block_until_ready(iy)
        res['smoke_canon_sec_full_est'] = (time.time() - t0) * sec.D / nt
    t0 = time.time()
    smp_s = {q: v_[:64] for q, v_ in list(smp.items())[:4]}
    res['smoke_fid'] = fidelity(lab, la_k, s_k, smp_s, lambda idx, v, iy: la_k[iy] - la_k[idx][:, None] + 0.01, la_s=la_k)
    res['smoke_fid_sec'] = time.time() - t0
    res['sec'] = time.time() - T00; dump(F, res); log('SMOKE DONE', json.dumps({q: res[q] for q in res if q.startswith('smoke') and q != 'smoke_fid'}))
    sys.exit(0)
if ARM in ('vit_warm', 'vit_scratch'):
    la_S = model.table(flat, sec.reps)
    ev = dict(table_sec=time.time() - te)
    ev['H_same'] = lab.H_site(la_S, s_k)
    Efn_S, u, info = sec.fn_solve(la_S, s_k); del u
    ev['E_FN'] = info['dE_FN_site']
    ev['frozen'] = (lab.frozen(la_k, s_k, la_S) - sec.E0) / N
    rl = lanczos_ritz(lambda x: sec.Hm(x), sec.vec(la_S, s_k), 1)[1]
    ev['lanczos_H'] = (rl['E'] - sec.E0) / N; del rl
    s_kr, _, _ = sec.krylov(sec.vec(la_S, jnp.ones_like(la_S)), s_k)
    ev['H_kry_own'] = lab.H_site(la_S, s_kr); del s_kr
    R = REFS
    ev['share_H'] = (R['H_base_sk'] - ev['H_same']) / (R['H_base_sk'] - R['H_stack'])
    ev['share_EFN'] = (R['E_FN_base_sk'] - ev['E_FN']) / (R['E_FN_base_sk'] - R['E_FN_stack'])
    ev['share_frozen'] = (R['frozen_base'] - ev['frozen']) / (R['frozen_base'] - R['frozen_stack'])
    ev['kept_preA'] = (H_START - ev['H_same']) / (H_START - R['H_stack'])
    log('[eval] ' + json.dumps(ev))
    res['eval'] = ev; dump(F, res)
    # continuation: stored FEAT net k+1 driven by the student's own one-hop quantities (no retraining)
    _, _, rc = loop_step(lab, la_S, s_k, load_loop_params(K_IT + 1), Efn=Efn_S, krylov=True)
    rc['frac_rel_table'] = rc['frac'] / R['continuation_frac_table']
    ev['continuation'] = rc
    log('[continuation] ' + json.dumps(rc))
    res['eval'] = ev; dump(F, res)
    ev['fidelity'] = fidelity(lab, la_k, s_k, smp, lambda idx, v, iy: la_S[iy] - la_S[idx][:, None], la_s=la_S)
    if SPEC.get('save_table'):
        np.save(os.path.join(OUT, 'la_student.npy'), np.asarray(la_S))
else:
    Rf = model._R
    ev = dict(fidelity=fidelity(lab, la_k, s_k, smp, lambda idx, v, iy: Rf(flat, sec.reps[idx]).astype(f64)))
    res['eval'] = ev; dump(F, res)
    if SPEC.get('continuation', True):
        # V_B, W_B on every orbit from one forward pass of R (neighbour signs from the table s_k)
        Vb = []; Wb = []; ch = SPEC.get('vw_chunk', 16384)

        @jax.jit
        def vw_chunk(fl, ii):
            x = sec.reps[ii]
            v = valid_bonds(x)
            y = jnp.where(v, x[:, None] ^ MASKS[None, :], x[:, None])
            iy = SS.canon(sec.T, sec.reps, y.reshape(-1)).reshape(x.shape[0], NB)
            kept = v & (s_k[ii][:, None] * s_k[iy] < 0)
            wr = jnp.where(v, 0.5 * JB[None, :] * jnp.exp(jnp.clip(Rf(fl, x).astype(f64), -60, 60)), 0.0)
            return jnp.sum(jnp.where(v & ~kept, wr, 0), 1), jnp.sum(jnp.where(kept, wr, 0), 1)
        tv = time.time()
        for i in range(0, sec.D, ch):
            ii = jnp.arange(i, i + ch); ii = jnp.minimum(ii, sec.D - 1)
            a_, b_ = vw_chunk(flat, ii); m = min(ch, sec.D - i)
            Vb.append(a_[:m]); Wb.append(b_[:m])
        Vb = jnp.concatenate(Vb); Wb = jnp.concatenate(Wb)
        ev['vw_sec'] = time.time() - tv
        lan, V, W = lab.VW(la_k, s_k)
        PA = lab.UA(la_k) ** 2
        okv = V > 1e-12
        ev['logV_rms_pa'] = float(jnp.sqrt(jnp.sum(jnp.where(okv, PA * jnp.log(jnp.maximum(Vb, 1e-300) / jnp.maximum(V, 1e-300)) ** 2, 0)) / jnp.sum(jnp.where(okv, PA, 0))))
        ev['logW_rms_pa'] = float(jnp.sqrt(jnp.sum(PA * jnp.log(jnp.maximum(Wb, 1e-300) / jnp.maximum(W, 1e-300)) ** 2)))
        del V, W, PA
        Efn_k = sec.E0 + N * REFS['E_FN_stack']
        _, _, rc = loop_step(lab, la_k, s_k, load_loop_params(K_IT + 1), Efn=Efn_k, feats=(lan, Vb, Wb), krylov=True)
        rc['frac_rel_table'] = rc['frac'] / REFS['continuation_frac_table']
        ev['continuation'] = rc
        log('[continuation B] ' + json.dumps(rc) + f'  logV {ev["logV_rms_pa"]:.4f} logW {ev["logW_rms_pa"]:.4f}')
ev['eval_sec'] = time.time() - te
res['eval'] = ev; res['sec'] = time.time() - T00; dump(F, res); log('DONE', res['sec'])
