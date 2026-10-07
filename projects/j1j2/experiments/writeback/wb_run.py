"""Write-back test, 6x6 J1-J2 (J2/J1 = 0.5), exact symmetric sector (k=0, A1, flip+), one GPU.

Question: can the first exact FN update from the symmetrised ViT psi_P be written back into a callable network
accurately enough to keep its energy gain?  Candidate (results/writeback/DESIGN.md, rank 1):
  b(x) = |psi_P(x)| * exp(f_theta(x)),   psi_P frozen, f_theta a fresh, exactly symmetric residual CNN (f = 0 at start),
  trained on the frozen-FN energy  E_f[b] = <b|H_FN[psi_P, s_P]|b>/<b|b>  (unrestricted minimiser: phi_FN).
Control: the same network and samples trained on the infidelity 1 - |<b|phi_FN>|^2/(<b|b><phi|phi>) (pointwise L2).

Both losses are estimated on configurations x ~ phi_FN^2 (exact sampling from the sector table) using the exact
ground-state-transform identity (A = H_FN stoquastic, A phi = E_FN phi, g = b/phi):
  E_f[b] - E_FN = 1/2 sum_x phi_x^2 sum_{y allowed} |H_xy| (phi_y/phi_x) (g_x - g_y)^2  /  sum_x phi_x^2 g_x^2
  1 - F       = 1 - (sum_x phi_x^2 g_x)^2 / sum_x phi_x^2 g_x^2
The sum over y uses K random valid bonds of x (importance weight nvalid/K); 'allowed' = s_x s_y = -1.
Every eval_every steps the network is evaluated on all 15.8M reps and both objectives are computed EXACTLY.

  python wb_run.py OUT SPEC.json
"""
import json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
for p_ in (HERE, os.path.join(HERE, '..', 'stall_6x6')):
    if p_ not in sys.path: sys.path.insert(0, p_)
import st6_sector as SS
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from jax.flatten_util import ravel_pytree
jax.config.update('jax_default_matmul_precision', 'highest')   # full fp32 (no TF32 on A40)

ROOT = '/project/theorie/a/A.Otaifi/chatty_stall6'
CSR = os.environ.get('ST6_CSR', ROOT + '/csr')
TABLE = os.environ.get('ST6_TABLE', ROOT + '/data/psi0_6x6_table.npz')
SYM = os.environ.get('ST6_SYM', ROOT + '/data/sym_tables.npz')
N = 36
log = SS.log
OUT = sys.argv[1]
SPEC = json.load(open(sys.argv[2]))
os.makedirs(OUT, exist_ok=True)
T00 = time.time()
AR = jnp.arange(N, dtype=jnp.uint64)
f64 = jnp.float64


def dump(name, obj):
    json.dump(obj, open(os.path.join(OUT, name), 'w'), indent=1, default=float)


def bits(S):
    return (((S[:, None] >> AR[None, :]) & jnp.uint64(1)).astype(jnp.float32) * 2 - 1)


# ============================================================================================ network
class ResCNN(nn.Module):
    """periodic residual CNN, per-site features summed over sites (translation invariant), zero-initialised head."""
    C: int = 32
    layers: int = 4

    @nn.compact
    def __call__(self, x):
        kw = dict(dtype=jnp.float32, param_dtype=jnp.float32)
        h = x.reshape(x.shape[0], 6, 6, 1)
        h = nn.gelu(nn.Conv(self.C, (3, 3), padding='CIRCULAR', **kw)(h))
        for _ in range(1, self.layers):
            h = h + nn.gelu(nn.Conv(self.C, (3, 3), padding='CIRCULAR', **kw)(nn.LayerNorm(**kw)(h)))
        h = h.sum(axis=(1, 2)) / 6.0
        return nn.Dense(1, kernel_init=nn.initializers.zeros, **kw)(h)[:, 0]


class Corr:
    """f(x) = 1/16 sum_{g in D4, flip} net(g x): exactly invariant under the full space group x spin flip."""

    def __init__(self, C, layers, seed=0, head_init=0.0):
        self.net = ResCNN(C, layers)
        params = self.net.init(jax.random.PRNGKey(seed), jnp.zeros((1, N), jnp.float32))['params']
        if head_init:
            params = jax.tree_util.tree_map(lambda q: q, params)
            k = params['Dense_0']['kernel']
            params['Dense_0']['kernel'] = head_init * jax.random.normal(jax.random.PRNGKey(seed + 7), k.shape, k.dtype)
        self.flat0, self.unravel = ravel_pytree(params)
        self.npar = int(self.flat0.size)
        inv = jnp.asarray(np.argsort(SS.space_group()[:8], axis=1))
        net, unravel = self.net, self.unravel
        app = jax.checkpoint(lambda p, X: net.apply({'params': p}, X))

        def f(flat, X):
            p = unravel(flat)

            def body(acc, k):
                Xk = X[:, inv[k // 2]] * jnp.where(k % 2 == 0, 1.0, -1.0).astype(jnp.float32)
                return acc + app(p, Xk), None
            acc, _ = jax.lax.scan(body, jnp.zeros(X.shape[0], jnp.float32), jnp.arange(16))
            return acc / 16.0
        self.f = f
        self._tab = jax.jit(lambda flat, S: f(flat, bits(S)))

    def table(self, flat, reps, batch=16384):
        D = reps.shape[0]; out = []
        for i in range(0, D, batch):
            s = reps[i:i + batch]; m = s.shape[0]
            if m < batch: s = jnp.concatenate([s, jnp.repeat(s[:1], batch - m)])
            out.append(self._tab(flat, s)[:m].astype(f64))
        return jnp.concatenate(out)


# ============================================================================================ setup (exact)
sec = SS.Sector(CSR, TABLE)
z = np.load(SYM)
lP = jnp.asarray(z['lP']); sP = jnp.asarray(z['sP'].astype(np.float32)); del z
res = dict(spec=SPEC)
Efn, u, info = sec.fn_solve(lP, sP)
res['FN_psiP'] = info
lphi = jnp.log(jnp.maximum(u / sec.sqrt_n, 1e-300))
p = u * u                                                     # sector probability of phi_FN (sum 1)
E_f0 = sec.fn_rayleigh(lP, sP, lP)
G0 = (E_f0 - Efn) / N                                         # frozen-FN gain of one exact iteration, per site
d = lphi - lP; mu = float(jnp.sum(p * d))
var_delta = float(jnp.sum(p * (d - mu) ** 2))


def Kop(x):                                                   # allowed-edge part |A_off| on sector vectors
    h1, h2 = sec.H2(x, sP * x)
    return 0.5 * (h1 - sP * h2)


Ku = Kop(u)
W = Ku / jnp.maximum(u, 1e-300)                                                    # = sum_{y allowed} |H_xy| phi_y/phi_x  (= D_FN - E_FN)
kappa = float(jnp.sum(p * W)) / N                            # white-noise cost per site per sigma^2 (FN metric)
gg = jnp.exp(-(d - mu))
c_edge = 0.5 * (gg * gg * u * Ku - 2 * gg * u * Kop(u * gg) + u * Kop(u * gg * gg))
G_edge = float(jnp.sum(c_edge) / jnp.sum(p * gg * gg)) / N
pc = p / sec.n                                                # per-configuration probability
dec = jnp.floor(jnp.log10(jnp.maximum(pc, 1e-300))).astype(jnp.int32)
tot_c = float(jnp.sum(c_edge)); tot_l2 = float(jnp.sum(p * (d - mu) ** 2)); tot_w = float(jnp.sum(p * W))
decs = []
for k in range(int(jnp.min(dec)), int(jnp.max(dec)) + 1):
    m = dec == k
    if not bool(jnp.any(m)): continue
    decs.append(dict(decade=k, mass=float(jnp.sum(jnp.where(m, p, 0))),
                     l2_share=float(jnp.sum(jnp.where(m, p * (d - mu) ** 2, 0))) / tot_l2,
                     gain_share=float(jnp.sum(jnp.where(m, c_edge, 0))) / tot_c,
                     whitenoise_share=float(jnp.sum(jnp.where(m, p * W, 0))) / tot_w))
res['diag'] = dict(E_FN_dE_site=(Efn - sec.E0) / N, frozen_psiP_dE_site=(E_f0 - sec.E0) / N, G0_frozen_gain_site=G0,
                   G0_edge_identity=G_edge, rms_delta=var_delta ** 0.5, kappa_FN_whitenoise=kappa,
                   delta_roughness_vs_white=G0 / (var_delta * kappa), decades=decs)
log('DIAG', json.dumps({k: v for k, v in res['diag'].items() if k != 'decades'}))
for r in decs: log('  decade', r)
del Ku, W, gg, c_edge, pc, dec
# reference guides: psi_P, phi_FN (own sign), phi_FN + Krylov (= exact loop iteration 2 guide)
s1, _, _ = sec.krylov(u, sP)
_, _, info2 = sec.fn_solve(lphi, s1)
res['ref'] = dict(H_psiP=sec.score(lP, sP)['dE_site'], H_phi_ownsign=sec.score(lphi, sP)['dE_site'],
                  H_phi_kry=sec.score(lphi, s1)['dE_site'], FN_phi_kry=info2['dE_FN_site'])
del s1
log('REF', res['ref'])
dump('wb.json', res)

cdf = jnp.cumsum(p)
dl = (lP - lphi + mu)                                         # log g at f = 0 (centred under phi^2)
MASKS = jnp.asarray(SS.MASKS); BI = jnp.asarray(SS.BI); BJ = jnp.asarray(SS.BJ); JB = jnp.asarray(SS.JB)


def make_batch(B, K):
    @jax.jit
    def batch(key):
        k1, k2 = jax.random.split(key)
        uu = jax.random.uniform(k1, (B,), f64) * cdf[-1]
        idx = jnp.clip(jnp.searchsorted(cdf, uu), 0, sec.D - 1)
        x = sec.reps[idx]
        valid = (((x[:, None] >> BI[None, :]) ^ (x[:, None] >> BJ[None, :])) & jnp.uint64(1)).astype(bool)
        nval = valid.sum(1)
        sc = jnp.where(valid, jax.random.uniform(k2, valid.shape), -1.0)
        _, bsel = jax.lax.top_k(sc, K)
        ok = jnp.take_along_axis(valid, bsel, 1)
        y = x[:, None] ^ MASKS[bsel]
        iy = SS.canon(sec.T, sec.reps, y.reshape(-1)).reshape(B, K)
        allowed = ok & (sP[idx][:, None] * sP[iy] < 0)
        w = jnp.where(allowed, 0.5 * JB[bsel] * jnp.exp(lphi[iy] - lphi[idx][:, None]), 0.0) * (nval[:, None] / K)
        return x, y, dl[idx], dl[iy], w
    return batch


def losses(model, flat, x, y, dlx, dly, w, need_y=True):
    B, K = y.shape
    if need_y:
        fa = model.f(flat, bits(jnp.concatenate([x, y.reshape(-1)]))).astype(f64)
        fx, fy = fa[:B], fa[B:].reshape(B, K)
    else:
        fx = model.f(flat, bits(x)).astype(f64); fy = None
    gx = jnp.exp(fx + dlx)
    den = jnp.mean(gx * gx)
    LI = 1.0 - jnp.mean(gx) ** 2 / den
    if fy is None:
        return jnp.nan, LI
    gy = jnp.exp(fy + dly)
    LE = 0.5 * jnp.mean(jnp.sum(w * (gx[:, None] - gy) ** 2, 1)) / den / N
    return LE, LI


def exact_eval(model, flat, final=False):
    sec.offload()
    fb = model.table(flat, sec.reps)
    sec.reload()
    la = lP + fb
    ef = sec.fn_rayleigh(lP, sP, la)
    b = sec.vec(la, jnp.ones_like(la))
    infid = 1.0 - float(b @ u) ** 2
    e = la - lphi; me = float(jnp.sum(p * e)); rms_e = float(jnp.sqrt(jnp.sum(p * (e - me) ** 2)))
    fm = float(jnp.sum(p * fb)); rms_f = float(jnp.sqrt(jnp.sum(p * (fb - fm) ** 2)))
    out = dict(frozenFN_minus_EFN_site=(ef - Efn) / N, frac_gain=(E_f0 - ef) / (E_f0 - Efn), infid=infid,
               rms_err_vs_phi=rms_e, rms_f=rms_f, H_ownsign=sec.score(la, sP)['dE_site'])
    out['err_roughness_vs_white'] = out['frozenFN_minus_EFN_site'] / max(rms_e ** 2 * kappa, 1e-300)
    if final:
        _, _, i1 = sec.fn_solve(la, sP); out['FN_ownsign'] = i1['dE_FN_site']
        v = sec.vec(la, jnp.ones_like(la)); sk, _, _ = sec.krylov(v, sP); del v
        out['H_kry'] = sec.score(la, sk)['dE_site']
        _, _, i2 = sec.fn_solve(la, sk); out['FN_kry'] = i2['dE_FN_site']
    del fb, la, b, e
    return out


# ============================================================================================ estimator check
if SPEC.get('check', True):
    t0 = time.time()
    chk = []
    for hi in (0.0, 0.05):
        mdl = Corr(32, 2, seed=3, head_init=hi)
        bt = make_batch(1024, 8)
        lf = jax.jit(lambda fl, x, y, a, b_, w: losses(mdl, fl, x, y, a, b_, w))
        est = np.array([[float(q) for q in lf(mdl.flat0, *bt(jax.random.PRNGKey(500 + i)))] for i in range(64)])
        ex = exact_eval(mdl, mdl.flat0)
        r = dict(head_init=hi, LE_est=est[:, 0].mean(), LE_se=est[:, 0].std() / 8, LE_exact=ex['frozenFN_minus_EFN_site'],
                 LI_est=est[:, 1].mean(), LI_se=est[:, 1].std() / 8, LI_exact=ex['infid'], rms_f=ex['rms_f'])
        chk.append(r); log('CHECK', r)
    res['check'] = chk; res['check_sec'] = time.time() - t0
    dump('wb.json', res)

# ============================================================================================ arms
res['arms'] = []
for arm in SPEC['arms']:
    tag = arm['tag']; t0 = time.time()
    model = Corr(arm['C'], arm['layers'], seed=arm.get('seed', 0))
    B, K, steps = arm.get('B', 1024), arm.get('K', 8), arm['steps']
    loss_kind = arm['loss']                                   # 'energy' | 'infid'
    sched = optax.warmup_cosine_decay_schedule(0.0, arm['lr'], arm.get('warmup', 200), steps, arm['lr'] * 0.02)
    opt = optax.adam(sched)
    flat = model.flat0; ost = opt.init(flat)
    bt = make_batch(B, K)
    need_y = loss_kind == 'energy'

    @jax.jit
    def step(fl, ost, key):
        x, y, dlx, dly, w = bt(key)

        def L(q):
            LE, LI = losses(model, q, x, y, dlx, dly, w, need_y=need_y)
            return (LE if loss_kind == 'energy' else LI), (LE, LI)
        (l, (LE, LI)), g = jax.value_and_grad(L, has_aux=True)(fl)
        upd, ost = opt.update(g, ost, fl)
        return optax.apply_updates(fl, upd), ost, LE, LI

    vb = make_batch(4096, 8)
    vkeys = [jax.random.PRNGKey(90_000 + i) for i in range(8)]
    vl = jax.jit(lambda fl, x, y, a, b_, w: losses(model, fl, x, y, a, b_, w))

    def val(fl):
        r = np.array([[float(q) for q in vl(fl, *vb(k))] for k in vkeys])
        return r.mean(0)
    log(f'== arm {tag}: loss {loss_kind} C {arm["C"]} layers {arm["layers"]} npar {model.npar} B {B} K {K} steps {steps}')
    hist = []
    ev = exact_eval(model, flat); ev.update(step=0, sec=0.0); hist.append(ev); log('  exact', ev)
    own = 'frozenFN_minus_EFN_site' if loss_kind == 'energy' else 'infid'
    best = (ev[own], flat, ev)
    key = jax.random.PRNGKey(1000 + arm.get('seed', 0))
    sec.offload()
    acc = []
    for it in range(1, steps + 1):
        key, k = jax.random.split(key)
        flat, ost, LE, LI = step(flat, ost, k)
        if it % 250 == 0:
            v = val(flat)
            acc.append(dict(step=it, val_LE=float(v[0]), val_LI=float(v[1]), sec=time.time() - t0))
            log('  val', acc[-1])
        if it % arm.get('eval_every', 1000) == 0 or it == steps:
            ev = exact_eval(model, flat, final=False); ev.update(step=it, sec=time.time() - t0)
            hist.append(ev); log('  exact', ev)
            if ev[own] < best[0]: best = (ev[own], flat, ev)
            sec.offload()
            if arm.get('max_sec') and time.time() - t0 > arm['max_sec']:
                log('  time limit for arm reached'); break
    np.save(os.path.join(OUT, f'params_{tag}_last.npy'), np.asarray(flat))
    np.save(os.path.join(OUT, f'params_{tag}_best.npy'), np.asarray(best[1]))
    fin_last = exact_eval(model, flat, final=arm.get('final', True))
    fin_best = exact_eval(model, best[1], final=arm.get('final', True))
    r = dict(tag=tag, arm=arm, npar=model.npar, hist=hist, val=acc, final_last=fin_last, final_best=fin_best,
             best_step=best[2]['step'], sec=time.time() - t0)
    res['arms'].append(r)
    log(f'== done {tag}', json.dumps(dict(last=fin_last, best=fin_best)))
    dump('wb.json', res)
    del model, flat, ost, best
res['sec'] = time.time() - T00
dump('wb.json', res)
log('DONE', res['sec'])
