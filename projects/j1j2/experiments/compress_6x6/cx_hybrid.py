"""Hybrid re-anchor test (README "Hybrid re-anchor test", pre-registered before any run). ONE iteration per job.
G_k = log A_theta_k (re-anchored ViT table) + c_k (FEAT net on ONE hop of the current ViT, refit each iteration to
the whole residual, not stacked).  Iteration k: exact clipped-Lanczos target la_t from (G_k, s_k); ViT fine-tune to
la_t (cx_reanchor.py in a subprocess, minSR, fit_only; k = 1 reuses ra_ms_s1_k1_17174477); correction c fit to
la_t - la_A on (log A, log V, log W, asinh E_L) of (la_A, s_{k+1}); guard; exact evaluation (<H>, own E_FN, ViT alone).
  python cx_hybrid.py OUT SPEC.json    SPEC: {"k": 1|2|3, "prev": hybrid run dir of k-1, "seed": 1}
"""
import os, sys, time, json, subprocess
import numpy as np
OUT = sys.argv[1]; SPEC = json.load(open(sys.argv[2])); os.makedirs(OUT, exist_ok=True)
K_IT = int(SPEC['k']); SEED = int(SPEC.get('seed', 1))
RUNS = '/project/theorie/a/A.Otaifi/chatty_compress6/runs'
F = os.path.join(OUT, 'cx_hybrid.json'); T00 = time.time()
res = dict(spec=SPEC)
HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- continuation rule, then the ViT re-anchor (subprocess)
if K_IT >= 2:
    PREV = SPEC['prev']; rp = json.load(open(os.path.join(PREV, 'cx_hybrid.json')))
    res['prev_share'] = rp['iter']['share']
    if not (rp['iter']['share'] >= 0.30):
        res['stopped'] = f'continuation rule: previous share {rp["iter"]["share"]:.3f} < 0.30'
        json.dump(res, open(F, 'w'), indent=1); print(res['stopped']); sys.exit(0)
    VDIR = os.path.join(OUT, 'vit'); os.makedirs(VDIR, exist_ok=True)
    vspec = os.path.join(OUT, 'vit_spec.json')
    json.dump(dict(k=K_IT, seed=SEED, opt='minsr', state=PREV, fit_only=True), open(vspec, 'w'))
    t0 = time.time()
    r = subprocess.run([sys.executable, os.path.join(HERE, 'cx_reanchor.py'), VDIR, vspec])
    res['vit_subprocess_sec'] = time.time() - t0
    if r.returncode != 0: raise RuntimeError('ViT re-anchor subprocess failed')
else:
    VDIR = os.path.join(RUNS, 'ra_ms_s1_k1_17174477')

from cx_lib import *
from scipy.optimize import minimize_scalar
from lanczos_lib import lanczos_ritz
EPS = 0.3
lab = Lab(); sec = lab.sec
if K_IT == 1:
    la_G, s_G = lab.lP0, lab.sP0
else:
    la_G = jnp.asarray(np.load(os.path.join(PREV, 'G_table.npy')))
    s_G = jnp.asarray(np.load(os.path.join(PREV, 's_next.npy')).astype(np.float32))


def clipped_lanczos(la, s):
    _, V, W = lab.VW(la, s); v = sec.vec(la, s); E = float(v @ sec.H(v)); del v
    EL = lab.D0 + V - W; del V, W

    def build(a):
        f = 1.0 + a * (EL - E); d = jnp.log(jnp.maximum(jnp.abs(f), EPS))
        return la + d, s * jnp.where(f >= 0, 1.0, -1.0).astype(s.dtype), d
    sd = float(jnp.sqrt(jnp.sum(sec.vec(la, s) ** 2 * (EL - E) ** 2)))
    r_ = minimize_scalar(lambda a: lab.H_site(*build(a)[:2]), bounds=(-3.0 / sd, 0.0), method='bounded',
                         options=dict(xatol=1e-4 / sd))
    la2, s2, d = build(float(r_.x))
    return la2, s2, d, float(r_.x)


H_G = lab.H_site(la_G, s_G)
la_t, s_t, d, alpha = clipped_lanczos(la_G, s_G)
H_t = lab.H_site(la_t, s_t)
s_v = np.load(os.path.join(VDIR, 's_next.npy')).astype(np.float32)
assert np.array_equal(s_v, np.asarray(s_t)), 'ViT run target sign differs'
la_A = jnp.asarray(np.load(os.path.join(VDIR, 'la_table.npy')))
it = dict(k=K_IT, H_G=H_G, H_target=H_t, alpha=alpha, exact_gain=H_G - H_t, vit_dir=VDIR)
it['vit_H'] = lab.H_site(la_A, s_t); it['vit_share'] = (H_G - it['vit_H']) / (H_G - H_t)
log(f'[k={K_IT}] <H> {H_G:.4e} -> target {H_t:.4e}; ViT alone {it["vit_H"]:.4e} (share {it["vit_share"]:.3f})')
res['iter'] = it; dump(F, res)

# ---------------------------------------------------------------- correction net on one hop of the current ViT
resid = la_t - la_A
lanA, VA, WA = lab.VW(la_A, s_t)
vA = sec.vec(la_A, s_t); EA = float(vA @ sec.H(vA)); ELA = lab.D0 + VA - WA
sdA = float(jnp.sqrt(jnp.sum(vA ** 2 * (ELA - EA) ** 2))); del vA
_, _, meas = lab.proposal_q(resid, la_t, s_t)
FEAT = lab.feat_table(lanA, VA, WA, meas)
u_ = jnp.arcsinh((ELA - EA) / sdA); m_ = float(jnp.sum(meas * u_)); sdu = float(jnp.sqrt(jnp.sum(meas * (u_ - m_) ** 2))) + 1e-30
FEAT = jnp.concatenate([FEAT, ((u_ - m_) / sdu).astype(jnp.float32)[:, None]], 1)
del lanA, VA, WA, ELA, u_, meas
mc, flc, fit = robust_fit(lab, lambda sd_: FeatNet(FEAT, seed=sd_), la_t, s_t, resid, SPEC.get('steps_c', 30000),
                          700 + K_IT + 1000 * SEED, 3e-3, f'c{K_IT}', heldout_curve=True)
it['fit_c'] = fit
c = mc.table(flc, sec.reps, lab.TIDX) if flc is not None else jnp.zeros_like(la_A)
if flc is not None: np.save(os.path.join(OUT, 'params_c.npy'), np.asarray(flc))
G_new = la_A + c
PT = sec.vec(la_t, s_t) ** 2
ex = G_new - la_t; ex = ex - jnp.sum(PT * ex)
it['guard_weight_excess_gt1'] = float(jnp.sum(jnp.where(ex > 1.0, sec.vec(G_new, s_t) ** 2, 0.0)))
it['guard_rejected'] = bool(it['guard_weight_excess_gt1'] > 1e-3)
if it['guard_rejected']:
    G_new = la_A
    ex = G_new - la_t; ex = ex - jnp.sum(PT * ex)
dec, _ = lab.decade_of(la_t)
it['ptwise_rms_err_by_decade'] = {int(q): float(jnp.sqrt(jnp.sum(jnp.where(dec == q, PT * ex ** 2, 0)) /
                                                         max(float(jnp.sum(jnp.where(dec == q, PT, 0))), 1e-300)))
                                  for q in range(-4, -16, -1)}
del ex
it['resid_ptrms'] = float(jnp.sqrt(jnp.sum(PT * (resid - jnp.sum(PT * resid)) ** 2)))
it['H'] = lab.H_site(G_new, s_t)
it['share'] = (H_G - it['H']) / (H_G - H_t)
it['bond_rms_err'] = bond_rms(lab, la_t, G_new - la_t)
_, u, info = sec.fn_solve(G_new, s_t); del u; it['E_FN'] = info['dE_FN_site']
_, u, info = sec.fn_solve(la_A, s_t); del u; it['vit_E_FN'] = info['dE_FN_site']
_, u, info = sec.fn_solve(la_t, s_t); del u; it['target_E_FN'] = info['dE_FN_site']
rl = lanczos_ritz(lambda x: sec.Hm(x), sec.vec(G_new, s_t), 1)
it['lanczos_p1'] = (rl[1]['E'] - sec.E0) / N
log(f'== hybrid k={K_IT}: share {it["share"]:.3f}  <H> {H_G:.4e} -> {it["H"]:.4e} (target {H_t:.4e}; ViT alone '
    f'{it["vit_H"]:.4e})  E_FN {it["E_FN"]:.4e} (ViT alone {it["vit_E_FN"]:.4e}, target {it["target_E_FN"]:.4e})  '
    f'+L1 {it["lanczos_p1"]:.4e}  guard {it["guard_rejected"]}')
np.save(os.path.join(OUT, 'G_table.npy'), np.asarray(G_new, np.float64))
np.save(os.path.join(OUT, 's_next.npy'), np.asarray(s_t, np.int8))
np.save(os.path.join(OUT, 'vit_params.npy'), np.load(os.path.join(VDIR, 'params.npy')))
np.save(os.path.join(OUT, 'vit_table.npy'), np.asarray(la_A, np.float64))
res['sec'] = time.time() - T00; dump(F, res); log('DONE', res['sec'])
