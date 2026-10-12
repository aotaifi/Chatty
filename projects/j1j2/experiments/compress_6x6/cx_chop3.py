"""Step-size stage 2 (README "Step size", stage 2): the round-4 constant-hop loop with the CLIPPED-LANCZOS target.
Iteration k from the callable guide G_k = log|psi_P| + sum_{j<k} h_j(x; spins, log a_P, V_P, W_P), sign s_k (table):
  (t) target, exact on the tables: one-hop quantities of (G_k, s_k): V, W, E_L = D0 + V - W, E = <H>;
      f = 1 + alpha (E_L - E),  d = log max(|f|, eps),  s_{k+1} = s_k sgn f,
      alpha energy-optimal: bounded scalar minimisation of <H>(G_k + d, s_k sgn f) (as cx_step.lanczos_state).
  (w) write-back: FEAT net on G_k's own (log a, V, W) [+ E_L if feat_EL] fit to d by robust_fit (edge loss in the
      metric of the target state (G_k + d, s_{k+1}) [metric 'target'] or of (G_k, s_k) ['guide'], tail proposal of d in
      that metric, Adam lr 3e-3, steps_w, clipping, held-out acceptance) -> la_w = G_k + d_fit.
      mode 'direct': no write-back net, la_w = G_k + d (d is an explicit one-hop function of G_k; reported arm).
  (r) residual re-base exactly as round 4: h_k on psi_P's one-hop features fit to la_w - G_k in the metric of
      (la_w, s_{k+1}); G_{k+1} = G_k + h_k.
Per iteration (exact): <H> of guide / exact target / write-back / re-based guide, capture, retention, iteration share,
bond-rms of the write-back error and of the re-base error, E_FN, + 1 / + 2 Lanczos steps, sign error w_s, time.
  python cx_chop3.py OUT SPEC.json
SPEC: n_loop (4), seed (0), eps (0.3), feat_EL (false), metric ('target'), mode ('net'), steps_w (20000),
      steps_r (20000, k <= 2), steps_r_small (40000, k >= 3), alpha_scale ({}: per-k multiplier of alpha*, diagnosis),
      C_w (32: write-back net width)
"""
import os, sys, time
import numpy as np
from scipy.optimize import minimize_scalar
from cx_lib import *
from lanczos_lib import lanczos_ritz

OUT = sys.argv[1]; SPEC = json.load(open(sys.argv[2])); os.makedirs(OUT, exist_ok=True)
F = os.path.join(OUT, 'cx_chop3.json'); T00 = time.time()
SEED = SPEC.get('seed', 0); EPS = SPEC.get('eps', 0.3); FEAT_EL = SPEC.get('feat_EL', False)
METRIC = SPEC.get('metric', 'target'); MODE = SPEC.get('mode', 'net'); C_W = SPEC.get('C_w', 32)
ASCALE = {int(k): float(v) for k, v in SPEC.get('alpha_scale', {}).items()}
lab = Lab(); sec = lab.sec
FEAT_BASE = base_features(lab)
J8 = json.load(open(os.path.join(LOOP8, 'wt_loop.json')))
# exact ground state, kept on the host as q = p0 * s0 (sign error w_s = (1 - |sum q s|) / 2, as Sector.score)
_z = np.load(TABLE); _v0 = np.asarray(_z['amp'], np.float64) * np.sqrt(np.asarray(sec.n)); del _z
_v0 /= np.linalg.norm(_v0); Q0 = _v0 * np.abs(_v0); del _v0


def w_s(s):
    return float(max(0.0, (1.0 - abs(float(np.dot(Q0, np.asarray(s, np.float64))))) / 2))


def lanczos(la, s):
    rl = lanczos_ritz(lambda x: sec.Hm(x), sec.vec(la, s), 2)
    return (rl[1]['E'] - sec.E0) / N, (rl[2]['E'] - sec.E0) / N


def clipped_lanczos(la, s, EL, E, scale=1.0):
    """exact clipped-Lanczos target (cx_step.lanczos_state, eps = EPS); alpha energy-optimal, then x scale."""
    def build(alpha):
        f = 1.0 + alpha * (EL - E)
        d = jnp.log(jnp.maximum(jnp.abs(f), EPS))
        return la + d, s * jnp.where(f >= 0, 1.0, -1.0).astype(s.dtype), d
    sd = float(jnp.sqrt(jnp.sum(sec.vec(la, s) ** 2 * (EL - E) ** 2)))
    r = minimize_scalar(lambda a: lab.H_site(*build(a)[:2]), bounds=(-3.0 / sd, 0.0), method='bounded',
                        options=dict(xatol=1e-4 / sd))
    a = float(r.x) * scale
    la2, s2, d = build(a)
    return la2, s2, d, float(r.x), a, sd


G, s = lab.lP0, lab.sP0
refs = dict(H_start=H_START, lanczos_psiP=[2.551e-5, 8.59e-6, 3.82e-6, 1.95e-6], rbm_pp=RBM_PP,
            stage1_Lc03=[2.619e-5, 1.157e-5, 7.065e-6, 4.905e-6])
log('refs', refs, 'spec', SPEC)
res = dict(spec=SPEC, iters=[], refs=refs,
           table_loop=[dict(it=i + 1, H=it['H_kry'], E_FN=it['E_FN_next']) for i, it in enumerate(J8['iters'])])
n_nets = 0
H_G = lab.H_site(G, s)
res['start'] = dict(H=H_G, w_s=w_s(s))
for k in range(1, SPEC.get('n_loop', 4) + 1):
    t0 = time.time(); rec = dict(it=k, guide_H=H_G, guide_w_s=w_s(s))
    # ---------------- (t) exact clipped-Lanczos target from the callable guide
    lan, V, W = lab.VW(G, s)
    v = sec.vec(G, s); E = float(v @ sec.H(v)); del v
    EL = lab.D0 + V - W
    la_t, s_t, d, a_opt, a_use, sdE = clipped_lanczos(G, s, EL, E, ASCALE.get(k, 1.0))
    rec.update(alpha_opt=a_opt, alpha=a_use, energy_sd=sdE, H_target=lab.H_site(la_t, s_t),
               H_G_newsign=lab.H_site(G, s_t), target_rough=bond_rms(lab, G, d),
               flips=float(jnp.sum(sec.n * (s_t != s)) / jnp.sum(sec.n)),
               flip_weight=float(jnp.sum(sec.vec(G, s) ** 2 * (s_t != s))),
               clipped_weight=float(jnp.sum(sec.vec(G, s) ** 2 * (jnp.abs(1.0 + a_use * (EL - E)) < EPS))))
    rec['exact_gain'] = H_G - rec['H_target']
    log(f'[it {k}] alpha* {a_opt:.4f} (used {a_use:.4f})  <H> {H_G:.4e} -> exact target {rec["H_target"]:.4e}  '
        f'flips {rec["flips"]:.3f}  rough {rec["target_rough"]:.4f}')
    # ---------------- (w) write-back
    if MODE == 'direct':
        d_fit = d; rec['fit_w'] = None
    else:
        la_m, s_m = (la_t, s_t) if METRIC == 'target' else (G, s)
        _, _, meas = lab.proposal_q(d, la_m, s_m)
        FEAT = lab.feat_table(lan, V, W, meas)
        if FEAT_EL:
            m_ = float(jnp.sum(meas * EL)); s_ = float(jnp.sqrt(jnp.sum(meas * (EL - m_) ** 2))) + 1e-30
            FEAT = jnp.concatenate([FEAT, ((EL - m_) / s_).astype(jnp.float32)[:, None]], 1)
        del meas
        mw, flw, rec['fit_w'] = robust_fit(lab, lambda sd_: FeatNet(FEAT, seed=sd_, C=C_W), la_m, s_m, d,
                                           SPEC.get('steps_w', 20000), k + 1000 * SEED, 3e-3, f'w{k}',
                                           heldout_curve=True)
        d_fit = mw.table(flw, sec.reps, lab.TIDX) if flw is not None else jnp.zeros_like(G)
        if flw is not None: np.save(os.path.join(OUT, f'params_w{k}.npy'), np.asarray(flw))
        del mw, FEAT
    del lan, V, W, EL
    la_w = G + d_fit
    rec['H_w'] = lab.H_site(la_w, s_t)
    rec['capture'] = (H_G - rec['H_w']) / rec['exact_gain']
    rec['w_err_bond_rms'] = bond_rms(lab, la_t, d_fit - d)          # write-back error in the target metric
    del d_fit, d
    # ---------------- (r) residual re-base onto psi_P's one-hop features (round 4)
    SR = SPEC.get('steps_r_small', 40000) if k >= 3 else SPEC.get('steps_r', 20000)
    mr, flr, rec['fit_r'] = robust_fit(lab, lambda sd_: FeatNet(FEAT_BASE, seed=sd_), la_w, s_t, la_w - G, SR,
                                       50 + k + 1000 * SEED, 1e-3, f'r{k}', heldout_curve=True)
    rec['steps_r'] = SR
    if flr is not None:
        G_new = G + mr.table(flr, sec.reps, lab.TIDX); n_nets += 1
        np.save(os.path.join(OUT, f'params_h{k}.npy'), np.asarray(flr))
    else:
        G_new = G
    del mr
    rec['n_small_nets'] = n_nets
    rec['H'] = lab.H_site(G_new, s_t)
    rec['retention'] = (H_G - rec['H']) / (H_G - rec['H_w'])          # share of the write-back's gain kept
    rec['iter_share'] = (H_G - rec['H']) / rec['exact_gain']          # share of the exact clipped-Lanczos step
    rec['rebase_bond_rms'] = bond_rms(lab, la_w, G_new - la_w)
    rec['total_err_bond_rms'] = bond_rms(lab, la_t, G_new - la_t)
    del la_w, la_t
    Efn, u, info = sec.fn_solve(G_new, s_t); del u
    rec['E_FN'] = info['dE_FN_site']
    rec['lanczos_p1'], rec['lanczos_p2'] = lanczos(G_new, s_t)
    rec['w_s'] = w_s(s_t)
    np.save(os.path.join(OUT, f's_{k}.npy'), np.asarray(s_t, np.int8))
    rec['sec'] = time.time() - t0
    log(f'== it {k}: exact {rec["H_target"]:.4e}  capture {rec["capture"]:.3f}  <H>_w {rec["H_w"]:.4e} -> {rec["H"]:.4e} '
        f'(ret {rec["retention"]:.3f}, share {rec["iter_share"]:.3f})  E_FN {rec["E_FN"]:.4e}  +L1 {rec["lanczos_p1"]:.4e} '
        f'+L2 {rec["lanczos_p2"]:.4e}  w_s {rec["w_s"]:.2e}  bond-rms w/r {rec["w_err_bond_rms"]:.4f}/'
        f'{rec["rebase_bond_rms"]:.4f}  | stage-1 {refs["stage1_Lc03"][k - 1] if k <= 4 else float("nan"):.4e}  '
        f'({rec["sec"]:.0f}s)')
    res['iters'].append(rec); dump(F, res)
    G, s, H_G = G_new, s_t, rec['H']
res['sec'] = time.time() - T00; dump(F, res); log('DONE', res['sec'])
