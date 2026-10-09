"""Test 1: CONSTANT-HOP LOOP. Every iteration k:
  (w) write-back from the callable guide G_k (exactly wt_loop / F-SI-Adam): FEAT net on G_k's own log a, V, W,
      semi-implicit target, guide-metric tail proposal, edge loss, Adam 20k x 256 x 8 -> la_w = G_k + f_k;
      Krylov sign s_{k+1} of la_w (as in the table loop);
  (r) re-base: distil la_w into a fresh base-feature net, G_{k+1} = log|psi_P| + g(x; spins, log a_P, V_P, W_P of the
      FROZEN psi_P) (the pre-test (a) 'featbase' arm: edge loss in the metric of (la_w, s_{k+1}), proposal = 1/2 n a_w
      + 1/2 Dirichlet node weight of la_w - log|psi_P|, Adam 20k x 256 x 8).
G_k therefore costs one hop of psi_P at every k, and its V, W two hops.
Exact referee per iteration: frac of the write-back, <H>, E_FN, one/two Lanczos steps on G_{k+1}, retention of the
re-base (step and accumulated).
  python cx_chop.py OUT SPEC.json      SPEC: n_loop, seed, steps_w, steps_r
"""
import os, sys, time
import numpy as np
from cx_lib import *
from lanczos_lib import lanczos_ritz

OUT = sys.argv[1]; SPEC = json.load(open(sys.argv[2])); os.makedirs(OUT, exist_ok=True)
F = os.path.join(OUT, 'cx_chop.json'); T00 = time.time()
SEED = SPEC.get('seed', 0); SW = SPEC.get('steps_w', 20000); SR = SPEC.get('steps_r', 20000)
lab = Lab(); sec = lab.sec
FEAT_BASE = base_features(lab)
J8 = json.load(open(os.path.join(LOOP8, 'wt_loop.json')))
res = dict(spec=SPEC, iters=[], table_loop=[dict(it=i + 1, H=it['H_kry'], E_FN=it['E_FN_next'], frac=it['frac'])
                                            for i, it in enumerate(J8['iters'])],
           refs=dict(H_start=H_START, lanczos_p1_psiP=2.55e-5, lanczos_p2_psiP=8.6e-6, rbm_pp=RBM_PP))


def lanczos(la, s):
    rl = lanczos_ritz(lambda x: sec.Hm(x), sec.vec(la, s), 2)
    return (rl[1]['E'] - sec.E0) / N, (rl[2]['E'] - sec.E0) / N


G, s = lab.lP0, lab.sP0
Efn, u, info = sec.fn_solve(G, s); del u
E_FN_G = info['dE_FN_site']
for k in range(1, SPEC.get('n_loop', 8) + 1):
    t0 = time.time(); rec = dict(it=k, guide_E_FN=E_FN_G, guide_H=lab.H_site(G, s))
    # ---------------- (w) write-back from the callable guide
    lan, V, W = lab.VW(G, s)
    op = FrozenOp(lab, G, s); Eg = op.energy(G)
    T = lab.T_of(V, W, Eg)
    CDF, LIW, DLR, meas = tail_proposal(lab, T, G, s)
    rec['ideal_gain_site'] = (Eg - Efn) / N
    rec['SI_target_frac'] = (Eg - op.energy(G + T)) / (Eg - Efn)
    FEAT = lab.feat_table(lan, V, W, meas); del lan, V, W, meas, T
    model = FeatNet(FEAT, seed=k + 1000 * SEED)
    sec.offload()
    flat, rec['curve_w'] = fit_edge(lab, model, G, s, CDF, LIW, DLR, SW, k + 1000 * SEED, tag=f'w{k}')
    sec.reload()
    fb = model.table(flat, sec.reps, lab.TIDX); del model, FEAT, CDF, LIW, DLR
    la_w = G + fb; del fb
    rec['frac'] = (Eg - op.energy(la_w)) / (Eg - Efn); del op
    s_new, _, _ = sec.krylov(sec.vec(la_w, jnp.ones_like(la_w)), s)
    rec['H_w'] = lab.H_site(la_w, s_new)                        # write-back table, before re-basing
    rec['H_G_newsign'] = lab.H_site(G, s_new)
    # ---------------- (r) re-base into a base-feature net (fresh)
    D = la_w - lab.lP0
    CDF, LIW, DLR, _ = tail_proposal(lab, D, la_w, s_new); del D
    mF = FeatNet(FEAT_BASE, seed=50 + k + 1000 * SEED)
    sec.offload()
    flF, rec['curve_r'] = fit_edge(lab, mF, la_w, s_new, CDF, LIW, DLR, SR, 50 + k + 1000 * SEED, tag=f'r{k}')
    sec.reload()
    g = mF.table(flF, sec.reps, lab.TIDX); del mF, CDF, LIW, DLR
    G_new = lab.lP0 + g; del g
    np.save(os.path.join(OUT, f'params_w{k}.npy'), np.asarray(flat)); np.save(os.path.join(OUT, f'params_r{k}.npy'), np.asarray(flF))
    rec['H'] = lab.H_site(G_new, s_new)
    rec['retention_step'] = (rec['H_G_newsign'] - rec['H']) / (rec['H_G_newsign'] - rec['H_w'])
    rec['retention_acc'] = (H_START - rec['H']) / (H_START - rec['H_w'])
    rec['rebase_bond_rms'] = bond_rms(lab, la_w, G_new - la_w)
    Efn, u, info = sec.fn_solve(G_new, s_new); del u
    rec['E_FN'] = E_FN_G = info['dE_FN_site']
    rec['lanczos_p1'], rec['lanczos_p2'] = lanczos(G_new, s_new)
    if SPEC.get('save_guides', True) and k in SPEC.get('save_at', [1, 3, 6]):
        np.save(os.path.join(OUT, f'G_{k}.npy'), np.asarray(G_new)); np.save(os.path.join(OUT, f'la_w_{k}.npy'), np.asarray(la_w))
        np.save(os.path.join(OUT, f's_{k}.npy'), np.asarray(s_new, np.int8))
    del la_w
    rec['sec'] = time.time() - t0
    tl = res['table_loop'][k - 1] if k <= len(res['table_loop']) else {}
    log(f'== it {k}: frac {rec["frac"]:.4f}  <H> table-step {rec["H_w"]:.4e} -> re-based {rec["H"]:.4e} '
        f'(ret step {rec["retention_step"]:.3f}, acc {rec["retention_acc"]:.3f})  E_FN {rec["E_FN"]:.4e}  '
        f'+L1 {rec["lanczos_p1"]:.4e} +L2 {rec["lanczos_p2"]:.4e}  | table loop {tl.get("H", float("nan")):.4e}  ({rec["sec"]:.0f}s)')
    res['iters'].append(rec); dump(F, res)
    G, s = G_new, s_new
res['sec'] = time.time() - T00; dump(F, res); log('DONE', res['sec'])
