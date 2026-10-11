"""Round 4: robust constant-hop loop with RESIDUAL re-bases (README round 4).
Iteration k from the callable guide G_k = log|psi_P| + sum_{j<=k} h_j(x; spins, log a_P, V_P, W_P):
  (w) write-back: FEAT net on G_k's own (log a, V, W), semi-implicit target, edge loss, 20k Adam steps (lr 3e-3);
  (r) residual re-base: h_{k+1} fit to the increment la_w - G_k (sampling by the increment's Dirichlet weight),
      20k steps for k <= 2 and 40k for k >= 3 (small increments), Adam lr 1e-3.
Both fits via robust_fit: 20% held-out orbits excluded from training, global-norm clipping 1.0, stall/collapse
detection with re-seeding (<= 3 tries, lr halved per retry), accepted only if the held-out edge loss beats f = 0;
a fit with no accepted try contributes f = 0 (write-back) / h = 0 (re-base) and is recorded.
  python cx_chop2.py OUT SPEC.json     SPEC: n_loop, seed
"""
import os, sys, time
import numpy as np
from cx_lib import *
from lanczos_lib import lanczos_ritz

OUT = sys.argv[1]; SPEC = json.load(open(sys.argv[2])); os.makedirs(OUT, exist_ok=True)
F = os.path.join(OUT, 'cx_chop2.json'); T00 = time.time()
SEED = SPEC.get('seed', 0)
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
E_FN_G = info['dE_FN_site']; n_nets = 0
for k in range(1, SPEC.get('n_loop', 8) + 1):
    t0 = time.time(); rec = dict(it=k, guide_E_FN=E_FN_G, guide_H=lab.H_site(G, s))
    # ---------------- (w) write-back
    lan, V, W = lab.VW(G, s)
    op = FrozenOp(lab, G, s); Eg = op.energy(G)
    T = lab.T_of(V, W, Eg)
    _, _, meas = lab.proposal_q(T, G, s)
    FEAT = lab.feat_table(lan, V, W, meas); del lan, V, W, meas
    rec['ideal_gain_site'] = (Eg - Efn) / N
    rec['SI_target_frac'] = (Eg - op.energy(G + T)) / (Eg - Efn)
    mw, flw, rec['fit_w'] = robust_fit(lab, lambda sd: FeatNet(FEAT, seed=sd), G, s, T, SPEC.get('steps_w', 20000),
                                       k + 1000 * SEED, 3e-3, f'w{k}')
    del T
    fb = mw.table(flw, sec.reps, lab.TIDX) if flw is not None else jnp.zeros_like(G)
    del mw, FEAT
    la_w = G + fb; del fb
    rec['frac'] = (Eg - op.energy(la_w)) / (Eg - Efn); del op
    s_new, _, _ = sec.krylov(sec.vec(la_w, jnp.ones_like(la_w)), s)
    rec['H_w'] = lab.H_site(la_w, s_new); rec['H_G_newsign'] = lab.H_site(G, s_new)
    # ---------------- (r) residual re-base onto psi_P's one-hop features
    SR = SPEC.get('steps_r_small', 40000) if k >= 3 else SPEC.get('steps_r', 20000)
    mr, flr, rec['fit_r'] = robust_fit(lab, lambda sd: FeatNet(FEAT_BASE, seed=sd), la_w, s_new, la_w - G, SR,
                                       50 + k + 1000 * SEED, 1e-3, f'r{k}')
    rec['steps_r'] = SR
    if flr is not None:
        G_new = G + mr.table(flr, sec.reps, lab.TIDX); n_nets += 1
        np.save(os.path.join(OUT, f'params_h{k}.npy'), np.asarray(flr))
    else:
        G_new = G
    del mr
    rec['n_small_nets'] = n_nets
    rec['H'] = lab.H_site(G_new, s_new)
    rec['retention_step'] = (rec['H_G_newsign'] - rec['H']) / (rec['H_G_newsign'] - rec['H_w']) if rec['H_G_newsign'] != rec['H_w'] else float('nan')
    rec['retention_acc'] = (H_START - rec['H']) / (H_START - rec['H_w'])
    rec['rebase_bond_rms'] = bond_rms(lab, la_w, G_new - la_w)
    Efn, u, info = sec.fn_solve(G_new, s_new); del u
    rec['E_FN'] = E_FN_G = info['dE_FN_site']
    rec['lanczos_p1'], rec['lanczos_p2'] = lanczos(G_new, s_new)
    np.save(os.path.join(OUT, f's_{k}.npy'), np.asarray(s_new, np.int8))
    del la_w
    rec['sec'] = time.time() - t0
    tl = res['table_loop'][k - 1]
    log(f'== it {k}: frac {rec["frac"]:.4f}  <H> {rec["H_w"]:.4e} -> {rec["H"]:.4e} (ret {rec["retention_step"]:.3f})  '
        f'E_FN {rec["E_FN"]:.4e}  +L1 {rec["lanczos_p1"]:.4e} +L2 {rec["lanczos_p2"]:.4e}  nets {n_nets}  '
        f'tries w/r {len(rec["fit_w"]["tries"])}/{len(rec["fit_r"]["tries"])}  | table {tl["H"]:.4e}  ({rec["sec"]:.0f}s)')
    res['iters'].append(rec); dump(F, res)
    G, s = G_new, s_new
res['sec'] = time.time() - T00; dump(F, res); log('DONE', res['sec'])
