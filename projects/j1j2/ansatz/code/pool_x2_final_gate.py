"""pool_x2_final_gate.py: FINAL gate of the pool expansion (PLAN_pfaffian.md "Pool expansion", lead 2026-11-06): Kish bond ESS on the ACTUAL combined root weights, per decade -2..-9, >= 3 P (P = 1.8e4).
 root weight of a training entry = w_in x mean_k rho_k  (pool.npz's own w_in, dec, ho; rho = W^B exp[(ln T_y - ln B_y) - (ln T_x - ln B_x)], sign sgn(M1), mode 'fix' labels), held-out entries (pool_fit.ho_hash of the orbit id) excluded
 = what the fit trains on; (w_in is already zero on held-out entries, as in pool_fit.pool_weights).  Config ESS = Kish(w_in).  Kish ESS = (sum w)^2 / sum w^2 over the decade's entries, NOT summed over parts.
 BROKEN version (must change the -9 ESS): rho -> W^B only (the label factor dropped).   usage: python pool_x2_final_gate.py POOL.npz OUT.json"""
import sys, json, numpy as np
P_ = 1.8e4; TARGET = 3 * P_; DECS = list(range(-2, -10, -1))
def kish(w):
    w = np.asarray(w, float); s2 = float(np.sum(w * w)); return float(np.sum(w) ** 2 / s2) if s2 > 0 else 0.0
Z = np.load(sys.argv[1]); dec = Z['dec']; w_in = Z['w_in']; ho = Z['ho']; rho = Z['rho']; WB = Z['WB']; chain = Z['chain']; K = int(Z['K']); nblk = Z['nblk']
def table(rho_mean, use_ho):
    out = {}
    for d in DECS:
        m = (dec == d) & (~ho if use_ho else True); wr = (w_in * rho_mean)[m]; wx = w_in[m]; o = np.sort(wr)[::-1]
        out[d] = dict(n=int(m.sum()), config_ess=kish(wx), bond_ess=kish(wr), ratio=kish(wr) / kish(wx) if kish(wx) > 0 else None, top1=float(o[0] / o.sum()) if o.size and o.sum() > 0 else 0.0,
                      per_rung=[int(np.sum(m & (chain == q))) for q in range(3)])
    return out
rm = rho.mean(1); tr = table(rm, True); br = table(np.where(nblk > 0, WB, 0.0), True)
res = dict(pool=sys.argv[1], S=int(len(dec)), K=K, target=TARGET, training=tr, broken_WB_only=br)
ok = all(tr[d]['bond_ess'] >= TARGET for d in DECS); worst = min(DECS, key=lambda d: tr[d]['bond_ess']); wmin = tr[worst]['bond_ess']
res['pass'] = bool(ok); res['worst_decade'] = int(worst); res['worst_bond_ess'] = float(wmin); res['min_over_target'] = float(wmin / TARGET)
res['broken_changes_m9'] = bool(abs(br[-9]['bond_ess'] - tr[-9]['bond_ess']) > 0.05 * tr[-9]['bond_ess'])
json.dump(res, open(sys.argv[2], 'w'), indent=1, default=float)
for d in DECS: print('dec %d n %7d  config ESS %9.0f  bond ESS %9.0f  ratio %.3f  top1 %.4f  | broken(W^B only) bond %9.0f | rung counts %s' % (d, tr[d]['n'], tr[d]['config_ess'], tr[d]['bond_ess'], tr[d]['ratio'], tr[d]['top1'], br[d]['bond_ess'], tr[d]['per_rung']))
print('FINAL GATE %s: min bond ESS over decades -2..-9 = %.3g at decade %d = %.2f x target 3P = %.3g (training entries, held-out excluded); broken (rho = W^B only) changes the -9 ESS %.3g -> %.3g (%s)' % (
    'PASS' if ok else 'FAIL', wmin, worst, wmin / TARGET, TARGET, tr[-9]['bond_ess'], br[-9]['bond_ess'], 'detected' if res['broken_changes_m9'] else 'NOT detected'))
