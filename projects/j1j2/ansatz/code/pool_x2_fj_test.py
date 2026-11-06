"""pool_x2_fj_test.py: checks of pool_x2_fj on 8x8 configs (usage: python pool_x2_fj_test.py CONFIGS.npz|random N OUT.json).
 (a) sign: lemma-ratio endpoint sign == direct sgn Re[w e^{i Im logB(y)}] (logB of y itself) on every valid endpoint; BROKEN versions ('w', 'ratio') must differ.
 (b) f_J(x): own formula == pp2_lemma.fJ_fixed(at_y=False) through M1LemmaSign (max |diff|), plus fJ_fixed with the broken sign must differ.
 (c) sx == decades8.make_sG8_np(alpha = 0, E, w) (the M1 code path at p = 0) on a few configs.
 (d) partners: empirical pick frequencies == q(b) = HW|r_b|/W^B on blocked bonds (chi2)."""
import sys, os, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import pool_x2_fj as F, pp2_lemma as PL, pool8 as P8
src, N, outp = sys.argv[1], int(sys.argv[2]), sys.argv[3]
if src == 'random':
    rng = np.random.default_rng(3); X = np.ones((N, 64)); [X.__setitem__((i, rng.permutation(64)[:32]), -1) for i in range(N)]
else:
    z = np.load(src); X = P8.unpack_words(z['words'][:N], 64).astype(np.float64)
fj = F.FJ8(); res = dict(src=src, N=int(N)); t = time.time(); ev = fj.evaluate(X); res['sec_evaluate_per_config'] = (time.time() - t) / N
# (a) direct signs at every valid endpoint
lat = fj.lat; M = len(X); sy_dir = np.zeros(ev['sy'].shape, np.int8); t = time.time()
for m in range(M):
    bs = np.nonzero(ev['valid'][m])[0]; Y = np.repeat(X[m][None], len(bs), 0); ar = np.arange(len(bs)); i_, j_ = fj.BI[bs], fj.BJ[bs]; Y[ar, i_], Y[ar, j_] = X[m, j_], X[m, i_]
    lBy = np.asarray(PL.logB(fj.p, Y)); sy_dir[m, bs] = np.where(np.real(fj.m1.w * np.exp(1j * np.imag(lBy))) >= 0, 1, -1)
res['sec_direct_sign_per_config'] = (time.time() - t) / M
v = ev['valid']; res['sign_endpoints'] = int(v.sum()); res['sign_mismatch'] = int((sy_dir[v] != ev['sy'][v]).sum()); res['guard_checked'] = fj.m1.n_checked; res['guard_fallbacks'] = fj.m1.n_fallback
res['min_abs_margin'] = float(np.min(np.abs(np.real(fj.m1.w * np.exp(1j * (np.imag(ev['lB'])[:, None] + np.angle(ev['r']))))[v])))
for brk in ('w', 'ratio'):
    m1b = F.M1Sign(fj.p, broken=brk); evb = fj.evaluate(X, m1=m1b); res['broken_' + brk + '_sign_mismatch'] = int((sy_dir[v] != evb['sy'][v]).sum())
    res['broken_' + brk + '_fx_maxdiff'] = float(np.nanmax(np.abs(np.where(np.isfinite(evb['fx']), evb['fx'], 0) - np.where(np.isfinite(ev['fx']), ev['fx'], 0))))
# (b) fJ_fixed cross-check
fx2, sx2, sy2 = fj.fj_fixed_x(X); f1 = np.where(np.isfinite(ev['fx']), ev['fx'], 0.0); f2 = np.where(np.isfinite(fx2), fx2, 0.0)
res['fx_vs_fJ_fixed_maxdiff'] = float(np.max(np.abs(f1 - f2))); res['fx_nonfinite_own'] = int((~np.isfinite(ev['fx'])).sum()); res['fx_nonfinite_fJ_fixed'] = int((~np.isfinite(fx2)).sum())
res['fx_stats'] = dict(min=float(np.min(f1)), median=float(np.median(f1)), max=float(np.max(f1)), frac_below_m0p5=float(np.mean(f1 < -0.5)))
# (c) M1 code path at alpha = 0 (few configs)
try:
    sys.path.insert(0, '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_8x8build/gates_cE/code'); sys.path.insert(0, '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_8x8build/decades8/run8L/code')
    import decades8 as D8
    al, E, w = D8.m1_consts(); sfn = D8.make_sG8_np(0.0, E, w); n_c = min(M, 24); s_m1 = sfn(X[:n_c]); res['sx_vs_make_sG8_np_alpha0_mismatch'] = int((s_m1 != ev['sx'][:n_c]).sum()); res['n_c'] = n_c
except Exception as e: res['sx_vs_make_sG8_np_alpha0_mismatch'] = 'ERR ' + repr(e)[:200]
# (d) partner frequencies
rng = np.random.default_rng(1); chi2 = 0.0; dof = 0; npk = 20000
for m in range(min(M, 4)):
    evm = {k: (v_[m:m + 1] if hasattr(v_, '__len__') and len(v_) == M else v_) for k, v_ in ev.items()}
    w = np.where(evm['blocked'][0], evm['A'][0], 0.0); q = w / w.sum(); Xr = np.repeat(X[m:m + 1], npk, 0)
    evr = {k: np.repeat(v_, npk, 0) for k, v_ in evm.items()}; pk, _ = fj.partners(Xr, evr, rng, K=1); cnt = np.bincount(pk[:, 0], minlength=fj.NB)
    ok = q * npk > 5; chi2 += float((((cnt - q * npk) ** 2 / (q * npk))[ok]).sum()); dof += int(ok.sum()) - 1
res['partner_chi2'] = chi2; res['partner_dof'] = dof
json.dump(res, open(outp, 'w'), indent=1, default=float)
print(json.dumps(res, indent=1, default=float))
