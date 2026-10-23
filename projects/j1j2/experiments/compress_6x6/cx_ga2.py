"""Gate G-A2 (README compress_6x6 "M1 + gate for the loop"): null control, splice the real residual r = log|a_2| - log|phi_1| (O2b round 1;
both tables normalised with norm_la first) into phi_1 on decade subsets.  Decade = Lab.decade_of(norm phi_1) (floor log10 of |phi_1|^2 per configuration, as cx_t1s).
  splice_full: r everywhere (= a_2: must reproduce H1's O2b numbers);  splice_tail: r on decades <= -9 only;  splice_bulk: r on decades >= -8 only.
For each: K_1, K_2 (kstep copied verbatim from cx_h1.py), sign s_2, E_FN;  tax_k = E_FN(K_k(splice)) - E_FN(K_k(phi_1)).
  python cx_ga2.py OUT SPEC.json   SPEC: refresh, o2b_table, target_la
"""

import os, sys, time, hashlib
import numpy as np
from scipy.optimize import minimize_scalar, minimize
from cx_lib import *
OUT = sys.argv[1]; SPEC = json.load(open(sys.argv[2])); os.makedirs(OUT, exist_ok=True)
F = os.path.join(OUT, 'cx_ga2.json'); T00 = time.time()
REF = dict(phi1=4.951226482141635e-06, o2b_a2=1.0881438700119065e-05, phi1_k2=3.244153e-06, o2b_k2=5.147486e-06)
res = dict(spec=SPEC, done=False, ref=REF, cases={})


def progress(msg):
    log(msg); open(os.path.join(OUT, 'progress.txt'), 'a').write(f'{time.strftime("%Y-%m-%dT%H:%M:%S")} PROGRESS {msg}\n')


lab = Lab(); sec = lab.sec


def norm_la(la):
    a = jnp.exp(la - jnp.max(la)); return la - jnp.max(la) - 0.5 * jnp.log(jnp.sum(sec.n * a * a))


def fn(la, s):
    _, u, info = sec.fn_solve(la, s, maxit=400); return info['dE_FN_site'], info


s2 = jnp.asarray(np.load(os.path.join(SPEC['refresh'], 's2.npy')).astype(np.float32))


def kstep(la, name):
    t0 = time.time(); lb = norm_la(la)
    v = sec.vec(lb, s2); Hv = sec.H(v); Eb = float(v @ Hv)
    u1 = Hv - Eb * v; Hu1 = sec.H(u1); u2 = Hu1 - Eb * u1
    R1 = u1 / v; R2 = u2 / v; del Hv, Hu1
    ok = bool(jnp.all(jnp.isfinite(R1)) & jnp.all(jnp.isfinite(R2)))
    sdE = float(jnp.sqrt(jnp.sum(u1 * u1)))                      # = sqrt(sum v^2 (E_L - E)^2), as hop()

    def la_c(a1, a2): return lb + jnp.log(jnp.maximum(jnp.abs(1.0 + a1 * R1 + a2 * R2), 1e-30))
    def Hs(a1, a2): return float(lab.H_site(la_c(a1, a2), s2))
    rec = dict(name=name, E_base=Eb, sdE=sdE, finite=ok)
    # k = 1: the O2b hop
    r1 = minimize_scalar(lambda a: Hs(a, 0.0), bounds=(-3.0 / sdE, 3.0 / sdE), method='bounded', options=dict(maxiter=100, xatol=1e-4 / sdE))
    a1V = float(r1.x); H1 = float(r1.fun); e1, i1 = fn(norm_la(la_c(a1V, 0.0)), s2)
    rec['k1'] = dict(alpha=[a1V], H=H1, EFN=e1, nfev=r1.nfev, at_bound=bool(abs(abs(a1V) - 3.0 / sdE) < 1e-3 * 3.0 / sdE), fn_res=i1['res'])
    progress(f'[{name}] k=1 alpha {a1V:.5f} <H>-E0 {H1:.5e} E_FN {e1:.6e} ({time.time() - t0:.0f}s)')
    # signed 2-step Ritz polynomial (reference + second start)
    Hu2 = sec.H(u2)
    B = jnp.stack([v, u1, u2], 1); Hm3 = np.asarray(B.T @ jnp.stack([u1 + Eb * v, u2 + Eb * u1, Hu2], 1)); Gm = np.asarray(B.T @ B)
    Hm3 = 0.5 * (Hm3 + Hm3.T)
    import scipy.linalg as sl
    w, c = sl.eigh(Hm3, Gm); c0 = c[:, 0] / c[0, 0]
    ritz = float((w[0] - sec.E0) / N)
    rec['ritz_signed_2'] = dict(alpha=[float(c0[1]), float(c0[2])], H=ritz)
    del Hu2, B
    sc = np.array([sdE, sdE ** 2])
    best = None
    for st, x0 in (('from_k1', np.array([a1V, 0.0]) * sc), ('from_ritz', np.array([c0[1], c0[2]]) * sc)):
        cnt = [0]

        def obj(x):
            cnt[0] += 1; return Hs(x[0] / sc[0], x[1] / sc[1]) * 1e6
        sim = np.array([x0, x0 + [0.15, 0.0], x0 + [0.0, 0.15]])
        r = minimize(obj, x0, method='Nelder-Mead', options=dict(initial_simplex=sim, xatol=1e-5, fatol=1e-9, maxiter=400, maxfev=500))
        rec.setdefault('k2_starts', {})[st] = dict(x=list(map(float, r.x)), H=float(r.fun) * 1e-6, nfev=cnt[0], start_H=float(obj(x0)) * 1e-6)
        if best is None or r.fun < best[0]: best = (float(r.fun), r.x, st)
    a1, a2 = best[1][0] / sc[0], best[1][1] / sc[1]; H2 = best[0] * 1e-6
    e2, i2 = fn(norm_la(la_c(a1, a2)), s2)
    rec['k2'] = dict(alpha=[float(a1), float(a2)], H=H2, EFN=e2, start=best[2], fn_res=i2['res'])
    progress(f'[{name}] k=2 alpha ({a1:.5f}, {a2:.5f}) <H>-E0 {H2:.5e} (signed Ritz {ritz:.5e}, start {best[2]}) E_FN {e2:.6e} ({time.time() - t0:.0f}s)')
    res['cases'][name] = rec; dump(F, res)
    return rec


def ctl(name, e, ref, tol=1e-6):
    r = abs(e - ref) / ref; ok = r < tol
    res.setdefault('control', {})[name] = dict(EFN=e, ref=ref, rel=r, ok=ok); dump(F, res)
    progress(f'CONTROL {name}: k=1 E_FN {e:.8e} vs {ref:.8e} (rel {r:.2e}) ok={ok}')
    return ok



# --------------------------------------------------------------- the splices
lphi = norm_la(jnp.asarray(np.load(SPEC['target_la']), jnp.float64))
la2 = norm_la(jnp.asarray(np.load(SPEC['o2b_table']), jnp.float64))
r = la2 - lphi
dec, lp = lab.decade_of(lphi); dec_h = np.asarray(dec)
p = np.asarray(sec.n * jnp.exp(lp)); p = p / p.sum()          # orbit probability of phi_1
r_h = np.asarray(r)
tail_m = dec_h <= -9; bulk_m = ~tail_m
def stat(m):
    w = p[m]; W = w.sum(); mu = float((w * r_h[m]).sum() / W)
    return dict(n_orbits=int(m.sum()), prob_mass=float(W), r_mean=mu, r_rms_about_mean=float(np.sqrt((w * (r_h[m] - mu) ** 2).sum() / W)),
                r_rms_raw=float(np.sqrt((w * r_h[m] ** 2).sum() / W)))
res['residual'] = dict(tail=stat(tail_m), bulk=stat(bulk_m), all=stat(np.ones_like(tail_m)),
                       finite=bool(np.isfinite(r_h).all()), decade_hist={int(d): int((dec_h == d).sum()) for d in np.unique(dec_h)})
dump(F, res); progress('residual ' + json.dumps({k: res['residual'][k] for k in ('tail', 'bulk', 'all')}))
assert res['residual']['finite']
rp = kstep(lphi, 'phi1'); ok = ctl('phi1', rp['k1']['EFN'], REF['phi1']) & ctl('phi1_k2', rp['k2']['EFN'], REF['phi1_k2'], 1e-5)
sp = {}
for nm, m in (('splice_full', np.ones_like(tail_m)), ('splice_tail', tail_m), ('splice_bulk', bulk_m)):
    la = lphi + r * jnp.asarray(m, jnp.float64)
    sp[nm] = kstep(la, nm); del la
    if nm == 'splice_full':
        ok &= ctl('full_vs_O2b_k1', sp[nm]['k1']['EFN'], REF['o2b_a2']) & ctl('full_vs_O2b_k2', sp[nm]['k2']['EFN'], REF['o2b_k2'], 1e-5)
        res['control']['ok'] = bool(ok); dump(F, res)
        if not ok: res['stop'] = 'CONTROL FAILED'; res['done'] = True; dump(F, res); progress('CONTROL FAILED: stop'); sys.exit(2)
summ = {}
for nm in sp:
    t1 = sp[nm]['k1']['EFN'] - rp['k1']['EFN']; t2 = sp[nm]['k2']['EFN'] - rp['k2']['EFN']
    summ[nm] = dict(tax_1=t1, tax_2=t2, ratio=t2 / t1); progress(f'TAX {nm}: tax_1 {t1:.4e} tax_2 {t2:.4e} ratio {t2 / t1:.3f}')
summ['tail_over_bulk_ratio'] = summ['splice_tail']['ratio'] / summ['splice_bulk']['ratio']
res['summary'] = summ; res['sec'] = time.time() - T00; res['done'] = True; dump(F, res)
progress(f'VERDICT DONE {res["sec"]:.0f}s ' + ' '.join(f'{k}:{v["ratio"]:.3f}' for k, v in summ.items() if isinstance(v, dict)) + f' tail/bulk {summ["tail_over_bulk_ratio"]:.3f}')
