"""Test H1 (README "Test H1"): does a second hop absorb the refit tax?  No training.
For each table a in {O2b round-1 a_2, F1 arm A a_2, F1 arm B a_2} and the target phi_1 (O2b round_1/target_la.npy), sign s_2:
  K_k(a) = |a| |1 + sum_{j=1..k} alpha_j (H - E)^j v / v|,  v = signed normalised vector of (a, s_2), E = <v|H|v>;
  k = 1 is exactly cx_o2b.hop() (alpha_V: bounded argmin of <H>, bound +-3/sdE, xatol 1e-4/sdE, maxiter 100);
  k = 2: (alpha_1, alpha_2) minimise <H> of the state (la_c, s_2) (Nelder-Mead on x1 = alpha_1 sdE, x2 = alpha_2 sdE^2, two starts:
         (alpha_V, 0) and the signed 2-step Lanczos-Ritz polynomial), the same <H> objective as k = 1; sign stays s_2.
  E_FN = sec.fn_solve(norm(la_c), s_2) (maxit 400), as hop().  tax_k = E_FN(K_k(a_2)) - E_FN(K_k(phi_1)).
F1 a_2 tables: the saved round_1/la_table.npy of runs/f1_{A,B} on Paderborn (cx_t1s table readout, no regeneration).
Controls (exit 2 before the K_2 runs): k = 1 of phi_1 vs 4.951226e-6, O2b a_2 vs 1.0881439e-5 (rel 1e-6); F1 A 1.05469569e-5, B 9.52744979e-6.
  python cx_h1.py OUT SPEC.json    SPEC: refresh, o2b_table, target_la, f1_table_A, f1_table_B, stage (control|full)
"""
import os, sys, time, hashlib
import numpy as np
from scipy.optimize import minimize_scalar, minimize
from cx_lib import *
OUT = sys.argv[1]; SPEC = json.load(open(sys.argv[2])); os.makedirs(OUT, exist_ok=True)
F = os.path.join(OUT, 'cx_h1.json'); T00 = time.time()
STAGE = SPEC.get('stage', 'full')
REF = dict(phi1=4.951226482141635e-06, o2b_a2=1.0881438700119065e-05, F1A=1.05469569072388e-05, F1B=9.527449787623729e-06)
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


# --------------------------------------------------------------- controls
dump(F, res)
tgt = jnp.asarray(np.load(SPEC['target_la']), jnp.float64)
ok = True
rp = kstep(tgt, 'phi1'); ok &= ctl('phi1', rp['k1']['EFN'], REF['phi1'])
ro = kstep(jnp.asarray(np.load(SPEC['o2b_table']), jnp.float64), 'o2b_a2'); ok &= ctl('o2b_a2', ro['k1']['EFN'], REF['o2b_a2'])
res['control']['ok'] = bool(ok); dump(F, res)
if not ok:
    res['stop'] = 'CONTROL FAILED'; res['sec'] = time.time() - T00; res['done'] = True; dump(F, res); progress('CONTROL FAILED: stop'); sys.exit(2)
if STAGE == 'control':
    res['sec'] = time.time() - T00; res['done'] = True; dump(F, res); progress(f'DONE control stage {res["sec"]:.0f}s ok'); sys.exit(0)
for arm in ('A', 'B'):
    tab = jnp.asarray(np.load(SPEC[f'f1_table_{arm}']), jnp.float64)
    r = kstep(tab, f'F1{arm}'); del tab
    res['control'][f'F1{arm}'] = ctl(f'F1{arm}', r['k1']['EFN'], REF[f'F1{arm}']) and True
    res['control']['ok'] = bool(res['control']['ok'] and res['control'][f'F1{arm}']); dump(F, res)
summ = {}
for nm in ('o2b_a2', 'F1A', 'F1B'):
    t1 = res['cases'][nm]['k1']['EFN'] - res['cases']['phi1']['k1']['EFN']; t2 = res['cases'][nm]['k2']['EFN'] - res['cases']['phi1']['k2']['EFN']
    summ[nm] = dict(tax_1=t1, tax_2=t2, ratio=t2 / t1)
    progress(f'TAX {nm}: tax_1 {t1:.4e} tax_2 {t2:.4e} ratio {t2 / t1:.3f}')
res['summary'] = summ; res['sec'] = time.time() - T00; res['done'] = True; dump(F, res)
progress(f'VERDICT DONE {res["sec"]:.0f}s ' + ' '.join(f'{k}:{v["ratio"]:.3f}' for k, v in summ.items()))
