"""C2-T0b at exact 6x6 (PLAN_ansatz.md "C2-T0b at 6x6"): is the positive pre-image effect specific to the J2 hop?
Phi_a = (1 + a K)^{-1} psi0 for K in {V_J2, H_J2 = V_J2 + D_J2, V_J1} (bond-class hop operators, sector_ops.py), a chosen so that
1 + aK is positive definite with cond in {1.5, 2, 3, 5, 10, 30, 100}, both signs of a.
  python c2_t0b_6x6.py OUT_DIR TABDIR [smoke]   (env C2B_TIMING=1: slice timing also in a full run)
env ST6_CSR, ST6_TABLE (sector), GPU.  Measurement logic (W_M, by-decade, tail) copied from c2_t0_6x6.py.
smoke: time the table build on a 1M-rep slice per class; if the full build extrapolates to <= 30 min per class,
build and save both tables and run checks (a)-(c), then stop.
"""
import os, sys, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
for p_ in (os.path.join(HERE, '..', '..', 'experiments', 'stall_6x6'), os.environ.get('ST6_CODE', ''), HERE):
    if p_ and os.path.isdir(p_) and p_ not in sys.path: sys.path.insert(0, p_)
import st6_sector as SS
import sector_ops as OPS
import jax, jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
OUT, TABDIR = sys.argv[1], sys.argv[2]; SMOKE = len(sys.argv) > 3 and sys.argv[3] == 'smoke'
os.makedirs(OUT, exist_ok=True); F = os.path.join(OUT, 'c2_t0b_smoke.json' if SMOKE else 'c2_t0b_6x6.json')
T0 = time.time(); log = SS.log
ROOT = '/project/theorie/a/A.Otaifi/Chatty/runs/stall6'
sec = SS.Sector(os.environ.get('ST6_CSR', ROOT + '/csr'), os.environ.get('ST6_TABLE', ROOT + '/data/psi0_6x6_table.npz'))
D = sec.D; N = 36
res = dict(D=int(D), E0_site=sec.E0 / N, smoke=SMOKE)
def dump(): json.dump(res, open(F, 'w'), indent=1)
D0 = OPS.diag0(sec)
DJ1 = OPS.diag0(sec, 0, len(SS.NN)); DJ2 = OPS.diag0(sec, len(SS.NN))       # bond-class diagonals
res['check_DJ1+DJ2=D0_maxabs'] = float(jnp.max(jnp.abs(DJ1 + DJ2 - D0)))
res['check_DJ2_misscaled_maxabs'] = float(jnp.max(jnp.abs(DJ1 + 1.01 * DJ2 - D0)))
log('[c2b] diag split check', res['check_DJ1+DJ2=D0_maxabs'], 'misscaled', res['check_DJ2_misscaled_maxabs'])

TIMING = SMOKE or os.environ.get('C2B_TIMING', '0') == '1'   # full job may also time a 1M slice first
if TIMING:
    res['slice_timing'] = {}
    for cls in ('J1', 'J2'):
        OPS.build_table(sec, cls, limit=1 << 14, log=log)            # compile
        t, info = OPS.build_table(sec, cls, limit=1 << 20, log=log)
        info['extrap_full_sec'] = info['sec'] * D / (1 << 20); res['slice_timing'][cls] = info
        log('[c2b] slice', info); dump()
    lim = 1800 if SMOKE else float(os.environ.get('C2B_BUILD_LIMIT', '3600'))
    if max(v['extrap_full_sec'] for v in res['slice_timing'].values()) > lim:
        res['stop'] = f'full build > {lim:.0f} s per class: size the job from slice_timing'; dump()
        log('[c2b] DONE STOP build too slow'); sys.exit(3)

tabs = {}
for cls in ('J1', 'J2'):
    t, info = OPS.load_or_build(sec, cls, TABDIR, log=log); res[f'table_{cls}'] = info; tabs[cls] = t; dump()
V = {c: OPS.BondOp(sec, tabs[c], c) for c in ('J1', 'J2')}; del tabs
ck = OPS.checks(sec, D0, V['J1'], V['J2']); res['checks'] = ck; dump(); log('[c2b] checks', ck)
if not (ck['a_pass'] and ck['b_fails'] and ck['c_pass'] and res['check_DJ1+DJ2=D0_maxabs'] < 1e-12
        and res['check_DJ2_misscaled_maxabs'] > 1e-6):
    res['verdict'] = 'CHECK_FAILED'; dump(); log('[c2b] DONE VERDICT CHECK_FAILED'); sys.exit(2)
if SMOKE:
    res['sec'] = time.time() - T0; dump(); log('[c2b] DONE SMOKE OK'); sys.exit(0)

# ---------------------------------------------------------------- measurement (as c2_t0_6x6.py)
MA = jnp.uint64(sum(1 << (x + 6 * y) for y in range(6) for x in range(6) if (x + y) % 2 == 0))
sN = jnp.where(jax.lax.population_count(sec.reps & MA) % 2 == 0, 1.0, -1.0)
v0 = sec.v0.astype(jnp.float64); v0 = v0 / jnp.linalg.norm(v0)
res['check_E0'] = [float(v0 @ sec.H(v0)), sec.E0]
def WM(v):
    v = v / jnp.linalg.norm(v); g = jnp.sign(jnp.sum(v * jnp.abs(v) * sN))
    return float(jnp.sum(v ** 2 * ((jnp.sign(v) * g) != sN)))
def by_dec(v):
    v = v / jnp.linalg.norm(v); g = jnp.sign(jnp.sum(v * jnp.abs(v) * sN)); bad = (jnp.sign(v) * g) != sN
    p_conf = v ** 2 / sec.n
    d = np.asarray(jnp.floor(jnp.log10(jnp.maximum(p_conf, 1e-300))).astype(jnp.int32))
    p = np.asarray(v ** 2); b = np.asarray(bad)
    return {str(k): [float(p[d == k].sum()), float((p * b)[d == k].sum() / max(p[d == k].sum(), 1e-300))]
            for k in range(int(d.max()), max(int(d[p > 0].min()), -20) - 1, -1) if (d == k).any()}
def tail(v):
    v = v / jnp.linalg.norm(v); p_conf = v ** 2 / sec.n
    return float(jnp.sum(jnp.where(p_conf < 1e-9, v ** 2, 0.0)))
res['W_M_psi0'] = WM(v0); res['psi0_by_decade'] = by_dec(v0); res['psi0_tail_w(p<1e-9)'] = tail(v0)
res['W_M_psi0_check'] = abs(res['W_M_psi0'] - 0.019558252288183215) < 1e-6
log('[c2b] W_M psi0', res['W_M_psi0'], res['W_M_psi0_check'], 'E', res['check_E0']); dump()

def lanczos_ext(op, m=150):
    q = jax.random.normal(jax.random.PRNGKey(1), (D,)); q = q / jnp.linalg.norm(q); qp = jnp.zeros(D); b = 0.0
    al, be = [], []
    for _ in range(m):
        z = op(q) - b * qp; a = float(q @ z); z = z - a * q; b = float(jnp.linalg.norm(z))
        al.append(a); be.append(b); qp, q = q, z / b
    T = np.diag(al) + np.diag(be[:-1], 1) + np.diag(be[:-1], -1); ev, ec = np.linalg.eigh(T)
    return float(ev[0]), float(ev[-1]), float(abs(be[-1] * ec[-1, 0])), float(abs(be[-1] * ec[-1, -1]))
def cg(op, b, tol=1e-10, maxit=3000):
    x = jnp.zeros_like(b); r = b; p = r; rs = float(r @ r); nb = float(jnp.linalg.norm(b)); indef = False
    for it in range(maxit):
        Ap = op(p); pAp = float(p @ Ap)
        if pAp <= 0: indef = True; break
        al = rs / pAp; x = x + al * p; r = r - al * Ap; rn = float(r @ r)
        if np.sqrt(rn) < tol * nb: break
        p = r + (rn / rs) * p; rs = rn
    return x, it + 1, float(jnp.linalg.norm(b - op(x)) / nb), indef

CONDS = (1.5, 2, 3, 5, 10, 30, 100)
res['scan'] = {}; res['spectrum'] = {}; res['best_le3'] = {}
KER = {'J2': V['J2'], 'HJ2': lambda x: V['J2'](x) + DJ2 * x, 'J1': V['J1']}   # HJ2 = V_J2 + D_J2 (lead, 2026-10-23)
for cls in ('J2', 'HJ2', 'J1'):
    K = KER[cls]; lmin, lmax, rmin, rmax = lanczos_ext(K)
    res['spectrum'][cls] = dict(lmin=lmin, lmax=lmax, ritz_res_min=rmin, ritz_res_max=rmax)
    log('[c2b]', cls, 'spectrum', res['spectrum'][cls]); res['scan'][cls] = []
    for c in CONDS:
        for sgn in (+1, -1):
            a = (c - 1) / (lmax - c * lmin) if sgn > 0 else (c - 1) / (lmin - c * lmax)
            t = time.time(); phi, nit, rel, indef = cg(lambda x: x + a * K(x), v0)
            cond = (1 + a * (lmax if a > 0 else lmin)) / (1 + a * (lmin if a > 0 else lmax))
            rec = dict(a=float(a), cond_target=c, cond=float(cond), cg_iters=nit, rel_resid=rel, indefinite=indef,
                       W_M=WM(phi), W_M_ratio=WM(phi) / res['W_M_psi0'], tail_w=tail(phi), sec=time.time() - t)
            res['scan'][cls].append(rec); log('[c2b]', cls, rec); dump()
    ok = [r for r in res['scan'][cls] if r['cond'] <= 3 + 1e-9 and not r['indefinite'] and r['rel_resid'] < 1e-8]
    best = min(ok, key=lambda r: r['W_M']); res['best_le3'][cls] = best
    phi = cg(lambda x: x + best['a'] * K(x), v0)[0]; res[f'best_le3_by_decade_{cls}'] = by_dec(phi)
    np.save(os.path.join(OUT, f'phi_best_le3_{cls}.npy'), np.asarray(phi / jnp.linalg.norm(phi), np.float32)); dump()
W0 = res['W_M_psi0']; r2 = res['best_le3']['J2']['W_M'] / W0; r1 = res['best_le3']['J1']['W_M'] / W0
rh = res['best_le3']['HJ2']['W_M'] / W0
res['ratio_J2_le3'] = r2; res['ratio_J1_le3'] = r1; res['ratio_HJ2_le3'] = rh
res['verdict'] = ('PASS' if (r2 <= 0.1 and r1 >= 0.5) else ('KILL' if r2 >= 0.5 else 'INCONCLUSIVE'))   # pre-registered (V_J2)
res['verdict_HJ2_same_rule'] = ('PASS' if (rh <= 0.1 and r1 >= 0.5) else ('KILL' if rh >= 0.5 else 'INCONCLUSIVE'))
res['sec'] = time.time() - T0; dump()
log('[c2b] DONE VERDICT', res['verdict'], 'ratio_J2', r2, 'ratio_J1', r1, 'ratio_HJ2', rh, res['verdict_HJ2_same_rule'])
