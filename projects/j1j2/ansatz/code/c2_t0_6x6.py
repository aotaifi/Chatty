"""C2-T0 at exact 6x6 (PLAN_ansatz.md): positive pre-image of psi0 under the local hop kernel 1 + a V,
V = H - diag(H) (the configuration-space off-diagonal part, restricted to the symmetric sector).
  python c2_t0_6x6.py OUT_DIR      (env ST6_CSR, ST6_TABLE as in cx_lib; GPU)
Measured per a: W_M(Phi_a) (Marshall-violating weight, total and by decade of Phi's per-config probability),
tail weight (decades <= -10), CG residual. Pass/kill: PLAN_ansatz.md C2-T0 (6x6).
"""
import os, sys, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
for p_ in (os.path.join(HERE, '..', '..', 'experiments', 'stall_6x6'), os.environ.get('ST6_CODE', '')):
    if p_ and os.path.isdir(p_) and p_ not in sys.path: sys.path.insert(0, p_)
import st6_sector as SS
import jax, jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); F = os.path.join(OUT, 'c2_t0_6x6.json'); T0 = time.time()
ROOT = '/project/theorie/a/A.Otaifi/chatty_stall6'
sec = SS.Sector(os.environ.get('ST6_CSR', ROOT + '/csr'), os.environ.get('ST6_TABLE', ROOT + '/data/psi0_6x6_table.npz'))
D = sec.D; N = 36
BI = jnp.asarray(SS.BI); BJ = jnp.asarray(SS.BJ); JB = jnp.asarray(SS.JB)
@jax.jit
def diag_chunk(x):
    valid = (((x[:, None] >> BI[None, :]) ^ (x[:, None] >> BJ[None, :])) & jnp.uint64(1)).astype(jnp.float64)
    return 0.25 * JB.sum() - 0.5 * (valid * JB[None, :]).sum(1)
D0 = jnp.concatenate([diag_chunk(sec.reps[i:i + (1 << 20)]) for i in range(0, D, 1 << 20)])
MA = jnp.uint64(sum(1 << (x + 6 * y) for y in range(6) for x in range(6) if (x + y) % 2 == 0))
sN = jnp.where(jax.lax.population_count(sec.reps & MA) % 2 == 0, 1.0, -1.0)
v0 = sec.v0.astype(jnp.float64); v0 = v0 / jnp.linalg.norm(v0)
def V(x): return sec.H(x) - D0 * x
res = dict(D=int(D), E0_site=sec.E0 / N)
# ---- checks: <v0|H|v0> = E0; V symmetric (random u, w); diag of the orbit basis vs D0 is NOT required equal
E = float(v0 @ sec.H(v0)); res['check_E0'] = [E, sec.E0]
k1, k2 = jax.random.split(jax.random.PRNGKey(0)); u = jax.random.normal(k1, (D,)); w = jax.random.normal(k2, (D,))
res['check_sym'] = float(abs(u @ V(w) - w @ V(u)) / abs(u @ V(w)))
def WM(v):
    v = v / jnp.linalg.norm(v); g = jnp.sign(jnp.sum(v * jnp.abs(v) * sN))
    return float(jnp.sum(v ** 2 * ((jnp.sign(v) * g) != sN)))
def by_dec(v):
    v = v / jnp.linalg.norm(v); g = jnp.sign(jnp.sum(v * jnp.abs(v) * sN)); bad = (jnp.sign(v) * g) != sN
    p_conf = v ** 2 / sec.n                                    # per-configuration probability
    d = np.asarray(jnp.floor(jnp.log10(jnp.maximum(p_conf, 1e-300))).astype(jnp.int32))
    p = np.asarray(v ** 2); b = np.asarray(bad)
    return {str(k): [float(p[d == k].sum()), float((p * b)[d == k].sum() / max(p[d == k].sum(), 1e-300))]
            for k in range(int(d.max()), max(int(d[p > 0].min()), -20) - 1, -1) if (d == k).any()}
def tail(v):
    v = v / jnp.linalg.norm(v); p_conf = v ** 2 / sec.n
    return float(jnp.sum(jnp.where(p_conf < 1e-9, v ** 2, 0.0)))
res['W_M_psi0'] = WM(v0); res['psi0_by_decade'] = by_dec(v0); res['psi0_tail_w(p<1e-9)'] = tail(v0)
print('[c2] W_M psi0', res['W_M_psi0'], 'E check', res['check_E0'], 'sym', res['check_sym'], flush=True)
# ---- extreme eigenvalues of V (Lanczos, 60 steps)
def lanczos_ext(op, m=60):
    q = jax.random.normal(jax.random.PRNGKey(1), (D,)); q = q / jnp.linalg.norm(q); qp = jnp.zeros(D); b = 0.0
    al, be = [], []
    for _ in range(m):
        z = op(q) - b * qp; a = float(q @ z); z = z - a * q; b = float(jnp.linalg.norm(z))
        al.append(a); be.append(b); qp, q = q, z / b
    T = np.diag(al) + np.diag(be[:-1], 1) + np.diag(be[:-1], -1); ev = np.linalg.eigvalsh(T)
    return float(ev[0]), float(ev[-1])
lmin, lmax = lanczos_ext(V); res['V_lmin'], res['V_lmax'] = lmin, lmax
print('[c2] V spectrum', lmin, lmax, flush=True)
def cg(op, b, x0=None, tol=1e-10, maxit=2000):
    x = jnp.zeros_like(b) if x0 is None else x0; r = b - op(x); p = r; rs = float(r @ r); nb = float(jnp.linalg.norm(b))
    for it in range(maxit):
        Ap = op(p); al = rs / float(p @ Ap); x = x + al * p; r = r - al * Ap; rn = float(r @ r)
        if np.sqrt(rn) < tol * nb: break
        p = r + (rn / rs) * p; rs = rn
    return x, it + 1, float(jnp.linalg.norm(b - op(x)) / nb)
def bicgstab(op, b, tol=1e-10, maxit=4000):
    x = jnp.zeros_like(b); r = b - op(x); rh = r; rho = al = om = 1.0; v = p = jnp.zeros_like(b); nb = float(jnp.linalg.norm(b))
    for it in range(maxit):
        rn = float(rh @ r); be = (rn / rho) * (al / om); rho = rn
        p = r + be * (p - om * v); v = op(p); al = rho / float(rh @ v); s_ = r - al * v; t = op(s_)
        om = float(t @ s_) / float(t @ t); x = x + al * p + om * s_; r = s_ - om * t
        if float(jnp.linalg.norm(r)) < tol * nb: break
    return x, it + 1, float(jnp.linalg.norm(b - op(x)) / nb)
if res['check_sym'] > 1e-8:
    print('[c2] V not symmetric in this basis: BiCGSTAB', flush=True); cg = bicgstab
# a range where 1 + a V is positive definite with cond <= 1e3: a in (-1/lmax, 1/|lmin|)
amax_neg = (1 - 1e-3) / lmax; amax_pos = (1 - 1e-3) / abs(lmin)
grid = sorted(set([-amax_neg * f for f in (0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)] +
                  [amax_pos * f for f in (0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)]))
res['scan'] = []
for a in grid:
    t = time.time(); phi, nit, rel = cg(lambda x: x + a * V(x), v0)
    cond = (1 + a * (lmax if a > 0 else lmin)) / (1 + a * (lmin if a > 0 else lmax))
    rec = dict(a=float(a), cond=float(cond), cg_iters=nit, rel_resid=rel, W_M=WM(phi), tail_w=tail(phi), sec=time.time() - t)
    res['scan'].append(rec); print('[c2]', rec, flush=True); json.dump(res, open(F, 'w'), indent=1)
best = min(res['scan'], key=lambda r: r['W_M']); res['best'] = best; res['ratio'] = best['W_M'] / res['W_M_psi0']
phi = cg(lambda x: x + best['a'] * V(x), v0)[0]; res['best_by_decade'] = by_dec(phi)
np.save(os.path.join(OUT, 'phi_best.npy'), np.asarray(phi / jnp.linalg.norm(phi), np.float32))
res['verdict'] = 'PASS' if res['ratio'] <= 0.1 else ('KILL' if res['ratio'] >= 0.5 else 'INCONCLUSIVE')
res['sec'] = time.time() - T0; json.dump(res, open(F, 'w'), indent=1); print('[c2] DONE', res['verdict'], res['ratio'], flush=True)
