"""Stage 0 of experiments/itfit_6x6 (README.md): stiffness of the frozen FN Hamiltonian F = H_FN[|psi_P|, s_P] and
exact (unprojected) imaginary-time step schemes.  No training, no sampling; float64 on one GPU.

  python itfit_exact.py OUT [SPEC.json]
Writes OUT/itfit_exact.json (incrementally).
"""
import json, os, sys, time
import numpy as np
from itfit_common import Exact, log, dump, N, f64, decade_table
import jax
import jax.numpy as jnp
import scipy.linalg

OUT = sys.argv[1]
SPEC = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else {}
os.makedirs(OUT, exist_ok=True)
F_OUT = os.path.join(OUT, 'itfit_exact.json')
T00 = time.time()
ex = Exact(); sec = ex.sec
lP, sP = ex.lP, ex.sP
res = dict(spec=SPEC)

# ------------------------------------------------------------------ start guide, FN ground state
t0 = time.time()
Efn, u, info = sec.fn_solve(lP, sP)
res['fn'] = info
D, w, Kw = ex.fn_diag(lP, sP)
b0 = w / jnp.linalg.norm(w)
E_f0, _ = ex.Ef(b0, D, sP, Kw / jnp.linalg.norm(w))
G0 = E_f0 - Efn
chk = sec.fn_rayleigh(lP, sP, lP)
res['start'] = dict(E_FN_dE_site=(Efn / N - (-0.50380965389088)), Ef0_dE_site=E_f0 / N + 0.50380965389088,
                    G0_site=G0 / N, Ef0_check_vs_fn_rayleigh=E_f0 - chk)
log('START', res['start'])
t_mv = time.time(); _ = ex.Kop(b0, sP); t_mv = time.time() - t_mv
res['sec_per_Kmatvec'] = t_mv

# ------------------------------------------------------------------ stiffness: spectrum, wall
Wall = D - ex.Hdiag                                         # sum_{viol} |H_xy| a_y/a_x (per configuration)
ex.Hdiag = None
tau_pos = 1.0 / float(jnp.max(D - Kw / w - E_f0))           # positivity of the explicit target a (1 - tau (E_L^F - E))
iEL = int(jnp.argmax(D - Kw / w))
Kb0 = Kw / jnp.linalg.norm(w)
del Kw
K1 = ex.Kop(jnp.ones_like(D), sP)
gersh = float(jnp.max(D + K1)); maxK1 = float(jnp.max(K1)); del K1
lan = ex.lanczos_ext(D, sP, m=SPEC.get('lanczos_m', 100))
lam_max = lan[-1]['lmax']
tau_stab = 2.0 / (lam_max - Efn)
p = u * u                                                    # phi^2 rep mass
lphi = jnp.log(jnp.maximum(u / ex.sqn, 1e-300))
delta = lphi - lP
mu = float(jnp.sum(p * delta))
Ku = ex.Kop(u, sP)
c_delta = ex.node_c(delta, u, Ku, sP)
del Ku
Qdelta = float(jnp.sum(c_delta)) / N
pc = (p / sec.n).astype(jnp.float32)                         # per-configuration phi^2
res['stiffness'] = dict(
    lambda_max=lam_max, lambda_min_lanczos=lan[-1]['lmin'], E_FN=Efn, lanczos=lan, gershgorin_max=gersh,
    max_D=float(jnp.max(D)), max_wall=float(jnp.max(Wall)), min_D_minus_EFN=float(jnp.min(D - Efn)),
    tau_stab=tau_stab, tau_pos=tau_pos, argmax_EL_wall=float(Wall[iEL]), argmax_EL_phi2_config=float(pc[iEL]),
    max_kept_rowsum=maxK1, Qdelta_site=Qdelta,
    wall_mean_phi2=float(jnp.sum(p * Wall)), wall_mean_a2=float(jnp.sum(b0 * b0 * Wall)),
    phi2_mass_wall_gt={str(t): float(jnp.sum(jnp.where(Wall > t, p, 0.0))) for t in (1, 10, 100, 1e3, 1e4)},
    gain_share_wall_gt={str(t): float(jnp.sum(jnp.where(Wall > t, c_delta, 0.0))) / (Qdelta * N) for t in (1, 10, 100, 1e3, 1e4)})
res['wall_decades'] = decade_table(jnp.maximum(Wall, 1e-300),
                                   dict(reps=jnp.ones_like(p), configs=sec.n, phi2=p, a2=b0 * b0, gain=c_delta,
                                        var=p * (delta - mu) ** 2), lo=-3, hi=5)
res['phi2_decades'] = decade_table(jnp.maximum(pc, 1e-300),
                                   dict(phi2=p, gain=c_delta, wall_x_phi2=p * Wall), lo=-13, hi=-5)
log('STIFF', {k: v for k, v in res['stiffness'].items() if k != 'lanczos'})
for r in res['wall_decades']: log('  wall', r)
dump(F_OUT, res)
del delta


# ------------------------------------------------------------------ exact step schemes
def frac(E):
    return (E_f0 - E) / G0


def neg(t):
    return float(jnp.sum(jnp.where(t < 0, t * t, 0.0)) / jnp.sum(t * t))


def n90_of(rows):
    fr = np.array([r['frac'] for r in rows]); n = np.array([r['n'] for r in rows])
    hit = np.where(fr >= 0.9)[0]
    if hit.size: return float(n[hit[0]]), False
    m = max(3, len(rows) // 5)
    y = np.log(np.maximum(1 - fr[-m:], 1e-12)); x = n[-m:]
    if len(x) < 3 or not np.all(np.isfinite(y)): return float('nan'), True
    s = np.polyfit(x, y, 1)[0]
    if s >= 0: return float('inf'), True
    return float(n[-1] + (np.log(0.1) - y[-1]) / s), True


def run(name, step, n_max, stop=None, every=1, extra=None):
    stop = SPEC.get('stop', 0.99) if stop is None else stop
    t0 = time.time(); nk0 = ex.nK
    b = b0
    E, Kb = ex.Ef(b, D, sP)
    rows = [dict(n=0, frac=0.0, dE=0.0)]
    inc = 0; Eprev = E
    for n in range(1, n_max + 1):
        b, Kb, inf = step(b, E, Kb)
        nb = jnp.linalg.norm(b); b = b / nb; Kb = Kb / nb
        E = float((b @ (D * b) - b @ Kb))
        if E > Eprev + 1e-12 * abs(E): inc += 1
        Eprev = E
        row = dict(n=n, frac=frac(E), neg_weight=neg(b), **inf)
        if n % every == 0 or n <= 10 or frac(E) >= stop or n == n_max: rows.append(row)
        if frac(E) >= stop: break
    n90, extrap = n90_of(rows)
    out = dict(scheme=name, rows=rows, n90=n90, n90_extrapolated=extrap, energy_increase_events=inc,
               final_frac=rows[-1]['frac'], steps=rows[-1]['n'], Kmatvecs=ex.nK - nk0, sec=time.time() - t0)
    if extra: out.update(extra(b))
    log(f'SCHEME {name}: frac {out["final_frac"]:.4f} at n={out["steps"]}, n90 {n90:.1f} (extrap {extrap}), '
        f'incr {inc}, matvecs {out["Kmatvecs"]}, {out["sec"]:.0f}s')
    return out


def residual_split(b):
    """where the still-missing gain sits: I2 node weight of e = log b - log phi per phi^2 decade and per wall decade."""
    e = jnp.log(jnp.maximum(b / ex.sqn, 1e-300)) - lphi
    Ku_ = ex.Kop(u, sP)
    ce = ex.node_c(e, u, Ku_, sP)
    return dict(Q_resid_site=float(jnp.sum(ce)) / N, resid_over_Qdelta=float(jnp.sum(ce)) / (Qdelta * N),
                resid_phi2_decades=decade_table(jnp.maximum(pc, 1e-300), dict(resid=ce), lo=-13, hi=-5),
                resid_wall_decades=decade_table(jnp.maximum(Wall, 1e-300), dict(resid=ce), lo=-3, hi=5))


def explicit(tau):
    def st(b, E, Kb):
        t = b - tau * (D * b - Kb - E * b)
        return t, ex.Kop(t, sP), {}
    return st


def opt_explicit(b, E, Kb):
    r = D * b - Kb - E * b
    Kr = ex.Kop(r, sP)
    V = [b, r]; KV = [Kb, Kr]
    A = np.array([[float(V[i] @ (D * V[j]) - V[i] @ KV[j]) for j in range(2)] for i in range(2)])
    B = np.array([[float(V[i] @ V[j]) for j in range(2)] for i in range(2)])
    ev, ec = scipy.linalg.eigh(0.5 * (A + A.T), B)
    c = ec[:, 0] / ec[0, 0]
    return b + c[1] * r, Kb + c[1] * Kr, dict(tau_eff=-float(c[1]))


def krylov2(b, E, Kb):
    v1 = b / jnp.linalg.norm(b); K1_ = Kb / jnp.linalg.norm(b)
    Fv1 = D * v1 - K1_; a1 = float(v1 @ Fv1)
    r = Fv1 - a1 * v1; b1 = float(jnp.linalg.norm(r)); v2 = r / b1
    K2 = ex.Kop(v2, sP); Fv2 = D * v2 - K2; a2 = float(v2 @ Fv2)
    r = Fv2 - a2 * v2 - b1 * v1; r = r - (v1 @ r) * v1 - (v2 @ r) * v2
    b2 = float(jnp.linalg.norm(r)); v3 = r / b2
    K3 = ex.Kop(v3, sP); Fv3 = D * v3 - K3; a3 = float(v3 @ Fv3)
    T = np.array([[a1, b1, float(v1 @ Fv3)], [b1, a2, float(v2 @ Fv3)], [float(v1 @ Fv3), float(v2 @ Fv3), a3]])
    ev, ec = np.linalg.eigh(0.5 * (T + T.T))
    c = ec[:, 0] * np.sign(ec[0, 0])
    return c[0] * v1 + c[1] * v2 + c[2] * v3, c[0] * K1_ + c[1] * K2 + c[2] * K3, {}


def semi(tau):
    def st(b, E, Kb):
        t = (b + tau * Kb) / (1.0 + tau * (D - E))
        return t, ex.Kop(t, sP), {}
    return st


def implicit(tau):
    def st(b, E, Kb):
        t, it, rr = ex.pcg(1.0 + tau * (D - E), tau, sP, b, tol=SPEC.get('cg_tol', 1e-8))
        return t, ex.Kop(t, sP), dict(cg_it=it, cg_res=rr)
    return st


# one-step captured gain vs tau
one = []
for tau in [tau_pos, 0.5 * tau_stab, tau_stab] + SPEC.get('one_step_explicit', [1e-3, 3e-3, 1e-2, 3e-2, 0.1]):
    E, Kb = ex.Ef(b0, D, sP, Kb0)
    t = b0 - tau * (D * b0 - Kb - E * b0)
    Et, _ = ex.Ef(t, D, sP)
    one.append(dict(scheme='explicit', tau=tau, frac=frac(Et), neg_weight=neg(t)))
for tau in SPEC.get('one_step_semi', [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 1e3]):
    E, Kb = ex.Ef(b0, D, sP, Kb0)
    t = (b0 + tau * Kb) / (1.0 + tau * (D - E))
    Et, _ = ex.Ef(t, D, sP)
    one.append(dict(scheme='semi', tau=tau, frac=frac(Et)))
res['one_step'] = one
for r in one: log('  one-step', r)
dump(F_OUT, res)

res['schemes'] = []
plan = SPEC.get('plan', [
    ['semi', 0.1, 150], ['semi', 0.3, 150], ['semi', 1.0, 150], ['semi', 3.0, 150], ['semi', 0.03, 150],
    ['opt_explicit', None, 150], ['krylov2', None, 60],
    ['implicit', 0.03, 15], ['implicit', 0.3, 15], ['implicit', 3.0, 15],
    ['explicit_stab', None, 600], ['explicit_pos', None, 200]])
for kind, tau, nmax in plan:
    if kind == 'semi':
        r = run(f'semi tau={tau}', semi(tau), nmax, extra=residual_split if tau in SPEC.get('split_taus', [0.3, 1.0]) else None)
    elif kind == 'implicit':
        r = run(f'implicit tau={tau}', implicit(tau), nmax)
    elif kind == 'opt_explicit':
        r = run('optimal explicit (1-hop RR)', opt_explicit, nmax, extra=residual_split)
    elif kind == 'krylov2':
        r = run('Krylov 2-hop RR', krylov2, nmax)
    elif kind == 'explicit_stab':
        r = run(f'explicit tau=0.9 tau_stab={0.9 * tau_stab:.3e}', explicit(0.9 * tau_stab), nmax, every=10)
    elif kind == 'explicit_tau':
        r = run(f'explicit tau={tau}', explicit(tau), nmax, every=10)
    elif kind == 'explicit_pos':
        r = run(f'explicit tau=tau_pos={tau_pos:.3e}', explicit(tau_pos), nmax, every=10)
    r.update(kind=kind, tau=tau)
    res['schemes'].append(r)
    dump(F_OUT, res)
res['sec'] = time.time() - T00
res['Kmatvecs'] = ex.nK
dump(F_OUT, res)
log('DONE', res['sec'])
