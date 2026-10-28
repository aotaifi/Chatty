"""Test G (PLAN.md "TEST G", board block 10): fit the TRUE E_FN of the complex PP + 3x3 backflow copy by
majorise-minimise (MM) with the Hellmann-Feynman (HF) gradient.  Model and warm start as cx_c1fit.py (C1 MODEL block,
copied unchanged: complex ppbf window 3, theta = [Re 1944, Im 1944]); review conditions C1-C8 of
projects/j1j2/reviews/2026-10-28_true_EFN_gradient_review.md built in.

  python cx_g1fit.py OUT SPEC.json [--mode probe|prod|dry]

Notation (sector = k=0, A1, flip-even orbit basis of st6_sector, D = 15.8M reps, normalised orbit states, real symmetric H):
  la_r      ln|A| at rep r (network table), normalised la_eff = ln max(a, 1e-15), a = exp(la) / sqrt(sum n a^2)  (= fn_solve floor)
  w_r       a_r sqrt(n_r);  s2 fixed sign (refresh_17409419/s2.npy)
  blocked   sector entry (r, r'), r != r', with s2_r H_rr' s2_r' > 0 -- from s2 and the CSR codes ONLY, never from A (C2)
  u = chi*  ground state of H_FN[A] (normalised, u >= 0 in the s2 frame), solved by gsolve = sec.fn_solve plus records
  L(theta; u) = sum_blocked H_rr' (u_r^2 w_r'/w_r + u_r'^2 w_r/w_r')
             = P + C,   P = sum_blocked H_rr' (u_r sqrt(w_r'/w_r) - u_r' sqrt(w_r/w_r'))^2  (pair form, float64, C8),
                        C = 2 sum_blocked H_rr' u_r u_r'  (independent of theta)
  <u|H_FN[A]|u> = <u|H_kept + diag H|u> + L  =>  E_FN(theta) <= <u_k|H_FN[A(theta)]|u_k>, equality at theta_k (MM, review b1)
  g_r = dL/dla_r at fixed u = dE_FN/dla_r (HF), sum_r g_r = 0 (scale invariance)
Outer loop:  table la_k -> u_k = chi*(la_k) -> k_inner sampled steps on L(.; u_k) -> table la_{k+1} -> EXACT check
  P(theta_{k+1}; u_k) < P(theta_k; u_k) (else backtrack: halve the block's parameter change, at most max_backtrack times, then
  revert) -> exact gauge fix -> re-solve.  Only the block's final L must be below its start (review C6).
Inner step (sampled, GGN = damped Gauss-Newton on the convex pair loss, the optimiser family of cx_c1fit):
  pairs drawn with probability proportional to their exact pair loss at the block start: rep r ~ T_r = u_r^2 V_r (exact, from the
  CSR pair kernel), then one blocked bond of a config of r ~ |H_xy| |A(y)| (canon lookup, as cx_c1fit.bonds_of).  For the physical
  pair e = {x, y}: t = ln|A(y)| - ln|A(x)|, a = u_r^2/n_r, b = u_r'^2/n_r', h = J_b/2,
     l_e = h (a e^t + b e^-t),  l'_e = h (sqrt(a) e^{t/2} - sqrt(b) e^{-t/2})(sqrt(a) e^{t/2} + sqrt(b) e^{-t/2}),  l''_e = l_e,
     importance weight c_e = L(theta_k)/l_e(theta_k)  (unbiased: E[(1/M) sum c_e l_e(theta)] = L(theta)).
  rows J_e = sqrt(c_e l_e / M) (dlnA(y) - dlnA(x))/dtheta,  r_e = sqrt(c_e/M) l'_e / sqrt(l_e)  (J^T r = sampled gradient of L)
  d = -(J^T J + lam I)^-1 J^T r, lam = lam_rel tr(J^T J)/P (lam_rel from lam0, x4 on reject, /2 on a significant gain, cap lam_cap).
  The gradient is the real gradient in (Re theta, Im theta): theta is a real vector [Re, Im], jax.grad of the real ln|A| (C4).
  Cap (C6): the step is scaled so the predicted blocked-bond change max_e |d t_e| on the direction batch is <= delta; it is
  rejected if the actual max |d t_e| on the acceptance batch exceeds 2 delta, or if the paired estimate of L(theta+d) - L(theta)
  on a fresh acceptance batch is > 0 (a step that raises L).
  Gauge (C5): A(c theta) = c^18 A(theta) exactly (F linear in theta).  After each accepted step theta <- theta exp(-D/18),
  D = estimate of sum_r PT_r (la_new - la_old) on a fixed sample of configurations drawn from PT = n phi^2 of the target
  (pure gauge: any error in D is harmless); at each re-solve D is taken exactly from the table.  No penalty term.
Readout per re-solve: E_FN(copy) per site, tax_0 = dE_FN(copy) - dE_FN(target) (target = round_1_target_la, solved once by the same
  solver; the split value 9.853014655922562e-6 is recorded next to it); absolute residual ||H_FN u - E u|| (||u|| = 1); per decade of
  u_r^2: max and rms of the local energy (H_FN u)_r/u_r - E; sign certificate (count and weight of s2-frame components < 0 before |.|);
  exact P before/after the block; exact max blocked-bond |d t| over the block; the plain iteration guide <- |chi*| (no network) from
  the same start, one solve per re-solve of the main loop, and its per-solve rate.
Stop: |E_FN change| < flat_tol per site for flat_n consecutive re-solves, or the time cap (a block is only started when its measured
  cost fits), or max_outer.
Modes:
  probe2 (lead 2026-10-28, optimiser check after probe 4049890) stored c1 table as A_0, chi_0 = FN[A_0], n_probe_steps inner steps on
         L_chi0 (mixture sampler + line search), then a fresh batch of B_mm pairs: paired estimate of L(theta_n; chi_0) - L(theta_0; chi_0),
         an upper bound on E_FN[A_n] - E_FN[A_0] (MM, block 10).  Acceptance (lead): >= 70% of steps lower L by > 2 sigma; bound < 0.
  gate   (lead 2026-10-28, replaces the 2-seed production) ONE frozen-chi* block, seed 1: full table A_0 (control (1) gate) ->
         chi_0 = FN[A_0] -> E_FN[chi_0] (chi_0 as guide amplitude, sign s2) -> k_inner sampled steps on L_chi0 (stopped early if the
         time left would not cover table + solve + margin) -> full table A_1 -> exact P(A_1; chi_0) vs P(A_0; chi_0) -> E_FN[A_1].
         Readout per site: E_FN[A_0], E_FN[chi_0], E_FN[A_1], Delta_0 = E_FN[A_0] - E_FN[chi_0], tau_0 = E_FN[A_1] - E_FN[chi_0].
  prod   full table at step 0 = control (1) gate (tax_0 vs ref_tax0 rel <= 1e-9, table vs the c1 run's la_table), then the loop.
  probe  rates + controls (2), (3): table on a subset (rate; agreement with the stored la_table), stored la_table as la_0 for
         control (1), HF finite-difference check in table space (2a) and its broken version (3), theta-space FD of the sampled loss
         incl. Re-only and Im-only directions (2b), sampled vs exact HF gradient (2c), n_probe_steps inner steps (s/step), Im-only check.
  dry    CPU plumbing (stored table, solver maxit dry_fn_maxit, small batches).  NOT a result.
"""
import os, sys, time, math, json, hashlib
from functools import partial
import numpy as np
HERE_ = os.path.dirname(os.path.abspath(__file__))
for p_ in (HERE_, os.path.join(HERE_, '..', '..', 'experiments', 'stall_6x6')):
    if os.path.isdir(p_) and p_ not in sys.path: sys.path.append(p_)
from cx_lib import *
import scipy.linalg

OUT = sys.argv[1]; SPEC = json.load(open(sys.argv[2])); os.makedirs(OUT, exist_ok=True)
_a = sys.argv[3:]
MODE = SPEC.get('mode', 'prod')
if '--mode' in _a: MODE = _a[_a.index('--mode') + 1]
assert MODE in ('prod', 'probe', 'dry', 'gate', 'probe2'), MODE
F = os.path.join(OUT, 'cx_g1.json'); PROG = os.path.join(OUT, 'progress.txt'); T00 = time.time()
TIME_SEC = float(SPEC.get('time_sec', 3000))          # time cap of the whole run (wall limit minus margin)
SEED = int(SPEC.get('seed', 1))
K_INNER = int(SPEC.get('k_inner', 20)); M = int(SPEC.get('B_pairs', 4096)); M_ACC = int(SPEC.get('B_acc', 4096))
DELTA = float(SPEC.get('delta', 0.5)); LAM0 = float(SPEC.get('lam0', 1e-2)); LAM_CAP = float(SPEC.get('lam_cap', 1.0)); LAM_MIN = 1e-6
Z_ACC = float(SPEC.get('z_acc', 2.0)); MAX_OUTER = int(SPEC.get('max_outer', 1000)); MAX_BT = int(SPEC.get('max_backtrack', 1))
FLAT_TOL = float(SPEC.get('flat_tol', 1e-8)); FLAT_N = int(SPEC.get('flat_n', 3)); M_GAUGE = int(SPEC.get('B_gauge', 2048))
FN_TOL = float(SPEC.get('fn_tol', 1e-10)); FN_MAXIT = int(SPEC.get('fn_maxit', 400))
FLOOR = 1e-15
SAMPLER = SPEC.get('sampler', 'mix'); MIXG = float(SPEC.get('mix_g', 0.5)) if SAMPLER == 'mix' else 0.0
LS = [float(q) for q in SPEC.get('line_search', [1.0, 0.5, 0.25, 0.125])]     # step fractions tried on the acceptance batch
REF_TAX0 = SPEC.get('ref_tax0'); REF_E_TARGET_SPLIT = float(SPEC.get('ref_dE_target_split', 9.853014655922562e-06))
res = dict(spec=SPEC, mode=MODE, done=False)


def progress(msg):
    with open(PROG, 'a') as fh: fh.write(f'{time.strftime("%Y-%m-%dT%H:%M:%S")} PROGRESS {msg}\n')
    log(msg)


def save():
    dump(F, res)


def tleft():
    return TIME_SEC - (time.time() - T00)


# code drift guard: the files actually run must be the ones hashed at submission
CODE_FILES = ('cx_g1fit.py', 'cx_lib.py', 'rbmpp_lib.py', 'st6_sector.py')
_where = {fn: next((os.path.join(d, fn) for d in sys.path if d and os.path.exists(os.path.join(d, fn))), None) for fn in CODE_FILES}
SHA_RUN = {fn: (hashlib.sha256(open(p, 'rb').read()).hexdigest() if p else None) for fn, p in _where.items()}
SHA_EXP = SPEC.get('code_sha256')
res['code_sha256'] = dict(expected=SHA_EXP, running=SHA_RUN, paths=_where)
if SHA_EXP is not None:
    bad = [fn for fn in CODE_FILES if SHA_EXP.get(fn) != SHA_RUN[fn]]
    res['code_sha256']['match'] = not bad
    if bad:
        res['stop'] = f'CODE DRIFT {bad}'; save(); progress(f'CODE DRIFT: sha256 mismatch for {bad}; exit 3'); sys.exit(3)
elif MODE != 'dry':
    res['stop'] = 'no code_sha256'; save(); progress('CODE DRIFT GUARD: no code_sha256 in the spec; exit 3'); sys.exit(3)
save()

lab = Lab(); sec = lab.sec
progress(f'[g1 {MODE}] lab ready {time.time() - T00:.0f}s  D {sec.D}')


def norm_la(la):
    a = jnp.exp(la - jnp.max(la)); return la - jnp.max(la) - 0.5 * jnp.log(jnp.sum(sec.n * a * a))


def la_eff(la):
    """normalised ln a with the fn_solve floor a >= 1e-15; and the floored mask."""
    lan = norm_la(la)
    fl = lan < math.log(FLOOR)
    return jnp.maximum(lan, math.log(FLOOR)), fl


RF = SPEC['refresh']
s2 = jnp.asarray(np.load(os.path.join(RF, 's2.npy')).astype(np.float32))
ls2 = norm_la(jnp.asarray(np.load(SPEC['target_la']), jnp.float64))
PT = sec.vec(ls2, s2) ** 2                                          # n phi^2 of the target, sums to 1 (gauge weight, readout)
DEC_T = jnp.asarray(lab.decade_of(ls2)[0])                             # per-configuration decade of phi^2 (target, as cx_c1fit)
DECU = lambda v2: np.clip(np.floor(np.log10(np.maximum(np.asarray(v2), 1e-300))), -40, 0).astype(int)

# ============================================================================================ model (cx_c1fit.py C1 MODEL, complex ppbf window 3)
from rbmpp_lib import Lattice
lat = Lattice(6, 0.5, 'full')
IMG = lat.IMG; FLP = lat.FL.astype(jnp.float64); NH = N // 2
xy = np.array([(i % 6, i // 6) for i in range(N)])
DISP = jnp.asarray(np.array([[((xy[j, 0] - xy[i, 0]) % 6) + 6 * ((xy[j, 1] - xy[i, 1]) % 6) for j in range(N)]
                             for i in range(N)], np.int32))            # DISP[i, j] = r_j - r_i
BF = True; COMPLEX = True; BFW = 3; ND9 = 9; HW = 1
NP_REAL = N * N + 2 * ND9 * N                                          # 1944
NBRW = NBR


# C1 MODEL BEGIN (copied from cx_c1fit.py, unchanged)
def log_one(f, u1, u2, zz):
    order = jnp.argsort(-zz, stable=True); Uo = order[:NH]; Do = order[NH:]
    Fm = f[Uo][:, Do]
    if BF:
        zn = zz[NBRW]
        M1 = zn[Uo] @ u1
        M2 = zn[Do] @ u2
        Fm = Fm + jnp.take_along_axis(M1, DISP[Uo][:, Do], 1) + jnp.take_along_axis(M2, DISP[Do][:, Uo], 1).T
    sg, ld = jnp.linalg.slogdet(Fm)
    up = (zz > 0).astype(jnp.int32); dn = 1 - up
    inv = jnp.sum(up * (jnp.cumsum(dn) - dn))
    if COMPLEX:
        return ld, (1.0 - 2.0 * (inv % 2).astype(jnp.float64)) * sg
    par = (jnp.where(sg < 0, 1, 0) + inv) % 2
    return ld, 1.0 - 2.0 * par


def A_one(th, sigma):
    if COMPLEX:
        tc = th[:NP_REAL] + 1j * th[NP_REAL:]
    else:
        tc = th
    f = tc[:N * N].reshape(N, N)
    if BF:
        u = tc[N * N:].reshape(2, ND9, N); u1, u2 = u[0], u[1]
    else:
        u1 = u2 = None
    Z = FLP[:, None] * sigma.astype(jnp.float64)[IMG]
    ld, sgn = jax.vmap(lambda zz: log_one(f, u1, u2, zz))(Z)
    m = jax.lax.stop_gradient(jnp.max(ld))
    return m, jnp.sum(sgn * jnp.exp(ld - m))


def logA_one(th, sigma):
    m, S_ = A_one(th, sigma)
    return m + jnp.log(jnp.maximum(jnp.abs(S_), 1e-300))
# C1 MODEL END


fwd_j = jax.jit(lambda th, Sx: jax.vmap(lambda s_: logA_one(th, s_))(bits(Sx).astype(jnp.float64)))
jac_j = jax.jit(jax.vmap(jax.grad(logA_one), in_axes=(None, 0)))     # real gradient in (Re theta, Im theta)
FCH, JCH = int(SPEC.get('fch', 512)), int(SPEC.get('jch', 64)); TCH = int(SPEC.get('tch', 512))
theta = jnp.asarray(np.load(SPEC['warm_theta']), jnp.float64)
NPAR = int(theta.size); assert NPAR == 2 * NP_REAL, NPAR
res['warm_theta_sha256'] = hashlib.sha256(open(SPEC['warm_theta'], 'rb').read()).hexdigest()
res['npar'] = NPAR


def _pad(Sx, ch):
    m = Sx.shape[0]
    return (jnp.concatenate([Sx, jnp.repeat(Sx[:1], ch - m)]) if m < ch else Sx), m


def fwd(th, Sx, ch=None):
    ch = ch or FCH; out = []
    for i in range(0, Sx.shape[0], ch):
        s_, m = _pad(Sx[i:i + ch], ch); out.append(fwd_j(th, s_)[:m])
    return jnp.concatenate(out)


@partial(jax.jit, donate_argnums=(0,))
def _put(Gb, blk, i0):
    return jax.lax.dynamic_update_slice(Gb, blk, (i0, 0))


def jac(th, Sx):
    R_ = Sx.shape[0]; Gb = jnp.zeros((R_, NPAR), f64)
    for i in range(0, R_, JCH):
        s_, m = _pad(Sx[i:i + JCH], JCH)
        Gb = _put(Gb, jac_j(th, bits(s_).astype(jnp.float64))[:m], i)
    return Gb


def table(th, idx=None):
    reps = sec.reps if idx is None else sec.reps[jnp.asarray(idx)]
    out = []; t0 = time.time(); tl = t0; D_ = reps.shape[0]
    for i in range(0, D_, TCH):
        s_, m = _pad(reps[i:i + TCH], TCH); out.append(fwd_j(th, s_)[:m])
        if time.time() - tl > 120:
            jax.block_until_ready(out[-1]); tl = time.time(); progress(f'  table {i + m}/{D_}  {tl - t0:.0f}s')
    return jnp.concatenate(out)


# ============================================================================================ exact sector pieces
@partial(jax.jit, donate_argnums=(0, 1))
def _pair_chunk(T, g, Q, rbase, inc, idx, cod, lw, au, sq, s):
    """one CSR chunk of the pair kernel: blocked = off-diagonal with s_r C s_r' > 0 (s2 and H only).  Sector H_rr' = C sqrt(n_r/n_r')
    (symmetric: st6_sector header).  Returns T (x-side terms u_r^2 H w_r'/w_r scattered to r), g (dL/dlog w), P (sum of squares), C."""
    row = rbase + jnp.cumsum(inc.astype(jnp.int32))
    c = cod.astype(jnp.float64) * 0.125
    blk = (idx != row) & (s[row] * s[idx] * c > 0)
    h = jnp.where(blk, jnp.abs(c) * sq[row] / sq[idx], 0.0)
    t = jnp.where(blk, lw[idx] - lw[row], 0.0)
    e = jnp.exp(0.5 * t)
    p = au[row] * e; q = au[idx] / e
    P = jnp.sum(h * (p - q) ** 2); Cc = jnp.sum(2.0 * h * au[row] * au[idx]); nb = jnp.sum(blk)
    T = T.at[row].add(h * p * p).at[idx].add(h * q * q)
    dt = h * (p - q) * (p + q)
    g = g.at[idx].add(dt).at[row].add(-dt)
    adt = jnp.abs(dt); Q = Q.at[idx].add(adt).at[row].add(adt)
    return T, g, Q, P, Cc, nb, jnp.sum(adt)


@jax.jit
def _dmax_chunk(rbase, inc, idx, cod, dl, s):
    row = rbase + jnp.cumsum(inc.astype(jnp.int32))
    c = cod.astype(jnp.float64) * 0.125
    blk = (idx != row) & (s[row] * s[idx] * c > 0)
    return jnp.max(jnp.where(blk, jnp.abs(dl[idx] - dl[row]), 0.0))


def pair_kernel(la, u):
    """exact pair quantities of (table la, chi u >= 0 normalised): P, C, L = P + C, T (per-rep x-side terms, sum = L),
    g = dL/dla (floored reps: 0), n_blocked (upper entries).  float64."""
    sec.reload()
    le, fl = la_eff(la)
    lw = le + 0.5 * jnp.log(sec.n)
    au = jnp.abs(u)
    T = jnp.zeros(sec.D, f64); g = jnp.zeros(sec.D, f64); Q = jnp.zeros(sec.D, f64); P = 0.0; Cc = 0.0; nb = 0; Gs = 0.0
    for (rb, inc, idx, cod) in sec.chunks:
        T, g, Q, p_, c_, n_, gs_ = _pair_chunk(T, g, Q, rb, inc, idx, cod, lw, au, sec.sqrt_n, s2)
        P += float(p_); Cc += float(c_); nb += int(n_); Gs += float(gs_)
    g = jnp.where(fl, 0.0, g)
    return dict(P=P, C=Cc, L=P + Cc, T=T, g=g, Q=Q, G=Gs, n_blocked_upper=nb, n_floored=int(jnp.sum(fl)))


def bond_dmax(la_old, la_new):
    sec.reload(); dl = la_new - la_old
    return max(float(_dmax_chunk(rb, inc, idx, cod, dl, s2)) for (rb, inc, idx, cod) in sec.chunks)


def gsolve(la, x0=None, tol=None, maxit=None):
    """sec.fn_solve (same algorithm, same floor, same tolerance convention) plus the records of review C7:
    absolute residual (||x|| = 1), relative residual, sign certificate (s2-frame components < 0 before |.|), local energy per
    decade of u_r^2.  Returns (E, u, info)."""
    tol = FN_TOL if tol is None else tol; maxit = FN_MAXIT if maxit is None else maxit
    t0 = time.time(); nm0 = sec.nmv; s = s2
    la = la - jnp.max(la)
    a = jnp.exp(la); a = a / jnp.sqrt(jnp.sum(sec.n * a * a))
    a = jnp.maximum(a, FLOOR)
    w = a * sec.sqrt_n
    del a, la
    h1, h2 = sec.H2(w, s * w)
    Dg = 0.5 * (h1 + s * h2) / w
    del h1, h2

    def A(x):
        h1, h2 = sec.H2(x, s * x)
        return Dg * x - 0.5 * (h1 - s * h2)

    x = (w if x0 is None else x0); x = x / jnp.linalg.norm(x)
    del w
    Ax = A(x); th = float(x @ Ax)
    den = jnp.maximum(Dg - th, 1.0)
    P = None; AP = None
    it = 0; rn = np.inf
    for it in range(1, maxit + 1):
        r = Ax - th * x
        rn = float(jnp.linalg.norm(r))
        if rn <= tol * abs(th): break
        z = r / den
        z = z - (x @ z) * x
        if P is not None: z = z - (P @ z) * P / (P @ P)
        z = z / jnp.linalg.norm(z)
        Az = A(z)
        Sb = [x, z] + ([P] if P is not None else [])
        ASb = [Ax, Az] + ([AP] if P is not None else [])
        m = len(Sb)
        del r
        GA = np.array([[float(Sb[i] @ ASb[j]) for j in range(m)] for i in range(m)])
        GB = np.array([[float(Sb[i] @ Sb[j]) for j in range(m)] for i in range(m)])
        GA = 0.5 * (GA + GA.T)
        try:
            ev, ec = scipy.linalg.eigh(GA, GB)
        except Exception:
            P = None; AP = None; continue
        c = ec[:, 0]
        xn = sum(c[i] * Sb[i] for i in range(m)); Axn = sum(c[i] * ASb[i] for i in range(m))
        del x, Ax, P, AP
        P = sum(c[i] * Sb[i] for i in range(1, m)); AP = sum(c[i] * ASb[i] for i in range(1, m))
        del Sb, ASb, z, Az
        nr = float(jnp.linalg.norm(xn)); x = xn / nr; Ax = Axn / nr
        pn = float(jnp.linalg.norm(P))
        if pn > 0: P = P / pn; AP = AP / pn
        th = float(x @ Ax)
    sgn = 1.0 if float(jnp.sum(x)) >= 0 else -1.0
    x = x * sgn
    negm = x < 0
    negw = float(jnp.sum(jnp.where(negm, x * x, 0.0))); nneg = int(jnp.sum(negm))
    dneg = np.asarray(DECU(jnp.where(negm, x * x, 1e-300)))[np.asarray(negm)] if nneg else np.zeros(0, int)
    u = jnp.abs(x); u = u / jnp.linalg.norm(u)
    Au = A(u); E = float(u @ Au)
    rabs = float(jnp.linalg.norm(Au - E * u))
    El = jnp.where(u > 0, Au / jnp.where(u > 0, u, 1.0) - E, 0.0)
    du = DECU(u * u); Eln = np.asarray(El); un = np.asarray(u)
    loc = {}
    for d in range(-1, -31, -1):
        m_ = (du == d) & (un > 0)
        if m_.any(): loc[str(d)] = dict(n=int(m_.sum()), max=float(np.max(np.abs(Eln[m_]))), rms=float(np.sqrt(np.mean(Eln[m_] ** 2))))
    m_ = (du < -30) & (un > 0)
    if m_.any(): loc['<-30'] = dict(n=int(m_.sum()), max=float(np.max(np.abs(Eln[m_]))), rms=float(np.sqrt(np.mean(Eln[m_] ** 2))))
    del El, Au
    info = dict(E_FN=E, E_FN_site=E / N, dE_FN_site=(E - sec.E0) / N, it=it, res_rel_lobpcg=rn / abs(th), res_abs_lobpcg=rn,
                res_abs_final=rabs, res_note='absolute = ||H_FN u - E u|| with ||u|| = 1 (E ~ -18.1); relative = absolute / |E|',
                sign_cert=dict(n_neg=nneg, weight_neg=negw, neg_by_decade_of_x2={str(k): int(v) for k, v in zip(*np.unique(dneg, return_counts=True))}),
                local_energy_by_decade_u2=loc, matvecs=sec.nmv - nm0, sec=time.time() - t0)
    return E, u, info


# ============================================================================================ sampler and pair loss
@jax.jit
def _bonds(idx, key, lae, u, wl, wg):
    """reps idx: one blocked bond (s2 only; partner in another orbit) per config x of rep r with probability proportional to
    wl * term_xy + wg * |l'_xy| (term_xy = chi_x^2 |H_xy| |A(y)|/|A(x)|, l'_xy = term_xy - term_yx); also V_r and sum_b |l'_b| config-side."""
    x = sec.reps[idx]; v = valid_bonds(x)
    y = jnp.where(v, x[:, None] ^ MASKS[None, :], x[:, None])
    iy = SS.canon(sec.T, sec.reps, y.reshape(-1)).reshape(x.shape[0], NB)
    blk = v & (s2[idx][:, None] * s2[iy] > 0) & (iy != idx[:, None])
    h = 0.5 * JB[None, :]; t = jnp.clip(lae[iy] - lae[idx][:, None], -300, 300)
    ax = (u[idx] ** 2 / sec.n[idx])[:, None]; by = u[iy] ** 2 / sec.n[iy]
    txy = jnp.where(blk, h * ax * jnp.exp(t), 0.0); tyx = jnp.where(blk, h * by * jnp.exp(-t), 0.0)
    Vc = jnp.sum(jnp.where(blk, h * jnp.exp(t), 0.0), 1); Qc = jnp.sum(jnp.abs(txy - tyx), 1)
    wt = wl * txy + wg * jnp.abs(txy - tyx)
    sel = jax.random.categorical(key, jnp.where(blk & (wt > 0), jnp.log(jnp.maximum(wt, 1e-300)), -jnp.inf), axis=1)
    return x, jnp.take_along_axis(y, sel[:, None], 1)[:, 0], jnp.take_along_axis(iy, sel[:, None], 1)[:, 0], sel, Vc, Qc


class PairSampler:
    """block-start distribution: undirected physical pair e with probability l_e(theta_k)/L(theta_k), over ALL 15.8M orbits (no
    decade band; every orbit with a blocked bond to another orbit has T_r > 0 and can be drawn).  Not cx_c1fit's band sampler."""

    def __init__(self, la, u, kern, seed):
        self.lae, _ = la_eff(la)
        self.u = u; self.L = kern['L']; self.G = kern['G']
        # SAMPLER 'loss': pair probability m_e = l_e/L.  'mix' (default): m_e = (1-MIXG) l_e/L + MIXG |l'_e|/G  (defensive mixture:
        # c_e l_e <= L/(1-MIXG) and c_e |l'_e| <= G/MIXG, so neither the curvature nor the gradient part has unbounded weights).
        self.wl = (1.0 - MIXG) / self.L; self.wg = 0.5 * MIXG / self.G
        R = self.wl * kern['T'] + self.wg * kern['Q']
        self.cdf = jnp.cumsum(R); self.Z = float(self.cdf[-1]); self.cdf = self.cdf / self.cdf[-1]
        self.rng = np.random.default_rng(seed); self.key = jax.random.PRNGKey(seed)

    def draw(self, Mp):
        U = jnp.asarray(self.rng.random(Mp))
        idx = jnp.minimum(jnp.searchsorted(self.cdf, U), sec.D - 1).astype(jnp.int32)
        parts = []
        for i in range(0, Mp, 2048):
            self.key, k = jax.random.split(self.key); parts.append(_bonds(idx[i:i + 2048], k, self.lae, self.u, self.wl, self.wg))
        x, y, iy, b, Vc, Qc = (jnp.concatenate([q[j] for q in parts]) for j in range(6))
        a = self.u[idx] ** 2 / sec.n[idx]; bb = self.u[iy] ** 2 / sec.n[iy]; h = 0.5 * JB[b]
        t0 = self.lae[iy] - self.lae[idx]
        l0, lp0 = pair_terms(t0, a, bb, h)
        m = (self.wl * l0 + 2 * self.wg * jnp.abs(lp0)) / self.Z          # probability of the undirected physical pair
        return dict(idx=idx, iy=iy, x=x, y=y, a=a, b=bb, h=h, c=1.0 / m, t0=t0, Vc=Vc, Qc=Qc)

    def coverage(self, bt):
        """draws with phi^2 (target, per configuration) < 1e-13 (decade <= -14), and with chi*^2 per configuration < 1e-13."""
        cx = self.u[bt['idx']] ** 2 / sec.n[bt['idx']]; cy = self.u[bt['iy']] ** 2 / sec.n[bt['iy']]
        dx = DEC_T[bt['idx']] <= -14; dy = DEC_T[bt['iy']] <= -14
        return dict(n=int(bt['idx'].shape[0]), phi2_x=int(jnp.sum(dx)), phi2_either=int(jnp.sum(dx | dy)),
                    chi2_x=int(jnp.sum(cx < 1e-13)), chi2_either=int(jnp.sum((cx < 1e-13) | (cy < 1e-13))))


def deep_prob(kern, u):
    """exact probability per draw that the sampled x has phi^2 < 1e-13 (target) or chi*^2 < 1e-13; and their share of L, of the orbits."""
    T = kern['T']; L_ = float(jnp.sum(T)); cx = u ** 2 / sec.n
    R = (1.0 - MIXG) * T / L_ + 0.5 * MIXG * kern['Q'] / kern['G']; R_ = float(jnp.sum(R))      # the sampler's rep weights
    mt = DEC_T <= -14; mc = cx < 1e-13
    return dict(sampler=SAMPLER, mix_g=MIXG, p_x_phi2_below_1e13_sampler=float(jnp.sum(jnp.where(mt, R, 0.0))) / R_,
                p_x_chi2_below_1e13_sampler=float(jnp.sum(jnp.where(mc, R, 0.0))) / R_,
                p_x_phi2_below_1e13=float(jnp.sum(jnp.where(mt, T, 0.0))) / L_, p_x_chi2_below_1e13=float(jnp.sum(jnp.where(mc, T, 0.0))) / L_,
                orbit_frac_phi2_below_1e13=float(jnp.mean(mt)), orbit_frac_chi2_below_1e13=float(jnp.mean(mc)),
                orbit_frac_T_zero=float(jnp.mean(T <= 0)))


@jax.jit
def pair_terms(t, a, b, h):
    sa = jnp.sqrt(a) * jnp.exp(0.5 * t); sb = jnp.sqrt(b) * jnp.exp(-0.5 * t)
    l = h * (sa * sa + sb * sb); lp = h * (sa - sb) * (sa + sb)
    return l, lp


def tvals(th, bt):
    S = fwd(th, jnp.concatenate([bt['x'], bt['y']])); Mp = bt['x'].shape[0]
    return S[Mp:] - S[:Mp], S


def Lhat(t, bt):
    l, _ = pair_terms(t, bt['a'], bt['b'], bt['h'])
    return float(jnp.mean(bt['c'] * l))


@jax.jit
def gn_dir(Gx, Gy, t, a, b, h, c, lam_rel, delta):
    Mp = t.shape[0]
    l, lp = pair_terms(t, a, b, h)
    Dg = Gy - Gx
    J = jnp.sqrt(c * l / Mp)[:, None] * Dg
    r = jnp.sqrt(c / Mp) * lp / jnp.sqrt(l)
    Sm = J.T @ J; gr = J.T @ r
    lam = lam_rel * jnp.trace(Sm) / Sm.shape[0]
    d = -jnp.linalg.solve(Sm + lam * jnp.eye(Sm.shape[0], dtype=f64), gr)
    pdt = Dg @ d; pmax = jnp.max(jnp.abs(pdt))
    sc = jnp.minimum(1.0, delta / jnp.maximum(pmax, 1e-30))
    d = d * sc
    return d, gr, sc, pmax, gr @ d, gr @ d + 0.5 * d @ (Sm @ d)


def dL_paired(t_old, t_new, bt):
    """paired per-pair change c_e (l_e(new) - l_e(old)), expm1 form (no cancellation)."""
    sa = jnp.sqrt(bt['a']) * jnp.exp(0.5 * t_old); sb = jnp.sqrt(bt['b']) * jnp.exp(-0.5 * t_old); dt = t_new - t_old
    v = np.asarray(bt['c'] * bt['h'] * (sa * sa * jnp.expm1(dt) + sb * sb * jnp.expm1(-dt)))
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(v.size))


# gauge sample: fixed configurations drawn from PT (target n phi^2)
_rg = np.random.default_rng(777 + SEED)
GIDX = jnp.asarray(np.minimum(np.searchsorted(np.cumsum(np.asarray(PT)) / float(jnp.sum(PT)), _rg.random(M_GAUGE)), sec.D - 1))
GX = sec.reps[GIDX]


def gauge_fix_sampled(th_new, Sg_old):
    Sg_new = fwd(th_new, GX); Dlt = float(jnp.mean(Sg_new - Sg_old))
    return th_new * math.exp(-Dlt / 18.0), Dlt


def gauge_check():
    """A(c theta) = c^18 A(theta): ln|A| shifts by 18 ln c (exact scale direction, review c1)."""
    xs = GX[:16]; c = 1.37
    d = np.asarray(fwd(theta * c, xs) - fwd(theta, xs)) - 18 * math.log(c)
    return float(np.max(np.abs(d)))


res['gauge_scale_check_maxdev'] = gauge_check(); save()
progress(f'[gauge] ln|A(c theta)| - ln|A(theta)| - 18 ln c: max dev {res["gauge_scale_check_maxdev"]:.2e}')

# ============================================================================================ inner step
STAT = dict(lam_rel=LAM0)


def inner_step(th, smp, Sg_old, n):
    ts = time.time()
    bt = smp.draw(M); Xall = jnp.concatenate([bt['x'], bt['y']])
    S = fwd(th, Xall); G = jac(th, Xall)
    t = S[M:] - S[:M]
    d, gr, sc, pmax, pred_lin, pred_quad = gn_dir(G[:M], G[M:], t, bt['a'], bt['b'], bt['h'], bt['c'], STAT['lam_rel'], DELTA)
    del G
    cov = smp.coverage(bt)
    ab = smp.draw(M_ACC)
    ta_old, _ = tvals(th, ab)
    tried = []
    for fr in LS:                                       # backtracking on the acceptance batch: first fraction that lowers Lhat
        cand = th + fr * d; ta_new, _ = tvals(cand, ab)
        dm, dse = dL_paired(ta_old, ta_new, ab)
        amax = float(jnp.max(jnp.abs(ta_new - ta_old))); tried.append(dict(frac=fr, dL=dm, se=dse, act_max_dt=amax))
        if np.isfinite(dm) and dm <= 0 and amax <= 2 * DELTA: break
    lam_used = STAT['lam_rel']
    if not np.isfinite(dm) or dm > 0 or amax > 2 * DELTA:
        ok = False; verdict = 'raises L' if (np.isfinite(dm) and dm > 0) else ('bond cap' if amax > 2 * DELTA else 'nonfinite')
        STAT['lam_rel'] = min(STAT['lam_rel'] * 4, LAM_CAP)
    else:
        ok = True; verdict = 'better (significant)' if dm < -Z_ACC * dse else 'better (point estimate)'
        if dm < -Z_ACC * dse: STAT['lam_rel'] = max(STAT['lam_rel'] / 2, LAM_MIN)
    Dlt = 0.0
    if ok:
        th, Dlt = gauge_fix_sampled(cand, Sg_old)
    rec = dict(step=n, ok=ok, verdict=verdict, lam_rel=lam_used, dL_acc=dm, dL_acc_se=dse, pred_lin=float(pred_lin), pred_quad=float(pred_quad),
               trust_scale=float(sc), pred_max_dt=float(pmax), act_max_dt_acc=amax, gauge_shift=Dlt, coverage_dir_batch=cov, line_search=tried,
               frac=tried[-1]['frac'], sec=time.time() - ts)
    return th, rec


# ============================================================================================ controls
def hf_fd_check(la0, u0, E0, g_hf, g_broken, eps_list, nd=3):
    """(2a) dE_FN/dla along random table directions w: central FD with gsolve (warm from u0) vs g_hf . w; (3) the same FD vs
    g_broken . w (chi* replaced by a random positive vector)."""
    rng = np.random.default_rng(4242)
    le, fl = la_eff(la0); err = np.asarray(le - ls2)
    du = DECU(np.asarray(u0) ** 2)
    W = []
    W.append(rng.standard_normal(sec.D))                                                   # (i) iid Gaussian per rep
    dc = rng.standard_normal(64)
    W.append(dc[-du] + 0.1 * rng.standard_normal(sec.D))                   # (ii) random constant per decade of u^2 + noise
    W.append(rng.standard_normal(sec.D) * err / max(np.std(err), 1e-30))                   # (iii) Gaussian x copy error ln|A| - ln phi
    out = []
    for k, w in enumerate(W[:nd]):
        w = jnp.asarray(w / np.max(np.abs(w)), f64)                                         # max |w| = 1
        an = float(g_hf @ w); br = float(g_broken @ w)
        rec = dict(direction=['iid gaussian', 'per-decade constant + noise', 'gaussian x copy error'][k], hf=an, broken=br, fd={})
        for eps in eps_list:
            Ep, _, ip = gsolve(la0 + eps * w, x0=u0); Em, _, im = gsolve(la0 - eps * w, x0=u0)
            fd = (Ep - Em) / (2 * eps)
            rec['fd'][str(eps)] = dict(fd=fd, rel_hf=abs(an - fd) / max(abs(fd), 1e-300), rel_broken=abs(br - fd) / max(abs(fd), 1e-300),
                                       it=(ip['it'], im['it']), sec=(ip['sec'], im['sec']))
            progress(f'  [2a/3] w{k} eps {eps:.0e}: FD {fd:.10e}  HF {an:.10e} (rel {rec["fd"][str(eps)]["rel_hf"]:.2e})  broken {br:.3e} '
                     f'(rel {rec["fd"][str(eps)]["rel_broken"]:.2e})  solves {ip["it"]},{im["it"]} it {ip["sec"]:.1f}+{im["sec"]:.1f}s')
        out.append(rec)
    return out


def theta_fd_check(th, smp, Mp=512, hs=(1e-5, 1e-6)):
    """(2b) theta-space FD of the sampled loss Lhat on a FIXED pair batch vs the code's GN-row gradient J^T r (Re, Im, mixed)."""
    bt = smp.draw(Mp); X = jnp.concatenate([bt['x'], bt['y']])
    S = fwd(th, X); G = jac(th, X); t = S[Mp:] - S[:Mp]
    _, gr, _, _, _, _ = gn_dir(G[:Mp], G[Mp:], t, bt['a'], bt['b'], bt['h'], bt['c'], 1e-2, 1e9)
    gr = np.asarray(gr)
    rng = np.random.default_rng(99); out = []
    for k, kind in enumerate(['mixed', 'mixed', 'mixed', 'Re only', 'Im only']):
        v = rng.standard_normal(NPAR)
        if kind == 'Re only': v[NP_REAL:] = 0
        if kind == 'Im only': v[:NP_REAL] = 0
        v = v / np.linalg.norm(v); an = float(gr @ v); rec = dict(kind=kind, analytic=an, fd={})
        for h_ in hs:
            tp, _ = tvals(th + h_ * jnp.asarray(v), bt); tm, _ = tvals(th - h_ * jnp.asarray(v), bt)
            fd = (Lhat(tp, bt) - Lhat(tm, bt)) / (2 * h_)
            rec['fd'][str(h_)] = dict(fd=fd, rel=abs(fd - an) / max(abs(fd), 1e-300))
        out.append(rec)
        progress(f'  [2b] {kind}: analytic {an:.8e}  ' + '  '.join(f'h {h_}: FD {rec["fd"][str(h_)]["fd"]:.8e} rel {rec["fd"][str(h_)]["rel"]:.1e}' for h_ in hs))
    return out


def im_only_check(th, smp, Mp=2048):
    """coordinator item 1: the GN step restricted to the Im parts lowers Lhat (fixed batch), and the change matches g.d."""
    bt = smp.draw(Mp); X = jnp.concatenate([bt['x'], bt['y']])
    S = fwd(th, X); G = jac(th, X); t = S[Mp:] - S[:Mp]
    d, gr, _, _, _, _ = gn_dir(G[:Mp], G[Mp:], t, bt['a'], bt['b'], bt['h'], bt['c'], STAT['lam_rel'], DELTA)
    out = {}
    for part, sl in (('Im only', slice(NP_REAL, None)), ('Re only', slice(0, NP_REAL))):
        dd = jnp.zeros(NPAR, f64).at[sl].set(d[sl]); rec = {}
        for s_ in (1.0, 0.1, 0.01):
            tn, _ = tvals(th + s_ * dd, bt)
            rec[str(s_)] = dict(dLhat=Lhat(tn, bt) - Lhat(t, bt), lin=float(s_ * (gr @ dd)))
        out[part] = rec
        progress(f'  [Im check] {part}: ' + '  '.join(f's {k}: dLhat {v["dLhat"]:.3e} (lin {v["lin"]:.3e})' for k, v in rec.items()))
    return out


def sampled_vs_exact(smp, g_hf, Mp=65536):
    """(2c) table space: sampled gradient (1/M) sum c l' (w_y - w_x) at the block start vs exact g . w (z-score), 3 directions;
    and V_r of the CSR kernel (T_r/u_r^2) vs the config-space sum over blocked bonds at the sampled reps."""
    bt = smp.draw(Mp); _, lp = pair_terms(bt['t0'], bt['a'], bt['b'], bt['h'])
    rng = np.random.default_rng(4243); out = []
    for k in range(3):
        w = jnp.asarray(rng.standard_normal(sec.D), f64)
        v = np.asarray(bt['c'] * lp * (w[bt['iy']] - w[bt['idx']]))
        ex = float(g_hf @ w); out.append(dict(exact=ex, sampled=float(v.mean()), se=float(v.std(ddof=1) / np.sqrt(Mp)),
                                             z=float((v.mean() - ex) / (v.std(ddof=1) / np.sqrt(Mp)))))
    Vk = np.asarray(smp_T[bt['idx']] / smp.u[bt['idx']] ** 2); Vc = np.asarray(bt['Vc'])
    vrel = float(np.max(np.abs(Vk - Vc) / np.maximum(Vc, 1e-300)))
    Qk = np.asarray(kern['Q'][bt['idx']] / sec.n[bt['idx']]); Qc = np.asarray(bt['Qc'])
    qrel = float(np.max(np.abs(Qk - Qc) / np.maximum(Qc, 1e-300)))
    return out, max(vrel, qrel)


# ============================================================================================ probe2: optimiser check
if MODE == 'probe2':
    la = jnp.asarray(np.load(SPEC['ref_table']), jnp.float64)
    E, u, inf = gsolve(la); res['solve_A0'] = {q: inf[q] for q in inf if q != 'local_energy_by_decade_u2'}
    progress(f'[probe2] E_FN[A_0] {inf["dE_FN_site"]:.12e}  it {inf["it"]} {inf["sec"]:.1f}s  res_abs {inf["res_abs_final"]:.1e}')
    kern = pair_kernel(la, u); res['deep_draw_prob0'] = deep_prob(kern, u)
    progress(f'[probe2] P {kern["P"]:.6e} L {kern["L"]:.6e} G {kern["G"]:.6e}  deep draw prob {res["deep_draw_prob0"]}')
    smp = PairSampler(la, u, kern, 10 + SEED); th = theta; Sg0 = fwd(th, GX); hist = []
    for n in range(1, int(SPEC.get('n_probe_steps', 20)) + 1):
        th, rec = inner_step(th, smp, Sg0, n); rec['sig2'] = bool(rec['ok'] and rec['dL_acc'] < -2 * rec['dL_acc_se']); hist.append(rec)
        if rec['ok']: Sg0 = fwd(th, GX)
        progress(f'  [inner {n}] {rec["verdict"]}  frac {rec["frac"]}  dL_acc {rec["dL_acc"]:.3e} +- {rec["dL_acc_se"]:.1e}  pred {rec["pred_lin"]:.3e}  '
                 f'max dt pred {rec["pred_max_dt"]:.3f} act {rec["act_max_dt_acc"]:.3f}  lam {rec["lam_rel"]:.1e}  {rec["sec"]:.2f}s  deep {rec["coverage_dir_batch"]}')
        res['probe_steps'] = hist; save()
        if tleft() < 60: break
    bt = smp.draw(int(SPEC.get('B_mm', 65536)))
    t_0, _ = tvals(theta, bt); t_n, _ = tvals(th, bt); dm, dse = dL_paired(t_0, t_n, bt)
    nsig = int(sum(h['sig2'] for h in hist))
    res['probe2'] = dict(steps=len(hist), accepted=int(sum(h['ok'] for h in hist)), sig2_lower=nsig, frac_sig2=nsig / max(len(hist), 1),
                         MM_bound_dE_total=dm, MM_bound_se=dse, MM_bound_dE_site=dm / N, MM_bound_se_site=dse / N, B_mm=int(bt['x'].shape[0]),
                         pass_70pct=bool(nsig >= 0.7 * len(hist)), pass_bound=bool(dm < -2 * dse),
                         sec_per_step_median=float(np.median([h['sec'] for h in hist])))
    np.save(os.path.join(OUT, 'theta_probe2.npy'), np.asarray(th))
    res['sec'] = time.time() - T00; res['done'] = True; save()
    progress(f'[probe2] steps {len(hist)}, accepted {res["probe2"]["accepted"]}, lower by > 2 sigma {nsig} ({res["probe2"]["frac_sig2"]:.2f}); '
             f'E_FN[A_n] - E_FN[A_0] <= {dm / N:.4e} +- {dse / N:.1e} per site (paired, {bt["x"].shape[0]} pairs)')
    progress(f'DONE probe2 {res["sec"]:.0f}s')
    sys.exit(0)

# ============================================================================================ step 0
res['controls'] = {}
REF_TABLE = SPEC.get('ref_table')
if SPEC.get('test_table'):            # CPU PLUMBING TEST ONLY (gate/prod paths): table = stored c1 table + network change on 256 orbits
    _ref = jnp.asarray(np.load(REF_TABLE), jnp.float64); _sub = jnp.asarray(np.sort(np.random.default_rng(6).choice(sec.D, 256, replace=False)))
    _th0 = theta; _f0 = fwd(theta, sec.reps[_sub])
    def table(th, idx=None):
        return _ref.at[_sub].add(fwd(th, sec.reps[_sub]) - _f0)
    res['TEST_TABLE'] = 'plumbing test: NOT a result'
if MODE in ('prod', 'gate'):
    t0 = time.time(); la = table(theta); jax.block_until_ready(la); res['table_sec_step0'] = time.time() - t0
    progress(f'[table] full {res["table_sec_step0"]:.0f}s')
    if REF_TABLE:
        lr = jnp.asarray(np.load(REF_TABLE), jnp.float64); dlt = np.asarray(norm_la(la) - norm_la(lr))
        res['controls']['table_vs_c1'] = dict(max_abs=float(np.max(np.abs(dlt)))); del lr, dlt
else:
    la = jnp.asarray(np.load(REF_TABLE), jnp.float64)
    nsub = int(SPEC.get('probe_table_n', 1 << 20)) if MODE == 'probe' else int(SPEC.get('dry_table_n', 512))
    sub = np.sort(np.random.default_rng(5).choice(sec.D, nsub, replace=False))
    t0 = time.time(); ls = table(theta, sub); jax.block_until_ready(ls); tsub = time.time() - t0
    dd = np.asarray(ls - la[jnp.asarray(sub)]); dd = dd - dd.mean()
    res['controls']['table_vs_c1_subset'] = dict(n=nsub, max_abs=float(np.max(np.abs(dd))), sec=tsub, est_full_table_sec=tsub * sec.D / nsub)
    progress(f'[table] subset {nsub}: {tsub:.1f}s -> full table est {tsub * sec.D / nsub:.0f}s; max |la - c1 la_table| {np.max(np.abs(dd)):.2e}')
save()
if MODE == 'dry': FN_MAXIT = int(SPEC.get('dry_fn_maxit', 3))
Et, ut, it_ = gsolve(ls2); del ut
res['target'] = dict(dE_FN_site=it_['dE_FN_site'], split_value=REF_E_TARGET_SPLIT, rel_vs_split=abs(it_['dE_FN_site'] / REF_E_TARGET_SPLIT - 1), info=it_)
progress(f'[target] dE_FN {it_["dE_FN_site"]:.12e} (split {REF_E_TARGET_SPLIT:.12e})  it {it_["it"]} {it_["sec"]:.1f}s  res_abs {it_["res_abs_final"]:.1e}')
E, u, inf = gsolve(la)
tax0 = inf['dE_FN_site'] - it_['dE_FN_site']
c1 = dict(tax0=tax0, ref_tax0=REF_TAX0, rel=(abs(tax0 / REF_TAX0 - 1) if REF_TAX0 else None), E_copy=inf['dE_FN_site'], solve=inf)
c1['ok'] = bool(c1['rel'] is not None and c1['rel'] <= 1e-9)
res['controls']['c1_tax0'] = c1; save()
progress(f'[control 1] tax_0 {tax0:.12e} vs {REF_TAX0} rel {c1["rel"]}  -> {"PASS" if c1["ok"] else "FAIL"}  (solve it {inf["it"]}, {inf["sec"]:.1f}s, '
         f'res_abs {inf["res_abs_final"]:.1e}, sign violations {inf["sign_cert"]["n_neg"]})')
if MODE in ('prod', 'gate') and not c1['ok'] and not SPEC.get('test_table'):
    res['stop'] = 'CONTROL 1 FAILED'; res['done'] = True; save(); progress('STOP control 1 failed'); sys.exit(2)

t0 = time.time(); kern = pair_kernel(la, u); res['kernel_sec'] = time.time() - t0
smp_T = kern['T']
sg = float(jnp.sum(kern['g'])); sga = float(jnp.sum(jnp.abs(kern['g'])))
res['deep_draw_prob0'] = deep_prob(kern, u)
progress(f'[sampler] exact per-draw probabilities {res["deep_draw_prob0"]}')
res['kernel0'] = dict(P=kern['P'], C=kern['C'], L=kern['L'], sum_g=sg, sum_abs_g=sga, n_blocked_upper=kern['n_blocked_upper'], n_floored=kern['n_floored'])
progress(f'[kernel] P {kern["P"]:.12e}  L {kern["L"]:.10e}  sum g {sg:.2e} (sum|g| {sga:.2e})  blocked {kern["n_blocked_upper"]}  floored {kern["n_floored"]}  {res["kernel_sec"]:.1f}s')
save()

# ============================================================================================ probe / dry: controls 2, 3 and rates
if MODE in ('probe', 'dry'):
    smp = PairSampler(la, u, kern, 10 + SEED)
    c2c, vrel = sampled_vs_exact(smp, kern['g'], Mp=int(SPEC.get('c2c_pairs', 65536 if MODE == 'probe' else 2048)))
    res['controls']['c2c_sampled_vs_exact'] = c2c; res['controls']['V_kernel_vs_config_maxrel'] = vrel
    progress(f'[2c] z-scores {[round(q["z"], 2) for q in c2c]}  V kernel vs config max rel {vrel:.2e}')
    res['controls']['c2b_theta_fd'] = theta_fd_check(theta, smp, Mp=int(SPEC.get('c2b_pairs', 512 if MODE == 'probe' else 16)))
    save()
    if MODE == 'probe':
        rng_b = np.random.default_rng(31337); ub = jnp.asarray(rng_b.random(sec.D) + 1e-3, f64); ub = ub / jnp.linalg.norm(ub)
        gb = pair_kernel(la, ub)['g']
        res['controls']['c2a_c3_hf_fd'] = hf_fd_check(la, u, E, kern['g'], gb, [float(e) for e in SPEC.get('fd_eps', [1e-3, 1e-4, 1e-2])])
        del gb; save()
    res['controls']['im_only'] = im_only_check(theta, smp, Mp=2048 if MODE == 'probe' else 16)
    save()
    if MODE == 'dry': M = M_ACC = int(SPEC.get('dry_pairs', 32))
    th = theta; Sg0 = fwd(th, GX); hist = []
    for n in range(1, int(SPEC.get('n_probe_steps', 6)) + 1):
        th, rec = inner_step(th, smp, Sg0, n); hist.append(rec)
        if rec['ok']: Sg0 = fwd(th, GX)
        progress(f'  [inner {n}] {rec["verdict"]}  dL_acc {rec["dL_acc"]:.3e} +- {rec["dL_acc_se"]:.1e}  pred {rec["pred_lin"]:.3e}  '
                 f'max dt pred {rec["pred_max_dt"]:.3f} act {rec["act_max_dt_acc"]:.3f}  lam {rec["lam_rel"]:.1e}  {rec["sec"]:.2f}s  deep draws {rec["coverage_dir_batch"]}')
    res['probe_steps'] = hist
    sp_ = [h['sec'] for h in hist]
    res['rates'] = dict(sec_per_inner_step_median_excl_first=float(np.median(sp_[1:])) if len(sp_) > 1 else None, sec_per_inner_step_first=sp_[0],
                        B_pairs=M, B_acc=M_ACC, sec_solve_cold_copy=inf['sec'], sec_solve_cold_target=it_['sec'],
                        sec_solve_warm=[r['fd'][e]['sec'] for r in res['controls'].get('c2a_c3_hf_fd', []) for e in r['fd']],
                        sec_kernel=res['kernel_sec'], est_full_table_sec=res['controls']['table_vs_c1_subset']['est_full_table_sec'])
    c2a = res['controls'].get('c2a_c3_hf_fd', [])
    if c2a:
        e_dec = str(float(SPEC.get('fd_eps', [1e-3])[0]))
        res['controls']['c2_pass'] = bool(all(r['fd'][e_dec]['rel_hf'] <= 1e-3 for r in c2a)
                                          and all(min(q['rel'] for q in r['fd'].values()) <= 1e-3 for r in res['controls']['c2b_theta_fd']))
        res['controls']['c3_broken_fails'] = bool(all(r['fd'][e_dec]['rel_broken'] > 1e-3 for r in c2a))
    res['sec'] = time.time() - T00; res['done'] = True; save()
    progress(f'DONE {MODE} {res["sec"]:.0f}s  rates {res["rates"]}  c1 {c1["ok"]} c2 {res["controls"].get("c2_pass")} c3 {res["controls"].get("c3_broken_fails")}')
    sys.exit(0)

# ============================================================================================ gate: one frozen-chi* block
if MODE == 'gate':
    rd = lambda i: {q: i[q] for q in i if q != 'local_energy_by_decade_u2'}
    la_chi = jnp.log(jnp.maximum(u, 1e-300) / sec.sqrt_n)                  # chi_0 as guide amplitude (per configuration), sign s2
    Ec, uc, ic = gsolve(la_chi, x0=u); del uc
    g = dict(E_FN_A0=inf['dE_FN_site'], E_FN_chi0=ic['dE_FN_site'], solve_A0=rd(inf), solve_chi0=rd(ic),
             local_energy_A0=inf['local_energy_by_decade_u2'], local_energy_chi0=ic['local_energy_by_decade_u2'])
    g['Delta_0'] = g['E_FN_A0'] - g['E_FN_chi0']
    res['gate'] = g; save()
    progress(f'[gate] E_FN[A_0] {g["E_FN_A0"]:.12e}  E_FN[chi_0] {g["E_FN_chi0"]:.12e}  Delta_0 {g["Delta_0"]:.6e}  (it {ic["it"]}, {ic["sec"]:.1f}s, res_abs {ic["res_abs_final"]:.1e})')
    smp = PairSampler(la, u, kern, 1000 + SEED)
    th0 = theta; P_old = kern['P']; Sg = fwd(theta, GX); recs = []
    t_table = float(res['table_sec_step0']); t_solve = float(inf['sec'])
    for n in range(1, K_INNER + 1):
        theta, rec = inner_step(theta, smp, Sg, n); recs.append(rec)
        if rec['ok']: Sg = fwd(theta, GX)
        if n % 10 == 0 or n == 1:
            progress(f'  [inner {n}] {rec["verdict"]}  dL_acc {rec["dL_acc"]:.3e} +- {rec["dL_acc_se"]:.1e}  pred {rec["pred_lin"]:.3e}  lam {rec["lam_rel"]:.1e}  '
                     f'{rec["sec"]:.2f}s  deep draws {rec["coverage_dir_batch"]}')
            res['inner'] = recs; save()
        need = t_table + 2 * t_solve + float(res['kernel_sec']) + 180 + 2 * rec['sec']
        if tleft() < need: g['inner_stop'] = f'time: {tleft():.0f}s left < {need:.0f}s after {n} steps'; break
    g['inner_steps'] = len(recs); g['inner_accepted'] = int(sum(r['ok'] for r in recs))
    g['sec_per_inner_step_median'] = float(np.median([r['sec'] for r in recs]))
    g['coverage_total'] = dict((q, int(sum(r['coverage_dir_batch'][q] for r in recs))) for q in recs[0]['coverage_dir_batch'])
    res['inner'] = recs; save()
    t0 = time.time(); la1 = table(theta); jax.block_until_ready(la1); g['table_sec_A1'] = time.time() - t0
    kn = pair_kernel(la1, u)
    g['P_A0_chi0'] = P_old; g['P_A1_chi0'] = kn['P']; g['MM_decrease'] = bool(kn['P'] < P_old)
    g['MM_bound_change_site'] = (kn['P'] - P_old) / N; g['bond_dmax_block'] = bond_dmax(la, la1)
    Dlt = float(jnp.sum(PT * la1) - jnp.sum(PT * la)); theta = theta * math.exp(-Dlt / 18.0); la1 = la1 - Dlt; g['gauge_shift'] = Dlt
    np.save(os.path.join(OUT, 'theta.npy'), np.asarray(theta)); np.save(os.path.join(OUT, 'la_table.npy'), np.asarray(la1, np.float64)); save()
    E1, u1, i1 = gsolve(la1, x0=u)
    g['E_FN_A1'] = i1['dE_FN_site']; g['solve_A1'] = rd(i1); g['local_energy_A1'] = i1['local_energy_by_decade_u2']
    g['tau_0'] = g['E_FN_A1'] - g['E_FN_chi0']; g['tax0_A0'] = tax0; g['tax0_A1'] = g['E_FN_A1'] - it_['dE_FN_site']
    g['E_FN_target'] = it_['dE_FN_site']; g['tau0_over_Delta0'] = g['tau_0'] / g['Delta_0']
    g['monotone'] = bool(g['E_FN_A1'] <= g['E_FN_A0'])
    res['sec'] = time.time() - T00; res['done'] = True; save()
    progress(f'[gate] E_FN[A_1] {g["E_FN_A1"]:.12e}  tau_0 {g["tau_0"]:.6e}  Delta_0 {g["Delta_0"]:.6e}  tau_0/Delta_0 {g["tau0_over_Delta0"]:.4f}  '
             f'MM decrease {g["MM_decrease"]} (P {P_old:.6e} -> {kn["P"]:.6e})  steps {len(recs)} ({g["inner_accepted"]} accepted)')
    progress(f'DONE gate {res["sec"]:.0f}s')
    sys.exit(0)

# ============================================================================================ production loop
G0 = float(jnp.sum(PT * la))                                           # gauge value at start (sum n phi^2 ln|A|)
outer = [dict(k=0, dE_FN_site=inf['dE_FN_site'], tax0=tax0, P=kern['P'], L=kern['L'], solve={q: inf[q] for q in inf if q != 'local_energy_by_decade_u2'},
              local_energy=inf['local_energy_by_decade_u2'], t=time.time() - T00)]
# plain iteration guide <- |chi*| (no network), same start, one solve per re-solve of the main loop
la_pl = la; u_pl = u; plain = [dict(j=0, dE_FN_site=inf['dE_FN_site'])]
res['outer'] = outer; res['plain_iteration'] = plain; res['inner'] = []; save()
np.save(os.path.join(OUT, 'theta_k0.npy'), np.asarray(theta))
cost = dict(table=res['table_sec_step0'], solve=inf['sec'], kernel=res['kernel_sec'], step=10.0)
flat_run = 0; stop = None; k = 0
while True:
    k += 1
    need = K_INNER * cost['step'] + cost['table'] + 2 * cost['solve'] + 2 * cost['kernel'] + 60
    if tleft() < need: stop = f'time cap: {tleft():.0f}s left < block cost {need:.0f}s'; break
    if k > MAX_OUTER: stop = 'max_outer'; break
    smp = PairSampler(la, u, kern, 1000 * k + SEED)
    th0 = theta; P_old = kern['P']; Sg = fwd(theta, GX); recs = []; tb = time.time()
    for n in range(1, K_INNER + 1):
        theta, rec = inner_step(theta, smp, Sg, n); rec['outer'] = k; recs.append(rec)
        if rec['ok']: Sg = fwd(theta, GX)
    cost['step'] = float(np.median([r['sec'] for r in recs]))
    res['inner'] += recs
    # exact block check: P(theta_new; u_k) < P(theta_k; u_k), with backtracking on the parameter change
    bt_n = 0; accepted = False
    while True:
        t0 = time.time(); la_new = table(theta); jax.block_until_ready(la_new); cost['table'] = time.time() - t0
        kn = pair_kernel(la_new, u)
        if kn['P'] < P_old: accepted = True; break
        if bt_n >= MAX_BT or tleft() < cost['table'] + 2 * cost['solve'] + 120: break
        bt_n += 1; theta = th0 + 0.5 * (theta - th0); progress(f'  [outer {k}] block raised P ({kn["P"]:.6e} >= {P_old:.6e}): backtrack {bt_n}')
    dmax = bond_dmax(la, la_new)
    if not accepted:
        theta = th0; STAT['lam_rel'] = min(STAT['lam_rel'] * 4, LAM_CAP)
        outer.append(dict(k=k, rejected=True, P_old=P_old, P_new=kn['P'], backtracks=bt_n, bond_dmax=dmax, t=time.time() - T00)); save()
        progress(f'[outer {k}] block REJECTED (P {kn["P"]:.6e} >= {P_old:.6e}); theta reverted'); continue
    # exact gauge fix (pure gauge: rescale theta, shift table)
    Dlt = float(jnp.sum(PT * la_new)) - G0
    theta = theta * math.exp(-Dlt / 18.0); la = la_new - Dlt; del la_new
    E_prev = outer[-1]['dE_FN_site'] if not outer[-1].get('rejected') else [o for o in outer if not o.get('rejected')][-1]['dE_FN_site']
    E, u, inf = gsolve(la, x0=u); cost['solve'] = inf['sec']
    t0 = time.time(); kern = pair_kernel(la, u); cost['kernel'] = time.time() - t0
    tax0 = inf['dE_FN_site'] - it_['dE_FN_site']
    acc = [r['ok'] for r in recs]
    rec = dict(k=k, dE_FN_site=inf['dE_FN_site'], tax0=tax0, dE_change_site=inf['dE_FN_site'] - E_prev, P_old_block=P_old, P_new_block=kn['P'],
               mm_bound_site=(kn['P'] - P_old) / N, backtracks=bt_n, bond_dmax_block=dmax, gauge_shift=Dlt, acc_rate=float(np.mean(acc)),
               P=kern['P'], L=kern['L'], sum_g=float(jnp.sum(kern['g'])), n_floored=kern['n_floored'], deep_draw_prob=deep_prob(kern, u),
               coverage_block=dict((q, int(sum(r_['coverage_dir_batch'][q] for r_ in recs))) for q in recs[0]['coverage_dir_batch']),
               solve={q: inf[q] for q in inf if q != 'local_energy_by_decade_u2'}, local_energy=inf['local_energy_by_decade_u2'],
               block_sec=time.time() - tb, t=time.time() - T00)
    outer.append(rec)
    # plain iteration, one solve
    la_pl = jnp.log(jnp.maximum(u_pl, 1e-300) / sec.sqrt_n); Epl, u_pl, ipl = gsolve(la_pl, x0=u_pl)
    plain.append(dict(j=len(plain), dE_FN_site=ipl['dE_FN_site'], it=ipl['it'], res_abs=ipl['res_abs_final'], sign_neg=ipl['sign_cert']['n_neg']))
    np.save(os.path.join(OUT, f'theta_k{k}.npy'), np.asarray(theta)); np.save(os.path.join(OUT, 'theta.npy'), np.asarray(theta))
    save()
    progress(f'[outer {k}] E_FN {inf["dE_FN_site"]:.10e}  tax_0 {tax0:.6e}  change {rec["dE_change_site"]:+.2e}/site  MM bound {rec["mm_bound_site"]:+.2e}  '
             f'acc {rec["acc_rate"]:.2f}  bond dmax {dmax:.3f}  res_abs {inf["res_abs_final"]:.1e}  sign neg {inf["sign_cert"]["n_neg"]}  '
             f'plain {ipl["dE_FN_site"]:.10e}  block {rec["block_sec"]:.0f}s')
    flat_run = flat_run + 1 if abs(rec['dE_change_site']) < FLAT_TOL else 0
    if flat_run >= FLAT_N: stop = f'E_FN flat (< {FLAT_TOL}/site) over {FLAT_N} re-solves'; break


def rate_of(seq):
    e = np.array(seq); d = np.diff(e)
    r = [d[i + 1] / d[i] for i in range(len(d) - 1) if d[i] != 0]
    return dict(ratios=r, median=float(np.median(r)) if r else None)


good = [o for o in outer if not o.get('rejected')]
res['summary'] = dict(stop_reason=stop, n_resolves=len(good) - 1, tax0_start=good[0]['tax0'], tax0_end=good[-1]['tax0'],
                      drop_rel=1 - good[-1]['tax0'] / good[0]['tax0'], E_FN_start=good[0]['dE_FN_site'], E_FN_end=good[-1]['dE_FN_site'],
                      monotone=bool(all(np.diff([o['dE_FN_site'] for o in good]) <= 1e-14)),
                      plain_E_end=plain[-1]['dE_FN_site'], rate_main=rate_of([o['dE_FN_site'] for o in good]),
                      rate_plain=rate_of([p['dE_FN_site'] for p in plain]), costs=cost)
np.save(os.path.join(OUT, 'la_table.npy'), np.asarray(la, np.float64)); np.save(os.path.join(OUT, 'theta.npy'), np.asarray(theta))
res['sec'] = time.time() - T00; res['done'] = True; save()
progress(f'[g1] stop: {stop}; tax_0 {good[0]["tax0"]:.6e} -> {good[-1]["tax0"]:.6e} (drop {res["summary"]["drop_rel"]:.3f}); plain {plain[-1]["dE_FN_site"]:.6e}')
progress(f'DONE {res["sec"]:.0f}s')
