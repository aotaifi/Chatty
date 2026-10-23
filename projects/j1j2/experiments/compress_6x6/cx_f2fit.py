"""Test F2 fit worker = a COPY of cx_f1fit.py (that file is unchanged) with ONE extra loss term (README "Test F2"):
  lambda_v * L_v,   L_v = sum_{violating bonds (x,y)} w_xy (log r_net(x,y) - log r_phi(x,y))^2 / sum w_xy,   w_xy = phi(x) phi(y) |H_xy| n_x,
  log r(x,y) = log A(y) - log A(x) (A the network, phi = exp(target_la)); a bond is VIOLATING iff it is a valid (antiparallel) bond and
  s_2(x) s_2(y) > 0 (H_xy = +JB/2 > 0): exactly the terms sec.fn_solve moves to the diagonal (Dg = (H1 + s H2)/2 w, st6_sector.py:241) and
  the bonds bonds_of / bond_chunk / cx_lib.VW already use.  Bonds are SAMPLED from the whole target table in proportion to w_xy (cdf over
  the training orbits from one pass over all orbits; held-out orbits give the held-out bond set), so the sample mean of d^2 is L_v;
  the offset c cancels in a ratio.  Rows of the damped Gauss-Newton step: sqrt(lambda_v / M_v) (G_y - G_x) with residual sqrt(lambda_v / M_v) d.
  The acceptance test adds lambda_v (L_v(new) - L_v(old)) on a fresh bond batch (paired; standard errors added in quadrature).
  The convergence rule is unchanged but its held-out loss L_ho is the TOTAL objective (old + lambda_v L_v); the components are logged
  (curve: L_ho_old, L_v_ho).  Spec keys: lambda_v (number; 0 = F1 exactly; "auto" = L_ho_old / L_v_ho at step 0), B_v (bonds per direction
  batch, 4096), B_v_acc (acceptance, = B_v), B_v_ho (held-out bonds, 8192), lv_check (brute-force controls, no fit), lv_check_reps.
  With lambda_v = 0 the random streams and the step are those of cx_f1fit.py (separate numpy/jax streams for the bond batches).
Original docstring of cx_f1fit.py follows.
Test F1 fit worker = a COPY of cx_t1s.py (that file is unchanged) with two ppbf options (README "Test F1"):
  bf_window (3 default = cx_t1s exactly | 5): backflow u1, u2 over a bf_window x bf_window window around i and j; a 3x3 warm_theta
      (f + u of 3x3, 1944 values) is embedded in the 5x5 layout (zeros outside the 3x3);
  freeze_outer (False): only f and the 3x3 slots of u are trained (the outer ring of a 5x5 window stays at its start value);
      the GN step is solved on the active columns only, so with window 5 + freeze_outer it is the window-3 step (control 2);
  max_steps (exists in cx_t1s): step cap; rate_only: stop after the fit (theta.npy, no table/readout/E_FN), to time a step.
The 'ppbf' u layout is (2, window^2, 36); the dy-outer, dx-inner order of cx_lib.NBR is kept for any window.
Original docstring of cx_t1s.py follows.
Test T1s (README "Test T1s", pre-registered 2026-10-21): decade-stratified supervised fit of log phi*(s_2).
  python cx_t1s.py OUT SPEC.json
One arm per job.  SPEC: arm in {'vit', 'vit_nostrat' (control 1), 'pp', 'ppbf'}, refresh (run dir with la_star_s2.npy,
s2.npy, vit/cx_reanchor.json), pp_params, pp_table, seed, fit_sec, and the optional knobs read below.

Target  log phi*(s_2) (refresh/la_star_s2.npy); the sign s_2 is fixed and not fitted.  Only log|A| is fitted.
Sampler decade-stratified: decades d = -5..-13 of |phi*|^2 per configuration (Lab.decade_of), ceil(B/9) configurations
        per decade per batch, uniform over configurations inside a decade (orbit reps drawn with weight n_r), training
        orbits only (Lab.HO = the 20% held-out orbits are never trained on).  'vit_nostrat' (decade weights off) draws
        the same B from |phi*|^2 over all training orbits and does not renormalise bond weights per decade.
Loss    L = mean_x L_x,  L_x = e_x^2 + (lam_b W_x / Wbar_d(x)) (1/K) sum_k (e_x - e_{y_k})^2,  e = log A - log phi* - c
        (c a fitted global offset).  Bonds: the bad bonds of s_2 at x (valid bonds with s_2(x) s_2(y) > 0, the
        FN-violating ones), weight w_xy = |H_xy| phi*_y/phi*_x, W_x = sum w_xy; K bonds per x drawn with probability
        w_xy/W_x; Wbar_d = mean of W_x over decade d (estimated once from a stratified sample) => per-decade renormalised.
        Equal counts per decade => every decade enters L with weight 1/9.
Optimiser damped Gauss-Newton (minSR in sample space when rows <= params, the same LM step in parameter space
        otherwise), lam = lam_rel tr(J^T J) / min(rows, params), lam_rel cap LAM_CAP; trust radius: the predicted rms
        change of log A on the batch is capped at TAU.
Acceptance: each step a FRESH stratified acceptance batch (B_acc configurations + K bonds each, independent of the
        direction batch); paired per-configuration difference Delta_x = L_x(theta + d) - L_x(theta); stratified mean and
        standard error.  Rejected only if significantly worse (mean > Z_ACC * SE): lam_rel x4 (cap).  Significantly
        better (mean < -Z_ACC * SE): accepted, lam_rel / 2.  Within noise: accepted, lam_rel unchanged (sample noise never
        drives the damping to its cap).
Held-out: one FIXED stratified batch from the held-out orbits (B_ho configurations), evaluated every HO_EVERY steps.
Converged (pre-registered): over the last 20% of steps acceptance >= 50%, lam_rel below its cap at every step, and the
        held-out loss flat (|L_ho(start of window) - L_ho(end)| / L_ho(end) < FLAT_TOL).  Checked at each held-out
        evaluation once n >= MIN_STEPS; the fit stops when converged or when fit_sec is spent.
Readings: the full-sector table of the fitted network; per-decade rms log error (count-weighted = n_r-weighted inside
        the decade, all orbits and held-out orbits, offset = equal-decade mean over -5..-13; and the T0 convention:
        |phi*|^2-weighted offset and weights, eps = rms 10^(d/2)); bond error per decade (cx_t0 bond_chunk);
        E_FN(s_2, fitted ratios), one sec.fn_solve (as cx_t0).
Controls: (1) arm 'vit_nostrat' (its decade profile is recorded next to the refresh fit's profile);
        (2) every arm: the readout code applied to the unchanged warm-start table (psi_P: lab.lP0; PP: sym_tables_PP)
        next to cx_t0.json's eps (vit 5.72e-7, pp 4.47e-7), and the warm network vs the warm table on held-out samples.
Models: 'vit' the symmetrised ViT of cx_reanchor (fp32, warm VIT_CKPT = psi_P);
        'pp'  pair product psi_PP(z) = eps(z) det F[U, D], F = f (36 x 36, the trained f of rpT1_pp, no RBM), projected
              on the whole space group x spin flip (576 images, as rbmpp_table.py / sym_tables_PP), float64;
        'ppbf' F_ij(z) = f_ij + B_ij(z), B_ij(z) = sum_{d in 3x3} [u1(d, r_j - r_i) z_{i+d} + u2(d, r_i - r_j) z_{j+d}]
              (local backflow, 3x3 receptive field around i and around j, translation invariant, u1 = u2 = 0 at start).
dry=True: CPU plumbing test (small batches, a few steps; the 'table' is the warm table with the network spliced in at
        a sample of orbits; bond readout on a few chunks; fn_solve only with dry_fn_maxit > 0).  NOT a result.
"""
import os, sys, time, math
from functools import partial
import numpy as np
from cx_lib import *

OUT = sys.argv[1]; SPEC = json.load(open(sys.argv[2])); os.makedirs(OUT, exist_ok=True)
F = os.path.join(OUT, 'cx_t1s.json'); PROG = os.path.join(OUT, 'progress.txt'); T00 = time.time()
ARM = SPEC['arm']; assert ARM in ('vit', 'vit_nostrat', 'pp', 'ppbf'), ARM
STRAT = ARM != 'vit_nostrat'; MODEL = 'vit' if ARM.startswith('vit') else ARM
DRY = bool(SPEC.get('dry', False)); SEED = int(SPEC.get('seed', 0))
DECS = list(range(-5, -14, -1)); ND = len(DECS)
B = int(SPEC.get('B', 2048)); K = int(SPEC.get('K', 2))
B_ACC = int(SPEC.get('B_acc', B)); B_HO = int(SPEC.get('B_ho', 4096))
LAM_B = float(SPEC.get('lam_b', 1.0)); FIT_SEC = float(SPEC.get('fit_sec', 1800)); MAX_STEPS = int(SPEC.get('max_steps', 10 ** 9))
LAM0 = float(SPEC.get('lam0', 1e-2)); LAM_CAP = float(SPEC.get('lam_cap', 1.0)); LAM_MIN = 1e-6
TAU = float(SPEC.get('tau', 0.05)); Z_ACC = float(SPEC.get('z_acc', 2.0))
# T1s-2 (README "Test T1s-2"): tau anneal TAU -> TAU_MIN, halved each time the held-out loss is flat (< FLAT_TOL) over
# ANNEAL_STEPS steps (and >= ANNEAL_STEPS steps since the last change); convergence additionally needs tau == TAU_MIN.
TAU_MIN = float(SPEC.get('tau_min', TAU)); ANNEAL_STEPS = int(SPEC.get('anneal_steps', 20))
WARM_THETA = SPEC.get('warm_theta')                   # T1s-2: start from a previous run's theta.npy
SPLICE_DEC = int(SPEC.get('splice_dec', -9))          # T1s-2: splice E_FN, fitted ratios for decades >= SPLICE_DEC
HO_EVERY = int(SPEC.get('ho_every', 10)); MIN_STEPS = int(SPEC.get('min_steps', 50)); FLAT_TOL = float(SPEC.get('flat_tol', 0.02))
BFW = int(SPEC.get('bf_window', 3)); FREEZE = bool(SPEC.get('freeze_outer', False)); RATE_ONLY = bool(SPEC.get('rate_only', False))
assert BFW in (3, 5), BFW
LAMV_SPEC = SPEC.get('lambda_v', 0.0)                 # F2: number or 'auto'
B_V = int(SPEC.get('B_v', 4096)); B_VACC = int(SPEC.get('B_v_acc', B_V)); B_VHO = int(SPEC.get('B_v_ho', 8192))
LV_CHECK = bool(SPEC.get('lv_check', False)); LV_DIAG = bool(SPEC.get('lv_diag', False)); LV_CHECK_REPS = int(SPEC.get('lv_check_reps', 2000))
V_FIXED = int(SPEC.get('v_fixed', 2 ** 17))            # F2 revision (Amendment F2-1): size of the FIXED importance-sampled training bond set (0 = fresh 4096-bond batches as the first F2 run)
B_VHO2 = int(SPEC.get('B_v_ho2', 2 ** 17))             # extra, logged-only held-out bond set (the decision set stays B_v_ho)
W_OLD = float(SPEC.get('old_weight', 1.0))              # F2-1 pilot (a): weight of the old loss (0 = L_v only)
MIN_MOVE = float(SPEC.get('min_move', 1e-3))           # README amendment T1s-1
MIN_WINDOW = int(SPEC.get('min_window', 50))          # amendment T1s-1: the window is the last 20% of steps, at least 50
NW = int(SPEC.get('n_wbar', 8192))                    # samples per decade for Wbar_d
T0JSON = SPEC.get('t0json', '/project/theorie/a/A.Otaifi/chatty_compress6/runs/t0_17410220/cx_t0.json')
res = dict(spec=SPEC, arm=ARM, dry=DRY, design=dict(B=B, K=K, B_acc=B_ACC, B_ho=B_HO, lam_b=LAM_B, lam0=LAM0, lam_cap=LAM_CAP,
                                                    tau=TAU, tau_min=TAU_MIN, anneal_steps=ANNEAL_STEPS, warm_theta=WARM_THETA, splice_dec=SPLICE_DEC, z_acc=Z_ACC, ho_every=HO_EVERY, min_steps=MIN_STEPS,
                                                    flat_tol=FLAT_TOL, lambda_v=LAMV_SPEC, v_fixed=V_FIXED, B_v_ho2=B_VHO2, old_weight=W_OLD, B_v=B_V, B_v_acc=B_VACC, B_v_ho=B_VHO, bf_window=BFW, freeze_outer=FREEZE, max_steps=MAX_STEPS, min_move=MIN_MOVE, min_window=MIN_WINDOW, fit_sec=FIT_SEC, decades=DECS, stratified=STRAT))


def progress(msg):
    with open(PROG, 'a') as fh: fh.write(f'{time.strftime("%Y-%m-%dT%H:%M:%S")} PROGRESS {msg}\n')
    log(msg)


# code drift guard (lead, 2026-10-21): the files actually run must be the ones hashed at submission
import hashlib
HERE_ = os.path.dirname(os.path.abspath(__file__))
CODE_FILES = ('cx_f2fit.py', 'cx_f2_lvcheck.py', 'cx_lib.py', 'rbmpp_lib.py')
SHA_RUN = {fn: (hashlib.sha256(open(os.path.join(HERE_, fn), 'rb').read()).hexdigest()
                if os.path.exists(os.path.join(HERE_, fn)) else None) for fn in CODE_FILES}
SHA_EXP = SPEC.get('code_sha256')
res['code_sha256'] = dict(expected=SHA_EXP, running=SHA_RUN, dir=HERE_)
if SHA_EXP is not None:
    bad = [fn for fn in CODE_FILES if SHA_EXP.get(fn) != SHA_RUN[fn]]
    res['code_sha256']['match'] = not bad
    if bad:
        res['code_sha256']['mismatch'] = bad; dump(F, res)
        progress(f'CODE DRIFT: sha256 mismatch for {bad}; exit 3 before fitting'); sys.exit(3)
elif not DRY:
    dump(F, res); progress('CODE DRIFT GUARD: no code_sha256 in the spec; exit 3'); sys.exit(3)
dump(F, res)
lab = Lab(); sec = lab.sec


def norm_la(la):
    a = jnp.exp(la - jnp.max(la)); return la - jnp.max(la) - 0.5 * jnp.log(jnp.sum(sec.n * a * a))


RF = SPEC['refresh']
TARGET = SPEC.get('target_la') or os.path.join(RF, 'la_star_s2.npy')     # O2b: phi_k of the loop instead of phi*(s_2)
res['target_la'] = TARGET
ls2 = norm_la(jnp.asarray(np.load(TARGET), jnp.float64))
s2 = jnp.asarray(np.load(os.path.join(RF, 's2.npy')).astype(np.float32))
dec, _ = lab.decade_of(ls2)
dec_h = np.asarray(dec); ho_h = np.asarray(lab.HO); n_h = np.asarray(sec.n); ls2_h = np.asarray(ls2)
res['decade_orbits'] = {str(d): dict(train=int(np.sum((dec_h == d) & ~ho_h)), heldout=int(np.sum((dec_h == d) & ho_h)))
                        for d in DECS}
progress(f'[t1s {ARM}] lab + target ready {time.time() - T00:.0f}s  orbits per decade {res["decade_orbits"]}')

# ============================================================================================ models
if MODEL == 'vit':
    import flax
    import vit_dt
    vit_dt.set_dtype('float32')
    vit = vit_dt.make_model_6x6()
    tmpl = vit.init(jax.random.PRNGKey(1234), jnp.zeros((1, N), jnp.float32))
    obj = flax.serialization.msgpack_restore(open(VIT_CKPT, 'rb').read())
    if 'variables' in obj: obj = obj['variables']
    tmpl = flax.serialization.from_state_dict(tmpl, obj)
    params = jax.tree_util.tree_map(lambda q: jnp.asarray(q, jnp.float32), tmpl['params'])
    theta0, unravel = ravel_pytree(params)
    REMAT = int(SPEC.get('remat', 4)); FCH, JCH, TCH = 2048, 128, 4096
    WARM = lab.lP0

    def logA_X(flat, X):                              # X (M, 36) float32 +-1  (cx_reanchor.sym_logA)
        p = {'params': unravel(flat)}; B_ = X.shape[0]
        Xg = jnp.stack([X[:, INV8[k // 2]] * SG2[k % 2] for k in range(16)], 0)
        body = jax.checkpoint(lambda Xi: vit_dt.logpsi_transl_2d(vit.apply, 2, p, Xi))
        zs = jax.lax.map(body, Xg.reshape(REMAT, (16 // REMAT) * B_, N)).reshape(16, B_)
        m = jnp.max(jnp.real(zs), 0)
        return m + jnp.log(jnp.abs(jnp.sum(jnp.exp(zs - m[None, :]), 0)))
    fwd_j = jax.jit(lambda fl, Sx: jnp.real(logA_X(fl, bits(Sx))).astype(f64))
    jac_j = jax.jit(jax.vmap(jax.grad(lambda fl, X1: jnp.real(logA_X(fl, X1[None]))[0]), in_axes=(None, 0)))
    jac_in = lambda Sx: bits(Sx)
else:
    from rbmpp_lib import Lattice
    lat = Lattice(6, 0.5, 'full')
    IMG = lat.IMG; FLP = lat.FL.astype(jnp.float64); NH = N // 2
    xy = np.array([(i % 6, i // 6) for i in range(N)])
    DISP = jnp.asarray(np.array([[((xy[j, 0] - xy[i, 0]) % 6) + 6 * ((xy[j, 1] - xy[i, 1]) % 6) for j in range(N)]
                                 for i in range(N)], np.int32))            # DISP[i, j] = r_j - r_i
    z = np.load(SPEC['pp_params'])
    assert not bool(z['rbm']), 'T1s PP arms: f-only PP expected'
    f0 = jnp.asarray(z['f'], jnp.float64); del z
    BF = MODEL == 'ppbf'
    ND9 = BFW * BFW; HW = BFW // 2
    NBRW = jnp.asarray(np.array([[((x + dx) % 6) + 6 * ((y + dy) % 6) for dy in range(-HW, HW + 1) for dx in range(-HW, HW + 1)]
                                 for y in range(6) for x in range(6)], np.int32))
    if BFW == 3: assert bool(jnp.all(NBRW == NBR))
    IN3 = np.array([(dy + HW) * BFW + (dx + HW) for dy in (-1, 0, 1) for dx in (-1, 0, 1)])   # 3x3 slots inside the window layout
    if SPEC.get('break_embed'): IN3 = IN3 + 1 if BFW == 5 else IN3     # TEST ONLY (check that the step-0 held-out loss detects a wrong embedding)
    theta0 = jnp.concatenate([f0.reshape(-1), jnp.zeros(2 * ND9 * N if BF else 0, jnp.float64)])
    FCH, JCH, TCH = 512, 64, 512
    WARM = jnp.asarray(np.load(SPEC['pp_table'])['lP'], jnp.float64)

    def log_one(f, u1, u2, zz):
        """log|psi_PP(z)| and its sign for one image z (rbmpp_lib.make_fns.log_one, real f; + local backflow)."""
        order = jnp.argsort(-zz, stable=True); Uo = order[:NH]; Do = order[NH:]
        Fm = f[Uo][:, Do]
        if BF:
            zn = zz[NBRW]                                                   # (36, bf_window^2) window around each site
            M1 = zn[Uo] @ u1                                                # (18, 36): [up site a, displacement]
            M2 = zn[Do] @ u2                                                # (18, 36): [down site b, displacement]
            Fm = Fm + jnp.take_along_axis(M1, DISP[Uo][:, Do], 1) + jnp.take_along_axis(M2, DISP[Do][:, Uo], 1).T
        sg, ld = jnp.linalg.slogdet(Fm)
        up = (zz > 0).astype(jnp.int32); dn = 1 - up
        inv = jnp.sum(up * (jnp.cumsum(dn) - dn))
        par = (jnp.where(sg < 0, 1, 0) + inv) % 2
        return ld, 1.0 - 2.0 * par

    def logA_one(th, sigma):
        f = th[:N * N].reshape(N, N)
        if BF:
            u = th[N * N:].reshape(2, ND9, N); u1, u2 = u[0], u[1]
        else:
            u1 = u2 = None
        Z = FLP[:, None] * sigma.astype(jnp.float64)[IMG]
        ld, sgn = jax.vmap(lambda zz: log_one(f, u1, u2, zz))(Z)
        m = jax.lax.stop_gradient(jnp.max(ld))
        return m + jnp.log(jnp.maximum(jnp.abs(jnp.sum(sgn * jnp.exp(ld - m))), 1e-300))
    fwd_j = jax.jit(lambda th, Sx: jax.vmap(lambda s_: logA_one(th, s_))(bits(Sx).astype(jnp.float64)))
    jac_j = jax.jit(jax.vmap(jax.grad(logA_one), in_axes=(None, 0)))
    jac_in = lambda Sx: bits(Sx).astype(jnp.float64)
NPAR = int(theta0.size)
theta_start = theta0
if WARM_THETA:
    _w = np.load(WARM_THETA)
    if _w.size != NPAR:                               # F1: a 3x3 theta (f, u of shape (2, 9, 36)) into the 5x5 layout
        assert MODEL == 'ppbf' and BFW == 5 and _w.size == N * N + 2 * 9 * N, (_w.size, NPAR)
        u5 = np.zeros((2, ND9, N)); u5[:, IN3, :] = _w[N * N:].reshape(2, 9, N)
        _w = np.concatenate([_w[:N * N], u5.reshape(-1)])
    theta_start = jnp.asarray(_w, theta0.dtype).reshape(theta0.shape); del _w
    res['warm_theta_sha256'] = __import__('hashlib').sha256(open(WARM_THETA, 'rb').read()).hexdigest()
ACT = None
if MODEL == 'ppbf' and FREEZE:
    _m = np.zeros((2, ND9, N), bool); _m[:, IN3, :] = True
    ACT = jnp.asarray(np.nonzero(np.concatenate([np.ones(N * N, bool), _m.reshape(-1)]))[0])
res['bf_window'] = BFW; res['freeze_outer'] = FREEZE; res['n_active'] = int(ACT.size) if ACT is not None else NPAR
res['npar'] = NPAR; res['npar_note'] = ('ViT (cx_reanchor)' if MODEL == 'vit' else
                                       f'f 36x36 ({N * N}, diagonal unused: {N * N - N} active)' + (f' + backflow u1,u2 2x9x36 ({2 * 9 * N})' if MODEL == 'ppbf' else ''))
progress(f'[t1s {ARM}] model {MODEL} npar {NPAR}')


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
    """per-configuration gradients written into one preallocated (donated) buffer: no 2x peak from a concatenation."""
    R_ = Sx.shape[0]
    Gb = jnp.zeros((R_, NPAR), theta0.dtype)
    for i in range(0, R_, JCH):
        s_, m = _pad(Sx[i:i + JCH], JCH)
        Gb = _put(Gb, jac_j(th, jac_in(s_))[:m].astype(theta0.dtype), i)
    return Gb


def table(th, idx=None):
    """full-sector log A (or at the orbit indices idx)."""
    reps = sec.reps if idx is None else sec.reps[jnp.asarray(idx)]
    out = []; t0 = time.time(); tl = t0; D_ = reps.shape[0]
    for i in range(0, D_, TCH):
        s_, m = _pad(reps[i:i + TCH], TCH); out.append(fwd_j(th, s_)[:m])
        if time.time() - tl > 120:
            jax.block_until_ready(out[-1]); tl = time.time(); progress(f'  table {i + m}/{D_}  {tl - t0:.0f}s')
    return jnp.concatenate(out)


# ============================================================================================ sampler and bonds
class Sampler:
    """stratified: ceil(M/9) configurations per decade -5..-13 (n_r-weighted inside the decade); else |phi*|^2 (n_r a^2)."""

    def __init__(self, eligible, seed, strat):
        self.rng = np.random.default_rng(seed); self.groups = []
        if strat:
            for d in DECS:
                idx = np.nonzero((dec_h == d) & eligible)[0]
                cdf = np.cumsum(n_h[idx]); self.groups.append((idx, cdf / cdf[-1]))
        else:
            idx = np.nonzero(eligible)[0]
            cdf = np.cumsum(n_h[idx] * np.exp(2 * (ls2_h[idx] - ls2_h.max()))); self.groups.append((idx, cdf / cdf[-1]))

    def draw(self, M):
        G = len(self.groups); m = -(-M // G)
        idx = np.concatenate([g[0][np.minimum(np.searchsorted(g[1], self.rng.random(m)), g[0].size - 1)] for g in self.groups])
        return idx, np.repeat(np.arange(G), m)


@jax.jit
def bonds_of(idx, key):
    """all bonds of the reps idx; bad bonds of s_2 with w = |H_xy| phi*_y/phi*_x; K of them drawn with prob w/W."""
    x = sec.reps[idx]; v = valid_bonds(x)
    y = jnp.where(v, x[:, None] ^ MASKS[None, :], x[:, None])
    iy = SS.canon(sec.T, sec.reps, y.reshape(-1)).reshape(x.shape[0], NB)
    bad = v & (s2[idx][:, None] * s2[iy] > 0)
    lw = jnp.where(bad, jnp.log(0.5 * JB[None, :]) + jnp.clip(ls2[iy] - ls2[idx][:, None], -60, 60), -jnp.inf)
    W = jnp.sum(jnp.exp(lw), 1)
    has = jnp.any(bad, 1)
    sel = jax.random.categorical(key, jnp.where(has[:, None], lw, 0.0), axis=1, shape=(K, x.shape[0])).T
    return x, jnp.take_along_axis(y, sel, 1), jnp.take_along_axis(iy, sel, 1), W


def wbar_of(sampler, key, n_per):
    """per-group mean of W_x (Wbar_d), from n_per configurations per group, in chunks of 2048."""
    G = len(sampler.groups); acc = np.zeros(G); cnt = np.zeros(G)
    for i in range(0, n_per, 2048 // G):
        idx, g = sampler.draw(min(2048 // G, n_per - i) * G); key, k = jax.random.split(key)
        W = np.asarray(bonds_of(jnp.asarray(idx), k)[3])
        np.add.at(acc, g, W); np.add.at(cnt, g, 1)
    return acc / cnt


def make_batch(sampler, M, key, wbar):
    idx, g = sampler.draw(M)
    parts = []
    for i in range(0, idx.size, 2052):                # bounded memory of the 288-image canonicalisation
        key, k_ = jax.random.split(key); parts.append(bonds_of(jnp.asarray(idx[i:i + 2052]), k_))
    x, yk, iyk, W = (jnp.concatenate([q[j] for q in parts]) for j in range(4))
    return dict(idx=idx, g=g, x=x, y=yk, lx=ls2[jnp.asarray(idx)], ly=ls2[iyk],
                wb=LAM_B * W / jnp.asarray(wbar)[jnp.asarray(g)] / K, W=W, ng=len(sampler.groups))


def S_of(th, bt):
    return fwd(th, jnp.concatenate([bt['x'], bt['y'].reshape(-1)]))


@jax.jit
def per_sample(S, lx, ly, wb, c):
    M = lx.shape[0]
    ex = S[:M] - lx - c; ey = S[M:].reshape(M, -1) - ly - c
    db = ex[:, None] - ey
    return ex * ex + wb * jnp.sum(db * db, 1), ex, db


def strat_mean_se(v, g, G):
    v = np.asarray(v); mu = []; var = []
    for k in range(G):
        q = v[g == k]; mu.append(q.mean()); var.append(q.var(ddof=1) / q.size)
    return float(np.mean(mu)), float(np.sqrt(np.sum(var)) / G)


def batch_loss(th, c, bt):
    L, ex, db = per_sample(S_of(th, bt), bt['lx'], bt['ly'], bt['wb'], c)
    return L, ex, db


def decade_diag(bt, ex, db):
    """per decade of x: rms of e (value) and wb-weighted rms of the bond differences (held-out diagnostics)."""
    d_ = dec_h[bt['idx']]; ex = np.asarray(ex); db = np.asarray(db); wb = np.asarray(bt['wb'])
    out = {}
    for d in DECS:
        m = d_ == d
        if not m.any(): continue
        sw = float(np.sum(wb[m])) * db.shape[1]
        out[str(d)] = dict(n=int(m.sum()), rms_e=float(np.sqrt(np.mean(ex[m] ** 2))),
                           bond_rms=float(np.sqrt(np.sum(wb[m][:, None] * db[m] ** 2) / sw)) if sw > 0 else None)
    return out


# ============================================================================================ F2: violating-bond sampler and L_v
VCH = 2048


@jax.jit
def wx_chunk(i0):
    """per rep x of the chunk: sum over its violating bonds of n_x phi_x phi_y |H_xy| (the weight bond_chunk uses)."""
    ii = jnp.minimum(i0 + jnp.arange(VCH), sec.D - 1); real = (i0 + jnp.arange(VCH)) < sec.D
    xx = sec.reps[ii]; vv = valid_bonds(xx)
    yy = jnp.where(vv, xx[:, None] ^ MASKS[None, :], xx[:, None])
    jj = SS.canon(sec.T, sec.reps, yy.reshape(-1)).reshape(VCH, NB)
    bd = vv & (s2[ii][:, None] * s2[jj] > 0) & real[:, None]
    w = jnp.where(bd, 0.5 * JB[None, :] * jnp.exp(ls2[ii][:, None] + ls2[jj]) * sec.n[ii][:, None], 0.0)
    return jnp.sum(w, 1)


@partial(jax.jit, static_argnums=(2,))
def vbond_of(idx, key, flip=False):
    """one violating bond per rep of idx, drawn with probability |H_xy| phi_y / sum (x fixed).  flip=True is the MISLABEL control:
    the flag is inverted (kept bonds s_2(x) s_2(y) < 0 are taken instead)."""
    x = sec.reps[idx]; v = valid_bonds(x)
    y = jnp.where(v, x[:, None] ^ MASKS[None, :], x[:, None])
    iy = SS.canon(sec.T, sec.reps, y.reshape(-1)).reshape(x.shape[0], NB)
    prod = s2[idx][:, None] * s2[iy]
    bad = v & ((prod < 0) if flip else (prod > 0))
    lw = jnp.where(bad, jnp.log(0.5 * JB[None, :]) + jnp.clip(ls2[iy] - ls2[idx][:, None], -60, 60), -jnp.inf)
    has = jnp.any(bad, 1)
    sel = jax.random.categorical(key, jnp.where(has[:, None], lw, 0.0), axis=1, shape=(1, x.shape[0])).T
    return x, jnp.take_along_axis(y, sel, 1)[:, 0], jnp.take_along_axis(iy, sel, 1)[:, 0], has


class VSampler:
    """bonds in proportion to w_xy over the eligible orbits: x ~ W_x (cdf, host), then the bond given x."""

    def __init__(self, Wx, eligible, seed, flip=False):
        self.rng = np.random.default_rng(seed); self.key = jax.random.PRNGKey(seed); self.flip = flip
        self.idx = np.nonzero(eligible & (Wx > 0))[0]
        cdf = np.cumsum(Wx[self.idx]); self.tot = float(cdf[-1]); self.cdf = cdf / cdf[-1]

    def draw(self, M):
        x_idx = self.idx[np.minimum(np.searchsorted(self.cdf, self.rng.random(M)), self.idx.size - 1)]
        parts = []
        for i in range(0, M, 2052):
            self.key, k_ = jax.random.split(self.key)
            parts.append(vbond_of(jnp.asarray(x_idx[i:i + 2052]), k_, self.flip))
        x, y, iy, has = (jnp.concatenate([q[j] for q in parts]) for j in range(4))
        assert bool(jnp.all(has)), 'sampled an orbit without a bond of the requested kind'
        ix = jnp.asarray(x_idx)
        return dict(ix=ix, x=x, y=y, iy=iy, lx=ls2[ix], ly=ls2[iy], M=M)


def v_resid(S, vb):
    """S = network log A on [x; y] of the bond batch.  d = (S_y - S_x) - (log phi_y - log phi_x); L_v = mean d^2 (sampling is already in proportion to w)."""
    M = vb['M']
    return (S[M:] - S[:M]) - (vb['ly'] - vb['lx'])


def Sv_of(th, vb):
    return fwd(th, jnp.concatenate([vb['x'], vb['y']]))


def lv_value(th, vb):
    d = v_resid(Sv_of(th, vb), vb); return float(jnp.mean(d * d)), d


def lv_exact(th, ridx, flip=False, chunk=512):
    """README formula by full enumeration over the reps ridx: sum w d^2 / sum w over ALL their violating bonds (flip: mislabelled flag).  Vectorised twin of check B."""
    num = 0.0; den = 0.0; nb_ = 0
    for i in range(0, ridx.size, chunk):
        ii = jnp.asarray(ridx[i:i + chunk]); xx = sec.reps[ii]; vv = valid_bonds(xx)
        yy = jnp.where(vv, xx[:, None] ^ MASKS[None, :], xx[:, None])
        jj = SS.canon(sec.T, sec.reps, yy.reshape(-1)).reshape(xx.shape[0], NB)
        prod = s2[ii][:, None] * s2[jj]
        bd = vv & ((prod < 0) if flip else (prod > 0))
        Sx = fwd(th, xx); Sy = fwd(th, sec.reps[jj.reshape(-1)]).reshape(xx.shape[0], NB)
        d = (Sy - Sx[:, None]) - (ls2[jj] - ls2[ii][:, None])
        w = jnp.where(bd, 0.5 * JB[None, :] * jnp.exp(ls2[ii][:, None] + ls2[jj]) * sec.n[ii][:, None], 0.0)
        num += float(jnp.sum(w * d * d)); den += float(jnp.sum(w)); nb_ += int(jnp.sum(bd))
    return num / den, nb_


# ---- F2-1: FIXED bond sets.  A bond set is drawn once (importance sampling in w, so L_v = plain mean of d^2); the configurations are deduplicated
#      (orbit indices of x and of y; the network is invariant under the space group x spin flip, so fwd(reps[iy]) = fwd(y); checked in lv_check D).
def make_fixed(vb):
    ix = np.asarray(vb['ix']); iy = np.asarray(vb['iy']); M = ix.size
    U, inv = np.unique(np.concatenate([ix, iy]), return_inverse=True)
    return dict(M=M, nU=int(U.size), cfg=sec.reps[jnp.asarray(U)], px=jnp.asarray(inv[:M]), py=jnp.asarray(inv[M:]), t=jnp.asarray(np.asarray(vb['ly'] - vb['lx'])))


def d_fixed(S, FX, swap=False):
    a, b = (FX['px'], FX['py']) if not swap else (FX['py'], FX['px'])
    return (S[b] - S[a]) - FX['t']


def lv_fixed(S, FX):
    d = d_fixed(S, FX); return float(jnp.mean(d * d))


GCH = 8192


def gn_fixed(th, FX, S, lamv, swap=False):
    """normal equations of the rows sqrt(lamv/M) (G_y - G_x), residual sqrt(lamv/M) d over the whole fixed set: (Sm (P,P), g (P), sum r^2).  Chunked: no (M, P) matrix."""
    Gu = jac(th, FX['cfg']); d = d_fixed(S, FX, swap).astype(f64); M = FX['M']
    a_, b_ = (FX['px'], FX['py']) if not swap else (FX['py'], FX['px'])
    Sm = jnp.zeros((Gu.shape[1], Gu.shape[1]), f64); g = jnp.zeros(Gu.shape[1], f64)
    for i in range(0, M, GCH):
        dg = (Gu[b_[i:i + GCH]] - Gu[a_[i:i + GCH]]).astype(f64)
        Sm = Sm + dg.T @ dg; g = g + dg.T @ d[i:i + GCH]
    del Gu
    q = lamv / M
    return Sm * q, g * q, q * jnp.sum(d * d)


# ============================================================================================ GN / minSR direction
@jax.jit
def solve_dir(G, S, lx, ly, wb, c, lam_rel, tau, Jx=None, rx=None, Qv=None, wo=1.0):
    """rows: value rows sv e_x (+ offset column -sv) and bond rows sb_x (e_x - e_yk); G = d log A / d theta on [x; y].
    Sample space (minSR): T = A G G^T A^T + cc cc^T with the 2-sparse row combination A applied by gathers."""
    M = lx.shape[0]; P = G.shape[1]; dt = G.dtype
    ex = S[:M] - lx - c; ey = S[M:].reshape(M, -1) - ly - c; Kb = ey.shape[1]
    wos = jnp.sqrt(wo)                                 # F2-1: weight of the old loss (1.0: bit-identical to F1)
    sv = (1.0 / jnp.sqrt(M)) * wos; sb = jnp.sqrt(wb / M) * wos
    r = jnp.concatenate([sv * ex, (sb[:, None] * (ex[:, None] - ey)).reshape(-1)])
    cc = jnp.concatenate([jnp.full(M, -sv, f64), jnp.zeros(M * Kb, f64)])
    R = M * (1 + Kb)
    Gx = G[:M]
    if R <= P + 1:
        assert Jx is None and Qv is None, 'F2 L_v rows are implemented for the parameter-space branch only'                                        # minSR, sample space: J = A G never materialised
        rows = jnp.arange(M * Kb); xs = rows // Kb
        i1 = jnp.concatenate([jnp.arange(M), xs]); c1 = jnp.concatenate([jnp.full(M, sv, f64), sb[xs]])
        i2 = jnp.concatenate([jnp.zeros(M, jnp.int32), M + rows]); c2 = jnp.concatenate([jnp.zeros(M, f64), -sb[xs]])
        GG = (G @ G.T).astype(f64)
        H = c1[:, None] * GG[i1] + c2[:, None] * GG[i2]; del GG
        T = H[:, i1] * c1[None, :] + H[:, i2] * c2[None, :] + jnp.outer(cc, cc); del H
        lam = lam_rel * jnp.trace(T) / R
        T = T.at[jnp.arange(R), jnp.arange(R)].add(lam)
        a = jax.scipy.linalg.cho_solve(jax.scipy.linalg.cho_factor(T, lower=True), r); del T
        w = jnp.zeros(R, f64).at[i1].add(c1 * a).at[i2].add(c2 * a)        # A^T a
        dP = -(G.T @ w.astype(dt)).astype(f64); dc = -(cc @ a)
    else:                                                 # the same LM step in parameter space (PP arms)
        Jv = Gx * sv.astype(dt)
        Jb = ((Gx[:, None, :] - G[M:].reshape(M, Kb, P)) * sb.astype(dt)[:, None, None]).reshape(M * Kb, P)
        Jf = jnp.concatenate([jnp.concatenate([Jv, Jb]).astype(f64), cc[:, None]], 1); del Jv, Jb
        if Jx is not None:                             # F2: rows sqrt(lambda_v/M_v) (G_y - G_x); the offset column is 0 (c cancels in a ratio)
            Jf = jnp.concatenate([Jf, jnp.concatenate([Jx.astype(f64), jnp.zeros((Jx.shape[0], 1), f64)], 1)])
            r = jnp.concatenate([r, rx])
        Sm = Jf.T @ Jf; g = Jf.T @ r; r2 = jnp.sum(r * r)
        if Qv is not None:                             # F2-1: the fixed bond set's normal equations, accumulated in chunks (gn_fixed)
            Sm = Sm.at[:P, :P].add(Qv[0]); g = g.at[:P].add(Qv[1]); r2 = r2 + Qv[2]
        lam = lam_rel * jnp.trace(Sm) / (P + 1)
        d = -jnp.linalg.solve(Sm + lam * jnp.eye(P + 1, dtype=f64), g)
        dP = d[:P]; dc = d[P]
    pc = (Gx.astype(f64) @ dP); pc = pc - jnp.mean(pc)
    prms = jnp.sqrt(jnp.mean(pc * pc))
    sc = jnp.minimum(1.0, tau / jnp.maximum(prms, 1e-30))
    return dP * sc, dc * sc, prms, sc, (r2 if R > P + 1 else jnp.sum(r * r))


def dense_check(seed=0, M=6, Kb=2, P=40):
    """dry-run self-test: the gather form of T equals the dense A G G^T A^T (sample-space branch)."""
    rng = np.random.default_rng(seed); R = M * (1 + Kb)
    G = jnp.asarray(rng.normal(size=(R, P)), f64); S = jnp.asarray(rng.normal(size=R)); lx = jnp.asarray(rng.normal(size=M))
    ly = jnp.asarray(rng.normal(size=(M, Kb))); wb = jnp.asarray(rng.random(M)); c = 0.1
    dP, dc, prms, sc, _ = solve_dir(G, S, lx, ly, wb, c, 1e-2, 1e9)
    sv = 1 / np.sqrt(M); sb = np.sqrt(np.asarray(wb) / M); A = np.zeros((R, R))
    A[np.arange(M), np.arange(M)] = sv
    for i in range(M):
        for k in range(Kb):
            A[M + i * Kb + k, i] = sb[i]; A[M + i * Kb + k, M + i * Kb + k] = -sb[i]
    Gn = np.asarray(G); Sn = np.asarray(S); ex = Sn[:M] - np.asarray(lx) - c; ey = Sn[M:].reshape(M, Kb) - np.asarray(ly) - c
    r = np.concatenate([sv * ex, (sb[:, None] * (ex[:, None] - ey)).reshape(-1)]); cc = np.concatenate([np.full(M, -sv), np.zeros(M * Kb)])
    J = A @ Gn; T = J @ J.T + np.outer(cc, cc); T = T + 1e-2 * np.trace(T) / R * np.eye(R)
    a = np.linalg.solve(T, r)
    return float(np.max(np.abs(np.asarray(dP) + J.T @ a))), float(abs(float(dc) + cc @ a))


# ============================================================================================ readout
CHW = 2048


@jax.jit
def bond_chunk(i0, lb):                               # cx_t0.bond_chunk
    ii = jnp.minimum(i0 + jnp.arange(CHW), sec.D - 1); real = (i0 + jnp.arange(CHW)) < sec.D
    xx = sec.reps[ii]; vv = valid_bonds(xx)
    yy = jnp.where(vv, xx[:, None] ^ MASKS[None, :], xx[:, None])
    jj = SS.canon(sec.T, sec.reps, yy.reshape(-1)).reshape(CHW, NB)
    bd = vv & (s2[ii][:, None] * s2[jj] > 0) & real[:, None]
    w = jnp.where(bd, 0.5 * JB[None, :] * jnp.exp(ls2[ii][:, None] + ls2[jj]) * sec.n[ii][:, None], 0.0)
    dl = (lb[jj] - lb[ii][:, None]) - (ls2[jj] - ls2[ii][:, None])
    dx = dec[ii]
    out = []
    for q in range(-4, -16, -1):
        m = (dx == q)[:, None]
        out.append(jnp.array([jnp.sum(jnp.where(m, w, 0)), jnp.sum(jnp.where(m, w * dl ** 2, 0))]))
    return jnp.stack(out)


def bond_stats(lb, nchunks=None):
    rng_ = range(0, sec.D, CHW) if nchunks is None else range(0, nchunks * CHW, CHW)
    acc = np.asarray(sum(bond_chunk(i0, lb) for i0 in rng_)); tot = acc.sum(0)
    r = lambda a, b: float(np.sqrt(a / b)) if b > 0 else None
    return dict(total=r(tot[1], tot[0]), by_decade={str(q): r(acc[k, 1], acc[k, 0]) for k, q in enumerate(range(-4, -16, -1))},
                partial_chunks=nchunks)


PT = sec.vec(ls2, s2) ** 2
SEG = jnp.asarray(np.clip(-dec_h, 0, 63))


def decade_table(la_b):
    """per-decade rms of e = norm(la_b) - log phi*:  count (n_r-weighted, all / held-out; offset = equal-decade mean over
    -5..-13) and T0 convention (|phi*|^2-weighted offset and weights; eps = rms 10^(d/2), eps_median_tail over -9..-13)."""
    e = norm_la(la_b) - ls2
    ss = lambda v: np.asarray(jax.ops.segment_sum(v, SEG, num_segments=64))
    wn = ss(sec.n); me = ss(sec.n * e) / np.maximum(wn, 1e-300)
    c_s = float(np.mean([me[-d] for d in DECS]))
    ec = e - c_s
    rn = np.sqrt(ss(sec.n * ec * ec) / np.maximum(wn, 1e-300))
    hn = jnp.where(lab.HO, sec.n, 0.0); wh = ss(hn); rh = np.sqrt(ss(hn * ec * ec) / np.maximum(wh, 1e-300))
    e0 = e - jnp.sum(PT * e); wp = ss(PT); rp = np.sqrt(ss(PT * e0 * e0) / np.maximum(wp, 1e-300))
    out = {}
    for d in range(-4, -16, -1):
        if wn[-d] <= 0: continue
        out[str(d)] = dict(rms_count=float(rn[-d]), rms_count_heldout=float(rh[-d]) if wh[-d] > 0 else None,
                           mean_count=float(me[-d] - c_s), rms_T0=float(rp[-d]), eps_T0=float(rp[-d]) * 10 ** (d / 2))
    tail = [out[str(d)]['eps_T0'] for d in range(-9, -14, -1) if str(d) in out]
    return dict(by_decade=out, offset_count=c_s, eps_median_tail_T0=float(np.median(tail)))


def efn(la_b, maxit=400):
    _, u, info = sec.fn_solve(la_b, s2, maxit=maxit); del u
    return info


# ============================================================================================ control 2 (readout code on the warm table)
SKIP_C2 = bool(SPEC.get('skip_control2', False))   # O2b: the target is not phi*(s_2), so cx_t0.json does not apply
t0json = json.load(open(T0JSON)) if (os.path.exists(T0JSON) and not SKIP_C2) else None
base_key = 'vit' if MODEL == 'vit' else 'pp'
dt_warm = decade_table(WARM)
c2 = dict(skipped_t0_comparison=SKIP_C2, warm_table=('lab.lP0 (psi_P)' if MODEL == 'vit' else SPEC['pp_table']), eps_median_tail_T0=dt_warm['eps_median_tail_T0'],
          by_decade=dt_warm['by_decade'])
if t0json:
    ref = t0json['bases'][base_key]
    c2['t0json_eps_median_tail'] = ref['eps_median_tail']
    c2['t0json_rms_e_by_decade'] = {k: v['rms_e'] for k, v in ref['eps_by_decade'].items()}
    c2['rel_diff_eps'] = abs(dt_warm['eps_median_tail_T0'] / ref['eps_median_tail'] - 1)
res['control2'] = c2; dump(F, res)
progress(f'[control 2] warm-table eps_median_tail {c2["eps_median_tail_T0"]:.4e}  (cx_t0.json {c2.get("t0json_eps_median_tail")})')

# ============================================================================================ fit
key = jax.random.PRNGKey(100 + SEED)
S_tr = Sampler(~ho_h, 1000 + SEED, STRAT); S_ho = Sampler(ho_h, 5000 + SEED, STRAT)
key, k1 = jax.random.split(key)
WBAR = wbar_of(Sampler(~ho_h, 3000 + SEED, STRAT), k1, NW)
res['Wbar'] = WBAR.tolist(); progress(f'[wbar] {np.array2string(WBAR, precision=3)}')
key, k1 = jax.random.split(key)
HOB = make_batch(S_ho, B_HO, k1, WBAR)                 # fixed held-out batch
# warm network vs warm table on the held-out sample (control 2, second half)
Sw = fwd(theta0, HOB['x']); dw = np.asarray(Sw - WARM[jnp.asarray(HOB['idx'])]); dw = dw - dw.mean()
res['control2']['warm_net_vs_table_rms'] = float(np.sqrt(np.mean(dw ** 2)))
res['control2']['warm_net_vs_table_max'] = float(np.max(np.abs(dw)))
c = float(np.mean(np.asarray(Sw) - np.asarray(HOB['lx'])))          # offset start: mean over the stratified held-out batch
theta = theta_start
if WARM_THETA:
    c = float(np.mean(np.asarray(fwd(theta, HOB['x'])) - np.asarray(HOB['lx'])))
L_ho0, ex, db = batch_loss(theta, c, HOB); L_ho0 = float(jnp.mean(L_ho0))
# ---- F2 bond sets (own random streams; nothing above is touched).  One pass over all orbits for W_x = sum of the violating-bond weights.
t_w = time.time()
Wx = np.concatenate([np.asarray(wx_chunk(i0)) for i0 in range(0, sec.D, VCH)])[:sec.D]
progress(f'[F2] violating-bond weights W_x over {sec.D} orbits {time.time() - t_w:.0f}s; total weight {Wx.sum():.6e} (train {Wx[~ho_h].sum():.6e}, held-out {Wx[ho_h].sum():.6e}); '
         f'orbits with a violating bond {int((Wx > 0).sum())}')
VS_tr = VSampler(Wx, ~ho_h, 9000 + SEED); VS_acc = VSampler(Wx, ~ho_h, 9500 + SEED); VS_ho = VSampler(Wx, ho_h, 9900 + SEED)
VHO = VS_ho.draw(B_VHO)                                # fixed held-out bond set
Lv_ho0, _ = lv_value(theta, VHO)
FIX = None; VHO2 = None
if B_VHO2 > 0: VHO2 = make_fixed(VSampler(Wx, ho_h, 9901 + SEED).draw(B_VHO2))
LAMV = (L_ho0 / Lv_ho0) if LAMV_SPEC == 'auto' else float(LAMV_SPEC)
res['F2'] = dict(lambda_v=LAMV, lambda_v_auto_at_step0=L_ho0 / Lv_ho0, L_ho_old_step0=L_ho0, L_v_ho_step0=Lv_ho0, lambda_v_x_L_v_step0=LAMV * Lv_ho0,
                 W_total=float(Wx.sum()), n_orbits_with_violating_bond=int((Wx > 0).sum()), B_v=B_V, B_v_acc=B_VACC, B_v_ho=B_VHO)
dump(F, res)
progress(f'[F2] step 0 held-out: old loss {L_ho0:.6e}  L_v {Lv_ho0:.6e}  auto lambda_v = old/L_v = {L_ho0 / Lv_ho0:.6f}  used lambda_v = {LAMV:.6f}')
if LV_DIAG:                                            # F2 revision diagnostic (no fit): tail of d^2 under the w-sampling, batch noise of L_v
    t_d = time.time(); NBt, Bt = 256, 4096
    vbd = VSampler(Wx, ~ho_h, 7777).draw(NBt * Bt)
    dd_ = np.asarray(v_resid(Sv_of(theta, vbd), vbd)); d2 = dd_ * dd_; Nd = d2.size
    srt = np.sort(d2)[::-1]; cs = np.cumsum(srt); tot_ = float(cs[-1])
    top = {f'top_{q}': float(cs[max(int(q * Nd), 1) - 1] / tot_) for q in (0.0001, 0.001, 0.01, 0.1)}
    top['top_10_bonds'] = float(cs[9] / tot_); top['top_100_bonds'] = float(cs[99] / tot_)
    qq = {str(q): float(np.quantile(d2, q)) for q in (0.5, 0.9, 0.99, 0.999, 0.9999)}; qq['max'] = float(srt[0]); qq['mean'] = tot_ / Nd
    bm = d2.reshape(NBt, Bt).mean(1); bm17 = d2.reshape(8, Nd // 8).mean(1)
    ix_d = np.asarray(vbd['ix']); iy_d = np.asarray(vbd['iy']); pair = ix_d.astype(np.int64) * sec.D + iy_d
    ux, cx_ = np.unique(ix_d, return_counts=True)
    hod = np.asarray(v_resid(Sv_of(theta, VHO), VHO)) ** 2
    dgn = dict(n_bonds=int(Nd), quantiles_d2=qq, share_of_L_v=top, mean_d2=tot_ / Nd, rms_d=float(np.sqrt(tot_ / Nd)),
               n_eff=float(tot_ ** 2 / np.sum(d2 ** 2)), n_eff_frac=float(tot_ ** 2 / np.sum(d2 ** 2) / Nd),
               batch4096_rel_std=float(bm.std(ddof=1) / bm.mean()), batch4096_min_max_over_mean=[float(bm.min() / bm.mean()), float(bm.max() / bm.mean())],
               batch131072_rel_std_8=float(bm17.std(ddof=1) / bm17.mean()), unique_x_frac=float(ux.size / Nd), unique_pairs_frac=float(np.unique(pair).size / Nd),
               top_x_orbit_share_of_bonds=float(cx_.max() / Nd), heldout_B_v_ho=int(hod.size), heldout_L_v=float(hod.mean()),
               heldout_rel_se_from_train_var=float(d2.std(ddof=1) / math.sqrt(hod.size) / d2.mean()),
               heldout_top1pct_share=float(np.sort(hod)[::-1][:max(hod.size // 100, 1)].sum() / hod.sum()), sec=time.time() - t_d)
    res['lv_diag'] = dgn; res['lv_check_pass'] = None; res['F2']['lv_diag'] = True; res['done'] = True; dump(F, res)
    progress('[lv_diag] ' + json.dumps(dgn)); sys.exit(0)
if LV_CHECK:
    exec(open(os.path.join(HERE_, 'cx_f2_lvcheck.py')).read())
    sys.exit(0)
if LAMV > 0 and V_FIXED > 0:                           # F2-1: ONE fixed training bond set (own stream), drawn once, used for GN rows and acceptance
    FIX = make_fixed(VSampler(Wx, ~ho_h, 9700 + SEED).draw(V_FIXED))
    S_fix = fwd(theta, FIX['cfg']); Lv_fix0 = lv_fixed(S_fix, FIX)
    res['F2'].update(v_fixed=V_FIXED, v_fixed_unique_configs=FIX['nU'], L_v_fixed_step0=Lv_fix0)
    progress(f'[F2-1] fixed training bond set {V_FIXED} bonds, {FIX["nU"]} unique configs; L_v(fixed set, step 0) {Lv_fix0:.6e}')
if VHO2 is not None:
    Lv_ho2_0 = lv_fixed(fwd(theta, VHO2['cfg']), VHO2); res['F2'].update(B_v_ho2=B_VHO2, L_v_ho2_step0=Lv_ho2_0)
    progress(f'[F2-1] big held-out bond set {B_VHO2} bonds, {VHO2["nU"]} unique configs; L_v(big held-out, step 0) {Lv_ho2_0:.6e}')
dump(F, res)
L_ho_tot0 = W_OLD * L_ho0 + LAMV * Lv_ho0
curve = [dict(step=0, t=0.0, L_ho=L_ho_tot0, L_ho_old=L_ho0, L_v_ho=Lv_ho0, L_v_ho2=(Lv_ho2_0 if VHO2 is not None else None), L_v_fix=(Lv_fix0 if FIX is not None else None), c=c, by_decade=decade_diag(HOB, ex, db))]
progress(f'[fit {ARM}] start held-out loss {L_ho_tot0:.4e} (old {L_ho0:.4e}, L_v {Lv_ho0:.4e}, lambda_v {LAMV:.4f})  warm net vs table rms {res["control2"]["warm_net_vs_table_rms"]:.2e}')
tau = TAU; tau_changed = 0; tau_hist = [dict(step=0, tau=tau)]
if DRY: res['dense_check_maxdiff'] = dense_check(); progress(f'[dry] gather-vs-dense T check {res["dense_check_maxdiff"]}')
hist = []; lam_rel = LAM0; n = 0; t0 = time.time(); conv = dict(converged=False, reason='not checked')
sec.offload()


def check_converged():
    n_ = len(hist); w0 = max(0, n_ - max(int(math.ceil(0.2 * n_)), 2 * HO_EVERY, MIN_WINDOW))
    win = hist[w0:]
    acc = float(np.mean([h['ok'] for h in win]))
    lam_ok = all(h['lam_rel'] < LAM_CAP for h in win)
    c0 = [q for q in curve if q['step'] <= w0]; c0 = c0[-1] if c0 else curve[0]
    flat = abs(c0['L_ho'] - curve[-1]['L_ho']) / curve[-1]['L_ho']
    accd = [h['pred_rms'] * h['trust_scale'] for h in win if h['ok']]
    med_acc = float(np.median(accd)) if accd else 0.0
    moving = med_acc >= MIN_MOVE                      # README amendment T1s-1: a flat but frozen fit is not converged
    tau_ok = tau <= TAU_MIN * (1 + 1e-9)               # T1s-2: tau has reached its floor
    ok = bool(acc >= 0.5 and lam_ok and flat < FLAT_TOL and moving and tau_ok)
    why = f'window steps {w0 + 1}..{n_}: acceptance {acc:.2f} (>= 0.5), lam_rel < cap at all steps {lam_ok}, ' \
          f'held-out change {flat:.4f} (< {FLAT_TOL}) from step {c0["step"]}, ' \
          f'median predicted rms change per accepted step {med_acc:.2e} (>= {MIN_MOVE:.0e}), tau {tau:.4f} (== {TAU_MIN})'
    if not moving: why = 'stalled: ' + why
    return dict(converged=ok, stalled=not moving, tau=tau, tau_at_floor=tau_ok, median_move_accepted_window=med_acc, reason=why, acc_window=acc, lam_below_cap=lam_ok, heldout_rel_change=flat, window_start=w0 + 1,
                median_pred_rms_window=float(np.median([h['pred_rms'] * h['trust_scale'] for h in win])),
                median_lam_rel_window=float(np.median([h['lam_rel'] for h in win])))


while time.time() - t0 < FIT_SEC and n < MAX_STEPS:
    n += 1; ts = time.time()
    key, k1, k2 = jax.random.split(key, 3)
    bt = make_batch(S_tr, B, k1, WBAR)
    Xall = jnp.concatenate([bt['x'], bt['y'].reshape(-1)])
    S = fwd(theta, Xall); G = jac(theta, Xall)
    Jx = rx = Qv = None
    if FIX is not None:                                # F2-1: the fixed set's normal equations at the current theta (S_fix cached for it)
        Qv = gn_fixed(theta, FIX, S_fix, LAMV)
    elif LAMV > 0:                                       # F2: fresh violating-bond batch (own streams); rows sqrt(lambda_v/M_v) (G_y - G_x)
        vb = VS_tr.draw(B_V); Xv = jnp.concatenate([vb['x'], vb['y']])
        Sv = fwd(theta, Xv); Gv = jac(theta, Xv); sq = math.sqrt(LAMV / B_V)
        Jx = (Gv[B_V:] - Gv[:B_V]) * sq; rx = sq * v_resid(Sv, vb).astype(f64); del Gv, Sv, Xv
    if ACT is None:
        dP, dc, prms, sc, rr = solve_dir(G, S, bt['lx'], bt['ly'], bt['wb'], c, lam_rel, tau, Jx, rx, Qv, W_OLD)
    else:                                              # F1 freeze_outer: the step on the active columns only (= the window-3 step)
        dPa, dc, prms, sc, rr = solve_dir(G[:, ACT], S, bt['lx'], bt['ly'], bt['wb'], c, lam_rel, tau, None if Jx is None else Jx[:, ACT], rx,
                                       None if Qv is None else (Qv[0][ACT][:, ACT], Qv[1][ACT], Qv[2]), W_OLD)
        dP = jnp.zeros(NPAR, dPa.dtype).at[ACT].set(dPa)
    del G, Jx, Qv
    dc = float(dc)
    ab = make_batch(S_tr, B_ACC, k2, WBAR)              # fresh stratified acceptance batch
    L_old, _, _ = batch_loss(theta, c, ab)
    cand = theta + dP.astype(theta.dtype)
    L_new, _, _ = batch_loss(cand, c + dc, ab)
    dmean, dse = strat_mean_se(L_new - L_old, ab['g'], ab['ng']); dmean *= W_OLD; dse *= W_OLD
    dv_fix = None
    if FIX is not None:                                # F2-1: deterministic paired change of L_v on the fixed set (no sampling error, no fresh batch)
        S_cand = fwd(cand, FIX['cfg']); d_o = d_fixed(S_fix, FIX); d_n = d_fixed(S_cand, FIX)
        dv_fix = float(jnp.mean(d_n * d_n - d_o * d_o)); dmean += LAMV * dv_fix
    elif LAMV > 0:                                       # F2: paired L_v change on a fresh bond batch; error bars added in quadrature
        va = VS_acc.draw(B_VACC)
        d_o = v_resid(Sv_of(theta, va), va); d_n = v_resid(Sv_of(cand, va), va); dd = np.asarray(d_n * d_n - d_o * d_o)
        dmean += LAMV * float(dd.mean()); dse = math.sqrt(dse ** 2 + LAMV ** 2 * float(dd.var(ddof=1)) / dd.size)
    lam_used = lam_rel
    if not np.isfinite(dmean) or dmean > Z_ACC * dse:
        ok = False; verdict = 'worse'; lam_rel = min(lam_rel * 4, LAM_CAP)
    elif dmean < -Z_ACC * dse:
        ok = True; verdict = 'better'; lam_rel = max(lam_rel / 2, LAM_MIN)
    else:
        ok = True; verdict = 'noise'
    if ok:
        theta = cand; c = c + dc
        if FIX is not None: S_fix = S_cand
    hist.append(dict(step=n, ok=ok, verdict=verdict, lam_rel=lam_used, lam_next=lam_rel, L_acc_old=float(jnp.mean(L_old)),
                     dmean=dmean, dse=dse, dv_fix=dv_fix, pred_rms=float(prms), trust_scale=float(sc), tau=tau, loss_batch=float(rr), c=c,
                     sec=time.time() - ts))
    if n % HO_EVERY == 0:
        Lh, ex, db = batch_loss(theta, c, HOB)
        Lv_h, _ = lv_value(theta, VHO)
        Lv2 = lv_fixed(fwd(theta, VHO2['cfg']), VHO2) if VHO2 is not None else None
        curve.append(dict(step=n, t=time.time() - t0, L_ho=W_OLD * float(jnp.mean(Lh)) + LAMV * Lv_h, L_ho_old=float(jnp.mean(Lh)), L_v_ho=Lv_h, L_v_ho2=Lv2,
                          L_v_fix=(lv_fixed(S_fix, FIX) if FIX is not None else None), c=c, by_decade=decade_diag(HOB, ex, db)))
        acc10 = np.mean([h['ok'] for h in hist[-HO_EVERY:]])
        d11 = curve[-1]['by_decade'].get('-11', {}).get('rms_e')
        progress(f'  {ARM} step {n}: held-out {curve[-1]["L_ho"]:.4e} (old {curve[-1]["L_ho_old"]:.4e}, L_v {curve[-1]["L_v_ho"]:.4e}, L_v big-ho {curve[-1]["L_v_ho2"]}, L_v fixed-train {curve[-1]["L_v_fix"]})  rms_e(-11) {d11}  acc {acc10:.2f}  lam_rel {lam_rel:.1e}  '
                 f'tau {tau:.4f} trust {float(sc):.2f}  {hist[-1]["sec"]:.1f}s/step')
        if tau > TAU_MIN and n - tau_changed >= ANNEAL_STEPS:          # T1s-2 anneal
            cb = [q for q in curve if q['step'] <= n - ANNEAL_STEPS]
            if cb and abs(cb[-1]['L_ho'] - curve[-1]['L_ho']) / curve[-1]['L_ho'] < FLAT_TOL:
                tau = max(tau / 2, TAU_MIN); tau_changed = n; tau_hist.append(dict(step=n, tau=tau))
                progress(f'  {ARM} step {n}: held-out flat over {ANNEAL_STEPS} steps -> tau {tau:.4f}')
        if n >= MIN_STEPS:
            conv = check_converged()
            if conv['converged']: break
        res['fit'] = dict(hist=hist, curve=curve, tau_hist=tau_hist); dump(F, res)
if not conv['converged']:
    conv = check_converged() if len(hist) >= max(2 * HO_EVERY, 2) and len(curve) > 1 else dict(converged=False, reason='too few steps')
    if conv.get('stalled'):
        conv['reason'] = f'stalled (fit ended after {n} steps, {time.time() - t0:.0f}s of {FIT_SEC:.0f}s): ' + conv['reason']
    else:
        conv['reason'] = f'out of budget ({n} steps, {time.time() - t0:.0f}s of {FIT_SEC:.0f}s): ' + conv['reason']
    conv['converged'] = False
conv['steps'] = n; conv['fit_sec'] = time.time() - t0
conv['acceptance_all'] = float(np.mean([h['ok'] for h in hist])) if hist else None
conv['lam_rel_at_cap_steps'] = int(sum(h['lam_rel'] >= LAM_CAP for h in hist))
res['fit'] = dict(hist=hist, curve=curve, tau_hist=tau_hist); res['converged'] = conv; dump(F, res)
np.save(os.path.join(OUT, 'theta.npy'), np.asarray(theta)); np.save(os.path.join(OUT, 'offset_c.npy'), np.asarray(c))
progress(f'[fit {ARM}] converged {conv["converged"]}: {conv["reason"]}')

if RATE_ONLY:
    sp_ = [h['sec'] for h in hist]; res['rate'] = dict(steps=n, median_sec_per_step=float(np.median(sp_)), mean_sec_per_step=float(np.mean(sp_)),
                                                         median_excl_first=float(np.median(sp_[1:])) if len(sp_) > 1 else None)
    res['sec'] = time.time() - T00; res['done'] = True; dump(F, res); progress(f'DONE rate_only {res["rate"]}'); sys.exit(0)

# ============================================================================================ readout
t1 = time.time()
if DRY:
    nd = int(SPEC.get('dry_table_n', 400))
    idx_s = np.unique(np.concatenate([HOB['idx'], S_tr.draw(nd)[0]]))
    a_fit = fwd(theta, sec.reps[jnp.asarray(idx_s)]); a_0 = fwd(theta0, sec.reps[jnp.asarray(idx_s)])
    off = jnp.mean(WARM[jnp.asarray(idx_s)] - a_0)
    la_fit = WARM.at[jnp.asarray(idx_s)].set(a_fit + off)
    res['dry_splice_orbits'] = int(idx_s.size)
    np.save(os.path.join(OUT, 'la_table.npy'), np.asarray(la_fit, np.float64))     # dry: lets an O2b dry run chain rounds
else:
    la_fit = table(theta)
    np.save(os.path.join(OUT, 'la_table.npy'), np.asarray(la_fit, np.float64))
res['table_sec'] = time.time() - t1
rd = decade_table(la_fit)
res['readout'] = dict(rms_by_decade=rd['by_decade'], offset_count=rd['offset_count'], eps_median_tail_T0=rd['eps_median_tail_T0'])
res['readout']['primary_table'] = {str(d): rd['by_decade'][str(d)]['rms_count'] for d in DECS}
dump(F, res)
progress(f'[readout {ARM}] rms_count by decade ' + ' '.join(f'{d}:{rd["by_decade"][str(d)]["rms_count"]:.4f}' for d in DECS))
nbc = int(SPEC.get('dry_bond_chunks', 2)) if DRY else None
res['readout']['bond_error'] = bond_stats(la_fit, nbc); res['readout']['bond_error_warm'] = bond_stats(WARM, nbc); dump(F, res)
if ARM == 'vit_nostrat':                               # control 1: next to the refresh fit's profile (|phi*|^2 convention)
    rv = json.load(open(os.path.join(RF, 'vit', 'cx_reanchor.json')))['iter']['ptwise_rms_err_by_decade']
    res['control1'] = dict(this_rms_T0={str(d): rd['by_decade'][str(d)]['rms_T0'] for d in DECS if str(d) in rd['by_decade']},
                           refresh_fit_rms_T0={str(d): rv.get(str(d)) for d in DECS})
    dump(F, res)
sec.reload()
mx = int(SPEC.get('dry_fn_maxit', 0)) if DRY else 400
if mx > 0:
    info = efn(norm_la(la_fit), mx)
    res['E_FN'] = info['dE_FN_site']; res['E_FN_info'] = info
    progress(f'[E_FN {ARM}] {info["dE_FN_site"]:.4e}  (lobpcg it {info["lobpcg_it"]}, res {info["res"]:.1e})')
    # splice: fitted log-amplitudes on decades >= SPLICE_DEC (offset aligned to phi* there, |phi*|^2-weighted), exact phi* below
    lf = norm_la(la_fit); msk = dec >= SPLICE_DEC
    off = float(jnp.sum(jnp.where(msk, PT * (lf - ls2), 0.0)) / jnp.sum(jnp.where(msk, PT, 0.0)))
    la_sp = jnp.where(msk, lf - off, ls2); del lf
    info = efn(la_sp, mx); del la_sp
    res['E_FN_splice'] = info['dE_FN_site']; res['E_FN_splice_info'] = dict(info, splice_dec=SPLICE_DEC, offset=off)
    progress(f'[E_FN splice {ARM}] fitted >= {SPLICE_DEC}, exact below: {info["dE_FN_site"]:.4e}  (lobpcg it {info["lobpcg_it"]})')
res['sec'] = time.time() - T00; res['done'] = True; dump(F, res); progress(f'DONE {res["sec"]:.0f}s')
