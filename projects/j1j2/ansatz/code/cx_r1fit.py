"""R1 fit worker (PLAN_pfaffian.md "R1") = a COPY of cx_c1fit.py (sha e72fe1df...; that file is unchanged) with:
  arm 'rbmpp': A(x) = exp(R(x)) * sum_{576 images g} parity x det F[Uo, Do], F = f complex 36x36 (theta = [Re f 1296, Im f 1296, RBM]),
      NO backflow; R(x) = sum_g sum_{a=1..rbm_alpha} ln cosh(b_a + sum_i W_a(i) z_g(i)), real symmetric RBM over the same IMG/FLP images,
      RBM block (rbm_alpha, 37) = [W_a (36) | b_a]; ln|A| = R + ln|sum ...|.  Warm theta of the complex copy (3888 = [Re f, Re u, Im f, Im u]):
      Re f = w[:1296], Im f = w[1944:3240], W = b = 0 (no Im noise).  A theta of size NPAR is taken as is.
  sign: the target sign is read from spec key 'sign_path' (required; replaces refresh/s2.npy in bonds_of, bond_chunk, PT, efn, t_real).
  skip_efn: no E_FN / splice readout at the end (R1 does them on CPU).  self_test: CPU checks (a) W = 0 constant, (b) Jacobian vs central
      finite differences, (w) warm ln|A| vs check_table, then exit 0 (no fit).  break_rbm_grad: TEST ONLY (stop_gradient on R).
Original docstring of cx_c1fit.py follows.
Test C (complex vs real PP) fit worker = cx_f1fit.py (sha 6d51ccbe...) + options --complex / --warm_theta / --im_sigma / --seed / t_real.npy (see C1 block markers).
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
# ---- C1: command-line options override the spec keys: --complex  --warm_theta PATH  --im_sigma S  --seed N
_a = sys.argv[3:]; _i = 0
while _i < len(_a):
    if _a[_i] == '--complex': SPEC['complex'] = True; _i += 1
    elif _a[_i] in ('--warm_theta', '--im_sigma', '--seed'):
        SPEC[_a[_i][2:]] = (float(_a[_i + 1]) if _a[_i] == '--im_sigma' else int(_a[_i + 1]) if _a[_i] == '--seed' else _a[_i + 1]); _i += 2
    else: raise SystemExit(f'unknown option {_a[_i]}')
COMPLEX = bool(SPEC.get('complex', False)); IM_SIGMA = float(SPEC.get('im_sigma', 0.0))
F = os.path.join(OUT, 'cx_t1s.json'); PROG = os.path.join(OUT, 'progress.txt'); T00 = time.time()
ARM = SPEC['arm']; assert ARM in ('vit', 'vit_nostrat', 'pp', 'ppbf', 'rbmpp'), ARM
STRAT = ARM != 'vit_nostrat'; MODEL = 'vit' if ARM.startswith('vit') else ARM
DRY = bool(SPEC.get('dry', False)); SEED = int(SPEC.get('seed', 0))
DECS = list(range(-5, -14, -1)); ND = len(DECS)
B = int(SPEC.get('B', 2048)); K = int(SPEC.get('K', 2))
B_ACC = int(SPEC.get('B_acc', B)); B_HO = int(SPEC.get('B_ho', 4096))
LAM_B = float(SPEC.get('lam_b', 1.0)); FIT_SEC = float(SPEC.get('fit_sec', 1800)); MAX_STEPS = int(SPEC.get('max_steps', 10 ** 9))
LAM0 = float(SPEC.get('lam0', 1e-2)); LAM_CAP = float(SPEC.get('lam_cap', 1.0)); LAM_MIN = float(SPEC.get('lam_floor', 1e-6))   # R3-cont: lam_floor (default 1e-6 = old constant)
assert 0 < LAM_MIN <= LAM_CAP, (LAM_MIN, LAM_CAP)
TAU = float(SPEC.get('tau', 0.05)); Z_ACC = float(SPEC.get('z_acc', 2.0))
# T1s-2 (README "Test T1s-2"): tau anneal TAU -> TAU_MIN, halved each time the held-out loss is flat (< FLAT_TOL) over
# ANNEAL_STEPS steps (and >= ANNEAL_STEPS steps since the last change); convergence additionally needs tau == TAU_MIN.
TAU_MIN = float(SPEC.get('tau_min', TAU)); ANNEAL_STEPS = int(SPEC.get('anneal_steps', 20))
WARM_THETA = SPEC.get('warm_theta')                   # T1s-2: start from a previous run's theta.npy
SPLICE_DEC = int(SPEC.get('splice_dec', -9))          # T1s-2: splice E_FN, fitted ratios for decades >= SPLICE_DEC
HO_EVERY = int(SPEC.get('ho_every', 10)); MIN_STEPS = int(SPEC.get('min_steps', 50)); FLAT_TOL = float(SPEC.get('flat_tol', 0.02))
BFW = int(SPEC.get('bf_window', 3)); FREEZE = bool(SPEC.get('freeze_outer', False)); RATE_ONLY = bool(SPEC.get('rate_only', False))
assert BFW in (3, 5), BFW
MIN_MOVE = float(SPEC.get('min_move', 1e-3))           # README amendment T1s-1
SAVE_AT = {int(q) for q in SPEC.get('save_theta_at', [])}   # round 3: steps at which theta / offset c are saved (OUT/theta_step{N}.npy, offset_c_step{N}.npy)
MIN_WINDOW = int(SPEC.get('min_window', 50))          # amendment T1s-1: the window is the last 20% of steps, at least 50
NW = int(SPEC.get('n_wbar', 8192))                    # samples per decade for Wbar_d
T0JSON = SPEC.get('t0json', '/project/theorie/a/A.Otaifi/chatty_compress6/runs/t0_17410220/cx_t0.json')
res = dict(spec=SPEC, arm=ARM, dry=DRY, design=dict(B=B, K=K, B_acc=B_ACC, B_ho=B_HO, lam_b=LAM_B, lam0=LAM0, lam_cap=LAM_CAP, lam_floor=LAM_MIN,
                                                    tau=TAU, tau_min=TAU_MIN, anneal_steps=ANNEAL_STEPS, warm_theta=WARM_THETA, splice_dec=SPLICE_DEC, z_acc=Z_ACC, ho_every=HO_EVERY, min_steps=MIN_STEPS,
                                                    flat_tol=FLAT_TOL, bf_window=BFW, freeze_outer=FREEZE, max_steps=MAX_STEPS, min_move=MIN_MOVE, min_window=MIN_WINDOW, fit_sec=FIT_SEC, decades=DECS, stratified=STRAT))


def progress(msg):
    with open(PROG, 'a') as fh: fh.write(f'{time.strftime("%Y-%m-%dT%H:%M:%S")} PROGRESS {msg}\n')
    log(msg)


# code drift guard (lead, 2026-10-21): the files actually run must be the ones hashed at submission
import hashlib
HERE_ = os.path.dirname(os.path.abspath(__file__))
CODE_FILES = ('cx_r1fit.py', 'cx_lib.py', 'rbmpp_lib.py')
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
SIGN_PATH = SPEC['sign_path']; res['sign_path'] = SIGN_PATH          # R1: target sign (s1), replaces refresh/s2.npy
ST = jnp.asarray(np.load(SIGN_PATH).astype(np.float32))
res['sign_sha256'] = hashlib.sha256(open(SIGN_PATH, 'rb').read()).hexdigest()
assert ST.shape == (sec.D,) and bool(jnp.all(jnp.abs(ST) == 1)), 'sign_path: +-1 per orbit rep expected'
dec, _ = lab.decade_of(ls2)
dec_h = np.asarray(dec); ho_h = np.asarray(lab.HO); n_h = np.asarray(sec.n); ls2_h = np.asarray(ls2)
res['decade_orbits'] = {str(d): dict(train=int(np.sum((dec_h == d) & ~ho_h)), heldout=int(np.sum((dec_h == d) & ho_h)))
                        for d in DECS}
progress(f'[t1s {ARM}] lab + target ready {time.time() - T00:.0f}s  orbits per decade {res["decade_orbits"]}')

# ============================================================================================ models
RBMPP = False
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
    RBMPP = MODEL == 'rbmpp'; NA = int(SPEC.get('rbm_alpha', 4))       # R1
    if RBMPP: COMPLEX = True
    BREAK_R = bool(SPEC.get('break_rbm_grad', False))                   # TEST ONLY
    ND9 = BFW * BFW; HW = BFW // 2
    if COMPLEX and not RBMPP: assert BF and BFW == 3, 'C1: --complex is ppbf window 3 only (theta = [Re 1944, Im 1944])'
    NP_REAL = N * N + 2 * ND9 * N                                              # 1944
    NBRW = jnp.asarray(np.array([[((x + dx) % 6) + 6 * ((y + dy) % 6) for dy in range(-HW, HW + 1) for dx in range(-HW, HW + 1)]
                                 for y in range(6) for x in range(6)], np.int32))
    if BFW == 3: assert bool(jnp.all(NBRW == NBR))
    IN3 = np.array([(dy + HW) * BFW + (dx + HW) for dy in (-1, 0, 1) for dx in (-1, 0, 1)])   # 3x3 slots inside the window layout
    if SPEC.get('break_embed'): IN3 = IN3 + 1 if BFW == 5 else IN3     # TEST ONLY (check that the step-0 held-out loss detects a wrong embedding)
    theta0 = jnp.concatenate([f0.reshape(-1), jnp.zeros(2 * ND9 * N if BF else 0, jnp.float64)])
    if COMPLEX and not RBMPP: theta0 = jnp.concatenate([theta0, jnp.zeros(NP_REAL, jnp.float64)])         # [Re, Im], Im = 0
    if RBMPP: theta0 = jnp.concatenate([f0.reshape(-1), jnp.zeros(N * N + NA * (N + 1), jnp.float64)])   # R1: [Re f, Im f, RBM (NA, 37)]
    FCH, JCH, TCH = 512, 64, 512
    WARM = jnp.asarray(np.load(SPEC['pp_table'])['lP'], jnp.float64)

    # C1 MODEL BEGIN
    def log_one(f, u1, u2, zz):
        """log|psi_PP(z)| and its sign for one image z (rbmpp_lib.make_fns.log_one, real f; + local backflow).
        C1: with complex f, u1, u2 the det is complex: returns ld = ln|det| (real) and sgn = parity factor * det/|det| (complex)."""
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
        if COMPLEX:
            return ld, (1.0 - 2.0 * (inv % 2).astype(jnp.float64)) * sg      # parity factor multiplies the complex det phase
        par = (jnp.where(sg < 0, 1, 0) + inv) % 2
        return ld, 1.0 - 2.0 * par

    def lncosh(x):
        ax = jnp.abs(x); return ax + jnp.log1p(jnp.exp(-2.0 * ax)) - jnp.log(2.0)

    def A_one(th, sigma):
        """(m, S): A = exp(m) * S, S = sum over the 576 images of sgn * exp(ld - m) (real, or complex with --complex).
        R1 rbmpp: m = max ld (stop_gradient) + R(sigma), R = sum_g sum_a ln cosh(b_a + W_a . z_g) (differentiable)."""
        if RBMPP:
            f = (th[:N * N] + 1j * th[N * N:2 * N * N]).reshape(N, N)
            wb = th[2 * N * N:].reshape(NA, N + 1)
            Z = FLP[:, None] * sigma.astype(jnp.float64)[IMG]
            ld, sgn = jax.vmap(lambda zz: log_one(f, None, None, zz))(Z)
            m = jax.lax.stop_gradient(jnp.max(ld))
            Rx = jnp.sum(lncosh(Z @ wb[:, :N].T + wb[:, N][None, :]))
            if BREAK_R: Rx = jax.lax.stop_gradient(Rx)
            return m + Rx, jnp.sum(sgn * jnp.exp(ld - m))
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
    sgn_j = jax.jit(lambda th, Sx: jax.vmap(lambda s_: jnp.sign(jnp.real(A_one(th, s_)[1])))(bits(Sx).astype(jnp.float64)))   # C1: sgn Re A (real arm: sgn A)
    jac_j = jax.jit(jax.vmap(jax.grad(logA_one), in_axes=(None, 0)))
    jac_in = lambda Sx: bits(Sx).astype(jnp.float64)
NPAR = int(theta0.size)
theta_start = theta0
if WARM_THETA:
    _w = np.load(WARM_THETA)
    if RBMPP:                                        # R1: Re f, Im f of the complex copy (3888 = [Re f, Re u, Im f, Im u]); RBM = 0
        if _w.size != NPAR:
            assert _w.size == 2 * (N * N + 2 * 9 * N), (_w.size, NPAR)
            _w = np.concatenate([_w[:N * N], _w[N * N + 2 * 9 * N:2 * N * N + 2 * 9 * N], np.zeros(NA * (N + 1))])
    elif COMPLEX:                                      # C1: the real 1944 theta is Re; Im = s * rms(Re theta) * N(0,1), seed IM_SEED
        assert _w.size == NP_REAL, (_w.size, NP_REAL)
        _rng = np.random.default_rng(int(SPEC.get('im_seed', SPEC.get('seed', 0))))
        _w = np.concatenate([_w, IM_SIGMA * float(np.sqrt(np.mean(_w ** 2))) * _rng.standard_normal(NP_REAL)])
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
res['complex'] = COMPLEX; res['im_sigma'] = IM_SIGMA; res['bf_window'] = BFW; res['freeze_outer'] = FREEZE; res['n_active'] = int(ACT.size) if ACT is not None else NPAR
res['npar'] = NPAR; res['npar_note'] = ('ViT (cx_reanchor)' if MODEL == 'vit' else
                                       f'f 36x36 ({N * N}, diagonal unused: {N * N - N} active)' + (f' + backflow u1,u2 2x9x36 ({2 * 9 * N})' if MODEL == 'ppbf' else f' + Im f ({N * N}) + RBM alpha {NA} x 37 ({NA * (N + 1)})' if MODEL == 'rbmpp' else ''))
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


# ============================================================================================ R1 self test (CPU checks; exits)
if SPEC.get('self_test'):
    assert RBMPP
    _r = np.random.default_rng(int(SPEC.get('self_test_seed', 7))); st = dict(break_rbm_grad=BREAK_R)
    _ix = np.sort(_r.choice(int(sec.D), 1000, replace=False)); _X = sec.reps[jnp.asarray(_ix)]
    _th0 = np.asarray(theta_start, np.float64); _o = 2 * N * N
    st['warm_rbm_block_zero'] = bool(np.all(_th0[_o:] == 0))
    _l0 = np.asarray(fwd(theta_start, _X))
    _lc = lambda x: np.abs(x) + np.log1p(np.exp(-2 * np.abs(x))) - np.log(2.0)
    # (a) W = 0, b random: ln|A| - ln|A|(b = 0) must equal 576 sum_a ln cosh(b_a) on every rep
    _b = 0.7 * _r.standard_normal(NA); _wb = np.zeros((NA, N + 1)); _wb[:, N] = _b
    _thb = _th0.copy(); _thb[_o:] = _wb.ravel(); _d = np.asarray(fwd(jnp.asarray(_thb), _X)) - _l0
    _ex = 576 * float(np.sum(_lc(_b)))
    st['a'] = dict(n_reps=int(_ix.size), images=int(IMG.shape[0]), b=_b.tolist(), expect=_ex, max_abs_dev=float(np.max(np.abs(_d - _ex))), spread=float(_d.max() - _d.min()))
    # (a) control: W != 0 must NOT give a constant
    _wb2 = 0.05 * _r.standard_normal((NA, N + 1)); _thw = _th0.copy(); _thw[_o:] = _wb2.ravel()
    _d2 = np.asarray(fwd(jnp.asarray(_thw), _X)) - _l0
    st['a_control_W_nonzero_spread'] = float(_d2.max() - _d2.min())
    st['a_pass'] = bool(st['a']['max_abs_dev'] < 1e-8 * max(1.0, abs(_ex)) and st['a_control_W_nonzero_spread'] > 1e-3)
    # (w) warm ln|A| (= plain complex PP, R const) vs check_table (complex copy s1 with backflow) on the same 1000 reps
    if SPEC.get('check_table'):
        _t = np.load(SPEC['check_table'])[_ix]; _dd = _l0 - _t; _dd = _dd - _dd.mean()
        st['warm_vs_table'] = dict(table=SPEC['check_table'], rms=float(np.sqrt(np.mean(_dd ** 2))), max_abs=float(np.max(np.abs(_dd))), offset_removed='mean')
    # (b) Jacobian vs central finite differences, 10 f (5 Re, 5 Im, off-diagonal) + 10 RBM parameters, at theta = _thw (W, b != 0), 5 reps
    _X5 = _X[:5]; _G = np.asarray(jac(jnp.asarray(_thw), _X5))
    _offd = [i * N + j for i in range(N) for j in range(N) if i != j]
    _pf = list(_r.choice(_offd, 5, replace=False)) + [N * N + q for q in _r.choice(_offd, 5, replace=False)]
    _pr = list(_o + _r.choice(NA * (N + 1), 10, replace=False)); _P = [int(q) for q in _pf + _pr]
    _h = float(SPEC.get('fd_h', 1e-6)); _rows = []
    for q in _P:
        _tp = _thw.copy(); _tp[q] += _h; _tm = _thw.copy(); _tm[q] -= _h
        _fd = (np.asarray(fwd(jnp.asarray(_tp), _X5, 5)) - np.asarray(fwd(jnp.asarray(_tm), _X5, 5))) / (2 * _h)
        _rows.append((q, _G[:, q], _fd))
    _rel = [float(np.max(np.abs(g - fd) / np.maximum(np.abs(g), 1e-6))) for _, g, fd in _rows]
    _sc = float(max(np.max(np.abs(g)) for _, g, _f in _rows))
    st['b'] = dict(params=_P, kind=['Re f'] * 5 + ['Im f'] * 5 + ['RBM'] * 10, h=_h, max_rel_dev_per_param=_rel,
                   max_rel_dev=float(max(_rel)), max_abs_dev=float(max(np.max(np.abs(g - fd)) for _, g, fd in _rows)), scale_max_abs_J=_sc,
                   min_abs_J=float(min(np.min(np.abs(g)) for _, g, _f in _rows)))
    st['b']['max_rel_dev_vs_global_scale'] = st['b']['max_abs_dev'] / _sc
    st['b_pass'] = bool(st['b']['max_rel_dev'] < 1e-5)
    json.dump(st, open(os.path.join(OUT, 'self_test.json'), 'w'), indent=1, default=float)
    progress(f'[self_test] a_pass {st["a_pass"]} (dev {st["a"]["max_abs_dev"]:.2e}, W!=0 spread {st["a_control_W_nonzero_spread"]:.3f})  b_pass {st["b_pass"]} '
             f'(max rel {st["b"]["max_rel_dev"]:.2e})  warm_vs_table {st.get("warm_vs_table", {}).get("rms")}')
    sys.exit(0)


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
    bad = v & (ST[idx][:, None] * ST[iy] > 0)
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


# ============================================================================================ GN / minSR direction
@jax.jit
def solve_dir(G, S, lx, ly, wb, c, lam_rel, tau):
    """rows: value rows sv e_x (+ offset column -sv) and bond rows sb_x (e_x - e_yk); G = d log A / d theta on [x; y].
    Sample space (minSR): T = A G G^T A^T + cc cc^T with the 2-sparse row combination A applied by gathers."""
    M = lx.shape[0]; P = G.shape[1]; dt = G.dtype
    ex = S[:M] - lx - c; ey = S[M:].reshape(M, -1) - ly - c; Kb = ey.shape[1]
    sv = 1.0 / jnp.sqrt(M); sb = jnp.sqrt(wb / M)
    r = jnp.concatenate([sv * ex, (sb[:, None] * (ex[:, None] - ey)).reshape(-1)])
    cc = jnp.concatenate([jnp.full(M, -sv, f64), jnp.zeros(M * Kb, f64)])
    R = M * (1 + Kb)
    Gx = G[:M]
    if R <= P + 1:                                        # minSR, sample space: J = A G never materialised
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
        Sm = Jf.T @ Jf; g = Jf.T @ r
        lam = lam_rel * jnp.trace(Sm) / (P + 1)
        d = -jnp.linalg.solve(Sm + lam * jnp.eye(P + 1, dtype=f64), g)
        dP = d[:P]; dc = d[P]
    pc = (Gx.astype(f64) @ dP); pc = pc - jnp.mean(pc)
    prms = jnp.sqrt(jnp.mean(pc * pc))
    sc = jnp.minimum(1.0, tau / jnp.maximum(prms, 1e-30))
    return dP * sc, dc * sc, prms, sc, jnp.sum(r * r)


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
    bd = vv & (ST[ii][:, None] * ST[jj] > 0) & real[:, None]
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


PT = sec.vec(ls2, ST) ** 2
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
    _, u, info = sec.fn_solve(la_b, ST, maxit=maxit); del u
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
# R3-cont: ONE fixed batch from the TRAINING orbits, same size/construction as HOB; own sampler (seed 7000+SEED) and own key (fold_in), so the
# training sampler and the key stream of the fit loop are untouched.  L_tr is evaluated wherever L_ho is, with the same batch_loss and offset c.
TRB = make_batch(Sampler(~ho_h, 7000 + SEED, STRAT), B_HO, jax.random.fold_in(key, 777), WBAR)
# warm network vs warm table on the held-out sample (control 2, second half)
Sw = fwd(theta0, HOB['x']); dw = np.asarray(Sw - WARM[jnp.asarray(HOB['idx'])]); dw = dw - dw.mean()
res['control2']['warm_net_vs_table_rms'] = float(np.sqrt(np.mean(dw ** 2)))
res['control2']['warm_net_vs_table_max'] = float(np.max(np.abs(dw)))
c = float(np.mean(np.asarray(Sw) - np.asarray(HOB['lx'])))          # offset start: mean over the stratified held-out batch
theta = theta_start
if WARM_THETA:
    c = float(np.mean(np.asarray(fwd(theta, HOB['x'])) - np.asarray(HOB['lx'])))
L_ho0, ex, db = batch_loss(theta, c, HOB); L_ho0 = float(jnp.mean(L_ho0))
L_tr0 = float(jnp.mean(batch_loss(theta, c, TRB)[0]))
curve = [dict(step=0, t=0.0, L_ho=L_ho0, L_tr=L_tr0, c=c, by_decade=decade_diag(HOB, ex, db))]
progress(f'[fit {ARM}] start held-out loss {L_ho0:.4e}  train(fixed batch) loss {L_tr0:.4e}  warm net vs table rms {res["control2"]["warm_net_vs_table_rms"]:.2e}')
tau = TAU; tau_changed = 0; tau_hist = [dict(step=0, tau=tau)]
if DRY: res['dense_check_maxdiff'] = dense_check(); progress(f'[dry] gather-vs-dense T check {res["dense_check_maxdiff"]}')
hist = []; lam_rel = max(LAM0, LAM_MIN); n = 0; t0 = time.time(); conv = dict(converged=False, reason='not checked')
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
    if ACT is None:
        dP, dc, prms, sc, rr = solve_dir(G, S, bt['lx'], bt['ly'], bt['wb'], c, lam_rel, tau)
    else:                                              # F1 freeze_outer: the step on the active columns only (= the window-3 step)
        dPa, dc, prms, sc, rr = solve_dir(G[:, ACT], S, bt['lx'], bt['ly'], bt['wb'], c, lam_rel, tau)
        dP = jnp.zeros(NPAR, dPa.dtype).at[ACT].set(dPa)
    del G
    dc = float(dc)
    ab = make_batch(S_tr, B_ACC, k2, WBAR)              # fresh stratified acceptance batch
    L_old, _, _ = batch_loss(theta, c, ab)
    cand = theta + dP.astype(theta.dtype)
    L_new, _, _ = batch_loss(cand, c + dc, ab)
    dmean, dse = strat_mean_se(L_new - L_old, ab['g'], ab['ng'])
    lam_used = lam_rel
    if not np.isfinite(dmean) or dmean > Z_ACC * dse:
        ok = False; verdict = 'worse'; lam_rel = min(lam_rel * 4, LAM_CAP)
    elif dmean < -Z_ACC * dse:
        ok = True; verdict = 'better'; lam_rel = max(lam_rel / 2, LAM_MIN)
    else:
        ok = True; verdict = 'noise'
    if ok: theta = cand; c = c + dc
    if n in SAVE_AT: np.save(os.path.join(OUT, f'theta_step{n}.npy'), np.asarray(theta)); np.save(os.path.join(OUT, f'offset_c_step{n}.npy'), np.asarray(c))
    hist.append(dict(step=n, ok=ok, verdict=verdict, lam_rel=lam_used, lam_next=lam_rel, L_acc_old=float(jnp.mean(L_old)),
                     dmean=dmean, dse=dse, pred_rms=float(prms), trust_scale=float(sc), tau=tau, loss_batch=float(rr), c=c,
                     sec=time.time() - ts))
    if n % HO_EVERY == 0:
        Lh, ex, db = batch_loss(theta, c, HOB)
        Ltr = float(jnp.mean(batch_loss(theta, c, TRB)[0]))
        curve.append(dict(step=n, t=time.time() - t0, L_ho=float(jnp.mean(Lh)), L_tr=Ltr, c=c, by_decade=decade_diag(HOB, ex, db)))
        acc10 = np.mean([h['ok'] for h in hist[-HO_EVERY:]])
        d11 = curve[-1]['by_decade'].get('-11', {}).get('rms_e')
        progress(f'  {ARM} step {n}: held-out {curve[-1]["L_ho"]:.4e}  train {curve[-1]["L_tr"]:.4e}  rms_e(-11) {d11}  acc {acc10:.2f}  lam_rel {lam_rel:.1e}  '
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
if MODEL == 'ppbf' and not COMPLEX:                    # C1: t = sgn(Re A) * s2 per orbit rep (real arm only), int8
    _sg = []; _reps = sec.reps if not DRY else sec.reps[jnp.asarray(idx_s)]
    for _i in range(0, _reps.shape[0], TCH):
        _s, _m = _pad(_reps[_i:_i + TCH], TCH); _sg.append(sgn_j(theta, _s)[:_m])
    _sg = np.asarray(jnp.concatenate(_sg)); _s2 = np.asarray(ST) if not DRY else np.asarray(ST)[idx_s]
    np.save(os.path.join(OUT, 't_real.npy'), (_sg * _s2).astype(np.int8))
    if DRY: np.save(os.path.join(OUT, 't_real_idx.npy'), idx_s)
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
if SPEC.get('skip_efn'): mx = 0; res['E_FN_skipped'] = 'skip_efn (R1: E_FN readouts on CPU at s1)'
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
