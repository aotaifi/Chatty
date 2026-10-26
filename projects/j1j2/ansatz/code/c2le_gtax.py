"""C2-L exp form (c2le): the C2-L loop test with the EXPONENTIAL pre-image instead of the additive one.
   python c2le_gtax.py OUT SPEC.json       (theorie A40; tables of sector_ops in SPEC['tabdir'])
Copy of c2l_gtax.py (sha 15c42c58...) with only the pre-image form changed (SPEC "form": "exp"):
  Phi_t   = exp(+t H_J2) phi_0 / |.|  (Lanczos-Krylov, c2le_expm.py, adaptive m: change m -> m+10 < 1e-12 relative)
  readout = exp(-t H_J2)(stored), stored = s_N |Phi_t| (exact store) or s_N e^{f_fit} (fit); tax_set as in C2-L.
  stage 'E0' (spec_c2le0.json, "t_grid"): exact store at each t: W_M(Phi_t), ratio to W_M(phi_0), tail, decade table,
              tax_B(1), tax_B(2) with s_N |Phi_t|; signed control exp(-t H_J2) Phi_t -> E_FN(K_2 phi_0) rel < 1e-6.
  stage 'full' (spec_c2le1_rbm.json, "t_fixed"): C2-L-1 arm (ii) unchanged (positive RBM M 33, 400 steps, warm start A's RBM),
              target log|Phi_t| with s_N; plus the rebuilt state's log-amplitude error by decade next to the fit rms by decade.
Checks (fatal, exit 2 like ctl_fail): X1 round trip exp(-tH)exp(+tH) phi_0 = phi_0 to 1e-10 and NOT for exp(+tH)exp(+tH);
  X2 Krylov convergence at every call (and a capped Krylov run must report non-convergence); X3 signed-Phi control (and a
  readout at 0.9 t must miss it); X4 t = 0 readout of signed phi_0 gives E_FN(K_2 phi_0) = 3.181795e-7 (L0 machinery).
Everything below this line of the docstring is the C2-L description (additive form), kept for reference.
C2-L (PLAN_ansatz.md "C2-L"): does storing the sign-free J2 pre-image cut the loop's refit tax?
Copy of the G-tax driver runs/gate_tax/cx_gtax.py (commit 52e7f76, sha 96b75153...), unchanged up to phi_0 (setup, controls 1-2,
g_0 = K_2(A), phi_0 = FN(g_0), E_FN(K_k phi_0)); then:
  control C0  E_FN(K_2 phi_0) reproduces G-tax 3.1818e-7 to rel 1e-4 (stop exit 2 otherwise);
  operators   V_J1, V_J2 from sector_ops (checks a/b/c), H_J2 = V_J2 + D_J2;
  Phi         = (1 + a H_J2)^{-1} phi_0 (CG), a < 0 on the cond grid {1.5, 2, 3, 5}; a = argmin W_M (Marshall-violating weight);
              W_M, tail weight (per-config p < 1e-9), decade table; exact-Phi control: (1 + a H_J2) Phi gives back E_FN(K_2 phi_0)
              (rel 1e-6), and |Phi| with s_N (no fit) for context;
  fit         the G-tax RBM+PP refit (same steps, B, K, tau, seed, warm start A) with target log|Phi| (floored 1e-15) and the
              Marshall sign s_N in the loss's bond term (PLAN: "Fit ... to log|Phi| (Marshall sign)");
  readout     psi_fit = (1 + a H_J2)(sign e^{f_fit}), sign in {s_N, sign(Phi)};
              tax_B(k) = E_FN(K_k psi_fit) - E_FN(K_k phi_0), k = 1, 2 (signed K_k of G-tax, own sign of psi_fit).
Verdict (PLAN, binding; on s_N, sign(Phi) reported alongside): PASS tax_B(1) < 1.657e-6 and tax_B(2) < 0.82e-6;
KILL tax_B(2) >= 1.657e-6; else INCONCLUSIVE.
"""
import os, sys, time, math, hashlib
from functools import partial
import numpy as np
import scipy.linalg as sl
from c2l_cx_lib import *
import sector_ops as OPS
from c2l_net import make_net, make_rbm_net
from c2le_expm import expm_krylov

OUT = sys.argv[1]; SPEC = json.load(open(sys.argv[2])); os.makedirs(OUT, exist_ok=True)
F = os.path.join(OUT, 'cx_gtax.json'); PROG = os.path.join(OUT, 'progress.txt'); T00 = time.time()
STAGE = SPEC.get('stage', 'full'); STEPS = int(SPEC.get('steps', 400)); SEED = int(SPEC.get('seed', 0))
DECS = list(range(-5, -14, -1)); ND = len(DECS); FLOOR = 1e-15
B = int(SPEC.get('B', 8208)); K = int(SPEC.get('K', 2)); B_ACC = int(SPEC.get('B_acc', B)); B_HO = int(SPEC.get('B_ho', 4096))
LAM_B = 1.0; LAM0 = 1e-2; LAM_CAP = 1.0; LAM_MIN = 1e-6; TAU0 = 0.05; TAU_MIN = 0.01; Z_ACC = 2.0; HO_EVERY = 10; ANNEAL = 20; FLAT_TOL = 0.02
NW = int(SPEC.get('n_wbar', 8192)); FIT_BUDGET = float(SPEC.get('fit_budget_s', 4800))
res = dict(spec=SPEC, stage=STAGE, done=False, design=dict(B=B, K=K, B_acc=B_ACC, B_ho=B_HO, lam0=LAM0, lam_cap=LAM_CAP, tau0=TAU0, tau_min=TAU_MIN,
                                                           anneal_steps=ANNEAL, z_acc=Z_ACC, ho_every=HO_EVERY, steps=STEPS, decades=DECS))


def progress(msg):
    with open(PROG, 'a') as fh: fh.write(f'{time.strftime("%Y-%m-%dT%H:%M:%S")} PROGRESS {msg}\n')
    log(msg)


HERE_ = os.path.dirname(os.path.abspath(__file__))
CODE_FILES = ('c2le_gtax.py', 'c2le_expm.py', 'c2l_net.py','c2l_cx_lib.py', 'c2l_rbmpp_lib.py', 'sector_ops.py', 'st6_sector.py')
SHA_RUN = {fn: (hashlib.sha256(open(os.path.join(HERE_, fn), 'rb').read()).hexdigest() if os.path.exists(os.path.join(HERE_, fn)) else None) for fn in CODE_FILES}
SHA_EXP = SPEC.get('code_sha256') or {}
bad = [fn for fn in CODE_FILES if SHA_EXP.get(fn) != SHA_RUN[fn]]
res['code_sha256'] = dict(expected=SHA_EXP, running=SHA_RUN, dir=HERE_, match=not bad)
if bad:
    res['code_sha256']['mismatch'] = bad; res['stop'] = 'CODE DRIFT'; dump(F, res); progress(f'CODE DRIFT: {bad}; exit 3'); sys.exit(3)
dump(F, res)


def stop(code, msg):
    res['stop'] = msg; res['sec'] = time.time() - T00; dump(F, res); progress(msg); sys.exit(code)


# ---- c2le spec: exp form only; E0 needs t_grid, full needs a numeric t_fixed > 0 (the lead sets it; "TBD" refuses)
if SPEC.get('form') != 'exp': stop(6, f'SPEC ERROR: form {SPEC.get("form")!r} != "exp" (c2le_gtax.py runs the exp form only)')
if STAGE == 'E0':
    T_GRID = [float(t) for t in SPEC.get('t_grid', [])]
    if not T_GRID or min(T_GRID) <= 0: stop(6, f'SPEC ERROR: stage E0 needs t_grid of t > 0, got {SPEC.get("t_grid")!r}')
elif STAGE in ('full', 'control'):
    if not isinstance(SPEC.get('t_fixed'), (int, float)) or not SPEC['t_fixed'] > 0:
        stop(6, f'SPEC ERROR: t_fixed {SPEC.get("t_fixed")!r} is not a number > 0 (set it from the exact-store result)')
    T_FIX = float(SPEC['t_fixed'])
else:
    stop(6, f'SPEC ERROR: stage {STAGE!r} not in E0 / full / control')
KRY = dict(tol=float(SPEC.get('krylov_tol', 1e-12)), m0=10, dm=10, mmax=int(SPEC.get('krylov_mmax', 100)))
res['krylov'] = dict(KRY, calls=[])


lab = Lab(); sec = lab.sec
TA = json.load(open(SPEC['table_A_json']))
REF = dict(EFN_A=TA['guide']['dE_FN_site'], E1_var=TA['lanczos'][1]['dE_site'], EFN_K1=TA['lanczos'][1]['dE_FN_site'], E2_var=TA['lanczos'][2]['dE_site'],
           E_A=TA['E_re'])
res['ref'] = REF; res['im_weight_A'] = TA['im_weight']
if not TA['im_weight'] <= 1e-3: stop(5, f'IM_WEIGHT {TA["im_weight"]} > 1e-3 : stop')
progress(f'[setup] im_weight(A) = {TA["im_weight"]:.3e} (table_A.json, exact over all orbits)')
cA = TA['guide']['c']; THETA_PH = float(np.arctan2(cA[1], cA[0]))
z = np.load(SPEC['table_A']); laA = jnp.asarray(z['lP'], jnp.float64); sA = jnp.asarray(z['sP'].astype(np.float32)); del z


def norm_la(la):
    a = jnp.exp(la - jnp.max(la)); return la - jnp.max(la) - 0.5 * jnp.log(jnp.sum(sec.n * a * a))


laA = norm_la(laA)
ARM = SPEC.get('arm', 'rbmpp')
net = (make_rbm_net(SPEC['params'], M=int(SPEC.get('M', 33)), seed=SEED) if ARM == 'rbm' else make_net(SPEC['params'], THETA_PH)); theta0 = net['theta0']; NPAR = int(theta0.size)
FCH, JCH, TCH = 512, 64, 512
res['npar'] = NPAR; res['arm'] = ARM; res['theta_phase'] = THETA_PH


def _pad(Sx, ch):
    m = Sx.shape[0]
    return (jnp.concatenate([Sx, jnp.repeat(Sx[:1], ch - m)]) if m < ch else Sx), m


def fwd(th, Sx, ch=None):
    ch = ch or FCH; out = []
    for i in range(0, Sx.shape[0], ch):
        s_, m = _pad(Sx[i:i + ch], ch); out.append(net['fwd_j'](th, s_)[:m])
    return jnp.concatenate(out)


@partial(jax.jit, donate_argnums=(0,))
def _put(Gb, blk, i0):
    return jax.lax.dynamic_update_slice(Gb, blk, (i0, 0))


def jac(th, Sx):
    R_ = Sx.shape[0]; Gb = jnp.zeros((R_, NPAR), theta0.dtype)
    for i in range(0, R_, JCH):
        s_, m = _pad(Sx[i:i + JCH], JCH)
        Gb = _put(Gb, net['jac_j'](th, s_)[:m].astype(theta0.dtype), i)
    return Gb


def table(th):
    """all orbits: log|Re|, sign (cos), and the exact im weight sum n e^{2 Re l} sin^2 / sum n e^{2 Re l}."""
    reps = sec.reps; out = []; t0 = time.time(); tl = t0; D_ = reps.shape[0]
    for i in range(0, D_, TCH):
        s_, m = _pad(reps[i:i + TCH], TCH); out.append(net['parts_j'](th, s_)[:m])
        if time.time() - tl > 120:
            jax.block_until_ready(out[-1]); tl = time.time(); progress(f'  table {i + m}/{D_}  {tl - t0:.0f}s')
    P = jnp.concatenate(out); del out
    lre, cs, sn = P[:, 0], P[:, 1], P[:, 2]; del P
    la = lre + jnp.log(jnp.maximum(jnp.abs(cs), 1e-300)); sg = jnp.where(cs >= 0, 1.0, -1.0).astype(jnp.float32)
    w = sec.n * jnp.exp(2 * (lre - jnp.max(lre)))
    imw = float(jnp.sum(w * sn * sn) / jnp.sum(w))
    return la, sg, imw


# ---------------------------------------------------------------- signed Krylov polynomial
def kpoly(la, s, k):
    """signed K_k: lowest Ritz vector of span{v, (H-E)v, ..., (H-E)^k v}; returns (la', s', E_var_site, coefficients)."""
    v = sec.vec(la, s); Hv = sec.H(v); E = float(v @ Hv)
    Bs = [v]; HB = [Hv]; u = v; Hu = Hv
    for j in range(k):
        un = Hu - E * u; Hun = sec.H(un); Bs.append(un); HB.append(Hun); u, Hu = un, Hun
    m = k + 1
    Gm = np.array([[float(Bs[i] @ Bs[j]) for j in range(m)] for i in range(m)])
    Hm = np.array([[float(Bs[i] @ HB[j]) for j in range(m)] for i in range(m)]); Hm = 0.5 * (Hm + Hm.T)
    w_, c = sl.eigh(Hm, Gm); c0 = c[:, 0] / c[0, 0]
    return build_k(Bs, c0), float((w_[0] - sec.E0) / N), c0.tolist()


def build_k(Bs, c):
    w = sum(float(c[i]) * Bs[i] for i in range(len(Bs)))
    return (jnp.log(jnp.maximum(jnp.abs(w), 1e-300) / sec.sqrt_n), jnp.where(w >= 0, 1.0, -1.0).astype(jnp.float32))


def fn_of(la, s, tag, tol=1e-10):
    _, u, info = sec.fn_solve(la, s, tol=tol, maxit=(400 if tol >= 1e-10 else 200))   # tight solve: 200 its cap (res recorded)
    progress(f'[E_FN {tag}] {info["dE_FN_site"]:.6e} (lobpcg {info["lobpcg_it"]}, res {info["res"]:.1e}, neg_w {info["neg_weight"]:.1e}, {info["sec"]:.0f}s)')
    return info['dE_FN_site'], u, info


def ctl_fail(msg): stop(2, 'CONTROL FAILED: ' + msg)


# ================================================================= control 2 (+ guide)
IDX4096 = np.sort(np.random.default_rng(12345).choice(sec.D, 4096, replace=False)); I4 = jnp.asarray(IDX4096)
t0 = time.time()
(la1, s1), E1v, c1 = kpoly(laA, sA, 1)
e1, _, i1 = fn_of(la1, s1, 'K_1(A)')
r_v = abs(E1v - REF['E1_var']) / REF['E1_var']; r_f = abs(e1 - REF['EFN_K1']) / REF['EFN_K1']
res['control2'] = dict(E1_var=E1v, ref_E1_var=REF['E1_var'], rel_var=r_v, EFN_K1=e1, ref_EFN_K1=REF['EFN_K1'], rel_fn=r_f, coeffs=c1, ok=bool(r_v < 1e-6 and r_f < 1e-6))
# deliberately wrong hop (0.9 alpha): the check must fail
vv = sec.vec(laA, sA); Hv_ = sec.H(vv); E_ = float(vv @ Hv_); u1_ = Hv_ - E_ * vv
lw, sw = build_k([vv, u1_], [1.0, 0.9 * c1[1]]); ew, _, _ = fn_of(lw, sw, 'wrong hop 0.9 alpha (must fail)'); del vv, Hv_, u1_
res['control2']['broken_EFN'] = ew; res['control2']['broken_rel'] = abs(ew - REF['EFN_K1']) / REF['EFN_K1']
res['control2']['broken_fails_check'] = bool(res['control2']['broken_rel'] > 1e-6)
dump(F, res)
progress(f'CONTROL 2: K_1(A) variational {E1v:.8e} vs {REF["E1_var"]:.8e} (rel {r_v:.1e}); E_FN {e1:.8e} vs {REF["EFN_K1"]:.8e} (rel {r_f:.1e}); '
         f'broken hop E_FN {ew:.6e} rel {res["control2"]["broken_rel"]:.1e} (fails: {res["control2"]["broken_fails_check"]}) ok={res["control2"]["ok"]}')
if not (res['control2']['ok'] and res['control2']['broken_fails_check']): ctl_fail('control 2')
del la1, s1, lw, sw

# ---------------------------------------------------------------- guide g_0 = K_2(A) and phi_0
(lg0, sg0), E2v, c2 = kpoly(laA, sA, 2)
res['g0'] = dict(E_var=E2v, ref_lanczos_p2=REF['E2_var'], rel_vs_lanczos_p2=abs(E2v - REF['E2_var']) / REF['E2_var'], coeffs=c2,
                 sign_flip_weight_vs_A=float(jnp.sum(jnp.where(sg0 != sA, sec.n * jnp.exp(2 * (lg0 - jnp.max(lg0))), 0.0)) / jnp.sum(sec.n * jnp.exp(2 * (lg0 - jnp.max(lg0))))))
progress(f'[g_0 = K_2(A)] variational {E2v:.8e} (lanczos p=2 {REF["E2_var"]:.8e}, rel {res["g0"]["rel_vs_lanczos_p2"]:.1e}) coeffs {c2}  sign-flip weight vs A {res["g0"]["sign_flip_weight_vs_A"]:.3e}')
if res['g0']['rel_vs_lanczos_p2'] > 1e-6: ctl_fail('K_2(A) variational energy != lanczos p=2')
efn_g0, u, i_g0 = fn_of(lg0, sg0, 'g_0 = phi_0')
res['E_FN_g0'] = efn_g0; res['E_FN_g0_info'] = i_g0
lphi = norm_la(jnp.log(jnp.maximum(u, 1e-300) / sec.sqrt_n)); del u
ls2 = norm_la(jnp.log(jnp.maximum(jnp.exp(lphi), FLOOR))); s2 = sg0; del lphi
res['n_floored'] = int(jnp.sum(jnp.exp(norm_la(ls2)) < FLOOR))
dump(F, res)
efn_p0 = {}
TOL_TIGHT = 1e-12; STAB_TOL = 1e-8                         # c2le amendment 4: tax stable to 1e-8/site under a 100x tighter FN solve
for k in (1, 2):
    (lk, sk), Ev, ck = kpoly(ls2, s2, k); ek, _, ik = fn_of(lk, sk, f'K_{k}(phi_0)')
    ekt, _, ikt = fn_of(lk, sk, f'K_{k}(phi_0) tight tol {TOL_TIGHT:g}', tol=TOL_TIGHT)      # c2le amendment 4
    efn_p0[k] = dict(E_var=Ev, EFN=ek, EFN_tight=ekt, lobpcg_it=ik['lobpcg_it'], lobpcg_it_tight=ikt['lobpcg_it'], coeffs=ck, sign_flip_weight_vs_s_g0=float(jnp.sum(jnp.where(sk != s2, sec.n * jnp.exp(2 * (lk - jnp.max(lk))), 0.0)) / jnp.sum(sec.n * jnp.exp(2 * (lk - jnp.max(lk))))))
    del lk, sk
res['phi0'] = dict(EFN_zero_hop=efn_g0, K1=efn_p0[1], K2=efn_p0[2]); dump(F, res)
progress(f'[phi_0] t_setup {time.time() - t0:.0f}s  E_FN(phi_0) {efn_g0:.6e}  E_FN(K_1 phi_0) {efn_p0[1]["EFN"]:.6e}  E_FN(K_2 phi_0) {efn_p0[2]["EFN"]:.6e}')

# ================================================================= C2-L: control C0, H_J2, Phi
REF_K2_PHI0 = 3.1818e-7
res['C0'] = dict(EFN_K2_phi0=efn_p0[2]['EFN'], ref=REF_K2_PHI0, rel=abs(efn_p0[2]['EFN'] - REF_K2_PHI0) / REF_K2_PHI0)
res['C0']['ok'] = bool(res['C0']['rel'] < 1e-4); dump(F, res)
progress(f'CONTROL C0: E_FN(K_2 phi_0) {efn_p0[2]["EFN"]:.6e} vs G-tax {REF_K2_PHI0:.4e} rel {res["C0"]["rel"]:.1e} ok={res["C0"]["ok"]}')
if not res['C0']['ok']: ctl_fail('C0: phi_0 baseline not reproduced')
TABDIR = SPEC['tabdir']
tabs = {c: OPS.load_or_build(sec, c, TABDIR, log=progress)[0] for c in ('J1', 'J2')}
V1 = OPS.BondOp(sec, tabs['J1'], 'J1'); V2 = OPS.BondOp(sec, tabs['J2'], 'J2'); del tabs['J1']
ck = OPS.checks(sec, lab.D0, V1, V2); del V1
DJ1 = OPS.diag0(sec, 0, len(SS.NN)); DJ2 = OPS.diag0(sec, len(SS.NN))
ck['DJ1+DJ2-D0_maxabs'] = float(jnp.max(jnp.abs(DJ1 + DJ2 - lab.D0))); del DJ1
res['ops_checks'] = ck; dump(F, res)
progress(f'[ops] checks a {max(ck["a_rel"]):.1e} b(mis-scaled) {min(ck["b_rel_misscaled"]):.1e} c {ck["c_sym"]} diag split {ck["DJ1+DJ2-D0_maxabs"]:.1e}')
if not (ck['a_pass'] and ck['b_fails'] and ck['c_pass'] and ck['DJ1+DJ2-D0_maxabs'] < 1e-12): ctl_fail('operator checks')
HJ2 = lambda x: V2(x) + DJ2 * x
MA_ = jnp.uint64(sum(1 << (x + 6 * y) for y in range(6) for x in range(6) if (x + y) % 2 == 0))
sN = jnp.where(jax.lax.population_count(sec.reps & MA_) % 2 == 0, 1.0, -1.0)


def WM(v):
    v = v / jnp.linalg.norm(v); g = jnp.sign(jnp.sum(v * jnp.abs(v) * sN))
    return float(jnp.sum(v ** 2 * ((jnp.sign(v) * g) != sN)))


def by_dec(v):
    """c2le amendment 3: common bins = decade of phi_0's per-config probability |phi_0(x)|^2 (DEC0), for every decade table.
    {decade: [weight of v in the bin, Marshall-violating fraction of v's weight in the bin]}"""
    v = v / jnp.linalg.norm(v); g = jnp.sign(jnp.sum(v * jnp.abs(v) * sN)); bad = (jnp.sign(v) * g) != sN
    d = DEC0
    p = np.asarray(v ** 2); b = np.asarray(bad)
    return {str(k): [float(p[d == k].sum()), float((p * b)[d == k].sum() / max(p[d == k].sum(), 1e-300))]
            for k in range(int(d.max()), max(int(d[p > 0].min()), -20) - 1, -1) if (d == k).any()}


def tail(v):
    v = v / jnp.linalg.norm(v); return float(jnp.sum(jnp.where(v ** 2 / sec.n < 1e-9, v ** 2, 0.0)))


def lanczos_ext(op, m=150):
    q = jax.random.normal(jax.random.PRNGKey(1), (sec.D,)); q = q / jnp.linalg.norm(q); qp = jnp.zeros(sec.D); b = 0.0; al, be = [], []
    for _ in range(m):
        z = op(q) - b * qp; a = float(q @ z); z = z - a * q; b = float(jnp.linalg.norm(z)); al.append(a); be.append(b); qp, q = q, z / b
    ev = np.linalg.eigvalsh(np.diag(al) + np.diag(be[:-1], 1) + np.diag(be[:-1], -1)); return float(ev[0]), float(ev[-1])


def cg(op, b, tol=1e-11, maxit=3000):
    x = jnp.zeros_like(b); r = b; p = r; rs = float(r @ r); nb = float(jnp.linalg.norm(b))
    for it in range(maxit):
        Ap = op(p); al = rs / float(p @ Ap); x = x + al * p; r = r - al * Ap; rn = float(r @ r)
        if np.sqrt(rn) < tol * nb: break
        p = r + (rn / rs) * p; rs = rn
    return x, it + 1, float(jnp.linalg.norm(b - op(x)) / nb)


v_phi0 = sec.vec(ls2, s2)
DEC0 = np.asarray(jnp.floor(jnp.log10(jnp.maximum(v_phi0 ** 2 / sec.n, 1e-300))).astype(jnp.int32))   # common decade bins (|phi_0|^2 per config)
res['decade_bins'] = 'floor(log10 |phi_0(x)|^2), phi_0 normalised (common bins for all decade tables; c2le amendment 3)'
res['diagnostics_label'] = 'W_M, W_M ratios to W_M(phi_0), tails and decade tables of Phi_t are FN-target diagnostics: measured on phi_0 = FN(K_2(A)), not on psi0'
lmin, lmax = lanczos_ext(HJ2); res['HJ2_spectrum'] = [lmin, lmax]
res['W_M_phi0'] = WM(v_phi0); res['tail_phi0'] = tail(v_phi0); res['phi0_by_decade'] = by_dec(v_phi0)
progress(f'[H_J2] spectrum [{lmin:.6f}, {lmax:.6f}]; phi_0: W_M {res["W_M_phi0"]:.6e} tail {res["tail_phi0"]:.4e}')
res['HJ2_t_width'] = {'width': lmax - lmin}


# ================================================================= c2le: exp(+-t H_J2) by Lanczos-Krylov (check X2 on every call)
def expJ2(v, tau, tag, fatal=True, **kw):
    """exp(tau H_J2) v; records m, matvecs, last change; non-convergence is fatal (X2)."""
    t0_ = time.time(); w, inf = expm_krylov(HJ2, v, tau, **dict(dict(tol=KRY['tol'], m0=KRY['m0'], dm=KRY['dm'], mmax=KRY['mmax']), **kw))
    rec_ = dict(tag=tag, tau=float(tau), m=inf['m'], matvecs=inf['matvecs'], change=inf['change'], changes=inf['changes'], converged=inf['converged'],
                breakdown=inf['breakdown'], sec=time.time() - t0_)
    res['krylov']['calls'].append(rec_)
    if fatal and not inf['converged']: ctl_fail(f'X2 Krylov not converged ({tag}, tau {tau}, m {inf["m"]}, change {inf["change"]})')
    return w, rec_


T_READ = 0.0                                                  # readout time of lift(); set before every tax_set


def lift(la, s):
    """exp(-T_READ H_J2)(s e^{la} sqrt n) -> (la', s') of the result (c2le readout)."""
    v = sec.vec(la, s); w, _ = expJ2(v, -T_READ, f'readout t {T_READ:g}')
    return norm_la(jnp.log(jnp.maximum(jnp.abs(w), 1e-300) / sec.sqrt_n)), jnp.where(w >= 0, 1.0, -1.0).astype(jnp.float32)


STAB = []; res['stability'] = STAB                       # amendment 4 records


def tax_set(la, s, tag, keep=False, tight=True, read=True):
    lp, sp = (lift(la, s) if read else (la, s)); out = {}
    e0, _, _ = fn_of(lp, sp, f'{tag} zero hop'); out['EFN_0'] = e0
    for k in (1, 2):
        (lk, sk), Ev, ck_ = kpoly(lp, sp, k); ek, _, _ = fn_of(lk, sk, f'K_{k}({tag})')
        out[f'EFN_K{k}'] = ek; out[f'E_var_K{k}'] = Ev; out[f'tax_B{k}'] = ek - efn_p0[k]['EFN']
        if tight:                                            # amendment 4: same state, FN tol 1e-12; tax must move < 1e-8/site
            ekt, _, ikt = fn_of(lk, sk, f'K_{k}({tag}) tight tol {TOL_TIGHT:g}', tol=TOL_TIGHT); out[f'res_tight_K{k}'] = ikt['res']
            out[f'tax_B{k}_tight'] = ekt - efn_p0[k]['EFN_tight']; out[f'stab_B{k}'] = abs(out[f'tax_B{k}_tight'] - out[f'tax_B{k}'])
        del lk, sk
    if tight:
        out['stable'] = bool(max(out['stab_B1'], out['stab_B2']) < STAB_TOL); STAB.append(dict(tag=tag, d1=out['stab_B1'], d2=out['stab_B2'], ok=out['stable']))
        progress(f'[stability {tag}] |tax_B1 tight - tax_B1| {out["stab_B1"]:.1e} |tax_B2 tight - tax_B2| {out["stab_B2"]:.1e} (< {STAB_TOL:g}/site: {out["stable"]})')
    lp_h = np.asarray(lp) if keep else None
    del lp, sp; return (out, lp_h) if keep else out


def efn_k2(la, s, tag):
    """E_FN(K_2(readout)) only (checks X3-broken and X4)."""
    lp, sp = lift(la, s); (lk, sk), _, _ = kpoly(lp, sp, 2); ek, _, _ = fn_of(lk, sk, tag); return ek


def la_of(v): return norm_la(jnp.log(jnp.maximum(jnp.abs(v), 1e-300) / sec.sqrt_n))


def sg_of(v): return jnp.where(v >= 0, 1.0, -1.0).astype(jnp.float32)


def make_Phi(t):
    """Phi_t = exp(+t H_J2) phi_0 / |.|, with check X1 (round trip to 1e-10; same sign twice must NOT return phi_0)."""
    W, k_fw = expJ2(v_phi0, +t, f'Phi_t forward t {t:g}')
    U, k_bw = expJ2(W, -t, f'X1 back t {t:g}')
    rt = float(jnp.linalg.norm(U - v_phi0)); del U
    Uw, _ = expJ2(W, +t, f'X1 broken same sign t {t:g}')
    Uw = Uw / jnp.linalg.norm(Uw); wr = float(jnp.linalg.norm(Uw * jnp.sign(Uw @ v_phi0) - v_phi0)); del Uw
    x1 = dict(round_trip=rt, same_sign_twice=wr, ok=bool(rt < 1e-10), broken_fails=bool(wr > 1e-3))
    progress(f'CHECK X1 t {t:g}: |exp(-tH)exp(+tH)phi_0 - phi_0| {rt:.2e} (< 1e-10 {x1["ok"]}); same sign twice {wr:.2e} (> 1e-3 {x1["broken_fails"]}); '
             f'Krylov m fwd {k_fw["m"]} back {k_bw["m"]} change {k_fw["change"] or 0:.1e}/{k_bw["change"] or 0:.1e}; |exp(+tH)phi_0| {float(jnp.linalg.norm(W)):.6e}')
    if not (x1['ok'] and x1['broken_fails']): ctl_fail(f'X1 round trip at t {t:g}: {x1}')
    nW = float(jnp.linalg.norm(W)); Phi_ = W / nW; del W
    rec_ = dict(t=t, norm_exp_tH_phi0=nW, krylov_fwd=dict(m=k_fw['m'], change=k_fw['change'], matvecs=k_fw['matvecs']),
                krylov_back=dict(m=k_bw['m'], change=k_bw['change'], matvecs=k_bw['matvecs']), X1=x1,
                W_M=WM(Phi_), tail_w=tail(Phi_), by_decade=by_dec(Phi_))
    rec_['W_M_over_W_M_phi0'] = rec_['W_M'] / res['W_M_phi0']
    return Phi_, rec_


# ---- X2 broken: a capped Krylov run (m <= 4) at the largest t must report non-convergence
t_big = max(T_GRID) if STAGE == 'E0' else T_FIX
_, kb = expJ2(v_phi0, t_big, 'X2 broken: m capped at 4', fatal=False, m0=2, dm=2, mmax=4)
res['X2_broken'] = dict(t=t_big, converged=kb['converged'], change=kb['change'], fails=bool(not kb['converged'])); dump(F, res)
progress(f'CHECK X2 broken: Krylov capped at m 4, t {t_big:g}: converged {kb["converged"]} change {kb["change"] or 0:.2e} (must not converge: {res["X2_broken"]["fails"]})')
if not res['X2_broken']['fails']: ctl_fail('X2 broken Krylov (m <= 4) reported convergence')
# ---- X4: t = 0 readout of the signed phi_0 must give the L0 machinery's E_FN(K_2 phi_0) = 3.181795e-7
REF_L0_K2 = 3.181795e-7                                      # run_17466418 progress.txt [E_FN K_2(phi_0)]
Phi0_, _ = expJ2(v_phi0, 0.0, 'X4 forward t 0'); T_READ = 0.0
e4 = efn_k2(la_of(Phi0_), sg_of(Phi0_), 'X4 t 0 readout of signed phi_0, K_2'); del Phi0_
res['X4'] = dict(EFN_K2=e4, EFN_K2_phi0=efn_p0[2]['EFN'], rel_vs_phi0=abs(e4 - efn_p0[2]['EFN']) / efn_p0[2]['EFN'], ref_L0=REF_L0_K2, rel_vs_L0=abs(e4 - REF_L0_K2) / REF_L0_K2)
res['X4']['ok'] = bool(res['X4']['rel_vs_phi0'] < 1e-6 and res['X4']['rel_vs_L0'] < 1e-5); dump(F, res)
progress(f'CHECK X4: t 0 readout E_FN(K_2) {e4:.6e} vs this run {efn_p0[2]["EFN"]:.6e} (rel {res["X4"]["rel_vs_phi0"]:.1e} < 1e-6) vs L0 {REF_L0_K2:.6e} (rel {res["X4"]["rel_vs_L0"]:.1e} < 1e-5) ok={res["X4"]["ok"]}')
if not res['X4']['ok']: ctl_fail('X4 t = 0 readout')


def x3_signed(Phi_, t, rec_, broken=False):
    """X3: exp(-t H_J2) Phi_t (signed, exact) must give back E_FN(K_2 phi_0) to rel < 1e-6; broken: readout at 0.9 t must miss."""
    global T_READ
    T_READ = t; ts = tax_set(la_of(Phi_), sg_of(Phi_), f'exact Phi_t signed, t {t:g}', tight=False)
    r = abs(ts['EFN_K2'] - efn_p0[2]['EFN']) / efn_p0[2]['EFN']; rec_['signed_Phi'] = dict(ts, rel_K2=r, ok=bool(r < 1e-6))
    progress(f'CHECK X3 t {t:g}: E_FN(K_2 exp(-tH) Phi_t) {ts["EFN_K2"]:.6e} vs E_FN(K_2 phi_0) {efn_p0[2]["EFN"]:.6e} rel {r:.1e} (< 1e-6)')
    if broken:
        T_READ = 0.9 * t; eb = efn_k2(la_of(Phi_), sg_of(Phi_), f'X3 broken: readout at 0.9 t {t:g}, K_2'); T_READ = t
        rb = abs(eb - efn_p0[2]['EFN']) / efn_p0[2]['EFN']; rec_['signed_Phi']['broken_0p9t'] = dict(EFN_K2=eb, rel=rb, fails=bool(rb > 1e-6))
        progress(f'CHECK X3 broken (readout at 0.9 t): E_FN(K_2) {eb:.6e} rel {rb:.1e} (must be > 1e-6: {rb > 1e-6})')
        if not rb > 1e-6: ctl_fail('X3 broken readout (0.9 t) reproduced phi_0')
    if not r < 1e-6: ctl_fail(f'X3 signed-Phi control at t {t:g}')


def s_N_of(Phi_):
    g_ = float(jnp.sign(jnp.sum(Phi_ * sN))); return (g_ * sN).astype(jnp.float32)


# ================================================================= stage E0: exact store s_N |Phi_t| on the t grid, no fit
if STAGE == 'E0':
    res['E0'] = []
    for i_t, t in enumerate(T_GRID):
        Phi, rec = make_Phi(t); dump(F, res)
        progress(f'[Phi_t] t {t:g} (FN-target diagnostics): W_M {rec["W_M"]:.4e} (phi_0 {res["W_M_phi0"]:.4e}, ratio {rec["W_M_over_W_M_phi0"]:.3e}) tail {rec["tail_w"]:.4e} (phi_0 {res["tail_phi0"]:.4e})')
        x3_signed(Phi, t, rec, broken=(i_t == 0)); dump(F, res)
        T_READ = t; rec.update(tax_set(la_of(Phi), s_N_of(Phi), f'E0 t {t:g} exact |Phi_t| s_N')); del Phi
        res['E0'].append(rec); dump(F, res)
        progress(f'[E0] t {t:g}: W_M ratio {rec["W_M_over_W_M_phi0"]:.3e} tail {rec["tail_w"]:.4f} tax_B1 {rec["tax_B1"]:.4e} tax_B2 {rec["tax_B2"]:.4e}')
    # amendment 1 (lead, before any run): fit t = the SMALLEST t with exact-store tax_B1 < 1.657e-6 AND tax_B2 < 0.82e-6;
    # if none passes both, the t with the lowest tax_B2; KILL (close the route) if that lowest tax_B2 >= 1.657e-6
    m2 = min(r['tax_B2'] for r in res['E0']); res['min_tax_B2_E0'] = m2
    both = sorted([r for r in res['E0'] if r['tax_B1'] < 1.657e-6 and r['tax_B2'] < 0.82e-6], key=lambda r: r['t'])
    if both: res['t_fit'] = both[0]['t']; res['t_fit_rule'] = 'smallest t passing both exact-store bars'
    else: res['t_fit'] = min(res['E0'], key=lambda r: r['tax_B2'])['t']; res['t_fit_rule'] = 'no t passes both bars: t with the lowest tax_B2'
    res['verdict_E0'] = 'KILL' if m2 >= 1.657e-6 else 'CONTINUE'
    res['stable_all'] = bool(all(q['ok'] for q in STAB))
    if not res['stable_all']: res['verdict_E0'] += ' (UNSTABLE: a tax moved >= 1e-8/site under the tight FN solve)'
    res['sec'] = time.time() - T00; res['done'] = True; dump(F, res)
    progress(f'VERDICT DONE E0 {res["verdict_E0"]} min tax_B2 {m2:.4e}; t_fit {res["t_fit"]} ({res["t_fit_rule"]}); stability all {res["stable_all"]} {res["sec"]:.0f}s'); sys.exit(0)
# ================================================================= stage full: Phi at t_fixed, exact-store controls, then the fit
Phi, rec = make_Phi(T_FIX); res['Phi'] = rec; dump(F, res)
progress(f'[Phi_t] t {T_FIX:g} (FN-target diagnostics): W_M {rec["W_M"]:.4e} (phi_0 {res["W_M_phi0"]:.4e}, ratio {rec["W_M_over_W_M_phi0"]:.3e}) tail {rec["tail_w"]:.4e} (phi_0 {res["tail_phi0"]:.4e})')
x3_signed(Phi, T_FIX, rec, broken=True); res['exact_Phi'] = {'sign_Phi': rec['signed_Phi'], 'rel_K2': rec['signed_Phi']['rel_K2']}; dump(F, res)
sN_f = s_N_of(Phi); sPhi = sg_of(Phi); la_Phi = la_of(Phi); T_READ = T_FIX
res['exact_Phi']['sign_N'], lp_exact_h = tax_set(la_Phi, sN_f, 'exact |Phi_t|, s_N', keep=True); dump(F, res)
progress(f'[exact |Phi_t| with s_N] tax_B1 {res["exact_Phi"]["sign_N"]["tax_B1"]:.4e} tax_B2 {res["exact_Phi"]["sign_N"]["tax_B2"]:.4e}')
np.save(os.path.join(OUT, 'Phi.npy'), np.asarray(Phi, np.float32))
del v_phi0, Phi, V2, DJ2
# ---- fit target: log|Phi_t| (floored like phi_0) with the Marshall sign
s_phi0, ls_phi0 = s2, ls2
ls2 = norm_la(jnp.log(jnp.maximum(jnp.exp(la_Phi), FLOOR))); s2 = sN_f
res['n_floored_Phi'] = int(jnp.sum(jnp.exp(la_Phi) < FLOOR)); dump(F, res)

# ================================================================= fit machinery (cx_t1s.py, model swapped)
dec, _ = lab.decade_of(ls2)
dec_h = np.asarray(dec); ho_h = np.asarray(lab.HO); n_h = np.asarray(sec.n); ls2_h = np.asarray(ls2)
res['decade_orbits'] = {str(d): dict(train=int(np.sum((dec_h == d) & ~ho_h)), heldout=int(np.sum((dec_h == d) & ho_h))) for d in DECS}
progress(f'[fit setup] orbits per decade {res["decade_orbits"]}')


class Sampler:
    def __init__(self, eligible, seed):
        self.rng = np.random.default_rng(seed); self.groups = []
        for d in DECS:
            idx = np.nonzero((dec_h == d) & eligible)[0]
            cdf = np.cumsum(n_h[idx]); self.groups.append((idx, cdf / cdf[-1]))

    def draw(self, M):
        G = len(self.groups); m = -(-M // G)
        idx = np.concatenate([g[0][np.minimum(np.searchsorted(g[1], self.rng.random(m)), g[0].size - 1)] for g in self.groups])
        return idx, np.repeat(np.arange(G), m)


@jax.jit
def bonds_of(idx, key):
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
    G = len(sampler.groups); acc = np.zeros(G); cnt = np.zeros(G)
    for i in range(0, n_per, 2048 // G):
        idx, g = sampler.draw(min(2048 // G, n_per - i) * G); key, k = jax.random.split(key)
        W = np.asarray(bonds_of(jnp.asarray(idx), k)[3]); np.add.at(acc, g, W); np.add.at(cnt, g, 1)
    return acc / cnt


def make_batch(sampler, M, key, wbar):
    idx, g = sampler.draw(M); parts = []
    for i in range(0, idx.size, 2052):
        key, k_ = jax.random.split(key); parts.append(bonds_of(jnp.asarray(idx[i:i + 2052]), k_))
    x, yk, iyk, W = (jnp.concatenate([q[j] for q in parts]) for j in range(4))
    return dict(idx=idx, g=g, x=x, y=yk, lx=ls2[jnp.asarray(idx)], ly=ls2[iyk], wb=LAM_B * W / jnp.asarray(wbar)[jnp.asarray(g)] / K, W=W, ng=len(sampler.groups))


def S_of(th, bt): return fwd(th, jnp.concatenate([bt['x'], bt['y'].reshape(-1)]))


@jax.jit
def per_sample(S, lx, ly, wb, c):
    M = lx.shape[0]; ex = S[:M] - lx - c; ey = S[M:].reshape(M, -1) - ly - c; db = ex[:, None] - ey
    return ex * ex + wb * jnp.sum(db * db, 1), ex, db


def strat_mean_se(v, g, G):
    v = np.asarray(v); mu = []; var = []
    for k in range(G):
        q = v[g == k]; mu.append(q.mean()); var.append(q.var(ddof=1) / q.size)
    return float(np.mean(mu)), float(np.sqrt(np.sum(var)) / G)


def batch_loss(th, c, bt): return per_sample(S_of(th, bt), bt['lx'], bt['ly'], bt['wb'], c)


def decade_diag(bt, ex, db):
    d_ = dec_h[bt['idx']]; ex = np.asarray(ex); db = np.asarray(db); wb = np.asarray(bt['wb']); out = {}
    for d in DECS:
        m = d_ == d
        if not m.any(): continue
        sw = float(np.sum(wb[m])) * db.shape[1]
        out[str(d)] = dict(n=int(m.sum()), rms_e=float(np.sqrt(np.mean(ex[m] ** 2))), bond_rms=float(np.sqrt(np.sum(wb[m][:, None] * db[m] ** 2) / sw)) if sw > 0 else None)
    return out


@jax.jit
def solve_dir(G, S, lx, ly, wb, c, lam_rel, tau):
    """parameter-space LM step of cx_t1s.solve_dir (rows = B (1 + K) > params)."""
    M = lx.shape[0]; P = G.shape[1]; dt = G.dtype
    ex = S[:M] - lx - c; ey = S[M:].reshape(M, -1) - ly - c; Kb = ey.shape[1]
    sv = 1.0 / jnp.sqrt(M); sb = jnp.sqrt(wb / M)
    r = jnp.concatenate([sv * ex, (sb[:, None] * (ex[:, None] - ey)).reshape(-1)])
    cc = jnp.concatenate([jnp.full(M, -sv, f64), jnp.zeros(M * Kb, f64)])
    Gx = G[:M]
    Jv = Gx * sv.astype(dt)
    Jb = ((Gx[:, None, :] - G[M:].reshape(M, Kb, P)) * sb.astype(dt)[:, None, None]).reshape(M * Kb, P)
    Jf = jnp.concatenate([jnp.concatenate([Jv, Jb]).astype(f64), cc[:, None]], 1); del Jv, Jb
    Sm = Jf.T @ Jf; g = Jf.T @ r
    lam = lam_rel * jnp.trace(Sm) / (P + 1)
    d = -jnp.linalg.solve(Sm + lam * jnp.eye(P + 1, dtype=f64), g)
    dP = d[:P]; dc = d[P]
    pc = (Gx.astype(f64) @ dP); pc = pc - jnp.mean(pc)
    prms = jnp.sqrt(jnp.mean(pc * pc)); sc = jnp.minimum(1.0, tau / jnp.maximum(prms, 1e-30))
    return dP * sc, dc * sc, prms, sc, jnp.sum(r * r)


_cache = {}


def fit_setup():
    if _cache: return _cache
    key = jax.random.PRNGKey(100 + SEED)
    S_tr = Sampler(~ho_h, 1000 + SEED); S_ho = Sampler(ho_h, 5000 + SEED)
    key, k1 = jax.random.split(key)
    WBAR = wbar_of(Sampler(~ho_h, 3000 + SEED), k1, NW)
    key, k1 = jax.random.split(key)
    HOB = make_batch(S_ho, B_HO, k1, WBAR)
    _cache.update(S_tr=S_tr, WBAR=WBAR, HOB=HOB, key=key)
    progress(f'[wbar] {np.array2string(WBAR, precision=3)}')
    return _cache


def run_fit(theta_start, steps, tag):
    """FIXED number of steps (no convergence rule).  steps = 0 returns theta_start unchanged."""
    cs = fit_setup(); S_tr, WBAR, HOB = cs['S_tr'], cs['WBAR'], cs['HOB']; key = jax.random.fold_in(cs['key'], steps + 7)
    S0 = fwd(theta_start, HOB['x'])
    c = float(np.mean(np.asarray(S0) - np.asarray(HOB['lx'])))
    Lh, ex, db = batch_loss(theta_start, c, HOB)
    curve = [dict(step=0, t=0.0, L_ho=float(jnp.mean(Lh)), c=c, by_decade=decade_diag(HOB, ex, db))]
    progress(f'[fit {tag}] start held-out loss {curve[0]["L_ho"]:.4e}')
    theta = theta_start; tau = TAU0; tau_changed = 0; tau_hist = [dict(step=0, tau=tau)]; hist = []; lam_rel = LAM0; t0 = time.time()
    if steps > 0: sec.offload()
    for n in range(1, steps + 1):
        ts = time.time(); key, k1, k2 = jax.random.split(key, 3)
        bt = make_batch(S_tr, B, k1, WBAR); Xall = jnp.concatenate([bt['x'], bt['y'].reshape(-1)])
        S = fwd(theta, Xall); G = jac(theta, Xall)
        dP, dc, prms, sc, rr = solve_dir(G, S, bt['lx'], bt['ly'], bt['wb'], c, lam_rel, tau); del G
        dc = float(dc)
        ab = make_batch(S_tr, B_ACC, k2, WBAR)
        L_old, _, _ = batch_loss(theta, c, ab); cand = theta + dP.astype(theta.dtype); L_new, _, _ = batch_loss(cand, c + dc, ab)
        dmean, dse = strat_mean_se(L_new - L_old, ab['g'], ab['ng']); lam_used = lam_rel
        if not np.isfinite(dmean) or dmean > Z_ACC * dse: ok = False; verdict = 'worse'; lam_rel = min(lam_rel * 4, LAM_CAP)
        elif dmean < -Z_ACC * dse: ok = True; verdict = 'better'; lam_rel = max(lam_rel / 2, LAM_MIN)
        else: ok = True; verdict = 'noise'
        if ok: theta = cand; c = c + dc
        hist.append(dict(step=n, ok=ok, verdict=verdict, lam_rel=lam_used, dmean=dmean, dse=dse, pred_rms=float(prms), trust_scale=float(sc), tau=tau, c=c, sec=time.time() - ts))
        if n == 10 and tag == 'fit' and steps > 10:                       # C2-L timing guard (A40 fp64): no change to the fit itself
            est = (time.time() - t0) / 10 * steps; res['fit_time_estimate_s'] = est
            progress(f'[fit timing] 10 steps {time.time() - t0:.0f}s -> {steps} steps ~{est:.0f}s (budget {FIT_BUDGET:.0f}s)')
            if est > FIT_BUDGET:
                np.save(os.path.join(OUT, 'theta_10steps.npy'), np.asarray(theta)); stop(4, f'DONE STOP fit too slow: est {est:.0f}s > {FIT_BUDGET:.0f}s')
        if n % HO_EVERY == 0:
            Lh, ex, db = batch_loss(theta, c, HOB)
            curve.append(dict(step=n, t=time.time() - t0, L_ho=float(jnp.mean(Lh)), c=c, by_decade=decade_diag(HOB, ex, db)))
            acc10 = np.mean([h['ok'] for h in hist[-HO_EVERY:]]); d11 = curve[-1]['by_decade'].get('-11', {}).get('rms_e')
            progress(f'  {tag} step {n}: held-out {curve[-1]["L_ho"]:.4e}  rms_e(-11) {d11}  acc {acc10:.2f}  lam_rel {lam_rel:.1e}  tau {tau:.4f} trust {float(sc):.2f}  {hist[-1]["sec"]:.1f}s/step')
            if tau > TAU_MIN and n - tau_changed >= ANNEAL:
                cb = [q for q in curve if q['step'] <= n - ANNEAL]
                if cb and abs(cb[-1]['L_ho'] - curve[-1]['L_ho']) / curve[-1]['L_ho'] < FLAT_TOL:
                    tau = max(tau / 2, TAU_MIN); tau_changed = n; tau_hist.append(dict(step=n, tau=tau)); progress(f'  {tag} step {n}: held-out flat -> tau {tau:.4f}')
            res['fit_' + tag] = dict(hist=hist, curve=curve, tau_hist=tau_hist); dump(F, res)
    wall = time.time() - t0
    out = dict(theta=theta, c=c, hist=hist, curve=curve, tau_hist=tau_hist, wall=wall,
               s_per_step=(wall / steps if steps else None), s_per_step_median=(float(np.median([h['sec'] for h in hist])) if hist else None),
               acceptance=(float(np.mean([h['ok'] for h in hist])) if hist else None))
    res['fit_' + tag] = dict({k: v for k, v in out.items() if k != 'theta'}); dump(F, res)
    return out


def ctl1_check(theta, label):
    la_n = fwd(theta, sec.reps[I4]); d = np.asarray(la_n - laA[I4]); dm = d - d.mean()
    P = net['parts_j'](theta, sec.reps[I4][:512]); sg = np.sign(np.asarray(P[:, 1]))
    agree = float(abs(np.mean(sg * np.asarray(sA[I4][:512]))))
    return dict(label=label, max_abs_dev_after_offset=float(np.max(np.abs(dm))), rms_dev=float(np.sqrt(np.mean(dm ** 2))), offset=float(d.mean()), sign_agree_512=agree)


# ================================================================= control 1
r0 = run_fit(theta0, 0, 'ctl0')
c1a = ctl1_check(r0['theta'], 'fit code, 0 steps')
c1a['theta_bitwise_equal_A'] = bool(np.array_equal(np.asarray(r0['theta']), np.asarray(theta0)))
pert = theta0 + 1e-6 * jax.random.normal(jax.random.PRNGKey(5), theta0.shape, jnp.float64)
c1b = ctl1_check(run_fit(pert, 0, 'ctl0b')['theta'], 'broken: theta perturbed by 1e-6 (must FAIL)')
if ARM == 'rbm':      # positive-RBM arm: warm start is A's RBM factor, not A; control = 0 steps returns theta0, perturbation changes logA
    dpert = float(jnp.max(jnp.abs(fwd(pert, sec.reps[I4]) - fwd(theta0, sec.reps[I4]))))
    res['control1'] = dict(zero_steps=c1a, broken=c1b, arm='rbm', pert_max_dlogA=dpert, ok=bool(c1a['theta_bitwise_equal_A']),
                           broken_fails=bool(dpert >= 1e-12)); dump(F, res)
else:
    res['control1'] = dict(zero_steps=c1a, broken=c1b, ok=bool(c1a['max_abs_dev_after_offset'] < 1e-12 and c1a['theta_bitwise_equal_A'] and c1a['sign_agree_512'] == 1.0),
                           broken_fails=bool(c1b['max_abs_dev_after_offset'] >= 1e-12)); dump(F, res)
progress(f'CONTROL 1: 0-step fit returns A: bitwise {c1a["theta_bitwise_equal_A"]}, max|dlogA| on 4096 orbits {c1a["max_abs_dev_after_offset"]:.2e} (<1e-12), '
         f'sign agree {c1a["sign_agree_512"]}; broken max {c1b["max_abs_dev_after_offset"]:.2e} fails={res["control1"]["broken_fails"]}  ok={res["control1"]["ok"]}')
if not (res['control1']['ok'] and res['control1']['broken_fails']): ctl_fail('control 1')
res['controls_ok'] = True; dump(F, res)

# ================================================================= fit
nsteps = 10 if STAGE == 'control' else STEPS
fr = run_fit(theta0, nsteps, 'fit')
np.save(os.path.join(OUT, 'theta.npy'), np.asarray(fr['theta']))
cv = fr['curve']; Ls = [q['L_ho'] for q in cv]
res['heldout'] = dict(steps=[q['step'] for q in cv], L_ho=Ls, falls=bool(Ls[-1] < Ls[0]), ratio_end_over_start=Ls[-1] / Ls[0], monotone=bool(all(b <= a for a, b in zip(Ls, Ls[1:]))),
                      s_per_step=fr['s_per_step'], s_per_step_median=fr['s_per_step_median'], acceptance=fr['acceptance'], tau_hist=fr['tau_hist'])
dump(F, res)
progress(f'[fit] {nsteps} steps in {fr["wall"]:.0f}s = {fr["s_per_step"]:.2f} s/step (median {fr["s_per_step_median"]:.2f}); held-out {Ls[0]:.4e} -> {Ls[-1]:.4e} (falls {Ls[-1] < Ls[0]}), acceptance {fr["acceptance"]:.2f}')
if STAGE == 'control':
    res['sec'] = time.time() - T00; res['done'] = True; dump(F, res); progress(f'DONE control stage {res["sec"]:.0f}s'); sys.exit(0)

# ================================================================= readout, K_k of a_1, tax
sec.reload()
la_a1, sg_a1, imw = table(fr['theta']); la_a1 = norm_la(la_a1)
res['a1'] = dict(im_weight=imw, im_weight_ok=bool(imw <= 1e-3)); dump(F, res)
w2 = sec.n * jnp.exp(2 * (ls2 - jnp.max(ls2)))
sg_rel = jnp.where(sg_a1 * sA > 0, 0.0, 1.0); sg_rel = jnp.minimum(sg_rel, 1.0)
agree_w = float(jnp.sum(w2 * (1 - sg_rel)) / jnp.sum(w2)); res['a1']['sign_weight_agreeing_with_A_phi0_weighted'] = max(agree_w, 1 - agree_w)
e_ = la_a1 - ls2; off = float(jnp.sum(w2 * e_) / jnp.sum(w2)); res['a1']['rms_log_err_phi0_weighted'] = float(jnp.sqrt(jnp.sum(w2 * (e_ - off) ** 2) / jnp.sum(w2)))
res['a1']['loss_dev_by_decade'] = {}
for d in DECS:
    m = jnp.asarray(dec_h == d); wn = jnp.where(m, sec.n, 0.0)
    if float(jnp.sum(wn)) > 0:
        me = float(jnp.sum(wn * e_) / jnp.sum(wn)); res['a1']['loss_dev_by_decade'][str(d)] = dict(mean=me, rms=float(jnp.sqrt(jnp.sum(wn * (e_ - me) ** 2) / jnp.sum(wn))))
progress(f'[a_1] im_weight {imw:.3e}; rms log error (phi_0-weighted) {res["a1"]["rms_log_err_phi0_weighted"]:.4f}; rms by decade ' +
         ' '.join(f'{d}:{v["rms"]:.3f}' for d, v in res['a1']['loss_dev_by_decade'].items())); dump(F, res)
res['a1']['loss_dev_by_decade_bins'] = 'fit-target (|Phi_t|^2) decades, as C2-L-1 (fit stratification bins)'
# amendment 3: fit rms by decade on the common |phi_0|^2 bins (n-weighted, mean removed per bin, as loss_dev_by_decade)
D0j = jnp.asarray(DEC0); res['a1']['fit_rms_by_phi0_decade'] = {}
for d in sorted(set(DEC0.tolist()), reverse=True):
    m = D0j == d; wn = jnp.where(m, sec.n, 0.0)
    me = float(jnp.sum(wn * e_) / jnp.sum(wn)); res['a1']['fit_rms_by_phi0_decade'][str(d)] = dict(n_orbits=int(jnp.sum(m)), mean=me, rms=float(jnp.sqrt(jnp.sum(wn * (e_ - me) ** 2) / jnp.sum(wn))))
dump(F, res)
tbl = OPS.load_or_build(sec, 'J2', TABDIR, log=progress)[0]; V2 = OPS.BondOp(sec, tbl, 'J2'); del tbl
DJ2 = OPS.diag0(sec, len(SS.NN)); HJ2 = lambda x: V2(x) + DJ2 * x
res['tax_B'] = {}; T_READ = T_FIX
res['tax_B']['sign_N'], lp_fit_h = tax_set(la_a1, sN_f, 'psi_fit sign_N', keep=True); dump(F, res)
# amendment 2: oracle sign(Phi_t) with the SAME fitted f -- DIAGNOSTIC ONLY, never a pass
res['tax_B']['sign_Phi'] = tax_set(la_a1, sPhi, 'psi_fit sign_Phi (oracle sign, diagnostic only)'); res['tax_B']['sign_Phi']['label'] = 'diagnostic only (oracle sign(Phi_t)); never a pass'
dump(F, res)
for tag in ('sign_N', 'sign_Phi'):
    progress(f'[tax_B {tag}] tax_B1 {res["tax_B"][tag]["tax_B1"]:.4e} tax_B2 {res["tax_B"][tag]["tax_B2"]:.4e} '
             f'(E_FN K_1 {res["tax_B"][tag]["EFN_K1"]:.6e}, K_2 {res["tax_B"][tag]["EFN_K2"]:.6e})')
# amendment 3: actual-fit-error control, no decoder: phi_0 e^{eps}, eps = log|Phi_fit| - log|Phi_t| (both normalised), phi_0's sign
eps = la_a1 - la_Phi
res['fit_error_control'] = tax_set(norm_la(ls_phi0 + eps), s_phi0, 'phi_0 e^eps (fit error, no decoder)', read=False)
res['fit_error_control']['eps_rms_phi0_weighted'] = float(jnp.sqrt(jnp.sum(sec.n * jnp.exp(2 * (ls_phi0 - jnp.max(ls_phi0))) * (eps - jnp.sum(sec.n * jnp.exp(2 * (ls_phi0 - jnp.max(ls_phi0))) * eps) / jnp.sum(sec.n * jnp.exp(2 * (ls_phi0 - jnp.max(ls_phi0))))) ** 2) / jnp.sum(sec.n * jnp.exp(2 * (ls_phi0 - jnp.max(ls_phi0))))))
del eps; dump(F, res)
fc = res['fit_error_control']; ex_ = res['exact_Phi']['sign_N']; tN = res['tax_B']['sign_N']
res['tax_side_by_side'] = {'exact_store s_N|Phi_t|': [ex_['tax_B1'], ex_['tax_B2']], 'fit readout exp(-tH)(s_N e^f)': [tN['tax_B1'], tN['tax_B2']],
                           'fit error, no decoder phi_0 e^eps': [fc['tax_B1'], fc['tax_B2']], 'oracle sign(Phi_t) e^f (diagnostic only)': [res['tax_B']['sign_Phi']['tax_B1'], res['tax_B']['sign_Phi']['tax_B2']]}
progress('[tax side by side, (tax_B1, tax_B2)] ' + '; '.join(f'{k}: {v[0]:.4e}, {v[1]:.4e}' for k, v in res['tax_side_by_side'].items()))
# rebuilt state's log-amplitude error by |phi_0|^2 decade: |log|psi_fit| - log|psi_ref||, |phi_0|^2 weighted;
# psi_fit = exp(-tH)(s_N e^f), refs: exact store exp(-tH)(s_N |Phi_t|) and phi_0 itself
w0 = np.asarray(sec.n * jnp.exp(2 * (ls_phi0 - jnp.max(ls_phi0)))); w0 = w0 / w0.sum(); nn_ = np.asarray(sec.n)
res['rebuilt_err_by_phi0_decade'] = {}
for ref_name, lref in (('vs_exact_store', lp_exact_h), ('vs_phi0', np.asarray(ls_phi0))):
    e = lp_fit_h - lref; tab_ = {}
    for d in sorted(set(DEC0.tolist()), reverse=True):
        m = DEC0 == d; wd = w0[m]; ed = e[m]
        if wd.sum() <= 0: continue
        tab_[str(d)] = dict(w=float(wd.sum()), mean_abs=float(np.sum(wd * np.abs(ed)) / wd.sum()), rms=float(np.sqrt(np.sum(wd * ed ** 2) / wd.sum())),
                            rms_n_meanremoved=float(np.sqrt(np.sum(nn_[m] * (ed - np.sum(nn_[m] * ed) / nn_[m].sum()) ** 2) / nn_[m].sum())))
    res['rebuilt_err_by_phi0_decade'][ref_name] = dict(by_decade=tab_, mean_abs=float(np.sum(w0 * np.abs(e))), rms=float(np.sqrt(np.sum(w0 * e ** 2))))
    del e
dump(F, res)
fr_ = res['a1']['fit_rms_by_phi0_decade']; rb_ = res['rebuilt_err_by_phi0_decade']['vs_exact_store']['by_decade']
progress('[decade |phi_0|^2: fit rms | rebuilt |dlog| vs exact store] ' + ' '.join(f'{d}:{fr_[d]["rms"]:.3f}|{rb_[d]["mean_abs"]:.3f}' for d in fr_ if d in rb_))
del ls_phi0


def verdict(t):
    return 'PASS' if (t['tax_B1'] < 1.657e-6 and t['tax_B2'] < 0.82e-6) else ('KILL' if t['tax_B2'] >= 1.657e-6 else 'INCONCLUSIVE')


# held-out slope over the last 50 steps (curve every 10 steps): mean fall per 10 steps relative to the final loss
st_ = res['heldout']['steps']; Lh_ = res['heldout']['L_ho']
if STEPS - 50 in st_:
    La, Lb = Lh_[st_.index(STEPS - 50)], Lh_[st_.index(STEPS)]; res['heldout']['fall_per10_last50_rel'] = (La - Lb) / 5 / Lb
else: res['heldout']['fall_per10_last50_rel'] = None
sl_ = res['heldout']['fall_per10_last50_rel']
res['verdict'] = verdict(res['tax_B']['sign_N'])
if res['verdict'] == 'KILL' and sl_ is not None and sl_ > 0.01: res['verdict'] = 'INCONCLUSIVE (cause: warm start, held-out still falling > 1% per 10 steps)'
res['verdict_sign_Phi_diagnostic'] = verdict(res['tax_B']['sign_Phi']) + ' (diagnostic only)'
res['stable_all'] = bool(all(q['ok'] for q in STAB))
if not res['stable_all']: res['verdict'] += ' (UNSTABLE: a tax moved >= 1e-8/site under the tight FN solve)'
res['baseline_G_tax'] = dict(tax_1=5.107e-6, tax_2=1.657e-6, rms_by_decade={'-8': 0.011, '-10': 0.031, '-11': 0.061, '-12': 0.129, '-13': 0.273})
res['sec'] = time.time() - T00; res['done'] = True; dump(F, res)
tP = res['tax_B']['sign_Phi']
progress(f'VERDICT DONE {res["verdict"]} (t {T_FIX:g}; s_N: tax_B1 {tN["tax_B1"]:.4e} tax_B2 {tN["tax_B2"]:.4e}; held-out fall/10 steps last 50 {sl_}; '
         f'oracle sign(Phi) diagnostic: tax_B1 {tP["tax_B1"]:.4e} tax_B2 {tP["tax_B2"]:.4e}; fit-error control {fc["tax_B1"]:.4e} {fc["tax_B2"]:.4e}; stability {res["stable_all"]}) {res["sec"]:.0f}s')
