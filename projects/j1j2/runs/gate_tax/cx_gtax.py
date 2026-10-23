"""G-tax (README compress_6x6 "M1 + gate for the loop"): ONE two-hop refit tax measured on RBM+PP.   python cx_gtax.py OUT SPEC.json
Setup  A = RBM+PP (rbmpp_6x6, rpT4_sr2 params_final); amplitude |Re(e^{-i th} psi_P)|, sign = sgn of it (th from table_A.json c).
       log im_weight; stop (exit 5) if > 1e-3.  A's exact table = sym_tables_A.npz (rbmpp_table.py, job 17168764, same params).
Guide  g_0 = K_2(A): SIGNED two-step Krylov polynomial of (la_A, s_A): the variational-energy optimum (lowest Ritz vector) of
       span{v, (H-E)v, (H-E)^2 v}; w = sum c_i b_i with its own signs (so sign s_g0 = sgn w; amplitude |w|/sqrt(n)).  K_1 likewise
       (= the exact Lanczos p=1 state).  (cx_h1.py k=2 keeps the sign s_2 and is the unsigned polynomial; this is the signed one.)
phi_0  = sec.fn_solve(la_g0, s_g0) ground state of H_FN[g_0] (floored at 1e-15, renormalised, as cx_o2b).
Refit  A -> a_1: cx_t1s.py loss/sampler/damped GN/acceptance, model RBM+PP (all 2480 real params: f, Re/Im b, Re/Im W), warm = A, target
       log phi_0 on its decades -5..-13, sign fixed = s_g0 (only the amplitude is fitted).  FIXED STEPS (no convergence rule, no stop);
       held-out loss (fixed stratified batch of the held-out orbits) every 10 steps; tau anneal 0.05 -> 0.01 as cx_o2b (a step-size rule,
       not a stopping rule).
tax_k  = E_FN(K_k(a_1; s_g0)) - E_FN(K_k(phi_0; s_g0)), k = 2 (the gate item), k = 1 and k = 0 for context.  a_1's table = |Re| amplitude over all
       15.8M orbits; its own Re-sign is not used (the guide sign rule: s_g0), its weight of disagreement with A's sign is logged.
Controls (exit 2 before the fit): (1) run_fit with 0 steps returns A bitwise and its log-amplitudes on 4096 orbits equal the saved A table to 1e-12
       (after the offset); a perturbed theta (1e-6) must FAIL this check.  (2) E_FN(K_1(A)) = 5.573833e-6 (table_A.json lanczos p=1), variational
       7.3798e-6; K_2 variational = lanczos p=2 2.4851e-6; a deliberately wrong hop (0.9 alpha) must FAIL control 2.
SPEC: params, table_A, table_A_json, stage ('control' = controls + 10 timing steps, 'full'), steps (400), code_sha256, seed.
"""
import os, sys, time, math, hashlib
from functools import partial
import numpy as np
import scipy.linalg as sl
from cx_lib import *
from gtax_net import make_net

OUT = sys.argv[1]; SPEC = json.load(open(sys.argv[2])); os.makedirs(OUT, exist_ok=True)
F = os.path.join(OUT, 'cx_gtax.json'); PROG = os.path.join(OUT, 'progress.txt'); T00 = time.time()
STAGE = SPEC.get('stage', 'full'); STEPS = int(SPEC.get('steps', 400)); SEED = int(SPEC.get('seed', 0))
DECS = list(range(-5, -14, -1)); ND = len(DECS); FLOOR = 1e-15
B = int(SPEC.get('B', 8208)); K = int(SPEC.get('K', 2)); B_ACC = int(SPEC.get('B_acc', B)); B_HO = int(SPEC.get('B_ho', 4096))
LAM_B = 1.0; LAM0 = 1e-2; LAM_CAP = 1.0; LAM_MIN = 1e-6; TAU0 = 0.05; TAU_MIN = 0.01; Z_ACC = 2.0; HO_EVERY = 10; ANNEAL = 20; FLAT_TOL = 0.02
NW = int(SPEC.get('n_wbar', 8192))
res = dict(spec=SPEC, stage=STAGE, done=False, design=dict(B=B, K=K, B_acc=B_ACC, B_ho=B_HO, lam0=LAM0, lam_cap=LAM_CAP, tau0=TAU0, tau_min=TAU_MIN,
                                                           anneal_steps=ANNEAL, z_acc=Z_ACC, ho_every=HO_EVERY, steps=STEPS, decades=DECS))


def progress(msg):
    with open(PROG, 'a') as fh: fh.write(f'{time.strftime("%Y-%m-%dT%H:%M:%S")} PROGRESS {msg}\n')
    log(msg)


HERE_ = os.path.dirname(os.path.abspath(__file__))
CODE_FILES = ('cx_gtax.py', 'gtax_net.py', 'cx_lib.py', 'rbmpp_lib.py')
SHA_RUN = {fn: (hashlib.sha256(open(os.path.join(HERE_, fn), 'rb').read()).hexdigest() if os.path.exists(os.path.join(HERE_, fn)) else None) for fn in CODE_FILES}
SHA_EXP = SPEC.get('code_sha256') or {}
bad = [fn for fn in CODE_FILES if SHA_EXP.get(fn) != SHA_RUN[fn]]
res['code_sha256'] = dict(expected=SHA_EXP, running=SHA_RUN, dir=HERE_, match=not bad)
if bad:
    res['code_sha256']['mismatch'] = bad; res['stop'] = 'CODE DRIFT'; dump(F, res); progress(f'CODE DRIFT: {bad}; exit 3'); sys.exit(3)
dump(F, res)


def stop(code, msg):
    res['stop'] = msg; res['sec'] = time.time() - T00; dump(F, res); progress(msg); sys.exit(code)


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
net = make_net(SPEC['params'], THETA_PH); theta0 = net['theta0']; NPAR = int(theta0.size)
FCH, JCH, TCH = 512, 64, 512
res['npar'] = NPAR; res['theta_phase'] = THETA_PH


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


def fn_of(la, s, tag):
    _, u, info = sec.fn_solve(la, s, maxit=400)
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
for k in (1, 2):
    (lk, sk), Ev, ck = kpoly(ls2, s2, k); ek, _, ik = fn_of(lk, sk, f'K_{k}(phi_0)')
    efn_p0[k] = dict(E_var=Ev, EFN=ek, coeffs=ck, sign_flip_weight_vs_s_g0=float(jnp.sum(jnp.where(sk != s2, sec.n * jnp.exp(2 * (lk - jnp.max(lk))), 0.0)) / jnp.sum(sec.n * jnp.exp(2 * (lk - jnp.max(lk))))))
    del lk, sk
res['phi0'] = dict(EFN_zero_hop=efn_g0, K1=efn_p0[1], K2=efn_p0[2]); dump(F, res)
progress(f'[phi_0] t_setup {time.time() - t0:.0f}s  E_FN(phi_0) {efn_g0:.6e}  E_FN(K_1 phi_0) {efn_p0[1]["EFN"]:.6e}  E_FN(K_2 phi_0) {efn_p0[2]["EFN"]:.6e}')

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
e_a1, _, _ = fn_of(la_a1, s2, 'a_1 zero hop (s_g0)')
res['a1']['EFN_zero_hop'] = e_a1; dump(F, res)
res['a1']['K'] = {}
for k in (1, 2):
    (lk, sk), Ev, ck = kpoly(la_a1, s2, k); ek, _, _ = fn_of(lk, sk, f'K_{k}(a_1)')
    res['a1']['K'][str(k)] = dict(E_var=Ev, EFN=ek, coeffs=ck); del lk, sk; dump(F, res)
tax = {'0': e_a1 - efn_g0}
for k in (1, 2): tax[str(k)] = res['a1']['K'][str(k)]['EFN'] - efn_p0[k]['EFN']
res['tax'] = dict(tax_0=tax['0'], tax_1=tax['1'], tax_2=tax['2'], tax_2_over_tax_1=tax['2'] / tax['1'],
                  components=dict(E_FN_A=REF['EFN_A'], E_FN_g0=efn_g0, E_FN_K2_phi0=efn_p0[2]['EFN'], E_FN_K2_a1=res['a1']['K']['2']['EFN'], E_FN_a1=e_a1,
                                  E_FN_K1_phi0=efn_p0[1]['EFN'], E_FN_K1_a1=res['a1']['K']['1']['EFN'], exact_gain_phi0_vs_g0=efn_g0 - efn_p0[2]['EFN']))
res['sec'] = time.time() - T00; res['done'] = True; dump(F, res)
progress(f'TAX tax_2 {tax["2"]:.4e} = E_FN(K_2 a_1) {res["a1"]["K"]["2"]["EFN"]:.6e} - E_FN(K_2 phi_0) {efn_p0[2]["EFN"]:.6e}; tax_1 {tax["1"]:.4e}; tax_0 {tax["0"]:.4e}')
progress(f'VERDICT DONE {res["sec"]:.0f}s tax_2 {tax["2"]:.4e} heldout_falls {res["heldout"]["falls"]} im_weight_a1 {imw:.2e}')
