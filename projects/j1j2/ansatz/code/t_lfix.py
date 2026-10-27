"""Test T-Lfix (STORAGE_PROBLEM.md "Test T-Lfix", pre-registered 2026-10-27): does the pair loss L_fix (board eq. 15) store the
fixed-node state better than the log loss of our code (board eq. 13), same network?  Exact 4x4, J2 = 0.5, CPU, no Monte Carlo.

    python3 t_lfix.py OUT setup            reproduction of the C13 records + checks (L_fix = 0 at phi; eq. 5; gradient)
    python3 t_lfix.py OUT fit ARM SEED     ARM in {A, B, C, D}; 400 LM steps; -> OUT/fit_ARM_sSEED.json
    python3 t_lfix.py OUT summary          table over arms and seeds -> OUT/summary.json

Machinery reused (copied, not imported, so the C13 code stays untouched):
  lattice, H, orbit fast path, v = Re[w30 psi_P], best_w, fn_state, hop  <- experiments/gfmc_fn_lattice/l8/l8_loop4x4.py (C13)
  RBM+PP network (rbmpp_lib, guide30 params4.npz layout; ALL 392 parameters trained here: f and the complex RBM)
  K_k = lowest Ritz vector of span{v, (H-E)v, ..., (H-E)^k v} (signed), tax_k = E_FN(K_k fit) - E_FN(K_k phi) per site
                                                                         <- ansatz/code/c2le_gtax.py kpoly / tax_set
  optimiser = c2le_gtax.py run_fit/solve_dir: Gauss-Newton/LM in parameter space, lam_rel adaptive (x4 reject, /2 accept),
              trust region tau on the centred rms of the predicted change of log|psi_T| (tau0 0.05, halved when the loss is flat
              over 20 steps, min 0.01), 400 fixed steps.  Exact version: the stratified sample means are replaced by exact sums
              over all 12870 configurations / all pairs; accept iff the exact loss falls (no z-test needed, no noise).
Target phi = phi_0 of C13 variant ii = FN ground state of g_0 = K_1(v_0), v_0 = RBM-only E_v optimum (loop4x4/rbmonly/flat_opt.npy).
Warm start = v_0's parameters; seed s perturbs every parameter by SIGMA_SEED * N(0,1) (rng seed s).
Arms:
  A  log loss, c2le per_sample l.471-473 in exact expectation:
     L_A = (1/G) sum_d (1/n_d) sum_{x in d} [ e_x^2 + (1/wbar_d) sum_{y viol(phi)} |H_xy| |phi_y/phi_x| (e_x - e_y)^2 ],
     e = log|psi_T| - log|phi| - c, d = decade of phi(x)^2, wbar_d = mean_{x in d} W(x), W(x) = sum_{y viol(phi)} |H_xy||phi_y/phi_x|
     (c2le: B samples per decade stratum, K = 2 bonds per sample drawn prop. to |H| phi_y/phi_x, weight W/wbar/K; viol = target sign).
  B  L_fix = sum_{{x,y} viol(psi_T)} |H_xy| (psi_T(x) phi(y) - phi(x) psi_T(y))^2 / |psi_T(x) psi_T(y)|   (viol: psi_T(x) H_xy psi_T(y) > 0)
  C  L_fix on a log-only net: positive symmetrised RBM (M = 23, 391 params) x Marshall sign; random init (seed).
  D  ideal ceiling (lead amendment): minimise the exact E_FN(K_2(fit)) itself, HF force, natural gradient (see run_fit_D).
"""
import os, sys, json, time, hashlib
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sla, scipy.linalg as sl
R = '/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/experiments'
sys.path.insert(0, R + '/rbmpp_6x6')
import jax, jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
from jax.flatten_util import ravel_pytree
from rbmpp_lib import Lattice, make_fns, logcosh, lse_c

OUT = sys.argv[1]; MODE = sys.argv[2]; os.makedirs(OUT, exist_ok=True)
PROG = os.path.join(OUT, 'progress.txt'); T00 = time.time()
STEPS = int(os.environ.get('STEPS', 400)); SIGMA_SEED = float(os.environ.get('SIGMA_SEED', 1e-3))
LAM0, LAM_CAP, LAM_MIN, TAU0, TAU_MIN, ANNEAL, FLAT_TOL, CHK_EVERY = 1e-2, 1.0, 1e-6, 0.05, 0.01, 20, 0.02, 10   # c2le values
C13 = '/scratch/a/A.Otaifi/l8/loop4x4/rbmonly'
REC = dict(E0N=-0.5286202094621785, EFN_v0N=-0.5275259429021022, EFN_g0N=-0.5282946486911766,       # opt.json
           EFN_i1_vN=-0.5274793555140624, EFN_ii1_gN=-0.5283338719945195)                          # loop_i/ii.json round 1


def log(*a):
    s = time.strftime('%Y-%m-%dT%H:%M:%S') + ' PROGRESS ' + ' '.join(str(x) for x in a)
    print(s, flush=True)
    with open(PROG, 'a') as fh: fh.write(s + '\n')


def jd(o): return json.dumps(o, indent=1, default=lambda a: a.tolist() if isinstance(a, np.ndarray) else float(a))


# ============================================================ setup (l8_loop4x4.py)
L = 4; N = 16; lat = Lattice(L, 0.5, 'translations')
BI = np.asarray(lat.BI); BJ = np.asarray(lat.BJ); JB = np.asarray(lat.JB, float)
basis = np.array([s for s in range(1 << N) if bin(s).count('1') == N // 2], np.int64)
Dd = len(basis); lookup = -np.ones(1 << N, np.int64); lookup[basis] = np.arange(Dd)
X = ((basis[:, None] >> np.arange(N)) & 1).astype(np.int8) * 2 - 1
anti = X[:, BI] != X[:, BJ]; wi, bi = np.nonzero(anti)
nbr = lookup[basis[wi] ^ (1 << BI[bi]) ^ (1 << BJ[bi])]
KH = 0.5 * JB[bi]                                                    # H_xy = +K > 0 on every exchange pair
diag = 0.25 * np.sum(JB[None] * X[:, BI] * X[:, BJ], 1)
H = (sp.coo_matrix((KH, (wi, nbr)), shape=(Dd, Dd)).tocsr() + sp.diags(diag)).tocsr()
ev0, vec0 = sla.eigsh(H, k=1, which='SA', tol=1e-14); E0 = ev0[0]; psi0 = vec0[:, 0]
und = wi < nbr; PX, PY, PK, PNNN = wi[und], nbr[und], KH[und], (bi[und] >= 32)   # unordered pairs {x,y}, NN bonds first 32
MA = sum(1 << (x + 4 * y) for y in range(4) for x in range(4) if (x + y) % 2 == 0)
sN = np.array([1.0 - 2.0 * (bin(int(b) & MA).count('1') % 2) for b in basis])

z0 = np.load('/scratch/a/A.Otaifi/l8/check4x4/params4.npz'); KEYS = ('f', 'bre', 'bim', 'Wre', 'Wim')
p0 = {k: jnp.asarray(z0[k]) for k in KEYS}
flat_g30, unrav = ravel_pytree(p0); flat_g30 = np.asarray(flat_g30); P = flat_g30.size
_IMG = np.asarray(lat.IMG); _FL = np.asarray(lat.FL)
ZI = _FL[None, :, None] * X[:, _IMG].astype(np.float64)
_ord = np.argsort(-ZI, axis=2, kind='stable'); UI = _ord[..., :N // 2]; DI = _ord[..., N // 2:]
_up = (ZI > 0).astype(np.int64); _dn = 1 - _up; INVP = np.sum(_up * (np.cumsum(_dn, axis=2) - _dn), axis=2) % 2
_codes = np.sum(((ZI > 0).astype(np.int64)) << np.arange(N), axis=2); _mins = _codes.min(axis=1)
REPS, REPOF = np.unique(_mins, return_inverse=True); REPOF = REPOF.ravel(); _first = np.array([np.nonzero(_mins == m)[0][0] for m in REPS])
ZI, UI, DI, INVP = ZI[_first], UI[_first], DI[_first], INVP[_first]; NORB = len(REPS)
ZIj = jnp.asarray(ZI); UIj = jnp.asarray(UI); DIj = jnp.asarray(DI); INVPj = jnp.asarray(INVP)


def lp_orb(fl):
    """log psi_P (complex) on the orbit representatives, ALL parameters live (f and RBM)."""
    p = unrav(fl)
    F = p['f'][UIj[..., :, None], DIj[..., None, :]]; sg, ld = jnp.linalg.slogdet(F)
    a = ld + 1j * jnp.pi * ((jnp.where(sg < 0, 1, 0) + INVPj) % 2)
    r = jnp.sum(logcosh((p['bre'] + 1j * p['bim']) + ZIj.astype(jnp.complex128) @ (p['Wre'] + 1j * p['Wim'])), axis=-1)
    return lse_c(a + r)


fns = make_fns(lat, jnp.float64, True)
lpv_brute = jax.vmap(lambda fl, s: fns['logpsi'](unrav(fl), s), in_axes=(None, 0))
_lp_orb_j = jax.jit(lp_orb)
def _lp_one(fl, zi, ui, di, invp):
    """one orbit: (Re, Im) log psi_P; vmapped jacrev = 120x faster than jacfwd of lp_orb (identical to 2e-14, checked 2026-10-27)."""
    p = unrav(fl)
    F = p['f'][ui[:, :, None], di[:, None, :]]; sg, ld = jnp.linalg.slogdet(F)
    a = ld + 1j * jnp.pi * ((jnp.where(sg < 0, 1, 0) + invp) % 2)
    r = jnp.sum(logcosh((p['bre'] + 1j * p['bim']) + zi.astype(jnp.complex128) @ (p['Wre'] + 1j * p['Wim'])), axis=-1)
    l = lse_c(a + r); return jnp.stack([jnp.real(l), jnp.imag(l)])
_jo = jax.jit(jax.vmap(jax.jacrev(_lp_one), in_axes=(None, 0, 0, 0, 0)))
_jac_orb_j = lambda fl: jnp.transpose(_jo(fl, ZIj, UIj, DIj, INVPj), (1, 0, 2))


def best_w(fl):
    l = np.asarray(_lp_orb_j(jnp.asarray(fl)))[REPOF]; ps = np.exp(l - l.real.max()); Rr, Ii = ps.real, ps.imag
    A = np.array([[Rr @ (H @ Rr), Rr @ (H @ Ii)], [Ii @ (H @ Rr), Ii @ (H @ Ii)]]); B = np.array([[Rr @ Rr, Rr @ Ii], [Ii @ Rr, Ii @ Ii]])
    ev, ec = sl.eigh(0.5 * (A + A.T), B); c = ec[:, 0]; w = c[0] - 1j * c[1]; return w / abs(w)


w30 = best_w(flat_g30); W30 = complex(w30)


class RBMPP:
    """psi_T(x) = Re[w30 psi_P(x)];  S = log|psi_T| = Re l + log|Re(w e^{i Im l})|, no max shift (no overflow at 4x4)."""
    P = P

    def __init__(self, impose=None): self.impose = impose          # review amendment 1: impose the target's sign (None = own sign)

    def S_sign(self, fl):
        l = np.asarray(_lp_orb_j(jnp.asarray(fl))); ph = W30 * np.exp(1j * l.imag); c = ph.real
        return (l.real + np.log(np.maximum(np.abs(c), 1e-300)))[REPOF], (np.where(c >= 0, 1.0, -1.0)[REPOF] if self.impose is None else self.impose.copy())

    def S_sign_jac(self, fl):
        """S, sign and dS/dtheta (Dd x P) = Re(w psi dlog psi)/Re(w psi) from the orbit Jacobian."""
        l = np.asarray(_lp_orb_j(jnp.asarray(fl))); Jri = np.asarray(_jac_orb_j(jnp.asarray(fl)))
        dl = Jri[0] + 1j * Jri[1]; ph = W30 * np.exp(1j * l.imag); c = ph.real
        dS = (ph[:, None] * dl).real / c[:, None]                       # d log|Re(w psi)| = Re(w psi dl)/Re(w psi)
        sg = np.where(c >= 0, 1.0, -1.0)[REPOF] if self.impose is None else self.impose.copy()
        return (l.real + np.log(np.maximum(np.abs(c), 1e-300)))[REPOF], sg, dS[REPOF]


# ---- log-only net for arm C: positive RBM symmetrised over the same 2N images, times the Marshall sign
MC = 23


def lp_pos(th):
    b = th[:MC]; W = th[MC:].reshape(N, MC)
    return jax.scipy.special.logsumexp(jnp.sum(logcosh(b + ZIj @ W).real, axis=-1), axis=-1)   # real input -> logcosh real


_lpp_j = jax.jit(lp_pos); _jpp_j = jax.jit(jax.jacfwd(lp_pos))


class PosRBM:
    P = MC + N * MC

    def S_sign(self, th): return np.asarray(_lpp_j(jnp.asarray(th)))[REPOF], sN.copy()

    def S_sign_jac(self, th):
        return np.asarray(_lpp_j(jnp.asarray(th)))[REPOF], sN.copy(), np.asarray(_jpp_j(jnp.asarray(th)))[REPOF]


def vec_of(S, s):
    v = s * np.exp(S - S.max()); return v / np.linalg.norm(v)


def fn_state(g):
    """exact FN ground state with guide g (C13 fn_state): violating = g_x H_xy g_y > 0 moved to the diagonal."""
    rr = g[nbr] / g[wi]; keep = rr < 0
    dFN = diag + np.bincount(wi[~keep], weights=(KH * rr)[~keep], minlength=Dd)
    HFN = (sp.coo_matrix((KH[keep], (wi[keep], nbr[keep])), shape=(Dd, Dd)) + sp.diags(dFN)).tocsr()
    e, ph = sla.eigsh(HFN, k=1, which='SA', tol=1e-14); ph = ph[:, 0] * np.sign(ph[:, 0] @ g)
    return float(e[0]), ph / np.linalg.norm(ph)


def kpoly(v, k):
    """signed K_k (c2le_gtax.kpoly): lowest Ritz vector of span{v, (H-E)v, ..., (H-E)^k v}."""
    v = v / np.linalg.norm(v)
    if k == 0: return v
    Hv = H @ v; E = v @ Hv; Bs = [v]; HB = [Hv]; u, Hu = v, Hv
    for _ in range(k):
        un = Hu - E * u; Hun = H @ un; Bs.append(un); HB.append(Hun); u, Hu = un, Hun
    m = k + 1; Gm = np.array([[Bs[i] @ Bs[j] for j in range(m)] for i in range(m)])
    Hm = np.array([[Bs[i] @ HB[j] for j in range(m)] for i in range(m)]); Hm = 0.5 * (Hm + Hm.T)
    w_, c = sl.eigh(Hm, Gm); c0 = c[:, 0] / c[0, 0]; w = sum(c0[i] * Bs[i] for i in range(m)); return w / np.linalg.norm(w)


def efn_k(v, k): return fn_state(kpoly(v, k))[0] / N


# ---- the target (C13 variant ii, round 0)
x_opt = np.load(os.path.join(C13, 'flat_opt.npy'))
net_own = RBMPP(); S_v0, s_v0 = net_own.S_sign(x_opt); v0 = vec_of(S_v0, s_v0)
g0 = kpoly(v0, 1); EFN_g0, phi = fn_state(g0); EFN_v0 = fn_state(v0)[0]
lphi = np.log(np.abs(phi)); sphi = np.sign(phi)
EFN_phi = {k: efn_k(phi, k) for k in (0, 1, 2)}
net0 = RBMPP(impose=sphi) if os.environ.get('SIGN', 'target') == 'target' else net_own
p2 = phi ** 2; DEC = np.floor(np.log10(p2)).astype(int); DECS = sorted(set(DEC.tolist()), reverse=True); G = len(DECS)


def tax(v): return {f'tax_{k}': efn_k(v, k) - EFN_phi[k] for k in (0, 1, 2)}


# ---- arm A operator (fixed: depends on the target only).  Rows: sqrt(p_x) e_x and sqrt(p_x |H| r / wbar_d) (e_x - e_y)
pd = np.zeros(Dd); wbar = {}
viol_t = phi[wi] * phi[nbr] > 0                                       # H > 0: violating for the target = same sign
Wx = np.bincount(wi[viol_t], weights=(KH * np.abs(phi[nbr] / phi[wi]))[viol_t], minlength=Dd)
for d in DECS:
    m = DEC == d; pd[m] = 1.0 / (G * m.sum()); wbar[d] = Wx[m].mean()
_wbx = np.array([wbar[d] for d in DEC]); wb_dir = pd[AXv := wi[viol_t]] * KH[viol_t] * np.abs(phi[nbr[viol_t]] / phi[AXv]) / _wbx[AXv]   # directed (x -> y) bond weights
AX, AY = wi[viol_t], nbr[viol_t]
nb_A = AX.size
A_rows = sp.vstack([sp.diags(np.sqrt(pd)), sp.csr_matrix((np.concatenate([np.sqrt(wb_dir), -np.sqrt(wb_dir)]),
                    (np.concatenate([np.arange(nb_A)] * 2), np.concatenate([AX, AY]))), shape=(nb_A, Dd))]).tocsr()
MA_ = (A_rows.T @ A_rows).tocsr(); a1 = np.sqrt(pd)                    # A^T A and A 1 (= sqrt(p) on the x rows, 0 on bonds)


def loss_A(S, c):
    e = S - lphi - c; r = A_rows @ e; return float(r @ r)


def LA_terms(S, c):
    e = S - lphi - c; return float(np.sum(pd * e * e)), float(np.sum(wb_dir * (e[AX] - e[AY]) ** 2))


# ---- L_fix (numpy) and its residual form
def lfix_parts(v, ph=phi):
    """residuals on the pairs violating for v; returns (L, mask, r, dr/dv_x, dr/dv_y)."""
    vx, vy = v[PX], v[PY]; m = vx * vy > 0                               # H > 0
    vx, vy, fx, fy, k = vx[m], vy[m], ph[PX][m], ph[PY][m], PK[m]
    q = np.sqrt(np.abs(vx * vy)); num = vx * fy - fx * vy; sk = np.sqrt(k)
    r = sk * num / q
    drx = sk * (fy / q - 0.5 * num / (q * vx)); dry = sk * (-fx / q - 0.5 * num / (q * vy))
    return float(r @ r), m, r, drx, dry


def lfix(v, ph=phi): return lfix_parts(v, ph)[0]


def eq5_sum(v, ph=phi):
    """board eq. 5 summed over the pairs violating for v: sum 2|H phi_x phi_y| (cosh(eps_y - eps_x) - 1), v = phi e^eps (same signs)."""
    m = v[PX] * v[PY] > 0; eps = np.log(np.abs(v)) - np.log(np.abs(ph))
    return float(np.sum(2 * PK[m] * np.abs(ph[PX][m] * ph[PY][m]) * (np.cosh(eps[PY][m] - eps[PX][m]) - 1)))


# ---- diagnostics of a fit
def diagnostics(S, s):
    v = vec_of(S, s); gs = np.sign(np.sum(v * phi)); v = v * gs
    e = S - lphi; off = np.sum(p2 * e); ew = e - off
    out = dict(tax(v), sign_wrong_weight_phi2=float(p2[np.sign(v) != sphi].sum()), rms_log_err_phi2=float(np.sqrt(np.sum(p2 * ew ** 2))),
               L_fix=lfix(v), fidelity=float((v @ phi) ** 2))
    cA = float(np.sum(pd * (S - lphi)) / np.sum(pd)); out['L_A'] = loss_A(S, cA); out['L_A_c_opt'] = cA
    # review amendment 5: tax from the tail (phi^2 < 1e-7) vs the bulk: keep the fit's log error in one region, exact |phi| in the other
    tail = p2 < 1e-7; eb = e - np.sum(p2[~tail] * e[~tail]) / np.sum(p2[~tail])
    for nm, keep in (('bulk_err_only', ~tail), ('tail_err_only', tail)):
        h_ = sphi * np.abs(phi) * np.exp(np.where(keep, eb, 0.0)); out[nm] = tax(h_ / np.linalg.norm(h_))
    out['tail_def'] = dict(phi2_below=1e-7, n=int(tail.sum()), weight=float(p2[tail].sum()))
    out['fit_rms_by_phi2_decade'] = {}
    for d in DECS:
        m = DEC == d; me = e[m].mean()
        out['fit_rms_by_phi2_decade'][str(d)] = dict(n=int(m.sum()), weight=float(p2[m].sum()), rms=float(np.sqrt(np.mean((e[m] - me) ** 2))),
                                                     sign_wrong_frac=float(np.mean(np.sign(v[m]) != sphi[m])))
    vt = phi[PX] * phi[PY] > 0; de = e[PX] - e[PY]; wgt = PK * np.abs(phi[PX] * phi[PY]); out['pair_class'] = {}
    for name, m in (('viol_target_NN', vt & ~PNNN), ('viol_target_NNN', vt & PNNN), ('nonviol_target_NN', ~vt & ~PNNN), ('nonviol_target_NNN', ~vt & PNNN)):
        out['pair_class'][name] = dict(n=int(m.sum()), rms_de_wHpp=float(np.sqrt(np.sum(wgt[m] * de[m] ** 2) / np.sum(wgt[m]))),
                                       sign_flip_frac=float(np.mean((np.sign(v[PX[m]] * v[PY[m]]) != np.sign(phi[PX[m]] * phi[PY[m]])))))
    return out


# ============================================================ LM fit (c2le run_fit / solve_dir, exact sums)
def gn_system(arm, net, th, c):
    """returns loss, Sm (P+1 x P+1), g (P+1), Gx = dS/dtheta (for the trust region)."""
    S, s, dS = net.S_sign_jac(th)
    if arm == 'A':
        e = S - lphi - c; r = A_rows @ e; Ar = A_rows.T @ r
        Jt = dS; MJ = MA_ @ Jt; Sm_tt = Jt.T @ MJ; Sm_tc = -(Jt.T @ (A_rows.T @ np.concatenate([a1, np.zeros(nb_A)])))
        Sm_cc = float(a1 @ a1)
        Sm = np.block([[Sm_tt, Sm_tc[:, None]], [Sm_tc[None, :], np.array([[Sm_cc]])]])
        gvec = np.concatenate([Jt.T @ Ar, [-float(a1 @ r[:Dd])]])
        return float(r @ r), Sm, gvec, dS
    v = s * np.exp(S - S.max())                                          # constant scale: L_fix is scale invariant
    Lf, m, r, drx, dry = lfix_parts(v)
    jx = drx * v[PX][m]; jy = dry * v[PY][m]                              # dr/dS = dr/dv * v
    nr = r.size; A = sp.csr_matrix((np.concatenate([jx, jy]), (np.concatenate([np.arange(nr)] * 2), np.concatenate([PX[m], PY[m]]))), shape=(nr, Dd))
    M = (A.T @ A).tocsr(); Sm_tt = dS.T @ (M @ dS); gt = dS.T @ (A.T @ r)
    Sm = np.zeros((net.P + 1, net.P + 1)); Sm[:net.P, :net.P] = Sm_tt; gvec = np.concatenate([gt, [0.0]])
    return Lf, Sm, gvec, dS


def loss_of(arm, net, th, c):
    S, s = net.S_sign(th)
    return loss_A(S, c) if arm == 'A' else lfix(s * np.exp(S - S.max()))


# ---- arm D (lead amendment 2026-10-27): ideal ceiling, train on the exact E_FN(K_2(fit)), Hellmann-Feynman force with the exact
#      FN ground state chi of the current K_2(fit) (recomputed every step); natural gradient (metric = QGT of the fit, v^2 weights),
#      same lam_rel / trust-region tau / accept-iff-loss-falls rules as A and B.  loss_D = E_FN(K_2 fit) - E0 (total).
jwi, jnb, jK = jnp.asarray(wi), jnp.asarray(nbr), jnp.asarray(KH); jdg = jnp.asarray(diag); REPOFj = jnp.asarray(REPOF)
def Hmv(x): return jdg * x + jax.ops.segment_sum(jK * x[jnb], jwi, num_segments=Dd)
def g2_jax(fl, freeze_c=False):
    l = lp_orb(fl)[REPOFj]; l = l - jax.lax.stop_gradient(jnp.max(jnp.real(l))); v = jnp.real(W30 * jnp.exp(l))
    if net0.impose is not None: v = jnp.asarray(sphi) * jnp.abs(v)          # imposed target sign
    v = v / jnp.sqrt(v @ v)
    Hv = Hmv(v); E = v @ Hv; u1 = Hv - E * v; Hu1 = Hmv(u1); u2 = Hu1 - E * u1; Hu2 = Hmv(u2)
    B = jnp.stack([v, u1, u2]); HB = jnp.stack([Hv, Hu1, Hu2]); Gm = B @ B.T; Hm = B @ HB.T; Hm = 0.5 * (Hm + Hm.T)
    Lc = jnp.linalg.cholesky(Gm); Li = jnp.linalg.inv(Lc); w_, Y = jnp.linalg.eigh(Li @ Hm @ Li.T); c = Li.T @ Y[:, 0]; c = c / c[0]
    if freeze_c: c = jax.lax.stop_gradient(c)                             # BROKEN control: Ritz coefficients not differentiated
    g = c @ B; return g / jnp.sqrt(g @ g)
def hf_surr(fl, chi2, nk, freeze_c=False):
    g = g2_jax(fl, freeze_c); rr = g[jnb] / g[jwi]; return chi2 @ jax.ops.segment_sum(jK * rr * nk, jwi, num_segments=Dd)
_hf_grad = jax.jit(jax.grad(hf_surr), static_argnums=3)
def efn2_force(th, freeze_c=False):
    """exact E_FN(K_2 fit) (total) and its HF gradient."""
    S, s = net0.S_sign(th); g = kpoly(vec_of(S, s), 2); e, chi = fn_state(g)
    nk = (g[nbr] / g[wi] >= 0).astype(np.float64)
    gr = np.asarray(_hf_grad(jnp.asarray(th), jnp.asarray(chi ** 2), jnp.asarray(nk), freeze_c)); return e, gr
def efn2(th): S, s = net0.S_sign(th); return fn_state(kpoly(vec_of(S, s), 2))[0]
def efn0(th): S, s = net0.S_sign(th); return fn_state(vec_of(S, s))[0]
def efn0_force(th, broken=False):
    """stage (c) of l8_loop4x4.py (cfc2681, efn0_vg) on all 392 parameters: exact E_FN(fit) (total) and the HF gradient
    sum_{x, y viol} chi_x^2 H_xy d(v_y/v_x)/dtheta = sum chi_x^2 K |v_y/v_x| (dS_y - dS_x).  BROKEN: ratio factor dropped."""
    S, s, dS = net0.S_sign_jac(th); v = vec_of(S, s); e, chi = fn_state(v); m = v[nbr] / v[wi] >= 0
    cxy = (chi[wi] ** 2 * KH * (1.0 if broken else np.exp(S[nbr] - S[wi])))[m]
    a = np.bincount(nbr[m], weights=cxy, minlength=Dd) - np.bincount(wi[m], weights=cxy, minlength=Dd)
    return e, dS.T @ a
FORCE = {'D0': (efn0_force, efn0), 'D2': (efn2_force, efn2)}


def run_fit_D(kind, th0, steps, tag):
    force, efn = FORCE[kind]
    th = np.array(th0, dtype=np.float64); Lcur = efn(th) - E0; curve = [dict(step=0, loss=Lcur)]; hist = []
    tau = TAU0; tau_changed = 0; tau_hist = [dict(step=0, tau=tau)]; lam_rel = LAM0; t0 = time.time(); snaps = {}
    for n in range(1, steps + 1):
        ts = time.time(); e, gr = force(th); Lval = e - E0
        S, s, dS = net0.S_sign_jac(th); w2 = np.exp(2 * (S - S.max())); w2 /= w2.sum()
        Oc = dS - w2 @ dS; Sm = (Oc * w2[:, None]).T @ Oc; Pp = Sm.shape[0]; lam = lam_rel * np.trace(Sm) / Pp
        dP = -np.linalg.solve(Sm + lam * np.eye(Pp), gr)
        pc = dS @ dP; pc = pc - np.sum(pd * pc) / np.sum(pd); prms = float(np.sqrt(np.sum(pd * pc * pc) / np.sum(pd)))
        sc = min(1.0, tau / max(prms, 1e-30)); cand = th + sc * dP; Lnew = efn(cand) - E0; lam_used = lam_rel
        if np.isfinite(Lnew) and Lnew < Lval: ok = True; lam_rel = max(lam_rel / 2, LAM_MIN); th, Lcur = cand, Lnew
        else: ok = False; lam_rel = min(lam_rel * 4, LAM_CAP)
        hist.append(dict(step=n, ok=ok, L_old=Lval, L_new=float(Lnew), lam_rel=lam_used, pred_rms=prms, trust_scale=sc, tau=tau, sec=time.time() - ts))
        if n % CHK_EVERY == 0:
            curve.append(dict(step=n, loss=Lcur))
            if tau > TAU_MIN and n - tau_changed >= ANNEAL:
                cb = [q for q in curve if q['step'] <= n - ANNEAL]
                if cb and abs(cb[-1]['loss'] - Lcur) / max(abs(Lcur), 1e-300) < FLAT_TOL:
                    tau = max(tau / 2, TAU_MIN); tau_changed = n; tau_hist.append(dict(step=n, tau=tau))
            log(f'[{tag}] step {n}: E_FN(K_{kind[1]})-E0 /N {Lcur / N:.6e} acc(10) {np.mean([h["ok"] for h in hist[-10:]]):.1f} lam_rel {lam_rel:.1e} tau {tau:.4f} {hist[-1]["sec"]:.2f}s/step')
        if n in (100, 200, 300, 400, 600, 800):
            S_, s_ = net0.S_sign(th); snaps[n] = tax(vec_of(S_, s_))
    return dict(theta=th, c=0.0, loss=Lcur, curve=curve, hist=hist, tau_hist=tau_hist, wall=time.time() - t0,
                acceptance=float(np.mean([h['ok'] for h in hist])) if hist else None, tax_snapshots=snaps)


def run_fit(arm, net, th0, steps, tag):
    th = np.array(th0, dtype=np.float64); S, s = net.S_sign(th)
    c = float(np.sum(pd * (S - lphi)) / np.sum(pd)) if arm == 'A' else 0.0
    Lcur = loss_of(arm, net, th, c); curve = [dict(step=0, loss=Lcur)]; hist = []
    tau = TAU0; tau_changed = 0; tau_hist = [dict(step=0, tau=tau)]; lam_rel = LAM0; t0 = time.time()
    snaps = {}
    for n in range(1, steps + 1):
        ts = time.time()
        Lval, Sm, gvec, Gx = gn_system(arm, net, th, c)
        Pp = Sm.shape[0]; lam = lam_rel * np.trace(Sm) / Pp
        d = -np.linalg.solve(Sm + lam * np.eye(Pp), gvec); dP, dc = d[:-1], d[-1]
        pc = Gx @ dP; pc = pc - np.sum(pd * pc) / np.sum(pd); prms = float(np.sqrt(np.sum(pd * pc * pc) / np.sum(pd)))
        sc = min(1.0, tau / max(prms, 1e-30))
        cand = th + sc * dP; cc = c + sc * dc; Lnew = loss_of(arm, net, cand, cc); lam_used = lam_rel
        if np.isfinite(Lnew) and Lnew < Lval: ok = True; lam_rel = max(lam_rel / 2, LAM_MIN); th, c, Lcur = cand, cc, Lnew
        else: ok = False; lam_rel = min(lam_rel * 4, LAM_CAP)
        hist.append(dict(step=n, ok=ok, L_old=Lval, L_new=float(Lnew), lam_rel=lam_used, pred_rms=prms, trust_scale=sc, tau=tau, sec=time.time() - ts))
        if n % CHK_EVERY == 0:
            curve.append(dict(step=n, loss=Lcur))
            if tau > TAU_MIN and n - tau_changed >= ANNEAL:
                cb = [q for q in curve if q['step'] <= n - ANNEAL]
                if cb and abs(cb[-1]['loss'] - Lcur) / max(abs(Lcur), 1e-300) < FLAT_TOL:
                    tau = max(tau / 2, TAU_MIN); tau_changed = n; tau_hist.append(dict(step=n, tau=tau))
            log(f'[{tag}] step {n}: loss {Lcur:.6e} acc(10) {np.mean([h["ok"] for h in hist[-10:]]):.1f} lam_rel {lam_rel:.1e} tau {tau:.4f} {hist[-1]["sec"]:.2f}s/step')
        if n in (100, 200, 300):
            S_, s_ = net.S_sign(th); snaps[n] = tax(vec_of(S_, s_))
    return dict(theta=th, c=c, loss=Lcur, curve=curve, hist=hist, tau_hist=tau_hist, wall=time.time() - t0,
                acceptance=float(np.mean([h['ok'] for h in hist])) if hist else None, tax_snapshots=snaps)


def sha(fn): return hashlib.sha256(open(fn, 'rb').read()).hexdigest()


CODE_SHA = sha(os.path.abspath(__file__))
HEAD = dict(sign=('target (imposed)' if net0.impose is not None else 'own'), code_sha256=CODE_SHA, jax=jax.__version__, Dd=Dd, orbits=NORB, P_rbmpp=P, P_pos=PosRBM.P, E0N=E0 / N, steps=STEPS, sigma_seed=SIGMA_SEED,
            decades={str(d): int((DEC == d).sum()) for d in DECS}, n_pairs=int(PX.size), n_viol_pairs_target=int(np.sum(phi[PX] * phi[PY] > 0)),
            EFN_phi_site=EFN_phi, target='phi_0 = FN(K_1(v_0)), C13 variant ii round 0, ' + os.path.join(C13, 'flat_opt.npy'))

# ============================================================ MODE setup: reproduction + checks
if MODE == 'setup':
    res = dict(HEAD); chk = {}
    # fast path vs brute rbmpp_lib on all params perturbed (FASTCHK as in C13, but every parameter live)
    xp = x_opt + 1e-2 * np.random.default_rng(1).standard_normal(P)
    a = np.asarray(_lp_orb_j(jnp.asarray(xp)))[REPOF]; b = np.asarray(lpv_brute(jnp.asarray(xp), jnp.asarray(X.astype(np.float64))))
    chk['fastchk_max_dpsi'] = float(np.max(np.abs(np.exp(a - a.real.max()) - np.exp(b - b.real.max()))))
    log(f'orbits {NORB}; P {P}; FASTCHK all params live, perturbed: max|psi_fast - psi_brute| {chk["fastchk_max_dpsi"]:.2e}')
    # reproduction of the C13 records (loop4x4/rbmonly)
    xi1 = np.load(os.path.join(C13, 'flat_i_1.npy')); xii1 = np.load(os.path.join(C13, 'flat_ii_1.npy'))
    vi1 = vec_of(*net_own.S_sign(xi1)); vii1 = vec_of(*net_own.S_sign(xii1))
    rep = dict(E0N=(E0 / N, REC['E0N']), EFN_v0N=(EFN_v0 / N, REC['EFN_v0N']), EFN_g0N=(EFN_g0 / N, REC['EFN_g0N']),
               EFN_round1_i_vN=(fn_state(vi1)[0] / N, REC['EFN_i1_vN']), EFN_round1_ii_K1N=(efn_k(vii1, 1), REC['EFN_ii1_gN']))
    res['reproduction'] = {k: dict(this=v[0], record=v[1], dev=v[0] - v[1]) for k, v in rep.items()}
    for k, v in res['reproduction'].items(): log(f'REPRO {k}: this {v["this"]:.13f} record {v["record"]:.13f} dev {v["dev"]:.1e}')
    res['C13_fidelity_refit_ii1_own_sign'] = diagnostics(*net_own.S_sign(xii1)); res['warm_start_v0_own_sign'] = diagnostics(S_v0, s_v0)
    res['C13_fidelity_refit_ii1'] = diagnostics(*net0.S_sign(xii1)); res['warm_start_v0'] = diagnostics(*net0.S_sign(x_opt))
    res['sign_floor_absphi_x_sign_v0'] = tax(np.abs(phi) * s_v0 * np.sign(np.sum(np.abs(phi) * s_v0 * phi)))
    log(f'sign floor |phi| x sign(v_0): {res["sign_floor_absphi_x_sign_v0"]}')
    log(f'C13 fidelity refit a_1(ii) (RBM-only, L-BFGS to convergence): tax_0 {res["C13_fidelity_refit_ii1"]["tax_0"]:.4e} '
        f'tax_1 {res["C13_fidelity_refit_ii1"]["tax_1"]:.4e} tax_2 {res["C13_fidelity_refit_ii1"]["tax_2"]:.4e}')
    log(f'warm start v_0: tax_0 {res["warm_start_v0"]["tax_0"]:.4e} tax_1 {res["warm_start_v0"]["tax_1"]:.4e} tax_2 {res["warm_start_v0"]["tax_2"]:.4e}')
    # check 1: L_fix = 0 at psi_T = phi; broken formula (indices swapped) must not vanish; perturbed phi must not vanish
    def lfix_broken_swap(v, ph=phi):
        m = v[PX] * v[PY] > 0; return float(np.sum(PK[m] * (v[PX][m] * ph[PX][m] - ph[PY][m] * v[PY][m]) ** 2 / np.abs(v[PX][m] * v[PY][m])))
    pert = phi * np.exp(1e-3 * np.random.default_rng(2).standard_normal(Dd))
    chk['L_fix_at_phi'] = lfix(phi); chk['L_fix_at_phi_scaled_by_7'] = lfix(7.0 * phi); chk['BROKEN_swapped_formula_at_phi'] = lfix_broken_swap(phi)
    chk['L_fix_at_phi_eps1e-3'] = lfix(pert)
    chk['check1_pass'] = bool(chk['L_fix_at_phi'] < 1e-28 and chk['L_fix_at_phi_scaled_by_7'] < 1e-26 and chk['BROKEN_swapped_formula_at_phi'] > 1e-6 and chk['L_fix_at_phi_eps1e-3'] > 0)
    log('CHECK 1 L_fix(phi) = 0:', json.dumps({k: chk[k] for k in ('L_fix_at_phi', 'L_fix_at_phi_scaled_by_7', 'BROKEN_swapped_formula_at_phi', 'L_fix_at_phi_eps1e-3', 'check1_pass')}))
    # check 2: L_fix(phi e^eps) = eq. 5 summed (identity), and dE_FN / L_fix ~ 0.65 (fn_weight_check.py, around psi0 and around phi)
    rows = []
    for base_name, base in (('psi0_exact_GS (fn_weight_check setting)', psi0 * np.sign(psi0 @ phi)), ('phi (this target)', phi)):
        Eb = fn_state(base)[0]
        for sig in (0.01, 0.03, 0.1, 0.3):
            eps = np.random.default_rng(0).normal(0, sig, Dd); pt = base * np.exp(eps)
            Lf = lfix(pt, base); e5 = eq5_sum(pt, base); dE = fn_state(pt)[0] - Eb
            m_all = np.ones(PX.size, bool)
            broken = float(np.sum(2 * PK * np.abs(base[PX] * base[PY]) * (np.cosh(eps[PY] - eps[PX]) - 1)))   # BROKEN: all pairs, not violating
            rows.append(dict(base=base_name, sigma=sig, L_fix=Lf, eq5_sum=e5, rel_identity=abs(Lf - e5) / e5, dE_FN_total=dE, ratio_dE_over_Lfix=dE / Lf,
                             BROKEN_all_pairs_sum=broken, BROKEN_rel=abs(Lf - broken) / Lf))
            log(f'CHECK 2 [{base_name}] sigma {sig}: L_fix {Lf:.6e} eq5 {e5:.6e} rel {abs(Lf - e5) / e5:.1e} | dE_FN {dE:.4e} ratio {dE / Lf:.3f} | BROKEN all-pairs rel {abs(Lf - broken) / Lf:.2f}')
    chk['eq5'] = rows; chk['check2_identity_pass'] = bool(max(r['rel_identity'] for r in rows) < 1e-10 and min(r['BROKEN_rel'] for r in rows) > 1e-3)
    # check 3: gradient of L_fix by autodiff (end-to-end jax, pair set frozen at theta) vs central FD on 3 parameters; GN gradient vs autodiff
    xc = x_opt + 1e-3 * np.random.default_rng(3).standard_normal(P)
    net_gc = net_own; S_c, s_c = net_gc.S_sign(xc); vref = s_c * np.exp(S_c - S_c.max()); mset = jnp.asarray(vref[PX] * vref[PY] > 0); scale = float(np.exp(S_c.max()))
    PXj, PYj, PKj, phij = jnp.asarray(PX), jnp.asarray(PY), jnp.asarray(PK), jnp.asarray(phi)
    def lfix_jax(fl, broken=False):
        l = lp_orb(fl)[jnp.asarray(REPOF)]; v = jnp.real(W30 * jnp.exp(l)) / scale
        vx, vy = v[PXj], v[PYj]; den = jnp.abs(vx * vy)
        if broken: den = jax.lax.stop_gradient(den)                         # BROKEN: denominator not differentiated
        return jnp.sum(jnp.where(mset, PKj * (vx * phij[PYj] - phij[PXj] * vy) ** 2 / jnp.where(mset, den, 1.0), 0.0))
    gj = np.asarray(jax.grad(lfix_jax)(jnp.asarray(xc))); gb = np.asarray(jax.grad(lambda f: lfix_jax(f, True))(jnp.asarray(xc)))
    _, _, gvec, _ = gn_system('B', net_gc, xc, 0.0); g_gn = 2 * gvec[:P]
    idx3 = [int(np.argmax(np.abs(gj[:64]))), 136 + int(np.argmax(np.abs(gj[136:]))), 128 + int(np.argmax(np.abs(gj[128:136])))]   # one Wim, one f, one bre/bim
    fd = []
    for i in idx3:
        h = 1e-6; e_ = np.zeros(P); e_[i] = h
        fp = float(lfix_jax(jnp.asarray(xc + e_))); fm = float(lfix_jax(jnp.asarray(xc - e_)))
        f_np = (lfix(vec_of(*net_gc.S_sign(xc + e_))) - lfix(vec_of(*net_gc.S_sign(xc - e_)))) / (2 * h)   # numpy L_fix, set re-evaluated
        fd.append(dict(param=i, fd_jax=(fp - fm) / (2 * h), fd_numpy_set_free=f_np, autodiff=float(gj[i]), gn=float(g_gn[i]), BROKEN=float(gb[i])))
    for q in fd:
        q['rel_autodiff'] = abs(q['autodiff'] - q['fd_jax']) / abs(q['fd_jax']); q['rel_gn'] = abs(q['gn'] - q['fd_jax']) / abs(q['fd_jax'])
        q['rel_numpy_fd'] = abs(q['fd_numpy_set_free'] - q['fd_jax']) / abs(q['fd_jax']); q['BROKEN_rel'] = abs(q['BROKEN'] - q['fd_jax']) / abs(q['fd_jax'])
        log(f'CHECK 3 param {q["param"]}: FD {q["fd_jax"]:.8e} autodiff {q["autodiff"]:.8e} (rel {q["rel_autodiff"]:.1e}) GN {q["gn"]:.8e} (rel {q["rel_gn"]:.1e}) '
            f'FD numpy {q["fd_numpy_set_free"]:.8e} (rel {q["rel_numpy_fd"]:.1e}) BROKEN {q["BROKEN"]:.4e} (rel {q["BROKEN_rel"]:.2f})')
    chk['grad'] = fd; chk['check3_pass'] = bool(all(q['rel_autodiff'] < 1e-5 and q['rel_gn'] < 1e-5 for q in fd) and all(q['BROKEN_rel'] > 1e-2 for q in fd))
    chk['grad_full_rel_gn_vs_autodiff'] = float(np.linalg.norm(g_gn - gj) / np.linalg.norm(gj))
    # control: 0 LM steps returns the warm start bitwise and its tax
    z = run_fit('A', net0, x_opt, 0, 'zero-step A'); zb = run_fit('B', net0, x_opt, 0, 'zero-step B')
    chk['zero_step'] = dict(A_bitwise=bool(np.array_equal(z['theta'], x_opt)), B_bitwise=bool(np.array_equal(zb['theta'], x_opt)))
    o1 = run_fit('A', net0, x_opt, 1, 'one-step A (BROKEN zero-step)'); chk['zero_step']['BROKEN_one_step_A_bitwise'] = bool(np.array_equal(o1['theta'], x_opt))
    chk['zero_step_pass'] = bool(chk['zero_step']['A_bitwise'] and chk['zero_step']['B_bitwise'] and not chk['zero_step']['BROKEN_one_step_A_bitwise'])
    log('CHECK zero-step', json.dumps(chk['zero_step']))
    res['checks'] = chk; res['sec'] = time.time() - T00
    np.save(os.path.join(OUT, 'phi.npy'), phi)
    open(os.path.join(OUT, 'setup.json'), 'w').write(jd(res))
    ok = chk['check1_pass'] and chk['check2_identity_pass'] and chk['check3_pass'] and chk['zero_step_pass'] and chk['fastchk_max_dpsi'] < 1e-10 \
        and all(abs(v['dev']) < 1e-9 for v in res['reproduction'].values())
    log(f'VERDICT SETUP {"ALL PASS" if ok else "FAIL"} ({res["sec"]:.0f}s)'); sys.exit(0 if ok else 2)

# ============================================================ MODE fit
if MODE == 'fit':
    ARM = sys.argv[3]; SEED = int(sys.argv[4]); tag = f'{ARM} s{SEED}'
    ph_saved = np.load(os.path.join(OUT, 'phi.npy'))
    if not np.array_equal(ph_saved, phi): log('TARGET DRIFT vs setup phi.npy; exit 3'); sys.exit(3)
    rng = np.random.default_rng(SEED)
    if ARM in ('D0', 'D2'):                                           # HF force check vs FD of the exact E_FN(K_k) before training
        force, efn = FORCE[ARM]
        th0 = x_opt + SIGMA_SEED * rng.standard_normal(P); e, gr = force(th0); _, grb = force(th0, True); hchk = []
        for i in (int(np.argmax(np.abs(gr[:64]))), 136 + int(np.argmax(np.abs(gr[136:])))):
            h = 1e-5; ei = np.zeros(P); ei[i] = h; fd = (efn(th0 + ei) - efn(th0 - ei)) / (2 * h)
            hchk.append(dict(param=i, fd=fd, hf=float(gr[i]), rel=abs(gr[i] - fd) / abs(fd), BROKEN_c_frozen=float(grb[i]), BROKEN_rel=abs(grb[i] - fd) / abs(fd)))
            log(f'[{tag}] HF CHECK param {i}: FD {fd:.8e} HF {gr[i]:.8e} rel {hchk[-1]["rel"]:.1e}; BROKEN (D2: Ritz c frozen; D0: ratio dropped) rel {hchk[-1]["BROKEN_rel"]:.2e}')
        if not all(q['rel'] < 1e-4 and q['BROKEN_rel'] > 1e-3 for q in hchk): log(f'[{tag}] HF CHECK FAILED; exit 6'); sys.exit(6)
        net = net0; arm_loss = ARM
    elif ARM in ('A', 'B'):
        net = net0; th0 = x_opt + SIGMA_SEED * rng.standard_normal(P); arm_loss = ARM
    else:
        net = PosRBM(); th0 = 1e-2 * rng.standard_normal(PosRBM.P); arm_loss = 'B'
    S0, s0 = net.S_sign(th0); start = diagnostics(S0, s0)
    log(f'[{tag}] P {net.P} start: tax_0 {start["tax_0"]:.4e} tax_1 {start["tax_1"]:.4e} tax_2 {start["tax_2"]:.4e} L_A {start["L_A"]:.4e} L_fix {start["L_fix"]:.4e}')
    fr = run_fit_D(ARM, th0, STEPS, tag) if ARM in ('D0', 'D2') else run_fit(arm_loss, net, th0, STEPS, tag)
    S1, s1 = net.S_sign(fr['theta']); fin = diagnostics(S1, s1)
    np.save(os.path.join(OUT, f'theta_{ARM}_s{SEED}.npy'), fr['theta'])
    res = dict(HEAD, arm=ARM, seed=SEED, P=net.P, hf_check=(hchk if ARM in ('D0', 'D2') else None), start=start, final=fin, loss_final=fr['loss'], c_final=fr['c'], acceptance=fr['acceptance'],
               wall=fr['wall'], curve=fr['curve'], tau_hist=fr['tau_hist'], tax_snapshots=fr['tax_snapshots'],
               hist=[{k: h[k] for k in ('step', 'ok', 'L_old', 'L_new', 'lam_rel', 'pred_rms', 'trust_scale', 'tau')} for h in fr['hist']])
    open(os.path.join(OUT, f'fit_{ARM}_s{SEED}.json'), 'w').write(jd(res))
    log(f'VERDICT FIT {tag} DONE: tax_0 {fin["tax_0"]:.4e} tax_1 {fin["tax_1"]:.4e} tax_2 {fin["tax_2"]:.4e} L_A {fin["L_A"]:.4e} L_fix {fin["L_fix"]:.4e} '
        f'acc {fr["acceptance"]:.2f} {fr["wall"]:.0f}s')
    sys.exit(0)

# ============================================================ MODE summary
if MODE == 'summary':
    import glob
    S = json.load(open(os.path.join(OUT, 'setup.json'))); tab = {}
    for fn in sorted(glob.glob(os.path.join(OUT, 'fit_*_s*.json'))):
        r = json.load(open(fn)); tab.setdefault(r['arm'], []).append(r)
    ws = S['warm_start_v0']
    out = dict(reference_C13=S['C13_fidelity_refit_ii1'], warm_start=ws, sign_floor=S['sign_floor_absphi_x_sign_v0'], arms={})
    print('warm start (imposed sign): ' + ' '.join(f'{k} {ws[k]:.4e}' for k in ('tax_0', 'tax_1', 'tax_2')) +
          f"  | bulk-only tax_0 {ws['bulk_err_only']['tax_0']:.4e} tail-only tax_0 {ws['tail_err_only']['tax_0']:.4e}")
    print('sign floor |phi| x sign(v_0): ' + ' '.join(f'{k} {v:.4e}' for k, v in S['sign_floor_absphi_x_sign_v0'].items()))
    print('C13 fidelity refit a_1(ii), imposed sign: ' + ' '.join(f'{k} {S["C13_fidelity_refit_ii1"][k]:.4e}' for k in ('tax_0', 'tax_1', 'tax_2')))
    for arm, rs in sorted(tab.items()):
        o = dict(seeds=[r['seed'] for r in rs], P=rs[0]['P'], steps=rs[0]['steps'])
        for k in ('tax_0', 'tax_1', 'tax_2', 'L_A', 'L_fix', 'rms_log_err_phi2', 'fidelity'):
            a = np.array([r['final'][k] for r in rs]); o[k] = dict(mean=float(a.mean()), min=float(a.min()), max=float(a.max()), values=a.tolist())
        for reg in ('bulk_err_only', 'tail_err_only'):
            a = np.array([r['final'][reg]['tax_0'] for r in rs]); o[reg + '_tax_0'] = dict(mean=float(a.mean()), min=float(a.min()), max=float(a.max()))
        o['R_vs_warm_start'] = {k: o[k]['mean'] / ws[k] for k in ('tax_0', 'tax_2')}
        o['fit_rms_by_phi2_decade_mean'] = {d: float(np.mean([r['final']['fit_rms_by_phi2_decade'][d]['rms'] for r in rs])) for d in rs[0]['final']['fit_rms_by_phi2_decade']}
        o['pair_class_rms_mean'] = {c: float(np.mean([r['final']['pair_class'][c]['rms_de_wHpp'] for r in rs])) for c in rs[0]['final']['pair_class']}
        o['acceptance'] = [r['acceptance'] for r in rs]
        out['arms'][arm] = o
        print(f'arm {arm} (P {o["P"]}, steps {o["steps"]}, seeds {o["seeds"]}): ' + '  '.join(f'{k} {o[k]["mean"]:.4e} [{o[k]["min"]:.4e}, {o[k]["max"]:.4e}]' for k in ('tax_0', 'tax_1', 'tax_2', 'L_A', 'L_fix'))
              + f'  | bulk-only tax_0 {o["bulk_err_only_tax_0"]["mean"]:.4e} tail-only {o["tail_err_only_tax_0"]["mean"]:.4e} | R(tax_0, tax_2) vs warm {o["R_vs_warm_start"]["tax_0"]:.3f} {o["R_vs_warm_start"]["tax_2"]:.3f}')
    open(os.path.join(OUT, 'summary.json'), 'w').write(jd(out))
