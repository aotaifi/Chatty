"""Exact 4x4 loop test: guide -> exact FN ground state -> fidelity refit (same class) -> next guide.  No Monte Carlo.
   python l8_loop4x4.py OUTDIR STAGE     env MASK = 'f' (f-only) | 'Wim,Wre,bim,bre' (RBM-only)
   STAGE opt   : converged E_v optimum of v = Re[w psi_P] (w = best_w(guide30), fixed), from the 4x4 guide30 (params4.npz)
                 -> OUTDIR/flat_opt.npy, opt.json
   STAGE loopi : variant (i)  no hop : phi_k = FN ground state with guide v(a_k);              a_{k+1} = argmax F(a, phi_k)
   STAGE loopiii: variant (iii) = (i) with a node-weighted fidelity (see node_weight)
   STAGE loopii: variant (ii) one hop: g_k = v_k + alpha_k (H - E_k) v_k (Ritz alpha), phi_k = FN(g_k); a_{k+1} = argmax F(a, phi_k)
                 5 rounds, refit = chunked L-BFGS to convergence, warm start a_k.  Controls at round 0 (zero-step refit + 1-step broken,
                 1%-weight sign-flipped target, identity G_g = alpha dE_v + broken wrong-E hop).  -> OUTDIR/loop_<variant>.json
   Setup (lattice, H, ansatz, v, best_w, Ritz hop, FN Hamiltonian) copied from l8_idtest4x4.py.
"""
import os, sys, json, time
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sla, scipy.linalg as sl, scipy.optimize as so
R = '/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/experiments'
for d in (R + '/gfmc_fn_lattice/l8', R + '/gfmc_fn_lattice', R + '/rbmpp_6x6'): sys.path.insert(0, d)
import jax, jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
from jax.flatten_util import ravel_pytree
from rbmpp_lib import Lattice, make_fns

OUT = sys.argv[1]; STAGE = sys.argv[2]; os.makedirs(OUT, exist_ok=True)
MASK = os.environ.get('MASK', 'f'); NROUND = int(os.environ.get('NROUND', 5))
MAXEV_OPT = int(os.environ.get('MAXEV_OPT', 30000)); MAXEV_FIT = int(os.environ.get('MAXEV_FIT', 8000)); CHUNK = int(os.environ.get('CHUNK', 500))
TOL_OPT = float(os.environ.get('TOL_OPT', 1e-11)); TOL_FIT = float(os.environ.get('TOL_FIT', 1e-11))
PROG = os.path.join(OUT, 'progress.txt')


def log(*a):
    s = time.strftime('%Y-%m-%dT%H:%M:%S') + ' PROGRESS ' + ' '.join(str(x) for x in a)
    print(s, flush=True)
    with open(PROG, 'a') as fh: fh.write(s + '\n')


L = 4; N = 16; lat = Lattice(L, 0.5, 'translations')
BI = np.asarray(lat.BI); BJ = np.asarray(lat.BJ); JB = np.asarray(lat.JB, float)
basis = np.array([s for s in range(1 << N) if bin(s).count('1') == N // 2], np.int64)
Dd = len(basis); lookup = -np.ones(1 << N, np.int64); lookup[basis] = np.arange(Dd)
X = ((basis[:, None] >> np.arange(N)) & 1).astype(np.int8) * 2 - 1
anti = X[:, BI] != X[:, BJ]; wi, bi = np.nonzero(anti)
nbr = lookup[basis[wi] ^ (1 << BI[bi]) ^ (1 << BJ[bi])]
diag = 0.25 * np.sum(JB[None] * X[:, BI] * X[:, BJ], 1)
H = (sp.coo_matrix((0.5 * JB[bi], (wi, nbr)), shape=(Dd, Dd)).tocsr() + sp.diags(diag)).tocsr()
E0 = sla.eigsh(H, k=1, which='SA', tol=1e-14)[0][0]
jwi, jnb, jK, jdg = jnp.asarray(wi), jnp.asarray(nbr), jnp.asarray(0.5 * JB[bi]), jnp.asarray(diag)
def Hmv(x): return jdg * x + jax.ops.segment_sum(jK * x[jnb], jwi, num_segments=Dd)

z0 = np.load('/scratch/a/A.Otaifi/l8/check4x4/params4.npz')
KEYS = ('f', 'bre', 'bim', 'Wre', 'Wim')
p0 = {k: jnp.asarray(z0[k]) for k in KEYS}
flat_g30, unrav = ravel_pytree(p0); flat_g30 = np.asarray(flat_g30); P = flat_g30.size
fns = make_fns(lat, jnp.float64, True)
Xj = jnp.asarray(X.astype(np.float64))
lpv = jax.vmap(lambda fl, s: fns['logpsi'](unrav(fl), s), in_axes=(None, 0))   # brute force (rbmpp_lib), the reference
from rbmpp_lib import logcosh, lse_c
# ---- fast path: images, up/down site lists and the parity are parameter independent -> precomputed once; the part of the ansatz
#      outside the class (MASK) is frozen at the start parameters and cached per image.  Checked against lpv below (FASTCHK).
_IMG = np.asarray(lat.IMG); _FL = np.asarray(lat.FL)
ZI = _FL[None, :, None] * X[:, _IMG].astype(np.float64)                     # (Dd, nI, N) images z_g of every configuration
_ord = np.argsort(-ZI, axis=2, kind='stable'); UI = _ord[..., :N // 2]; DI = _ord[..., N // 2:]
_up = (ZI > 0).astype(np.int64); _dn = 1 - _up; INVP = np.sum(_up * (np.cumsum(_dn, axis=2) - _dn), axis=2) % 2
# psi_P is invariant under the projection group (translations x spin flip, trivial characters): evaluate it on one representative per
# orbit and broadcast (REPOF).  The orbit of x = the set of its images; representative = smallest image code.  Checked by FASTCHK.
_codes = np.sum(((ZI > 0).astype(np.int64)) << np.arange(N), axis=2); _mins = _codes.min(axis=1)
REPS, REPOF = np.unique(_mins, return_inverse=True); _first = np.array([np.nonzero(_mins == m)[0][0] for m in REPS])
ZI, UI, DI, INVP = ZI[_first], UI[_first], DI[_first], INVP[_first]; REPOFj = jnp.asarray(REPOF.ravel())
ZIj = jnp.asarray(ZI); UIj = jnp.asarray(UI); DIj = jnp.asarray(DI); INVPj = jnp.asarray(INVP)
_MASKSET = set(os.environ.get('MASK', 'f').split(','))
def _pp(f):
    F = f[UIj[..., :, None], DIj[..., None, :]]; sg, ld = jnp.linalg.slogdet(F)
    return ld + 1j * jnp.pi * ((jnp.where(sg < 0, 1, 0) + INVPj) % 2)
def _rbm(b, W): return jnp.sum(logcosh(b + ZIj.astype(jnp.complex128) @ W), axis=-1)
_p0 = unrav(jnp.asarray(flat_g30))
_PP0 = None if 'f' in _MASKSET else _pp(_p0['f'])
_RB0 = _rbm(_p0['bre'] + 1j * _p0['bim'], _p0['Wre'] + 1j * _p0['Wim']) if 'f' in _MASKSET else None
def lp_fast(fl):
    p = unrav(fl)
    a = _pp(p['f']) if 'f' in _MASKSET else _PP0
    r = _RB0 if 'f' in _MASKSET else _rbm(p['bre'] + 1j * p['bim'], p['Wre'] + 1j * p['Wim'])
    return lse_c(a + r)[REPOFj]
def psi_all(fl):
    l = lp_fast(fl); return jnp.exp(l - jnp.max(l.real))
def vvec(fl, w): return jnp.real(w * psi_all(fl))
def Ev(fl, w):
    v = vvec(fl, w); return v @ Hmv(v) / (v @ v)
def logF(fl, w, t, W):
    """log of the (W-weighted) fidelity (sum W v t)^2 / (sum W v^2 sum W t^2); W = 1 is the plain fidelity."""
    v = vvec(fl, w); return 2 * jnp.log(jnp.abs(v @ (W * t))) - jnp.log(v @ (W * v)) - jnp.log(t @ (W * t))
_Ev_vg = jax.jit(jax.value_and_grad(Ev)); _lF_vg = jax.jit(jax.value_and_grad(logF)); _vv = jax.jit(vvec)
ONES = jnp.ones(Dd)

_lens = {k: int(np.asarray(p0[k]).size) for k in sorted(KEYS)}; _off = {}; o = 0
for k in sorted(KEYS): _off[k] = (o, o + _lens[k]); o += _lens[k]
IDX = np.concatenate([np.arange(*_off[k]) for k in MASK.split(',')])
Ev_vg = lambda fl, w: (lambda r: (float(r[0]), np.asarray(r[1])[IDX]))(_Ev_vg(jnp.asarray(fl), w))
lF_vg = lambda fl, w, t, W=ONES: (lambda r: (float(r[0]), np.asarray(r[1])[IDX]))(_lF_vg(jnp.asarray(fl), w, jnp.asarray(t), jnp.asarray(W)))
vv = lambda fl, w: np.asarray(_vv(jnp.asarray(fl), w))


def best_w(fl):
    ps = np.asarray(psi_all(jnp.asarray(fl))); Rr, Ii = ps.real, ps.imag
    A = np.array([[Rr @ (H @ Rr), Rr @ (H @ Ii)], [Ii @ (H @ Rr), Ii @ (H @ Ii)]]); B = np.array([[Rr @ Rr, Rr @ Ii], [Ii @ Rr, Ii @ Ii]])
    ev, ec = sl.eigh(0.5 * (A + A.T), B); c = ec[:, 0]; w = c[0] - 1j * c[1]; return w / abs(w)


def fn_state(g):
    """exact FN ground state with guide g (signs of g; sign-violating hops moved to the diagonal)."""
    rr = g[nbr] / g[wi]; K = 0.5 * JB[bi]; keep = rr < 0
    dFN = diag + np.bincount(wi[~keep], weights=(K * rr)[~keep], minlength=Dd)
    HFN = (sp.coo_matrix((K[keep], (wi[keep], nbr[keep])), shape=(Dd, Dd)) + sp.diags(dFN)).tocsr()
    e, ph = sla.eigsh(HFN, k=1, which='SA', tol=1e-14); ph = ph[:, 0] * np.sign(ph[:, 0] @ g)
    return float(e[0]), ph


def hop(v, dE=0.0):
    v = v / np.linalg.norm(v); Hv = H @ v; E = v @ Hv; u1 = Hv - E * v; u2 = H @ u1 - E * u1
    mu2 = u1 @ u1; mu3 = u1 @ u2; al = (mu3 - np.sqrt(mu3 ** 2 + 4 * mu2 ** 3)) / (2 * mu2 ** 2)
    g = v + al * u1 if dE == 0.0 else v + al * (Hv - (E + dE) * v)
    return g, float(al), float(E)


def state(fl, w):
    """everything about the network a: E_v, v, g (one hop), alpha, E_FN(v), E_FN(g), the FN states."""
    v = vv(fl, w); v = v / np.linalg.norm(v); g, al, E = hop(v)
    efv, phv = fn_state(v); efg, phg = fn_state(g)
    return dict(v=v, g=g, alpha=al, Ev=E, EFN_v=efv, EFN_g=efg, Eg=float(g @ (H @ g) / (g @ g)), phi_v=phv, phi_g=phg)


def chunked_lbfgs(f_vg, x0, maxev, tol, tag, maxiter_once=None):
    """minimise f over the class params IDX (others fixed at x0): L-BFGS-B in chunks of CHUNK iterations until the decrease over a
    chunk is < tol (or the chunk ends on its own convergence test) or maxev evaluations.  maxiter_once: a single call with that maxiter."""
    base = np.array(x0); k = [0]
    def full(x): b = base.copy(); b[IDX] = x; return b
    def fun(x):
        val, gr = f_vg(full(x)); k[0] += 1
        if k[0] % 200 == 0: log(f'{tag}: heartbeat evals {k[0]} f {val:.15g}')
        return val, np.asarray(gr, dtype=np.float64)
    if maxiter_once == 0:   # optimiser stopped at 0 steps (scipy's L-BFGS-B takes one step even with maxiter=0): evaluate, no step
        val, gr = fun(base[IDX]); xr = full(base[IDX])
        log(f'{tag}: zero steps: evals {k[0]}, f {val:.15g}')
        return xr, dict(nit=0, evals=k[0], f=float(val), gnorm=float(np.linalg.norm(gr)))
    if maxiter_once is not None:
        r = so.minimize(fun, base[IDX], jac=True, method='L-BFGS-B', options=dict(maxiter=maxiter_once, maxfun=max(3 * maxiter_once, 1), gtol=1e-14, ftol=1e-16))
        log(f'{tag}: single call maxiter={maxiter_once}: nit {r.nit}, evals {k[0]}, f {r.fun:.15g}')
        return full(r.x), dict(nit=int(r.nit), evals=k[0], f=float(r.fun), gnorm=float(np.linalg.norm(r.jac)))
    x = base[IDX].copy(); f_prev, _ = fun(x); hist = [f_prev]; t0 = time.time(); why = 'maxev'
    while k[0] < maxev:
        r = so.minimize(fun, x, jac=True, method='L-BFGS-B', options=dict(maxiter=CHUNK, maxfun=2 * CHUNK, gtol=1e-14, ftol=1e-16))
        x = r.x; dec = f_prev - r.fun; f_prev = r.fun; hist.append(float(r.fun))
        log(f'{tag}: evals {k[0]}  f {r.fun:.15g}  decrease/chunk {dec:.3e}  |grad| {np.linalg.norm(r.jac):.3e}  ({r.message})  {time.time() - t0:.0f}s')
        if dec < tol: why = f'decrease {dec:.2e} < tol {tol:.0e}'; break
    gn = float(np.linalg.norm(fun(x)[1]))
    log(f'{tag}: STOP ({why}) evals {k[0]} f {f_prev:.15g} |grad| {gn:.3e}')
    return full(x), dict(evals=k[0], f=float(f_prev), gnorm=gn, stop=why, hist=hist, sec=time.time() - t0)


def fid(v, t):
    return float((v @ t) ** 2 / ((v @ v) * (t @ t)))


def jd(o): return json.dumps(o, indent=1, default=lambda a: a.tolist() if isinstance(a, np.ndarray) else float(a))


def fastchk(fl, tag):
    a = np.asarray(lp_fast(jnp.asarray(fl))); b = np.asarray(lpv(jnp.asarray(fl), Xj))
    d = np.max(np.abs(np.exp(a - a.real.max()) - np.exp(b - b.real.max()))); log(f'FASTCHK {tag}: max |psi_fast - psi_brute| {d:.2e}'); return d
_d0 = fastchk(flat_g30, 'start params')
_xp = flat_g30.copy(); _IDXc = np.concatenate([np.arange(*_off[k]) for k in os.environ.get('MASK', 'f').split(',')])
_xp[_IDXc] += 1e-2 * np.random.default_rng(1).standard_normal(_IDXc.size); _d1 = fastchk(_xp, 'class params perturbed')
_xq = flat_g30.copy(); _oth = np.setdiff1d(np.arange(P), _IDXc); _xq[_oth] += 1e-2; _d2 = fastchk(_xq, 'BROKEN: frozen params perturbed (must differ)')
if not (_d0 < 1e-10 and _d1 < 1e-10 and _d2 > 1e-6): log('FASTCHK FAILED; exit 4'); sys.exit(4)
w30 = best_w(flat_g30)
log(f'orbits {len(REPS)} of {Dd} configurations'); log(f'class {MASK} params {len(IDX)} of {P}; E0/N {E0 / N:.10f}; w30 {w30}; jax {jax.__version__} devices {jax.devices()}')

if STAGE == 'time':
    t = time.time(); Ev_vg(flat_g30, w30); log(f'first eval (compile) {time.time() - t:.1f}s')
    t = time.time()
    for _ in range(5): Ev_vg(flat_g30, w30)
    log(f'Ev_vg per eval {(time.time() - t) / 5:.3f}s')
    tt = jnp.asarray(vv(flat_g30, w30)); lF_vg(flat_g30, w30, tt); t = time.time()
    for _ in range(5): lF_vg(flat_g30, w30, tt)
    log(f'lF_vg per eval {(time.time() - t) / 5:.3f}s')
    t = time.time(); S = state(flat_g30, w30); log(f'state (2 FN solves) {time.time() - t:.1f}s; E_FN(g)/N {S["EFN_g"] / N:.8f} (idtest -0.526639)')
    sys.exit(0)

if STAGE == 'opt':
    S = state(flat_g30, w30); log(f'start guide30 E_v/N {S["Ev"] / N:.10f}')
    x, info = chunked_lbfgs(lambda fl: Ev_vg(fl, w30), flat_g30, MAXEV_OPT, TOL_OPT, 'min E_v')
    np.save(os.path.join(OUT, 'flat_opt.npy'), x)
    S = state(x, w30)
    res = dict(mask=MASK, E0N=E0 / N, EvN=S['Ev'] / N, EFN_vN=S['EFN_v'] / N, EFN_gN=S['EFN_g'] / N, alpha=S['alpha'], opt=info)
    open(os.path.join(OUT, 'opt.json'), 'w').write(jd(res))
    log(f'VERDICT OPT DONE E_v/N {S["Ev"] / N:.10f} |grad| {info["gnorm"]:.3e} E_FN(v)/N {S["EFN_v"] / N:.10f} E_FN(g)/N {S["EFN_g"] / N:.10f}')
    sys.exit(0)

# ================= hop-aware training (lead follow-up): (a) min E_VMC(a*h_a), alpha by Ritz at every evaluation; (b) min E_FN(a*h_a)
def hopped(fl, w):
    v = vvec(fl, w); v = v / jnp.sqrt(v @ v); Hv = Hmv(v); E = v @ Hv; u1 = Hv - E * v; u2 = Hmv(u1) - E * u1
    mu2 = u1 @ u1; mu3 = u1 @ u2; al = (mu3 - jnp.sqrt(mu3 ** 2 + 4 * mu2 ** 3)) / (2 * mu2 ** 2)
    return v + al * u1, E, mu2, mu3
def Eg_ritz(fl, w):
    """variational energy of the hopped guide = lowest Ritz value in span{v, (H-E)v}: E + (r - sqrt(r^2 + 4 mu2))/2, r = mu3/mu2."""
    _, E, mu2, mu3 = hopped(fl, w); r = mu3 / mu2; return E + 0.5 * (r - jnp.sqrt(r * r + 4 * mu2))
def efn_hf(fl, w, phi2, nk):
    """Hellmann-Feynman surrogate: the a-dependent part of <phi|H_FN[g(a)]|phi> with phi (phi2 = phi^2 normalised) and the moved-bond
    set nk (1 = sign-violating hop moved to the diagonal) frozen; value differs from E_FN by an a-independent constant, gradient = dE_FN."""
    g, _, _, _ = hopped(fl, w); rr = g[jnb] / g[jwi]
    return phi2 @ jax.ops.segment_sum(jK * rr * nk, jwi, num_segments=Dd)
_Egr_vg = jax.jit(jax.value_and_grad(Eg_ritz)); _hf_vg = jax.jit(jax.value_and_grad(efn_hf))
Egr_vg = lambda fl, w: (lambda r: (float(r[0]), np.asarray(r[1])[IDX]))(_Egr_vg(jnp.asarray(fl), w))
def efn_vg(fl, w):
    g, _, _, _ = hopped(jnp.asarray(fl), w); g = np.asarray(g); e, ph = fn_state(g)
    nk = (g[nbr] / g[wi] >= 0).astype(np.float64); p2 = ph ** 2 / (ph @ ph)
    _, gr = _hf_vg(jnp.asarray(fl), w, jnp.asarray(p2), jnp.asarray(nk)); return e, np.asarray(gr)[IDX]

if STAGE in ('hopa', 'hopb', 'hopa_pert'):
    x0 = np.load(os.path.join(OUT, 'flat_opt.npy')); MAXEV = int(os.environ.get('MAXEV_HOP', 20000)); F = os.path.join(OUT, f'{STAGE}.json')
    S0 = state(x0, w30); res = dict(mask=MASK, stage=STAGE, E0N=E0 / N, start=dict(EvN=S0['Ev'] / N, EgN=S0['Eg'] / N, EFN_gN=S0['EFN_g'] / N))
    c = Egr_vg(x0, w30)[0]; res['check_ritz_vs_state'] = abs(c - S0['Eg']); log(f'Ritz E_g {c / N:.12f} vs state() {S0["Eg"] / N:.12f}')
    if STAGE == 'hopb':
        # check the HF gradient against a central finite difference of the exact E_FN along a random class direction; broken: alpha frozen
        rng = np.random.default_rng(7); dvec = np.zeros(P); dvec[IDX] = rng.standard_normal(len(IDX)); dvec /= np.linalg.norm(dvec)
        _, gr = efn_vg(x0, w30); h = 1e-5
        fd = (efn_vg(x0 + h * dvec, w30)[0] - efn_vg(x0 - h * dvec, w30)[0]) / (2 * h); an = float(gr @ dvec[IDX])
        g0, _, _, _ = hopped(jnp.asarray(x0), w30); g0 = np.asarray(g0); _, ph = fn_state(g0); nk = (g0[nbr] / g0[wi] >= 0).astype(float)
        al0 = S0['alpha']
        def brk(fl):   # BROKEN: alpha held fixed at its start value (misses d alpha / d a)
            v = vvec(fl, w30); v = v / jnp.sqrt(v @ v); Hv = Hmv(v); g = v + al0 * (Hv - (v @ Hv) * v); rr = g[jnb] / g[jwi]
            return jnp.asarray(ph ** 2 / (ph @ ph)) @ jax.ops.segment_sum(jK * rr * jnp.asarray(nk), jwi, num_segments=Dd)
        bk = float(np.asarray(jax.grad(brk)(jnp.asarray(x0)))[IDX] @ dvec[IDX])
        res['hf_check'] = dict(fd=fd, hf=an, rel=abs(fd - an) / abs(fd), BROKEN_alpha_frozen=bk, BROKEN_rel=abs(fd - bk) / abs(fd))
        log('HF gradient check', json.dumps(res['hf_check']))
    xs = x0
    if STAGE == 'hopa_pert':
        r5 = np.load(os.path.join(OUT, 'flat_ii_5.npy')); dn = np.linalg.norm((r5 - x0)[IDX]); rng = np.random.default_rng(11)
        dv = rng.standard_normal(len(IDX)); xs = x0.copy(); xs[IDX] += dn * dv / np.linalg.norm(dv); res['pert_norm'] = float(dn)
        log(f'perturbed start |dtheta| {dn:.4e} (= |a_5(ii) - a_0|)')
    fvg = (lambda fl: Egr_vg(fl, w30)) if STAGE != 'hopb' else (lambda fl: efn_vg(fl, w30))
    x, info = chunked_lbfgs(fvg, xs, MAXEV, float(os.environ.get('TOL_HOP', 1e-13)), STAGE)
    np.save(os.path.join(OUT, f'flat_{STAGE}.npy'), x); S = state(x, w30)
    res['opt'] = dict(EvN=S['Ev'] / N, EgN=S['Eg'] / N, EFN_gN=S['EFN_g'] / N, EFN_vN=S['EFN_v'] / N, alpha=S['alpha'],
                      evals=info['evals'], gnorm=info['gnorm'], stop=info['stop'], dtheta_from_opt=float(np.linalg.norm((x - x0)[IDX])))
    open(F, 'w').write(jd(res))
    log(f'VERDICT {STAGE} {MASK}: E_FN(g)/N {S["EFN_g"] / N:.10f} E_VMC(g)/N {S["Eg"] / N:.10f} E_v/N {S["Ev"] / N:.10f} evals {info["evals"]} |grad| {info["gnorm"]:.2e}')
    sys.exit(0)

VAR = {'loopi': 'i', 'loopii': 'ii', 'loopiii': 'iii'}[STAGE]
GKEY, PKEY = ('EFN_g', 'phi_g') if VAR == 'ii' else ('EFN_v', 'phi_v')
C0 = float(os.environ.get('NODE_C0', 0.05))


def node_weight(t):
    """variant (iii): W(x) = 1 / (c(x) + C0), c(x) = |sum_y H_xy t(y)| / sum_y |H_xy t(y)| over off-diagonal neighbours (N0 cancellation
    ratio, compress_6x6 README 'N0'), computed on the target t = phi_k (which carries the guide's sign); c -> 0 near the node.
    Lattice analogue of Caffarel-Pinar-Scemama 2609.01301 Eq. (25)-(26) (fit restricted to the nodal region), smoothed so every
    configuration keeps some weight.  W normalised to mean 1."""
    K = 0.5 * JB[bi]; a = K * t[nbr]
    c = np.abs(np.bincount(wi, weights=a, minlength=Dd)) / np.maximum(np.bincount(wi, weights=np.abs(a), minlength=Dd), 1e-300)
    W = 1.0 / (c + C0); return W / W.mean(), c
x0 = np.load(os.path.join(OUT, 'flat_opt.npy'))
F = os.path.join(OUT, f'loop_{VAR}.json')
res = dict(mask=MASK, variant=VAR, E0N=E0 / N, rounds=[], controls={})
S0 = state(x0, w30)
res['addon_EFN_gN'] = S0['EFN_g'] / N   # add-on = VMC optimum + one hop + FN = round 0 of variant ii


def rec(k, x, S):
    r = dict(k=k, EFN_guideN=S[GKEY] / N, EvN=S['Ev'] / N, EFN_vN=S['EFN_v'] / N, EFN_gN=S['EFN_g'] / N, EgN=S['Eg'] / N, alpha=S['alpha'],
             F_a_phi=fid(S['v'], S[PKEY]))
    res['rounds'].append(r); open(F, 'w').write(jd(res))
    log(f'ROUND {k}: E_FN(guide)/N {r["EFN_guideN"]:.10f}  E_v/N {r["EvN"]:.10f}  F(a_k, phi_k) {r["F_a_phi"]:.10f}  '
        f'[E_FN(v) {r["EFN_vN"]:.10f} E_FN(g) {r["EFN_gN"]:.10f} alpha {r["alpha"]:.6f}]')


def refit(x, phi, tag, **kw):
    if VAR == 'iii':
        W, c = node_weight(phi); wt = phi ** 2 / (phi @ phi)
        log(f'{tag}: node weight W = 1/(c + {C0}); phi^2-weighted share with c < 0.1: {wt[c < 0.1].sum():.4e}; W range {W.min():.3f}..{W.max():.3f}')
    else:
        W = np.ones(Dd)
    return chunked_lbfgs(lambda fl: (lambda r: (-r[0], -r[1]))(lF_vg(fl, w30, phi, W)), x, MAXEV_FIT, TOL_FIT, tag, **kw)


rec(0, x0, S0)
# ---------------- controls at round 0
C = res['controls']
gE = Ev_vg(x0, w30)[1]
if VAR == 'ii':
    _, Gg = lF_vg(x0, w30, S0['g']); gb, _, _ = hop(S0['v'], dE=0.05); _, Gb = lF_vg(x0, w30, gb); _, Gp = lF_vg(x0, w30, S0['phi_g'])
    C['identity'] = dict(rel_err=float(np.linalg.norm(Gg - S0['alpha'] * gE) / np.linalg.norm(Gg)),
                         rel_err_BROKEN_wrongE_hop=float(np.linalg.norm(Gb - S0['alpha'] * gE) / np.linalg.norm(Gb)),
                         norm_Gg=float(np.linalg.norm(Gg)), norm_alpha_dE=float(np.linalg.norm(S0['alpha'] * gE)), norm_dE=float(np.linalg.norm(gE)),
                         norm_G_broken=float(np.linalg.norm(Gb)), norm_G_phi=float(np.linalg.norm(Gp)))
    log('CONTROL identity', json.dumps(C['identity']))
x_z, iz = refit(x0, S0[PKEY], 'zero-step refit', maxiter_once=0)
Sz = state(x_z, w30)
x_1, i1 = refit(x0, S0[PKEY], 'one-step refit (broken zero-step)', maxiter_once=1)
S1 = state(x_1, w30)
C['zero_step'] = dict(bitwise_equal=bool(np.array_equal(x_z, x0)), max_abs_dtheta=float(np.max(np.abs(x_z - x0))), EFN_guideN=Sz[GKEY] / N,
                      EFN_unchanged=bool(Sz[GKEY] == S0[GKEY]),
                      BROKEN_one_step_bitwise_equal=bool(np.array_equal(x_1, x0)), BROKEN_max_abs_dtheta=float(np.max(np.abs(x_1 - x0))),
                      BROKEN_EFN_guideN=S1[GKEY] / N, BROKEN_EFN_unchanged=bool(S1[GKEY] == S0[GKEY]))
log('CONTROL zero-step', json.dumps(C['zero_step']))
open(F, 'w').write(jd(res))
# broken target: flip sign on a random 1% of phi_0's weight (whole configurations, random order until 1% of sum phi^2)
rng = np.random.default_rng(int(os.environ.get("BSEED", 20261026))); phi0 = S0[PKEY]; wt = phi0 ** 2 / (phi0 @ phi0); perm = rng.permutation(Dd)
nflip = int(np.searchsorted(np.cumsum(wt[perm]), 0.01)) + 1; flip = perm[:nflip]
phib = phi0.copy(); phib[flip] *= -1
xb, ib = refit(x0, phib, 'refit to BROKEN target (1% weight sign-flipped)')
Sb = state(xb, w30)
C['broken_target'] = dict(n_flipped=nflip, weight_flipped=float(wt[flip].sum()), EFN_guideN_after=Sb[GKEY] / N, EFN_guideN_before=S0[GKEY] / N,
                          F_true_phi_after=fid(Sb['v'], phi0), fit=dict(evals=ib['evals'], f=ib['f'], gnorm=ib['gnorm'], stop=ib['stop']))
log('CONTROL broken target', json.dumps(C['broken_target']))
open(F, 'w').write(jd(res))
# ---------------- the loop
x, S = x0, S0
for k in range(1, NROUND + 1):
    phi = S[PKEY]
    xn, info = refit(x, phi, f'round {k} refit')
    Sn = state(xn, w30)
    res['rounds'][-1]['F_after_refit'] = fid(Sn['v'], phi); res['rounds'][-1]['refit'] = dict(evals=info['evals'], f=info['f'], gnorm=info['gnorm'], stop=info['stop'], sec=info['sec'])
    if k == 1: C['broken_target']['EFN_guideN_true_round1'] = Sn[GKEY] / N
    np.save(os.path.join(OUT, f'flat_{VAR}_{k}.npy'), xn)
    x, S = xn, Sn; rec(k, x, S)
if NROUND == 0: log("VERDICT CONTROLS ONLY"); sys.exit(0)
E = [r["EFN_guideN"] for r in res["rounds"]]; d = [E[i - 1] - E[i] for i in range(1, len(E))]
res['gains'] = d
if d[0] <= 0: verdict = 'NO ROUND-1 GAIN'
elif d[2] >= 0.3 * d[0] and sum(d) >= 2 * d[0]: verdict = 'KEEPS GAINING'
elif d[1] < 0.1 * d[0]: verdict = 'STALLS'
else: verdict = 'SLOWS'
res['verdict'] = verdict; res['ratios'] = dict(r2_r1=d[1] / d[0], r3_r1=d[2] / d[0], cum_r1=sum(d) / d[0])
res['gap_fraction_closed'] = (res['addon_EFN_gN'] - E[-1]) / (res['addon_EFN_gN'] - E0 / N)
open(F, 'w').write(jd(res))
log(f'VERDICT LOOP {VAR} {MASK}: {verdict} gains {["%.3e" % g for g in d]} ratios {res["ratios"]} gap fraction {res["gap_fraction_closed"]:.4f}')
