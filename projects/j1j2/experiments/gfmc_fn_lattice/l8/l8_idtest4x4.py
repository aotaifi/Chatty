"""Exact 4x4 test of the 'no first-order gain from the hop' identity. Adversarial review of L8 (read-only on project files)."""
import os, sys, json, time
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sla, scipy.linalg as sl, scipy.optimize as so
R = '/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/experiments'
for d in (R + '/gfmc_fn_lattice/l8', R + '/gfmc_fn_lattice', R + '/rbmpp_6x6'): sys.path.insert(0, d)
import jax, jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
from jax.flatten_util import ravel_pytree
from rbmpp_lib import Lattice, make_fns
log = lambda *a: print(time.strftime('%H:%M:%S'), *a, flush=True)
L = 4; N = 16; lat = Lattice(L, 0.5, 'translations')
BI = np.asarray(lat.BI); BJ = np.asarray(lat.BJ); JB = np.asarray(lat.JB, float)
basis = np.array([s for s in range(1 << N) if bin(s).count('1') == N // 2], np.int64)
Dd = len(basis); lookup = -np.ones(1 << N, np.int64); lookup[basis] = np.arange(Dd)
X = ((basis[:, None] >> np.arange(N)) & 1).astype(np.int8) * 2 - 1
anti = X[:, BI] != X[:, BJ]; wi, bi = np.nonzero(anti)
nbr = lookup[basis[wi] ^ (1 << BI[bi]) ^ (1 << BJ[bi])]
diag = 0.25 * np.sum(JB[None] * X[:, BI] * X[:, BJ], 1)
H = (sp.coo_matrix((0.5 * JB[bi], (wi, nbr)), shape=(Dd, Dd)).tocsr() + sp.diags(diag)).tocsr()
E0 = sla.eigsh(H, k=1, which='SA')[0][0]
jwi, jnb, jK, jdg = jnp.asarray(wi), jnp.asarray(nbr), jnp.asarray(0.5 * JB[bi]), jnp.asarray(diag)
def Hmv(x): return jdg * x + jax.ops.segment_sum(jK * x[jnb], jwi, num_segments=Dd)

z0 = np.load('/scratch/a/A.Otaifi/l8/check4x4/params4.npz')
KEYS = ('f', 'bre', 'bim', 'Wre', 'Wim')
p0 = {k: jnp.asarray(z0[k]) for k in KEYS}
flat_g30, unrav = ravel_pytree(p0); flat_g30 = np.asarray(flat_g30); P = flat_g30.size
fns = make_fns(lat, jnp.float64, True)
Xj = jnp.asarray(X.astype(np.float64)); XA = [Xj]
lpv = jax.vmap(lambda fl, s: fns['logpsi'](unrav(fl), s), in_axes=(None, 0))
def logpsi_all(fl, Xa=None): return lpv(fl, XA[0] if Xa is None else Xa)
def psi_all(fl, Xa=None):
    l = logpsi_all(fl, Xa); return jnp.exp(l - jnp.max(l.real))
def vvec(fl, w, Xa=None): return jnp.real(w * psi_all(fl, Xa))
def Ev(fl, w, Xa):
    v = vvec(fl, w, Xa); return v @ Hmv(v) / (v @ v)
def Epsi(fl, Xa=None):
    ps = psi_all(fl, Xa); a, b = ps.real, ps.imag
    return (a @ Hmv(a) + b @ Hmv(b)) / (a @ a + b @ b)
def logF(fl, w, t, Xa=None):
    v = vvec(fl, w, Xa); return 2 * jnp.log(jnp.abs(v @ t)) - jnp.log(v @ v) - jnp.log(t @ t)
_Ev_vg = jax.jit(jax.value_and_grad(Ev)); _Ep_vg = jax.jit(jax.value_and_grad(Epsi)); _lF_vg = jax.jit(jax.value_and_grad(logF)); _vv = jax.jit(vvec)
_Ep = jax.jit(Epsi); _lF = jax.jit(logF)
Ev_vg = lambda fl, w: _Ev_vg(fl, w, Xj); Ep_vg = lambda fl: _Ep_vg(fl, Xj); lF_vg = lambda fl, w, t: _lF_vg(fl, w, t, Xj); vv = lambda fl, w: _vv(fl, w, Xj)
EpsiX = lambda fl: _Ep(fl, Xj); logFX = lambda fl, w, t: _lF(fl, w, t, Xj)
dlogv = jax.jit(jax.vmap(jax.grad(lambda fl, s, w: jnp.real(fns['logpsi'](unrav(fl), s)) + jnp.log(jnp.abs(jnp.real(w * jnp.exp(1j * jnp.imag(fns['logpsi'](unrav(fl), s))))))), in_axes=(None, 0, None)))

def best_w(fl):
    ps = np.asarray(psi_all(fl)); Rr, Ii = ps.real, ps.imag
    A = np.array([[Rr @ (H @ Rr), Rr @ (H @ Ii)], [Ii @ (H @ Rr), Ii @ (H @ Ii)]]); B = np.array([[Rr @ Rr, Rr @ Ii], [Ii @ Rr, Ii @ Ii]])
    ev, ec = sl.eigh(0.5 * (A + A.T), B); c = ec[:, 0]; w = c[0] - 1j * c[1]; return w / abs(w)

def targets(fl, w):
    v = np.asarray(vv(fl, w)); v = v / np.linalg.norm(v); Hv = H @ v; E = v @ Hv; u1 = Hv - E * v; u2 = H @ u1 - E * u1
    mu2 = u1 @ u1; mu3 = u1 @ u2; al = (mu3 - np.sqrt(mu3 ** 2 + 4 * mu2 ** 3)) / (2 * mu2 ** 2)
    g = v + al * u1
    rr = g[nbr] / g[wi]; K = 0.5 * JB[bi]; keep = rr < 0
    dFN = diag + np.bincount(wi[~keep], weights=(K * rr)[~keep], minlength=Dd)
    HFN = (sp.coo_matrix((K[keep], (wi[keep], nbr[keep])), shape=(Dd, Dd)) + sp.diags(dFN)).tocsr()
    e, ph = sla.eigsh(HFN, k=1, which='SA', tol=1e-13); ph = ph[:, 0] * np.sign(ph[:, 0] @ g)
    Eg = g @ (H @ g) / (g @ g)
    return dict(v=v, E=E, al=al, g=g, phi=ph, EFN=e[0], Eg=Eg)

def metric(fl, w, v):
    D = np.concatenate([np.asarray(dlogv(fl, Xj[i:i + 2048], w)) for i in range(0, Dd, 2048)])
    p = v ** 2 / (v @ v); Db = p @ D; S = ((D - Db) * p[:, None]).T @ (D - Db); return S

def nat(G, S, shift=1e-6):
    d = np.diag(S).copy(); d = np.maximum(d, 1e-12 * d.max()); sq = np.sqrt(d)
    Ss = S / np.outer(sq, sq) + shift * np.eye(len(d)); return (sl.solve(Ss, G / sq, assume_a='pos')) / sq

def report(tag, fl, w):
    T = targets(fl, w); v = T['v']
    Eg_, gE = Ev_vg(fl, w); gE = np.asarray(gE)
    _, Gg = lF_vg(fl, w, jnp.asarray(T['g'])); _, Gp = lF_vg(fl, w, jnp.asarray(T['phi']))
    Gg, Gp = np.asarray(Gg), np.asarray(Gp)
    _, gEp = Ep_vg(fl); gEp = np.asarray(gEp)
    gb = T['v'] + T['al'] * (H @ T['v'] - (T['E'] + 0.05) * T['v'])          # BROKEN target: wrong E in the hop
    _, Gb = lF_vg(fl, w, jnp.asarray(gb)); Gb = np.asarray(Gb)
    S = metric(fl, w, v)
    n = lambda a: float(np.sqrt(max(a @ nat(a, S), 0)))   # natural norm |a|_{S^-1}
    c = lambda a, b: float(a @ nat(b, S) / (n(a) * n(b) + 1e-300))
    out = dict(EvN=float(T['E'] / N), EgN=float(T['Eg'] / N), EFNN=float(T['EFN'] / N), EpsiN=float(EpsiX(fl) / N), alpha=float(T['al']),
               nat_Gg=n(Gg), nat_alpha_dE=n(T['al'] * gE), nat_Gphi=n(Gp), nat_Gphi_minus_Gg=n(Gp - Gg),
               rel_identity_err=float(np.linalg.norm(Gg - T['al'] * gE) / (np.linalg.norm(Gg) + 1e-300)),
               rel_identity_err_BROKEN=float(np.linalg.norm(Gb - T['al'] * gE) / (np.linalg.norm(Gb) + 1e-300)), nat_Gbroken=n(Gb),
               rel_identity_err_vs_dEpsi=float(np.linalg.norm(Gg - T['al'] * gEp) / (np.linalg.norm(Gg) + 1e-300)),
               cos_Gg_Gphi=c(Gg, Gp), nat_dEv=n(gE), nat_dEpsi=n(gEp),
               lin_gain_logF_g=0.5 * n(Gg) ** 2, lin_gain_logF_phi=0.5 * n(Gp) ** 2,
               F_v_g=float(np.exp(logFX(fl, w, jnp.asarray(T['g'])))), F_v_phi=float(np.exp(logFX(fl, w, jnp.asarray(T['phi'])))))
    log(tag, json.dumps({k: (float('%.6g' % x) if isinstance(x, float) else x) for k, x in out.items()}))
    return out, T, S

def lbfgs(f_vg, x0, it, tag):
    k = [0]
    def fun(x):
        val, gr = f_vg(jnp.asarray(x)); k[0] += 1; return float(val), np.asarray(gr, dtype=np.float64)
    r = so.minimize(fun, x0, jac=True, method='L-BFGS-B', options=dict(maxiter=it, maxfun=3 * it, gtol=1e-12, ftol=1e-16))
    log(f'{tag}: L-BFGS {r.nit} it, {k[0]} evals, f {r.fun:.12g}, |grad| {np.linalg.norm(r.jac):.2e} ({r.message})'); return r.x

MASK = os.environ.get('MASK', 'all')
_, unrav_ = ravel_pytree(p0); _lens = {k: int(np.asarray(p0[k]).size) for k in sorted(KEYS)}
_off = {}; o = 0
for k in sorted(KEYS): _off[k] = (o, o + _lens[k]); o += _lens[k]
IDX = np.arange(P) if MASK == 'all' else np.concatenate([np.arange(*_off[k]) for k in MASK.split(',')])
log('class', MASK, 'params', len(IDX), 'of', P)
_Evg0, _lFvg0, _Epvg0, _met0 = Ev_vg, lF_vg, Ep_vg, metric
Ev_vg = lambda fl, w: (lambda r: (r[0], np.asarray(r[1])[IDX]))(_Evg0(fl, w))
lF_vg = lambda fl, w, t: (lambda r: (r[0], np.asarray(r[1])[IDX]))(_lFvg0(fl, w, t))
Ep_vg = lambda fl: (lambda r: (r[0], np.asarray(r[1])[IDX]))(_Epvg0(fl))
metric = lambda fl, w, v: _met0(fl, w, v)[np.ix_(IDX, IDX)]
def lbfgs(f_vg, x0, it, tag):
    base = np.array(x0); k = [0]
    def full(x): b = base.copy(); b[IDX] = x; return b
    def fun(x):
        val, gr = f_vg(jnp.asarray(full(x))); k[0] += 1; return float(val), np.asarray(gr, dtype=np.float64)
    r = so.minimize(fun, base[IDX], jac=True, method='L-BFGS-B', options=dict(maxiter=it, maxfun=3 * it, gtol=1e-12, ftol=1e-16))
    log(f'{tag}: L-BFGS {r.nit} it, {k[0]} evals, f {r.fun:.12g}, |grad| {np.linalg.norm(r.jac):.2e} ({r.message})'); return full(r.x)
res = dict(E0N=E0 / N)
OUT = sys.argv[1] if len(sys.argv) > 1 else 'idtest.json'
w30 = best_w(flat_g30)
res['A_guide30'], T30, _ = report('A guide30 (the 4x4 control theta0)', flat_g30, w30)
# B: energy optimum of v = Re[w psi_P] (w fixed), started from guide30
flat_v = lbfgs(lambda x: Ev_vg(x, w30), flat_g30, int(os.environ.get('IT', 3000)), 'min E_v')
np.save(OUT + '.flat_v.npy', flat_v)
res['B_Evopt'], TB, SB = report('B E_v optimum', flat_v, w30)
# C: energy optimum of the complex psi_P (what SR training targets), then v with its best w
flat_p = lbfgs(lambda x: Ep_vg(x), flat_g30, int(os.environ.get('IT', 3000)), 'min E_psi')
np.save(OUT + '.flat_p.npy', flat_p)
wp = best_w(flat_p)
res['C_Epsiopt'], TC, _ = report('C E_psi optimum, v = Re[w psi]', flat_p, wp)
# D: second order: from B, maximise F to the fixed targets g_B and phi_B (full refit, exact), report R gain and energies
for name, t in (('g', TB['g']), ('phi', TB['phi'])):
    tj = jnp.asarray(t)
    fl2 = lbfgs(lambda x: (lambda r: (-r[0], -r[1]))(lF_vg(x, w30, tj)), flat_v, int(os.environ.get('IT2', 2000)), f'max F to {name}_B')
    F0 = float(np.exp(logFX(flat_v, w30, tj))); F1 = float(np.exp(logFX(fl2, w30, tj)))
    T2 = targets(fl2, w30); np.save(OUT + f'.flat_refit_{name}.npy', fl2)
    res[f'D_refit_to_{name}'] = dict(F0=F0, F1=F1, R=F1 / F0, EvN_after=float(T2['E'] / N), EFNN_after=float(T2['EFN'] / N), EgN_after=float(T2['Eg'] / N))
    log(f'D refit B -> {name}: F {F0:.8f} -> {F1:.8f} (R {F1 / F0:.6f}); E_v/N {TB["E"] / N:.8f} -> {T2["E"] / N:.8f}; E_FN(g)/N {TB["EFN"] / N:.8f} -> {T2["EFN"] / N:.8f}')
json.dump(res, open(OUT, 'w'), indent=1)
log('DONE')
