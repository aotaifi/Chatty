"""4x4 checks of the generalised walker guide (RBMPPGuideG: complex RBM, global phase w, p = 0 and p = 1, any projection) and of
the VMC moment estimator (vmc_moments.py) against exact numbers, with COMPLEX RBM parameters (the old check_4x4.py uses a real RBM
and cannot see the phase of psi_P).
  python check_general.py OUT [--proj translations] [--sweeps 600]
 G1 exact: psi_P on all 12870 states, optimal phase w (2x2 in {Re psi, Im psi}), v = Re(w psi), E, mu_2..mu_4, alpha (closed form and
    independent 2x2 eigh), psi_1 = v + alpha (H-E) v, E_1, var_1 (direct vectors vs formulas from moments), exact E_FN(v), E_FN(psi_1).
 G2 provider vs table: RBMPPGuideG.rows (p = 0, p = 1) against TableGuide rows on all configurations (rate, d_FN); the OLD RBMPPGuide
    (phase of psi_P(x) dropped) and a drop_phase version must FAIL (they are wrong when psi_P is complex).
 G3 VMC (vmc_moments): phase w, E, var, alpha, E_1, var_1 against G1 (z-scores from the chain-group jackknife); broken: moments
    without the E_L(y) neighbour term must FAIL.
 G4 walkers with RBMPPGuideG (p = 0 and p = 1) against the exact E_FN; the dropped-phase version must FAIL.
"""
import sys, os, json, time, argparse
import numpy as np
import scipy.sparse as sp, scipy.sparse.linalg as sla, scipy.linalg as sl
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'rbmpp_6x6'))
import jax, jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
from rbmpp_lib import Lattice, make_fns, init_params
from gfmc_fn_lib import TableGuide, RBMPPGuide, RBMPPGuideG, GFMC, blocked_error
import vmc_moments as VM

ap = argparse.ArgumentParser()
ap.add_argument('out'); ap.add_argument('--proj', default='translations'); ap.add_argument('--sweeps', type=int, default=120)
ap.add_argument('--chains', type=int, default=256); ap.add_argument('--blocks', type=int, default=400); ap.add_argument('--M', type=int, default=100)
ap.add_argument('--seed', type=int, default=5); ap.add_argument('--opt_iters', type=int, default=200); ap.add_argument('--rbm_scale', type=float, default=0.25)
args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)
L = 4; N = 16
lat = Lattice(L, 0.5, args.proj)
BI = np.asarray(lat.BI); BJ = np.asarray(lat.BJ); JB = np.asarray(lat.JB, float)
log = lambda *a: print(time.strftime('%H:%M:%S'), *a, flush=True)
res = {}; allok = True


def verdict(name, ok, msg=''):
    global allok
    allok &= bool(ok); log(('PASS ' if ok else 'FAIL ') + name, msg); res.setdefault('verdicts', {})[name] = bool(ok)


basis = np.array([s for s in range(1 << N) if bin(s).count('1') == N // 2], np.int64)
D = len(basis); lookup = -np.ones(1 << N, np.int64); lookup[basis] = np.arange(D)
X = ((basis[:, None] >> np.arange(N)) & 1).astype(np.int8) * 2 - 1
anti = X[:, BI] != X[:, BJ]; wi, bi = np.nonzero(anti)
nbr = lookup[basis[wi] ^ (1 << BI[bi]) ^ (1 << BJ[bi])]
diag = 0.25 * np.sum(JB[None] * X[:, BI] * X[:, BJ], 1)
H = (sp.coo_matrix((0.5 * JB[bi], (wi, nbr)), shape=(D, D)).tocsr() + sp.diags(diag)).tocsr()

# ---------------------------------------------------------------- G1 exact
p = init_params(lat, M=4, seed=args.seed, rbm_scale=args.rbm_scale)
fns = make_fns(lat, jnp.float64, True)
lp = jax.jit(jax.vmap(fns['logpsi'], in_axes=(None, 0)))
Xj = jnp.asarray(X.astype(np.float64))
# a reasonable complex RBM+PP guide: L-BFGS on the exact energy of the complex state (parameters real: f, Re/Im b, Re/Im W)
from jax.experimental import sparse as jsp
from jax.flatten_util import ravel_pytree
gpath = os.path.join(args.out, f'guide4x4_complex_{args.proj}.npz')
flat0, unrav = ravel_pytree(p)
Hb = jsp.BCOO.from_scipy_sparse(H)
lp_raw = jax.vmap(fns['logpsi'], in_axes=(None, 0))


@jax.jit
def loss_fn(fl, Xa):
    l = lp_raw(unrav(fl), Xa); l = l - jnp.max(jnp.real(l)); ps = jnp.exp(l)
    Hps = Hb @ ps
    return jnp.real(jnp.vdot(ps, Hps)) / jnp.real(jnp.vdot(ps, ps))


if os.path.exists(gpath):
    flat = jnp.asarray(np.load(gpath)['flat'])
else:
    from scipy.optimize import minimize
    vg = jax.jit(jax.value_and_grad(loss_fn)); cnt = [0]; t0 = time.time()

    def fun(z):
        e, g = vg(jnp.asarray(z), Xj); cnt[0] += 1
        if cnt[0] % 25 == 1: log('opt', cnt[0], float(e) / N, f'{time.time() - t0:.0f}s')
        return float(e), np.asarray(g)
    r = minimize(fun, np.asarray(flat0), jac=True, method='L-BFGS-B', options=dict(maxiter=args.opt_iters))
    flat = jnp.asarray(r.x); np.savez(gpath, flat=r.x)
p = unrav(flat)
log('guide energy/N (exact, complex state)', float(loss_fn(flat, Xj)) / N)
l_all = np.concatenate([np.asarray(lp(p, Xj[i:i + 1024])) for i in range(0, D, 1024)])
l_all -= l_all.real.max()
psi = np.exp(l_all); psi /= np.linalg.norm(psi)
Hpsi = H @ psi
Rr, Ii = psi.real, psi.imag
A = np.array([[Rr @ (H @ Rr), Rr @ (H @ Ii)], [Ii @ (H @ Rr), Ii @ (H @ Ii)]]); Bm = np.array([[Rr @ Rr, Rr @ Ii], [Ii @ Rr, Ii @ Ii]])
ev, ec = sl.eigh(0.5 * (A + A.T), Bm); c = ec[:, 0]
w = c[0] - 1j * c[1]; w /= abs(w)
v = (w * psi).real; v /= np.linalg.norm(v)
Hv = H @ v; E = v @ Hv; u1 = Hv - E * v; Hu1 = H @ u1; u2 = Hu1 - E * u1
mu2 = u1 @ u1; mu3 = u1 @ u2; mu4 = u2 @ u2
G = np.diag([1.0, mu2]); Hm = np.array([[0, mu2], [mu2, mu3]]); evr, ecr = sl.eigh(Hm, G); alpha_eig = float(ecr[1, 0] / ecr[0, 0])
cf = VM.lanczos_from_moments((E - E + 0.0, mu2, mu3, mu4), E)          # moments about E: m1 = 0
log(f'G1 exact: E/N {E / N:.8f} (E0 {sla.eigsh(H, k=1, which="SA")[0][0] / N:.8f}) var {mu2:.5f} mu3 {mu3:.5f} mu4 {mu4:.5f}; alpha eigh {alpha_eig:.8f} closed form {cf["alpha"]:.8f}; '
    f'|Im| weight {Ii @ Ii:.3e}; w {w:.6f}')
verdict('G1 closed-form alpha = 2x2 Ritz', abs(alpha_eig - cf['alpha']) < 1e-9 * abs(alpha_eig))
alpha = alpha_eig
psi1 = v + alpha * u1
E1d = float(psi1 @ (H @ psi1) / (psi1 @ psi1)); Hp1 = H @ psi1; var1d = float(Hp1 @ Hp1 / (psi1 @ psi1) - E1d ** 2)
verdict('G1 E_1, var_1 formulas = direct vectors', abs(E1d - cf['E1']) < 1e-9 and abs(var1d - cf['var1']) < 1e-8 * var1d,
        f'E_1 direct {E1d / N:.8f} formula {cf["E1"] / N:.8f}; var_1 direct {var1d:.6f} formula {cf["var1"]:.6f}; (E_1-E)/var = {(E1d - E) / mu2:.6f} vs alpha {alpha:.6f}')
res['exact'] = dict(E=E, var=mu2, mu3=mu3, mu4=mu4, alpha=alpha, E1=E1d, var1=var1d, w=[w.real, w.imag], im_weight=float(Ii @ Ii))


def fn_exact(vec):
    rr = vec[nbr] / vec[wi]; K = 0.5 * JB[bi]; keep = rr < 0
    d = diag + np.bincount(wi[~keep], weights=(K * rr)[~keep], minlength=D)
    F = sp.coo_matrix((np.r_[K[keep] * rr[keep], d], (np.r_[wi[keep], np.arange(D)], np.r_[nbr[keep], np.arange(D)])), shape=(D, D)).tocsr()
    return float(np.real(sla.eigs(F, k=1, which='SR', tol=1e-12, maxiter=100000)[0][0]))


EFN1 = fn_exact(psi1); EFN0 = fn_exact(v)
log(f'G1 exact E_FN: p=1 {EFN1 / N:.8f}  p=0 {EFN0 / N:.8f}  (E_1 {E1d / N:.8f}, E {E / N:.8f})')
res['exact'].update(EFN1=EFN1, EFN0=EFN0)

# ---------------------------------------------------------------- G2 provider vs table
pn = {k: np.asarray(p[k]) for k in p}
idx = np.random.default_rng(0).choice(D, 800, replace=False)
for name, vec, al in (('p=1', psi1, alpha), ('p=0', v, 0.0)):
    tg = TableGuide(N, BI, BJ, JB, vec); r0, d0 = tg.rows(X[idx])
    gg = RBMPPGuideG(pn, lat, al, E, w=w, chunk=256); r, d = gg.rows(X[idx])
    err = max(np.max(np.abs(r - r0)), np.max(np.abs(d - d0))); sc = max(np.max(np.abs(r0)), np.max(np.abs(d0)))
    verdict(f'G2 RBMPPGuideG {name} = table rows', err < 1e-9 * sc, f'max abs diff {err:.2e} (scale {sc:.3g}), {gg.n_eval} evals')
    gb = RBMPPGuideG(pn, lat, al, E, w=w, chunk=256, drop_phase=True); rb, db = gb.rows(X[idx])
    eb = max(np.max(np.abs(rb - r0)), np.max(np.abs(db - d0)))
    verdict(f'G2 broken (phase dropped) {name} FAILS as it must', eb > 1e-3 * sc, f'max abs diff {eb:.2e}')
go = RBMPPGuide(pn, lat, alpha, E, chunk=256, w=w); ro, do = go.rows(X[idx]); tg = TableGuide(N, BI, BJ, JB, psi1); r0, d0 = tg.rows(X[idx])
eo = max(np.max(np.abs(ro - r0)), np.max(np.abs(do - d0)))
log(f'G2 note: the old RBMPPGuide (6x6 path) on this complex psi_P differs from the table by {eo:.2e} (it assumes psi_P real; correct at 6x6 where Im weight is 1e-26)')
res['G2_old_provider_maxdiff'] = float(eo)

# ---------------------------------------------------------------- G3a per-configuration quantities of the estimator against the exact vectors
sm = VM.Sampler(pn, lat, True, chunk=256)
idx3 = np.random.default_rng(7).choice(D, 300, replace=False)
vt_, h1_, h2_, dv_ = VM.config_quantities(sm, X[idx3].astype(np.float64), w)
vu = (w * psi).real                                  # unnormalised v with psi normalised to 1: v/|psi| etc. are scale free in psi, v scale = psi scale
Hvu = H @ vu; H2vu = H @ Hvu; ap_ = np.abs(psi)
ex3 = np.stack([vu[idx3] / ap_[idx3], Hvu[idx3] / ap_[idx3], H2vu[idx3] / ap_[idx3]], 1)
got3 = np.stack([vt_, h1_, h2_], 1)
rel3 = float(np.max(np.abs(got3 - ex3)) / np.max(np.abs(ex3)))
verdict('G3a vt, h1, h2 per configuration = exact v, Hv, H^2 v over |psi| (300 configs)', rel3 < 1e-9, f'max rel diff {rel3:.2e}; fast-update ratio dev {dv_:.1e}')
bad = got3.copy(); bad[:, 2] = bad[:, 1] * 0.0 + bad[:, 2] * 1.01
verdict('G3a broken (h2 scaled by 1.01) FAILS as it must', float(np.max(np.abs(bad - ex3)) / np.max(np.abs(ex3))) > 1e-4)

# ---------------------------------------------------------------- G3 VMC estimator
class FS:     # object with the Sampler interface but via the 4x4 params
    pass


sm = VM.Sampler(pn, lat, True, chunk=256)
wA, Ec, infoA = VM.stage_phase(sm, 256, 40, 0, 60, log)
vA = (wA * psi).real; vA /= np.linalg.norm(vA); EA = float(vA @ (H @ vA))
verdict('G3 phase from VMC: energy of Re(w_VMC psi) within 2e-3 (total) of the exact optimum', EA - E < 2e-3,
        f'VMC w {wA:.4f} exact {w:.4f}; E(w_VMC) - E(w_exact) = {EA - E:.2e} (total), Im weight {Ii @ Ii:.2e}')
E_ref = E + 0.37                                         # deliberately off-centre reference: the shift algebra must make it irrelevant
sums, info = VM.stage_moments(sm, w, E_ref, args.chains, args.sweeps, 3, 60, 16, log)
out = VM.analyse(sums, E_ref, N, args.chains, args.sweeps)
log(f'G3 fast-update ratio vs separate evaluation max rel dev {info["ratio_vs_fastupdate_maxrel"]:.1e}; <h1^2>/<vt h2> {out["check_h2"]["s2_over_s2b"]:.5f} +- {out["check_h2"]["err"]:.5f}')
ex = dict(E=E, var=mu2, alpha=alpha, E1=E1d, var1=var1d, mu3=mu3, mu4=mu4)
worst = 0.0
for k in ('E', 'var', 'alpha', 'E1', 'var1', 'mu3', 'mu4'):
    z = (out[k]['value'] - ex[k]) / out[k]['err']; worst = max(worst, abs(z))
    log(f'  {k:6s} VMC {out[k]["value"]:+.6f} +- {out[k]["err"]:.1e}  exact {ex[k]:+.6f}  z {z:+.2f}')
verdict('G3 VMC moments = exact (|z| < 4 for E, var, alpha, E_1, var_1, mu3, mu4)', worst < 4.0, f'worst |z| {worst:.2f}')
res['G3'] = {k: dict(vmc=out[k]['value'], err=out[k]['err'], exact=ex[k]) for k in ex}
# broken estimator: drop the E_L(y) neighbour term of h2 (set E_y -> 0): must disagree with exact for mu3/mu4/alpha/E1
sb = sums.copy()
S2 = VM.Sampler(pn, lat, True, chunk=256); S2.start(args.chains, 3, 60); sbr = np.zeros_like(sums)
for it in range(args.sweeps // 4):
    Xs = S2.step(); E_, l_, rP_ = S2.at_x(Xs); wi_, bi_, Y_ = S2.neighbours(Xs); Ey_, ly_ = S2.at_y(Y_)
    u = w * np.exp(1j * l_.imag); vt = u.real; h1 = (u * E_).real
    Hvy = (u[wi_] * np.exp(ly_ - l_[wi_]) * 1.0).real                     # BROKEN: E_L(y) replaced by 1
    h2 = S2.diag(Xs) * h1 + np.bincount(wi_, weights=0.5 * S2.JB[bi_] * Hvy, minlength=Xs.shape[0])
    h1s = h1 - E_ref * vt; h2s = h2 - 2 * E_ref * h1 + E_ref ** 2 * vt
    q = np.stack([vt * vt, vt * h1s, h1s * h1s, h1s * h2s, h2s * h2s, vt * h2s], 1)
    for g in range(16): sbr[it, g] = q[(np.arange(args.chains) % 16) == g].sum(0)
ob = VM.analyse(sbr[:args.sweeps // 4], E_ref, N, args.chains, args.sweeps // 4)
zb = max(abs((ob[k]['value'] - ex[k]) / ob[k]['err']) for k in ('mu3', 'mu4', 'alpha', 'E1', 'var1'))
verdict('G3 broken estimator (E_L(y) -> 1) FAILS as it must', zb > 8, f'worst |z| {zb:.1f}')

# ---------------------------------------------------------------- G4 walkers
def run_chain(guide, M, dtau, nblocks, burn, seed, vec):
    r_ = np.random.default_rng(seed + 1000)
    X0 = X[r_.choice(D, M, p=vec ** 2 / np.sum(vec ** 2))]
    g = GFMC(guide, X0, dtau, seed); rec = []
    for b in range(nblocks + burn):
        r = g.block()
        if b >= burn: rec.append(r['emix'])
    m, errs, tau = blocked_error(np.array(rec)); return m, max(errs[-4:]), tau


def series_chain(guide, M, dtau, nblocks, seed, vec):
    r_ = np.random.default_rng(seed + 1000)
    X0 = X[r_.choice(D, M, p=vec ** 2 / np.sum(vec ** 2))]
    g = GFMC(guide, X0, dtau, seed)
    return np.array([g.block()['emix'] for _ in range(nblocks)])


for name, vec, al, EF in (('p=1', psi1, alpha, EFN1), ('p=0', v, 0.0, EFN0)):
    # G4a (strong, deterministic): same X0, same RNG seed, RBMPPGuideG vs TableGuide of the exact vector: identical rows -> identical walks
    nbd = 120
    gg = RBMPPGuideG(pn, lat, al, E, w=w, chunk=256); tg = TableGuide(N, BI, BJ, JB, vec)
    a_ = series_chain(gg, 64, 0.1, nbd, 31, vec); b_ = series_chain(tg, 64, 0.1, nbd, 31, vec)
    dmax = float(np.max(np.abs(a_ - b_)) / np.max(np.abs(b_)))
    verdict(f'G4a walkers {name}: RBMPPGuideG walk = TableGuide walk (same seed), {nbd} blocks', dmax < 1e-6, f'max rel difference of E_mix per block {dmax:.2e}')
    gb = RBMPPGuideG(pn, lat, al, E, w=w, chunk=256, drop_phase=True)
    c_ = series_chain(gb, 64, 0.1, nbd, 31, vec); dbk = float(np.max(np.abs(c_ - b_)) / np.max(np.abs(b_)))
    verdict(f'G4a broken (phase dropped) {name} walk differs from the table walk', dbk > 1e-3, f'max rel difference {dbk:.2e}')
    # G4b (statistical, informational with a pass window): walkers against the exact E_FN
    t0 = time.time()
    m, e, tau = run_chain(RBMPPGuideG(pn, lat, al, E, w=w, chunk=256), args.M, 0.1, args.blocks, 100, 21, vec)
    z = (m - EF) / e
    verdict(f'G4b walkers {name} = exact E_FN (|z| < 3)', abs(z) < 3, f'walker - exact = {(m - EF) / N:+.3e} +- {e / N:.1e} per site ({z:+.1f} sigma, tau {tau:.1f} blocks; E_FN minus VMC energy of the guide {(EF - (E1d if name == "p=1" else E)) / N:+.2e}) {time.time() - t0:.0f}s')
json.dump(res, open(os.path.join(args.out, f'check_general_{args.proj}.json'), 'w'), indent=1, default=float)
log('CHECK_GENERAL', 'ALL PASS' if allok else 'SOME FAIL', args.proj)
