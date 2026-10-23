"""4x4 check of the GFMC walker code against the exact lattice FN energy for the same kind of guide.

Guide: a (partly optimised) real RBM+PP on the 4x4 torus (rbmpp_lib.Lattice(4)), plus one signed Lanczos hop
psi_1 = v + alpha (H - E) v, alpha the exact p = 1 Ritz coefficient of the 4x4 Sz=0 sector (12870 states).
Stages (each printed with PASS/FAIL):
  S1 exact: H, v, E, var, alpha (2x2 Ritz), psi_1, exact E_FN by sparse eigsh of H_FN[psi_1].
  S2 provider vs table: RBMPPGuide.rows (GPU/CPU fast updates) against TableGuide rows on all 12870 configurations.
  S3 walkers (TableGuide, many blocks) vs exact E_FN, dtau and M scans, mixed and growth estimators.
  S4 broken versions (sign-violating hops dropped instead of moved to the diagonal; wrong alpha sign): must FAIL.
  S5 end-to-end short run with the RBMPPGuide on the same guide.
  python check_4x4.py OUTDIR [--stage ...]
"""
import sys, os, json, time, argparse
import numpy as np
import scipy.sparse as sp, scipy.sparse.linalg as sla
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'rbmpp_6x6'))
import jax, jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
from rbmpp_lib import Lattice, make_fns, init_params
from gfmc_fn_lib import TableGuide, RBMPPGuide, GFMC, blocked_error, iat

ap = argparse.ArgumentParser()
ap.add_argument('out'); ap.add_argument('--proj', default='simple'); ap.add_argument('--opt_iters', type=int, default=150)
ap.add_argument('--blocks', type=int, default=20000); ap.add_argument('--M', type=int, default=2000)
ap.add_argument('--stages', default='1,2,3,4,5'); ap.add_argument('--seed', type=int, default=11)
ap.add_argument('--scan', default='2000:0.1,2000:0.05,2000:0.2,500:0.1,4000:0.1')
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
stages = set(args.stages.split(','))
L = 4; N = 16
lat = Lattice(L, 0.5, args.proj)
BI = np.asarray(lat.BI); BJ = np.asarray(lat.BJ); JB = np.asarray(lat.JB, float)
log = lambda *a: print(time.strftime('%H:%M:%S'), *a, flush=True)
res = {}

# ---------------------------------------------------------------- exact sector
basis = np.array([s for s in range(1 << N) if bin(s).count('1') == N // 2], np.int64)
D = len(basis); lookup = -np.ones(1 << N, np.int64); lookup[basis] = np.arange(D)
X = ((basis[:, None] >> np.arange(N)) & 1).astype(np.int8) * 2 - 1
anti = X[:, BI] != X[:, BJ]
wi, bi = np.nonzero(anti)
nbr = lookup[basis[wi] ^ (1 << BI[bi]) ^ (1 << BJ[bi])]
diag = 0.25 * np.sum(JB[None] * X[:, BI] * X[:, BJ], 1)
Hoff = sp.coo_matrix((0.5 * JB[bi], (wi, nbr)), shape=(D, D)).tocsr()
H = (Hoff + sp.diags(diag)).tocsr()
E0 = float(sla.eigsh(H, k=1, which='SA', tol=1e-12)[0][0]); log('exact E0', E0, 'per site', E0 / N)

# ---------------------------------------------------------------- guide: real RBM + PP, optimised on the exact energy
fns = make_fns(lat, jnp.float64, True)
Xj = jnp.asarray(X.astype(np.float64))
lp_chunk = jax.jit(jax.vmap(fns['logpsi'], in_axes=(None, 0)))
CH = 1024


def psi_all(p):
    out = []
    for i in range(0, D, CH):
        out.append(np.real(np.asarray(jnp.exp(lp_chunk(p, Xj[i:i + CH])))))
    return np.concatenate(out)


def build_params(z):
    M = 4
    f = z[:N * N].reshape(N, N); b = z[N * N:N * N + M]; W = z[N * N + M:].reshape(N, M)
    return dict(f=jnp.asarray(f), bre=jnp.asarray(b), bim=jnp.zeros(M), Wre=jnp.asarray(W), Wim=jnp.zeros((N, M)))


Dpad = ((D + CH - 1) // CH) * CH
Xpad = jnp.concatenate([Xj, jnp.repeat(Xj[:1], Dpad - D, 0)])


@jax.jit
def fwd(zz, Xc):
    return jnp.real(jnp.exp(lp_chunk(build_params(zz), Xc)))


@jax.jit
def bwd(zz, Xc, g):
    return jax.vjp(lambda z_: fwd(z_, Xc), zz)[1](g)[0]


def loss_grad(z):
    zz = jnp.asarray(z)
    psi = np.concatenate([np.asarray(fwd(zz, Xpad[i:i + CH])) for i in range(0, Dpad, CH)])[:D]
    Hp = H @ psi; nn = psi @ psi; E = psi @ Hp / nn
    g = np.zeros(Dpad); g[:D] = 2 * (Hp - E * psi) / nn
    grad = np.zeros_like(z)
    for i in range(0, Dpad, CH):
        grad += np.asarray(bwd(zz, Xpad[i:i + CH], jnp.asarray(g[i:i + CH])))
    return E, grad


gpath = os.path.join(args.out, 'guide4x4.npz')
if os.path.exists(gpath):
    z = np.load(gpath)['z']
else:
    from scipy.optimize import minimize
    p0 = init_params(lat, M=4, seed=3)
    z0 = np.concatenate([np.asarray(p0['f']).ravel(), np.asarray(p0['bre']), np.asarray(p0['Wre']).ravel()])
    t0 = time.time(); cnt = [0]

    def fun(z):
        E, g = loss_grad(z); cnt[0] += 1
        if cnt[0] % 10 == 1: log('opt', cnt[0], E / N, 'dE/site', (E - E0) / N, f'{time.time()-t0:.0f}s')
        return E, g
    r = minimize(fun, z0, jac=True, method='L-BFGS-B', options=dict(maxiter=args.opt_iters))
    z = r.x; np.savez(gpath, z=z)
p = build_params(z)
v = psi_all(p); v /= np.linalg.norm(v)
Hv = H @ v; E = float(v @ Hv); var = float(np.sum((Hv - E * v) ** 2))
u1 = Hv - E * v
# exact p = 1 Ritz in span{v, u1}
G = np.array([[1, 0], [0, u1 @ u1]]); Hm = np.array([[0, u1 @ u1], [u1 @ u1, u1 @ (H @ u1) - E * (u1 @ u1)]])
w_, c_ = np.linalg.eigh(np.linalg.solve(np.linalg.cholesky(G), np.linalg.solve(np.linalg.cholesky(G), Hm).T).T) if False else (None, None)
import scipy.linalg as sl
ev, ec = sl.eigh(Hm, G); alpha = float(ec[1, 0] / ec[0, 0])
alpha_formula = float(ev[0] / var)
log(f'guide dE/site {(E-E0)/N:.4e} var/site {var/N:.4e} alpha {alpha:.6f} (E1-E)/var {alpha_formula:.6f}')
psi1 = v + alpha * u1
E1 = float(psi1 @ (H @ psi1) / (psi1 @ psi1)); log('p=1 state dE/site', (E1 - E0) / N)
res['guide'] = dict(dE_site=(E - E0) / N, var_site=var / N, alpha=alpha, alpha_formula=alpha_formula, dE1_site=(E1 - E0) / N)
assert abs(alpha - alpha_formula) < 1e-9 * abs(alpha), 'alpha = (E1-E)/var identity failed'


def fn_exact(psi, drop=False, lnk=None):
    s = np.sign(psi); a = np.abs(psi)
    rr = psi[nbr] / psi[wi]
    K = 0.5 * JB[bi]
    keep = rr < 0
    d = diag.copy()
    if not drop:
        d += np.bincount(wi[~keep], weights=(K * rr)[~keep], minlength=D)
    F = sp.coo_matrix((np.r_[-K[keep] * rr[keep] * -1, d], (np.r_[wi[keep], np.arange(D)], np.r_[nbr[keep], np.arange(D)])), shape=(D, D)).tocsr()
    # H_FN in the importance-sampled (non-symmetric) form: off = K r (negative), so ground value is the same as the symmetric one
    ew = sla.eigs(F, k=1, which='SR', tol=1e-12, maxiter=100000)[0]
    return float(np.real(ew[0]))


if '1' in stages:
    efn = fn_exact(psi1)
    log(f'S1 exact E_FN(psi_1) dE/site {(efn-E0)/N:.6e} (<H> {(E1-E0)/N:.6e}, E_FN(v) {(fn_exact(v)-E0)/N:.6e})')
    res['S1'] = dict(EFN_site=(efn - E0) / N, EFN_v_site=(fn_exact(v) - E0) / N)
EFN = fn_exact(psi1)
res['EFN_total'] = EFN

# ---------------------------------------------------------------- S2 provider vs table
tg = TableGuide(N, BI, BJ, JB, psi1)
if '2' in stages:
    gp = RBMPPGuide({k: np.asarray(p[k]) for k in p}, lat, alpha, E, chunk=256, w=1.0)
    # E in the provider is the (normalised) <v|H|v> of v = psi_P/|psi_P|_sector: psi_P has norm != 1 but E is a ratio: ok
    idx = np.random.default_rng(0).choice(D, 600, replace=False)
    t0 = time.time(); r, d = gp.rows(X[idx]); r0, d0 = tg.rows(X[idx])
    # provider guide is psi_P scaled; ratios are scale free
    err = max(np.max(np.abs(r - r0)), np.max(np.abs(d - d0)))
    scale = max(np.max(np.abs(r0)), np.max(np.abs(d0)))
    log(f'S2 provider vs table on {len(idx)} configs: max abs diff rate/dFN {err:.3e} (scale {scale:.3g}); max Im {gp.max_im:.2e}; {time.time()-t0:.1f}s')
    res['S2'] = dict(maxdiff=float(err), scale=float(scale), max_im=gp.max_im, PASS=bool(err < 1e-9 * scale))
    log('S2', 'PASS' if res['S2']['PASS'] else 'FAIL')


# ---------------------------------------------------------------- S3 walkers
def run_chain(guide, M, dtau, nblocks, burn, seed, X0=None):
    rng = np.random.default_rng(seed + 1000)
    if X0 is None:
        X0 = X[rng.choice(D, M, p=np.abs(psi1) ** 2 / np.sum(psi1 ** 2))]
    g = GFMC(guide, X0, dtau, seed)
    rec = []
    for b in range(nblocks + burn):
        r = g.block()
        if b >= burn: rec.append(r)
    return rec


def summarize(rec, EFN):
    em = np.array([r['emix'] for r in rec]); eg = np.array([r['egrow'] for r in rec])
    out = {}
    for name, x in (('mix', em), ('grow', eg)):
        m, errs, tau = blocked_error(x)
        out[name] = dict(dE_site=(m * 1.0 - 0) / N, err_site=max(errs[-4:]) / N, tau_blocks=tau,
                         diff_site=(m - EFN) / N, nsig=(m - EFN) / max(errs[-4:]))
    out['ess_mean'] = float(np.mean([r['ess'] for r in rec])); out['hops_per_walker_block'] = float(np.mean([r['hops'] for r in rec]) / len(rec[0:1]) )
    return out


if '3' in stages:
    scan = []
    for (M, dtau) in [(int(a_.split(':')[0]), float(a_.split(':')[1])) for a_ in args.scan.split(',')]:
        t0 = time.time(); rec = run_chain(tg, M, dtau, args.blocks, 100, args.seed)
        s = summarize(rec, EFN); s.update(M=M, dtau=dtau, sec=time.time() - t0)
        log(f"S3 M={M} dtau={dtau}: mixed {s['mix']['diff_site']:.3e} +- {s['mix']['err_site']:.1e} ({s['mix']['nsig']:+.1f} sig, tau {s['mix']['tau_blocks']:.1f} blocks) "
            f"growth {s['grow']['diff_site']:.3e} +- {s['grow']['err_site']:.1e} ({s['grow']['nsig']:+.1f} sig)  [diff = walker - exact, per site; EFN-E0 {(EFN-E0)/N:.4e}]  {s['sec']:.0f}s")
        scan.append(s)
    res['S3'] = scan

# ---------------------------------------------------------------- S4 broken versions
if '4' in stages:
    def brk_drop(self, X, anti, wi, bi, rr):
        K = 0.5 * self.JB[bi]; keep = rr < 0
        rate = np.zeros(anti.shape); rate[wi[keep], bi[keep]] = -K[keep] * rr[keep]
        return rate, self.diag0(X)                               # violating hops dropped, not moved to the diagonal
    tb = TableGuide(N, BI, BJ, JB, psi1, break_fn=brk_drop)
    rec = run_chain(tb, args.M, 0.1, args.blocks // 2, 100, 12)
    s = summarize(rec, EFN)
    log(f"S4 broken (violating hops dropped): mixed diff {s['mix']['diff_site']:.3e} +- {s['mix']['err_site']:.1e} ({s['mix']['nsig']:+.1f} sig) -> expect FAIL (|nsig|>5)")
    # wrong alpha sign in the guide, compared against the exact FN of the intended guide
    tw = TableGuide(N, BI, BJ, JB, v - alpha * u1)
    rec = run_chain(tw, args.M, 0.1, args.blocks // 2, 100, 13)
    s2 = summarize(rec, EFN)
    log(f"S4 wrong alpha sign: mixed diff {s2['mix']['diff_site']:.3e} +- {s2['mix']['err_site']:.1e} ({s2['mix']['nsig']:+.1f} sig) -> expect FAIL")
    res['S4'] = dict(dropped=s, wrong_alpha=s2)

# ---------------------------------------------------------------- S5 end to end with the provider
if '5' in stages:
    gp = RBMPPGuide({k: np.asarray(p[k]) for k in p}, lat, alpha, E, chunk=256)
    t0 = time.time(); rec = run_chain(gp, 200, 0.1, 400, 40, 21)
    s = summarize(rec, EFN); s['sec'] = time.time() - t0
    log(f"S5 RBMPP provider M=200 dtau=0.1 400 blocks: mixed diff {s['mix']['diff_site']:.3e} +- {s['mix']['err_site']:.1e} ({s['mix']['nsig']:+.1f} sig) {s['sec']:.0f}s; evals {gp.n_eval}")
    res['S5'] = s
json.dump(res, open(os.path.join(args.out, 'check_4x4.json'), 'w'), indent=1, default=float)
log('WROTE', os.path.join(args.out, 'check_4x4.json'))
