"""Test L8 controls at 4x4 (exact basis, CPU): (iv) dump regression, (ii) fidelity estimator vs exact, (iii) broken sign, (i) zero-step refit.

  python l8_check4x4.py OUT --guide GUIDE.npz [--blocks 300 --burn 50 --M 256 --dtau 0.1 --steps 30]
GUIDE.npz: check_general.py's L-BFGS 4x4 complex RBM+PP ('flat'; translations x spin flip, M = 4 hidden units), e.g. the 30-iteration guide
/scratch/a/A.Otaifi/gfmc_m13b/cg_final_30/guide4x4_complex_translations.npz (E/N -0.52374 vs exact -0.52862: room for a refit).
Every walker below is produced by the PRODUCTION path: gfmc_run.py (RBMPPGuideG, p = 1, consts from the exact 4x4 numbers) with --dump,
and every estimate by l8_lib / l8_refit.py, as at 8x8.  Exact: phi_0 = ground state of the FN Hamiltonian of g = v + alpha (H - E) v.
 (iv) gfmc_run.py with and without --dump, same seed: blocks.npz arrays (all but the wall-clock 'sec') and the final ckpt walkers + RNG state
      bitwise identical.  Broken: a dump that draws one extra random number per block must change them.
 (ii) for theta_0 (a = v_0 != g) and two perturbed parameter sets: E_pi(a/g) vs <a|phi>/<g|phi>, Q = <a|phi>/<v0|phi> and R(theta) vs exact,
      z = (estimate - exact) / error (bin jackknife over blocks of 2 chains; Ninv error from VMC sweep scatter); pass |z| < 3 for all.
 (iii) refit (steps SR steps) with weights a/g and with a/|g| (--sign_broken): exact R of each final parameter set; pass if the correct refit
      raises exact R and the broken one ends measurably lower (its gain < half of the correct gain).  Also held-out R of the correct refit vs exact.
 (i) l8_refit.py --steps 0: params_final arrays bitwise = params0, R = 1 exactly; broken: --steps 1 must change both.
"""
import argparse, json, os, sys, time, subprocess, pickle
import numpy as np
import scipy.sparse as sp, scipy.sparse.linalg as sla, scipy.linalg as sl
HERE = os.path.dirname(os.path.abspath(__file__)); GF = os.path.join(HERE, '..'); RB = os.path.join(HERE, '..', '..', 'rbmpp_6x6')
for d in (HERE, GF, RB): sys.path.insert(0, d)
import jax, jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
from jax.flatten_util import ravel_pytree
from rbmpp_lib import Lattice, make_fns, init_params
from gfmc_fn_lib import TableGuide, GFMC
import l8_lib as LL

ap = argparse.ArgumentParser()
ap.add_argument('out'); ap.add_argument('--guide', required=True)
ap.add_argument('--blocks', type=int, default=300); ap.add_argument('--burn', type=int, default=50); ap.add_argument('--M', type=int, default=256)
ap.add_argument('--dtau', type=float, default=0.1); ap.add_argument('--steps', type=int, default=30); ap.add_argument('--lr', type=float, default=0.3)
ap.add_argument('--maxn', type=float, default=0.02); ap.add_argument('--eps', default='0.1,0.3'); ap.add_argument('--vmc_sweeps', type=int, default=40)
ap.add_argument('--skip_chains', action='store_true', help='reuse the chains already in OUT')
ap.add_argument('--skip_ii', action='store_true', help='skip the (ii) parameter-set comparison (refit checks only)')
args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)
PY = sys.executable
log = lambda *a: print(time.strftime('%H:%M:%S'), *a, flush=True)
res = {}; allok = True


def verdict(name, ok, msg=''):
    global allok
    allok &= bool(ok); log(('PASS ' if ok else 'FAIL ') + name, msg); res.setdefault('verdicts', {})[name] = dict(ok=bool(ok), msg=msg)


L = 4; N = 16; lat = Lattice(L, 0.5, 'translations')
BI = np.asarray(lat.BI); BJ = np.asarray(lat.BJ); JB = np.asarray(lat.JB, float)
basis = np.array([s for s in range(1 << N) if bin(s).count('1') == N // 2], np.int64)
Dd = len(basis); lookup = -np.ones(1 << N, np.int64); lookup[basis] = np.arange(Dd)
X = ((basis[:, None] >> np.arange(N)) & 1).astype(np.int8) * 2 - 1
anti = X[:, BI] != X[:, BJ]; wi, bi = np.nonzero(anti)
nbr = lookup[basis[wi] ^ (1 << BI[bi]) ^ (1 << BJ[bi])]
diag = 0.25 * np.sum(JB[None] * X[:, BI] * X[:, BJ], 1)
H = (sp.coo_matrix((0.5 * JB[bi], (wi, nbr)), shape=(Dd, Dd)).tocsr() + sp.diags(diag)).tocsr()

# ---------------------------------------------------------------- guide params and exact quantities
p = init_params(lat, M=4, seed=5, rbm_scale=0.25); _, unrav = ravel_pytree(p)
p = unrav(jnp.asarray(np.load(args.guide)['flat']))
P0 = os.path.join(args.out, 'params4.npz')
np.savez(P0, **{k: np.asarray(p[k]) for k in LL.KEYS}, proj='translations', rbm=True)
z0 = np.load(P0); model = LL.Model(lat, z0, True, 256); flat0 = np.asarray(model.flat0)
Xf = X.astype(np.float64)
lall = model.logpsi(flat0, Xf)                                  # raw normalisation, as the walkers' code sees it
psi = np.exp(lall)
Rr, Ii = psi.real, psi.imag
A_ = np.array([[Rr @ (H @ Rr), Rr @ (H @ Ii)], [Ii @ (H @ Rr), Ii @ (H @ Ii)]]); B_ = np.array([[Rr @ Rr, Rr @ Ii], [Ii @ Rr, Ii @ Ii]])
ev, ec = sl.eigh(0.5 * (A_ + A_.T), B_); cc = ec[:, 0]; w = cc[0] - 1j * cc[1]; w /= abs(w)
v0 = (w * psi).real; Hv = H @ v0; E = float(v0 @ Hv / (v0 @ v0)); u1 = Hv - E * v0
vn = v0 / np.linalg.norm(v0); u1n = H @ vn - E * vn; u2n = H @ u1n - E * u1n
mu2 = u1n @ u1n; mu3 = u1n @ u2n
alpha = float((mu3 - np.sqrt(mu3 ** 2 + 4 * mu2 ** 3)) / (2 * mu2 ** 2))
g = v0 + alpha * u1                                              # = Re[w psi_P (1 + alpha (E_L - E))] in the raw normalisation
CJ = os.path.join(args.out, 'consts4.json')
json.dump(dict(alpha=alpha, E=E, w=[w.real, w.imag], source='l8_check4x4 exact'), open(CJ, 'w'), indent=1)
# FN Hamiltonian of g (symmetric), its ground state phi
rr = g[nbr] / g[wi]; K = 0.5 * JB[bi]; keep = rr < 0
dFN = diag + np.bincount(wi[~keep], weights=(K * rr)[~keep], minlength=Dd)
HFN = (sp.coo_matrix((K[keep], (wi[keep], nbr[keep])), shape=(Dd, Dd)) + sp.diags(dFN)).tocsr()
eFN, vFN = sla.eigsh(HFN, k=1, which='SA', tol=1e-13); EFN = float(eFN[0]); phi = vFN[:, 0]
if phi @ g < 0: phi = -phi
E0 = sla.eigsh(H, k=1, which='SA')[0][0]
log(f'exact 4x4: E(v)/N {E / N:.8f} alpha {alpha:.8f} w {w:.5f}  E_FN(g)/N {EFN / N:.8f}  E0/N {E0 / N:.8f}; min(phi g)/max {np.min(phi * g) / np.max(phi * g):.1e}; '
    f'h<0 fraction (|phi g|-weighted) {np.sum(np.abs(phi * g)[(g / v0) < 0]) / np.sum(np.abs(phi * g)):.2e}')
F = lambda a: (a @ phi) ** 2 / (a @ a)
res['exact'] = dict(E=E, alpha=alpha, EFN=EFN, E0=E0, F_v0=float(F(v0) / (phi @ phi)), F_g=float(F(g) / (phi @ phi)))
log(f'exact fidelities: F(v0) {res["exact"]["F_v0"]:.6f}  F(g) {res["exact"]["F_g"]:.6f}')

# ---------------------------------------------------------------- (iv) dump regression (production gfmc_run.py)
env = dict(os.environ, JAX_PLATFORMS='cpu')


def gfmc(name, seed, blocks, burn, dump, wait=False):
    out = os.path.join(args.out, name)
    cmd = [PY, os.path.join(GF, 'gfmc_run.py'), out, '--params', P0, '--consts', CJ, '--p', '1', '--L', '4', '--proj', 'translations',
           '--M', str(args.M), '--dtau', str(args.dtau), '--blocks', str(blocks), '--burn', str(burn), '--seed', str(seed), '--chunk', '256',
           '--ref', str(E0 / N)] + (['--dump'] if dump else [])
    pr = subprocess.Popen(cmd, env=env, stdout=open(out + '.log', 'w'), stderr=subprocess.STDOUT)
    return pr


if not args.skip_chains:
    for d in ('reg_dump', 'reg_nodump', 'c1', 'c2'):
        subprocess.run(['rm', '-rf', os.path.join(args.out, d)])
    procs = [gfmc('reg_dump', 7, 40, 10, True), gfmc('reg_nodump', 7, 40, 10, False),
             gfmc('c1', 1, args.blocks, args.burn, True), gfmc('c2', 2, args.blocks, args.burn, True)]
    t0 = time.time()
    for pr in procs: pr.wait()
    log(f'gfmc_run chains done ({time.time() - t0:.0f}s), exit codes {[pr.returncode for pr in procs]}')
a_ = np.load(os.path.join(args.out, 'reg_dump', 'blocks.npz')); b_ = np.load(os.path.join(args.out, 'reg_nodump', 'blocks.npz'))
keys = [k for k in a_.files if k != 'sec']
same = all(np.array_equal(a_[k], b_[k]) for k in keys)
ca = np.load(os.path.join(args.out, 'reg_dump', 'ckpt.npz'), allow_pickle=True); cb = np.load(os.path.join(args.out, 'reg_nodump', 'ckpt.npz'), allow_pickle=True)
same_ck = np.array_equal(ca['X'], cb['X']) and pickle.loads(ca['rng'].tobytes()) == pickle.loads(cb['rng'].tobytes())
ndump = len(os.listdir(os.path.join(args.out, 'reg_dump', 'dump')))
verdict('(iv) gfmc_run.py --dump vs no dump: blocks.npz bitwise identical (all arrays but sec) and final walkers + RNG identical',
        same and same_ck and ndump == 50, f'{len(a_["emix"])} blocks, arrays {keys}; {ndump} dump files; emix[-1] {a_["emix"][-1]!r} vs {b_["emix"][-1]!r}')
# broken dump (draws one extra random number per block) on the in-process walk: must differ; the correct keep=True must not
tg = TableGuide(N, BI, BJ, JB, g)
X0 = X[np.random.default_rng(3).choice(Dd, 128, p=g ** 2 / np.sum(g ** 2))]


def walk(mode, nb=40):
    G_ = GFMC(tg, X0.copy(), 0.1, 11); G_.keep = mode != 'off'; out = []
    for _ in range(nb):
        out.append(G_.block()['emix'])
        if mode == 'broken': G_.rng.random()
    return np.array(out)


e_off, e_on, e_br = walk('off'), walk('on'), walk('broken')
nd_br = int(np.sum(e_off != e_br))
verdict('(iv) in-process: keep=True walk identical to keep=False; broken dump (extra RNG draw) differs', np.array_equal(e_off, e_on) and nd_br > 0,
        f'keep=True differing blocks {int(np.sum(e_off != e_on))}/40; broken differing blocks {nd_br}/40 (first at {int(np.argmax(e_off != e_br))})')
res['iv'] = dict(identical=bool(same and same_ck), broken_differing_blocks=nd_br)

# ---------------------------------------------------------------- (ii) estimator vs exact
dirs = [os.path.join(args.out, 'c1'), os.path.join(args.out, 'c2')]
D = LL.load_dumps(dirs, 0.0, first_block=args.burn)
Gw = LL.guide_at(model, flat0, D['X'], alpha, E, w)
idx = lookup[((D['X'] > 0).astype(np.int64) * (1 << np.arange(N))).sum(1)]
gdev = np.max(np.abs(np.exp(Gw['lr0']) * Gw['cg'] - g[idx])) / np.max(np.abs(g))
verdict('(ii) guide values at the walkers (l8_lib.guide_at) = exact g', gdev < 1e-10, f'max rel dev {gdev:.1e}')
allset = np.ones(len(D['p']), bool)
WS = LL.WalkerSet(D, Gw, allset, w, binb=2); WSb = LL.WalkerSet(D, Gw, allset, w, binb=2, sign_broken=True)
per_chain = [LL.WalkerSet(D, Gw, D['chain'] == c, w, binb=2) for c in (0, 1)]
rng = np.random.default_rng(42)
xi1 = rng.standard_normal(model.P); xi2 = rng.standard_normal(model.P)
eps = [float(x) for x in args.eps.split(',')]
sets = [('theta0 (a = v0)', flat0), (f'theta0 + {eps[0]} xi1', flat0 + eps[0] * xi1 * np.abs(flat0).mean()),
        (f'theta0 + {eps[1]} xi2', flat0 + eps[1] * xi2 * np.abs(flat0).mean())]
nbin = WS.nbin; binw = np.bincount(WS.bin, weights=WS.pw, minlength=nbin)


def mean_jk(r, ws):
    A = np.bincount(ws.bin, weights=ws.pw * r, minlength=ws.nbin); Bw = np.bincount(ws.bin, weights=ws.pw, minlength=ws.nbin)
    jk = (A.sum() - A) / (Bw.sum() - Bw); n = ws.nbin
    return float(A.sum() / Bw.sum()), float(np.sqrt((n - 1) / n * np.sum((jk - jk.mean()) ** 2)))


res['ii'] = []
worst = 0.0
for name, fl in ([] if args.skip_ii else sets):
    la = model.logpsi(fl, Xf); a = (w * np.exp(la)).real
    r = WS.ratios(model.logpsi(fl, WS.X))
    m, me = mean_jk(r, WS); ex_m = (a @ phi) / (g @ phi)
    q, qe = WS.Q(r); ex_q = (a @ phi) / (v0 @ phi)
    vm = LL.VMC(model, 1024, 3, therm=50); nis = []
    for k in range(args.vmc_sweeps):
        S = vm.sample(fl); nis.append(LL.ninv(model, fl, flat0, S, w)[0])
    ni = float(np.mean(nis)); nie = float(np.std(nis, ddof=1) / np.sqrt(len(nis))) if not np.array_equal(fl, flat0) else 0.0
    ex_ni = (v0 @ v0) / (a @ a)
    R = q * q * ni; Re_ = R * np.sqrt((2 * qe / q) ** 2 + (nie / ni) ** 2); ex_R = F(a) / F(v0)
    zs = [(m - ex_m) / me, (q - ex_q) / qe if qe > 0 else 0.0, (R - ex_R) / Re_ if Re_ > 0 else 0.0]
    if qe == 0: zs[1] = 0.0 if q == ex_q or abs(q - ex_q) < 1e-12 else np.inf
    pc = [mean_jk(ws.ratios(model.logpsi(fl, ws.X)), ws) for ws in per_chain]
    qb, qbe = WSb.Q(WSb.ratios(model.logpsi(fl, WSb.X)))
    Rb = qb * qb * ni
    log(f'(ii) {name}: E_pi(a/g) {m:.6f} +- {me:.1e} exact {ex_m:.6f} z {zs[0]:+.2f} (chains {pc[0][0]:.6f} +- {pc[0][1]:.1e}, {pc[1][0]:.6f} +- {pc[1][1]:.1e}); '
        f'Q {q:.6f} +- {qe:.1e} exact {ex_q:.6f} z {zs[1]:+.2f}; Ninv {ni:.6f} +- {nie:.1e} exact {ex_ni:.6f}; R {R:.6f} +- {Re_:.1e} exact {ex_R:.6f} z {zs[2]:+.2f}; '
        f'BROKEN (a/|g|): Q {qb:.6f} R {Rb:.6f}')
    worst = max(worst, max(abs(x) for x in zs))
    res['ii'].append(dict(name=name, Epi=m, Epi_err=me, Epi_exact=ex_m, Q=q, Q_err=qe, Q_exact=ex_q, Ninv=ni, Ninv_err=nie, Ninv_exact=ex_ni,
                          R=R, R_err=Re_, R_exact=ex_R, z=zs, chains=pc, Q_broken=qb, R_broken=Rb))
if not args.skip_ii: verdict('(ii) E_pi(a/g), Q and R = exact for 3 parameter sets (|z| < 3)', worst < 3, f'worst |z| {worst:.2f}')
hh = WS.health(WS.r0); log(f'(ii) 4x4 weight health at theta0: {hh}')
res['health4'] = hh


# ---------------------------------------------------------------- (iii) refit correct vs broken sign, exact R; (i) zero step
def refit(name, extra):
    out = os.path.join(args.out, name); subprocess.run(['rm', '-rf', out])
    cmd = [PY, os.path.join(HERE, 'l8_refit.py'), out, '--params0', P0, '--consts', CJ, '--dumps', ','.join(dirs), '--L', '4', '--first_block', str(args.burn),
           '--chains', '1024', '--seed', '5', '--lr', str(args.lr), '--maxn', str(args.maxn), '--save_every', '1', '--final_sweeps', '16'] + extra
    rc = subprocess.run(cmd, env=env, stdout=open(out + '.log', 'w'), stderr=subprocess.STDOUT).returncode
    return out, rc


def exact_R(path):
    zz = np.load(path); m2 = LL.Model(lat, zz, True, 256); fl = np.asarray(m2.flat0)
    a = (w * np.exp(m2.logpsi(fl, Xf))).real
    return float(F(a) / F(v0))


t0 = time.time()
oc, rcc = refit('refit_ok', ['--steps', str(args.steps), '--no_gate'])
ob, rcb = refit('refit_broken', ['--steps', str(args.steps), '--no_gate', '--sign_broken', '--no_health_stop'])
Rc = exact_R(os.path.join(oc, 'params_final.npz')); Rb = exact_R(os.path.join(ob, 'params_final.npz'))
jc = json.load(open(os.path.join(oc, 'refit.json'))); jb = json.load(open(os.path.join(ob, 'refit.json')))
fc = jc['final']; fb = jb['final']
log(f'(iii) refits ({time.time() - t0:.0f}s, rc {rcc} {rcb}): correct: exact R {Rc:.6f}, held-out estimate {fc["R_held"]:.6f} +- {fc["R_held_err"]:.1e}; '
    f'broken a/|g|: exact R {Rb:.6f}, its own held-out estimate {fb["R_held"]:.6f} +- {fb["R_held_err"]:.1e}')
log('(iii) held-out curve correct: ' + ' '.join(f'{c_["it"]}:{c_["R_held"]:.5f}+-{c_["R_held_err"]:.4f}' for c_ in jc['curve']))
log('(iii) curve broken: ' + ' '.join(f'{c_["it"]}:{c_["R_held"]:.5f}' for c_ in jb['curve'][::5]))
verdict('(iii) correct refit raises exact R; broken sign (a/|g|) ends measurably lower (gain < half)', Rc > 1 and (Rb - 1) < 0.5 * (Rc - 1),
        f'exact R correct {Rc:.6f} vs broken {Rb:.6f}')
zf = (fc['R_held'] - Rc) / fc['R_held_err']
import glob as _g
exc = [(int(os.path.basename(f_)[8:11]), exact_R(f_)) for f_ in sorted(_g.glob(os.path.join(oc, 'params_t*.npz')))]
log('(iii) exact R per step of the correct refit: ' + ' '.join(f'{i_}:{r_:.5f}' for i_, r_ in exc))
sel = fc['selected_step']; nst = len(jc['curve']) - 1
best_exact = max(exc, key=lambda t: t[1]) if exc else (0, 1.0)
log(f'(iii) selected step {sel} of {nst} (held-out curve turned down after it: {sel < nst}); exact-R maximum at step {best_exact[0]} ({best_exact[1]:.5f})')
res['iii_exact_curve'] = exc; res['iii_selected'] = sel
verdict('(ii/iii) chain-split held-out R at the selected step = its exact R (|z| < 2)', abs(zf) < 2, f'estimate {fc["R_held"]:.6f} +- {fc["R_held_err"]:.1e} exact {Rc:.6f} z {zf:+.2f}')
res['iii'] = dict(R_exact_correct=Rc, R_exact_broken=Rb, final_correct=fc, final_broken=fb, curve_correct=[(c_['it'], c_['R_held'], c_['R_held_err']) for c_ in jc['curve']],
                  curve_broken=[(c_['it'], c_['R_held'], c_['R_held_err']) for c_ in jb['curve']])
o0, rc0 = refit('refit_zero', ['--steps', '0', '--no_gate'])
o1, rc1 = refit('refit_one', ['--steps', '1', '--no_gate', '--select', 'last'])
zz0 = np.load(os.path.join(o0, 'params_final.npz')); zz1 = np.load(os.path.join(o1, 'params_final.npz'))
bit0 = all(np.array_equal(zz0[k], z0[k]) for k in LL.KEYS); bit1 = all(np.array_equal(zz1[k], z0[k]) for k in LL.KEYS)
R0 = json.load(open(os.path.join(o0, 'refit.json')))['final']['R_held']; R1 = json.load(open(os.path.join(o1, 'refit.json')))['final']['R_held']
verdict('(i) zero-step refit: params bitwise = params0 and R = 1 exactly; broken (1 step) changes both', bit0 and R0 == 1.0 and (not bit1) and R1 != 1.0,
        f'0 steps: bitwise {bit0}, R {R0!r}; 1 step: bitwise {bit1}, R {R1!r}')
json.dump(res, open(os.path.join(args.out, 'l8_check4x4.json'), 'w'), indent=1, default=float)
log('L8_CHECK4X4', 'ALL PASS' if allok else 'SOME FAIL')
