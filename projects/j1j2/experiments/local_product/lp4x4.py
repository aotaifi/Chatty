"""Local-product Krylov (LP), Stage A at 4x4 (PREREG.md in this directory; rules fixed there).   Exact, no Monte Carlo.
   MASK=f|Wim,Wre,bim,bre  python lp4x4.py OUTDIR GUIDEDIR        (GUIDEDIR = /scratch/a/A.Otaifi/l8/loop4x4/{fonly,rbmonly})
Guide psi = v = Re[w30 psi_P(flat_opt)] (the VMC optimum of the restricted class, exactly as l8_loop4x4.py 'opt'; E_v/N checked
against GUIDEDIR/opt.json).  Setup (lattice, H, basis, ansatz, w30, FN solver, Ritz hop) is l8_loop4x4.py's own code, executed
up to its first STAGE branch (not copied).
  eps_b(x) = (H_b v)(x)/v(x) = J_b/4 s_i s_j + [s_i != s_j] J_b/2 v(x^b)/v(x);  sum_b eps_b = E_L (checked).
  G   : v (1 + alpha (E_L - E)), alpha by Ritz (hop()).
  LP  : v prod_b [1 + a_t (eps_b - e_t)], t = J1/J2, e_t = <eps_b>_v averaged over the bonds of type t; (a_J1, a_J2) minimise E_VMC.
  LPs : v prod_i [1 + a (eps_i - e_s)], eps_i = 1/2 sum_{b ni i} eps_b, e_s = <eps_i>; a minimises E_VMC.
Controls (each with a broken version): alpha = 0 returns v bitwise (broken: alpha 1e-6); broken locality (see SHIFT note);
first-order identity |n(LP(a,a)) - n(G(a))| = O(a^2) at two a (broken: J2 factors dropped -> O(a)).
Amendment 1 (Astra): shifting eps_b by a lattice vector only PERMUTES the factors (LP_shift == LP, printed, not a control);
broken locality = 'LPperm': eps_b taken from an unrelated configuration (fixed random permutation of the basis), alphas optimised.
Added arms: LPeq (alpha_J1 = alpha_J2), P_plaq (2x2 plaquettes, J1 weight 1/2, J2 weight 1, one alpha), LPunif (no locality,
v (1 + a (E_L - E)/N_B)^N_B).  Crossed: E_FN(|G| sgn LP), E_FN(|LP| sgn G); amplification max |n(LP)/n(psi)| and where.
Verdict rule (PREREG): best of LP, LPs only.
"""
import os, sys, json, time
import numpy as np
OUT, GDIR = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
SRC = '/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/experiments/gfmc_fn_lattice/l8/l8_loop4x4.py'
code = open(SRC).read(); cut = code.index("\nif STAGE == 'time':")
sys.argv = [SRC, os.path.join(OUT, 'l8setup'), 'none']
g = {'__name__': 'l8setup', '__file__': SRC}
exec(compile(code[:cut], SRC, 'exec'), g)
for k in ('H', 'E0', 'N', 'basis', 'lookup', 'X', 'BI', 'BJ', 'JB', 'vv', 'w30', 'fn_state', 'hop', 'MASK', 'L'):
    globals()[k] = g[k]
t00 = time.time()
RES = os.path.join(OUT, 'lp4x4.json'); res = dict(mask=MASK, guide_dir=GDIR, E0N=E0 / N)


def log(*a):
    s = time.strftime('%Y-%m-%dT%H:%M:%S') + ' PROGRESS ' + ' '.join(str(x) for x in a)
    print(s, flush=True); open(os.path.join(OUT, 'progress.txt'), 'a').write(s + '\n')


def dump(): open(RES, 'w').write(json.dumps(res, indent=1, default=float))


x_opt = np.load(os.path.join(GDIR, 'flat_opt.npy')); ref = json.load(open(os.path.join(GDIR, 'opt.json')))
v = vv(x_opt, w30); v = v / np.linalg.norm(v)
Ev = float(v @ (H @ v)); res['psi_check'] = dict(EvN=Ev / N, EvN_opt_json=ref['EvN'], diff=Ev / N - ref['EvN'])
log(f'guide {MASK}: E_v/N {Ev / N:.12f} (opt.json {ref["EvN"]:.12f}); min |v| {np.abs(v).min():.2e}')
if abs(Ev / N - ref['EvN']) > 1e-9: log('GUIDE MISMATCH; exit 3'); sys.exit(3)

Dd, NB = X.shape[0], BI.size
MB = (np.int64(1) << BI.astype(np.int64)) | (np.int64(1) << BJ.astype(np.int64))
anti = X[:, BI] != X[:, BJ]
nbr = np.where(anti, lookup[basis[:, None] ^ MB[None, :]], 0)
DG = 0.25 * JB[None, :] * X[:, BI] * X[:, BJ]
def off_of(u): return np.where(anti, 0.5 * JB[None, :] * u[nbr] / u[:, None], 0.0)
OFF = off_of(v); EPS = DG + OFF
EL = (H @ v) / v
res['check_sum_eps_EL'] = float(np.max(np.abs(EPS.sum(1) - EL) / np.maximum(1, np.abs(EL))))
TYPE = (JB < 0.75).astype(int)                       # 0 = J1, 1 = J2
p = v * v
def etype(E_): return np.array([np.mean((p @ E_)[TYPE == t]) for t in (0, 1)])
ET = etype(EPS)
log(f'sum_b eps_b vs E_L: max rel {res["check_sum_eps_EL"]:.2e}; e_J1 {ET[0]:.8f} e_J2 {ET[1]:.8f}; '
    f'32 e_J1 + 32 e_J2 = {np.sum(ET[TYPE]):.10f} vs E {Ev:.10f}')
# site densities
# shifted bond map b -> b + e_x (only for the literal-shift demonstration)
site = lambda x, y: (x % L) + L * (y % L)
key = {(int(min(a, b)), int(max(a, b))): k for k, (a, b) in enumerate(zip(BI, BJ))}
kb = lambda i, j: key[(min(i, j), max(i, j))]
sh = lambda i: site(i % L + 1, i // L)
SHIFT = np.array([kb(sh(a), sh(b)) for a, b in zip(BI, BJ)])
assert np.all(JB[SHIFT] == JB) and len(set(SHIFT)) == NB
# 2x2 plaquettes (Amendment 1): J1 bonds weight 1/2 (each in two plaquettes), J2 bonds weight 1 (one plaquette)
PLQ = np.zeros((NB, N)); INC = np.zeros((NB, N))
for yy in range(L):
    for xx in range(L):
        q = site(xx, yy); a_, b_, c_, d_ = site(xx, yy), site(xx + 1, yy), site(xx, yy + 1), site(xx + 1, yy + 1)
        for i, j in ((a_, b_), (a_, c_), (c_, d_), (b_, d_)): PLQ[kb(i, j), q] += 0.5
        for i, j in ((a_, d_), (c_, b_)): PLQ[kb(i, j), q] += 1.0
INC[np.arange(NB), BI] += 0.5; INC[np.arange(NB), BJ] += 0.5
assert np.allclose(PLQ.sum(1), 1) and np.allclose(INC.sum(1), 1)
PERM = np.random.default_rng(20261026).permutation(Dd)              # broken locality: densities of an unrelated configuration
EPSP = EPS[PERM]; ETP = etype(EPSP)
Hd = H.diagonal(); CN = np.abs(H @ v - Hd * v) / ((H @ np.abs(v)) - Hd * np.abs(v))   # N0 cancellation ratio of psi

def nrm(u): return u / np.linalg.norm(u)


def prod_state(F, a_cols):
    """v prod_cols [1 + a_col F_col], in log form; returns the normalised vector."""
    fac = 1.0 + a_cols[None, :] * F
    lg = np.sum(np.log(np.abs(fac)), 1); sg = np.prod(np.sign(fac), 1)
    lv = np.log(np.abs(v)) + lg
    return nrm(np.sign(v) * sg * np.exp(lv - lv.max()))


def LP(a1, a2, E_=None, et=None):
    E_ = EPS if E_ is None else E_; et = ET if et is None else et
    return prod_state(E_ - et[TYPE][None, :], np.where(TYPE == 0, a1, a2).astype(float))
def GRP(M, a, E_=None, et=None):
    E_ = EPS if E_ is None else E_; et = ET if et is None else et
    return prod_state((E_ - et[TYPE][None, :]) @ M, np.full(M.shape[1], float(a)))
def LPs(a): return GRP(INC, a)
def PPLQ(a): return GRP(PLQ, a)
def UNIF(a): return prod_state(np.repeat(((EL - Ev) / NB)[:, None], NB, 1), np.full(NB, float(a)))
def Gst(a): return nrm(v * (1 + a * (EL - Ev)))
def Evmc(u): return float(u @ (H @ u))


import scipy.optimize as so
GRID = np.concatenate([-np.logspace(-0.5, -3.5, 13), [0.0], np.logspace(-3.5, -0.5, 13)])   # signed log grid, |a| 3e-4 .. 0.32
def opt2(f, x0, tag):
    """grid scan (GRID x GRID) then Nelder-Mead from the grid minimum, initial simplex of the local grid size."""
    best = (np.inf, None)
    for a in GRID:
        for b in GRID:
            e = Evmc(f(a, b)); best = min(best, (e, (a, b)), key=lambda q: q[0])
    z0 = np.array(best[1]); h = np.maximum(1e-3, 0.3 * np.abs(z0))
    r = so.minimize(lambda z: Evmc(f(*z)), z0, method='Nelder-Mead',
                    options=dict(initial_simplex=np.array([z0, z0 + [h[0], 0], z0 + [0, h[1]]]), xatol=1e-8, fatol=1e-14, maxiter=4000))
    log(f'{tag}: grid min {best[0] / N:.10f} at {best[1]}; NM {r.fun / N:.12f} at {r.x} ({r.nfev} evals)')
    return r.x, float(r.fun)
def opt1(f, tag):
    es = [Evmc(f(a)) for a in GRID]; i = int(np.argmin(es)); lo, hi = GRID[max(i - 1, 0)], GRID[min(i + 1, GRID.size - 1)]
    r = so.minimize_scalar(lambda a: Evmc(f(a)), bounds=(lo, hi), method='bounded', options=dict(xatol=1e-10))
    log(f'{tag}: grid min {min(es) / N:.10f} at {GRID[i]:.3e}; refined {r.fun / N:.12f} at {r.x:.6e}')
    return float(r.x), float(r.fun)


def rec(name, u, **kw):
    efn, _ = fn_state(u); d = dict(EvmcN=Evmc(u) / N, EFN_N=efn / N, dEvmc=(Evmc(u) - E0) / N, dEFN=(efn - E0) / N,
                                    sign_flip_weight=float(np.sum(p[np.sign(u) != np.sign(v)])), **kw)
    res.setdefault('states', {})[name] = d; dump()
    log(f'{name:12s} E_VMC-E0 {d["dEvmc"]:.6e}  E_FN-E0 {d["dEFN"]:.6e}  /site  sign-flip weight(psi^2) {d["sign_flip_weight"]:.3e}  {kw}')
    return d


S = {}
S['psi'] = rec('psi', v)
g_, alG, _ = hop(v); uG = nrm(g_); S['G'] = rec('G', uG, alpha=alG)
xa, _ = opt2(LP, (alG, alG), 'LP'); uL = LP(*xa); S['LP'] = rec('LP', uL, alpha_J1=xa[0], alpha_J2=xa[1])
ae, _ = opt1(lambda a: LP(a, a), 'LP alpha_J1=alpha_J2'); S['LPeq'] = rec('LPeq', LP(ae, ae), alpha=ae)
asi, _ = opt1(LPs, 'LPs'); S['LPs'] = rec('LPs', LPs(asi), alpha=asi)
apl, _ = opt1(PPLQ, 'P_plaq'); S['Pplaq'] = rec('Pplaq', PPLQ(apl), alpha=apl)
# ---------------- crossed sign/amplitude diagnostics and the amplitude amplification
S['absG_sgnLP'] = rec('absG_sgnLP', np.abs(uG) * np.sign(uL)); S['absLP_sgnG'] = rec('absLP_sgnG', np.abs(uL) * np.sign(uG))
r = np.abs(uL) / np.abs(nrm(v)); im = int(np.argmax(r))
res['amplification'] = dict(max_ratio=float(r[im]), at_abs_psi_over_max=float(np.abs(v[im]) / np.abs(v).max()), at_c=float(CN[im]),
                            at_psi2=float(p[im]), LP_weight_ratio_gt10=float(np.sum(uL[r > 10] ** 2)), median_c=float(np.median(CN)),
                            corr_logr_logpsi=float(np.corrcoef(np.log(r), np.log(np.abs(v)))[0, 1]))
log('AMPLIFICATION |n(LP)/n(psi)|', json.dumps(res['amplification']))
# ---------------- controls
C = res['controls'] = {}
u0 = LP(0.0, 0.0); ub = LP(1e-6, 1e-6)
C['alpha0'] = dict(LP0_bitwise_equal_psi=bool(np.array_equal(u0, nrm(v))), LP0_EFN_minus_psi=(fn_state(u0)[0] - fn_state(v)[0]) / N,
                   LPs0_bitwise=bool(np.array_equal(LPs(0.0), nrm(v))), Pplaq0_bitwise=bool(np.array_equal(PPLQ(0.0), nrm(v))),
                   BROKEN_alpha1e6_bitwise_equal=bool(np.array_equal(ub, nrm(v))), BROKEN_max_abs_diff=float(np.max(np.abs(ub - nrm(v)))))
log('CONTROL alpha=0', json.dumps(C['alpha0']))
EPSs = EPS[:, SHIFT]; Ls = LP(*xa, E_=EPSs, et=etype(EPSs))
C['shift_literal_max_abs_diff_vs_LP'] = float(np.max(np.abs(Ls - uL)))
log(f'literal shift (permutes factors): max |LP_shift - LP| {C["shift_literal_max_abs_diff_vs_LP"]:.2e} (cannot fail; not a control)')
fp = lambda a, b: LP(a, b, E_=EPSP, et=ETP)
xp, _ = opt2(fp, (alG, alG), 'LP-perm (BROKEN: densities of an unrelated configuration)'); S['LPperm'] = rec('LPperm_BROKEN', fp(*xp), alpha_J1=xp[0], alpha_J2=xp[1])
au, _ = opt1(UNIF, 'LP-unif (no locality)'); S['LPunif'] = rec('LPunif', UNIF(au), alpha=au)
gG = S['psi']['dEFN'] - S['G']['dEFN']
C['locality'] = {k: dict(gain=S['psi']['dEFN'] - S[k]['dEFN'], gain_over_G=(S['psi']['dEFN'] - S[k]['dEFN']) / gG) for k in ('LP', 'LPeq', 'LPs', 'Pplaq', 'LPperm', 'LPunif')}
log('CONTROL locality (FN gain / G gain)', json.dumps({k: round(q['gain_over_G'], 4) for k, q in C['locality'].items()}))
fo = {}
for a in (1e-3, 5e-4):
    fo[str(a)] = dict(dist=float(np.linalg.norm(LP(a, a) - Gst(a))), BROKEN_noJ2_dist=float(np.linalg.norm(LP(a, 0.0) - Gst(a))))
r1 = fo['0.001']['dist'] / fo['0.0005']['dist']; r2 = fo['0.001']['BROKEN_noJ2_dist'] / fo['0.0005']['BROKEN_noJ2_dist']
C['first_order'] = dict(points=fo, ratio_halving=r1, BROKEN_ratio_halving=r2)
log(f'CONTROL first order: |n(LP(a,a)) - n(G(a))| at a=1e-3, 5e-4: {fo["0.001"]["dist"]:.3e}, {fo["0.0005"]["dist"]:.3e} ratio {r1:.3f} (O(a^2): 4); '
    f'BROKEN (J2 factors dropped) {fo["0.001"]["BROKEN_noJ2_dist"]:.3e}, {fo["0.0005"]["BROKEN_noJ2_dist"]:.3e} ratio {r2:.3f} (O(a): 2)')
# ---------------- verdict (PREREG rules, per guide)
best = min(S[k]['dEFN'] for k in ('LP', 'LPs'))   # PREREG: LP or LPs (LPeq, Pplaq reported, not in the rule); q = best / S['G']['dEFN']
verd = 'PASS' if q <= 1.0 else ('KILL' if q > 1.5 else 'MARGINAL')
res['verdict'] = dict(verdict=verd, ratio_bestLP_over_G=q, gain_ratio_LP_over_G=(S['psi']['dEFN'] - best) / gG); res['sec'] = time.time() - t00; dump()
log(f'VERDICT 4x4 {MASK}: {verd}  E_FN-E0 /site: psi {S["psi"]["dEFN"]:.4e} G {S["G"]["dEFN"]:.4e} LP {S["LP"]["dEFN"]:.4e} LPeq {S["LPeq"]["dEFN"]:.4e} LPs {S["LPs"]["dEFN"]:.4e} Pplaq {S["Pplaq"]["dEFN"]:.4e}  '
    f'(best LP)/G {q:.3f}; FN-gain ratio {res["verdict"]["gain_ratio_LP_over_G"]:.3f}; {res["sec"]:.0f}s')
