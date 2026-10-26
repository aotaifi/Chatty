"""Local-product Krylov (LP), Stage A at 6x6 (PREREG.md in this directory; rules fixed there).   python lp6x6.py OUTDIR
Exact on the 15.8M-orbit symmetric sector (k = 0, A1, flip even; st6_sector.py), one GPU.  Guide psi = RBM+PP A: the exact table
sym_tables_A.npz of rpTabA_17168764 (lP, sP; sha256 checked), the guide of results/rbmpp_6x6/runs/table_A.json (no forward pass).
Definitions and Amendment-1 arms (LPeq, P_plaq, LPperm broken control, crossed |G|sgnLP / |LP|sgnG, amplification) as lp4x4.py:  eps_b(x) = (H_b psi)(x)/psi(x), G = Ritz hop, LP = psi prod_b [1 + a_t (eps_b - e_t)] (t = J1/J2),
LPs = psi prod_i [1 + a (eps_i - e_s)].  LP keeps the sector symmetry (one alpha per bond type; bond sets map onto themselves).
Neighbour table: for every rep x and bond b, the rep of x^b (st6_sector.canon); off-diagonal part OFF_xb = J_b/2 psi(x^b)/psi(x)
stored float32 on the device (computed in float64); diagonal 0.25 J_b s_i s_j exact.  Check: sum_b eps_b = (H psi)/psi from the
sector matvec.  Controls and broken versions as lp4x4.py; FN = sec.fn_solve (the convention of table_A.json / cx_gtax.py).
Diagnostic: near-node configurations by the N0 cancellation ratio c(x) = |sum_y H_xy psi(y)| / sum_y |H_xy psi(y)| (compress_6x6
README 'Test N0'), c < 0.1 and c < 0.05: fraction flipped (sign != psi's) by G and by LP, overlap of the two flip sets.
"""
import os, sys, json, time, hashlib
import numpy as np
HERE = '/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/experiments'
sys.path.insert(0, HERE + '/stall_6x6')
import st6_sector as SS
import jax, jax.numpy as jnp
import scipy.optimize as so
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); T00 = time.time()
ROOT = '/project/theorie/a/A.Otaifi/Chatty/runs/stall6'
SYMA = '/project/theorie/a/A.Otaifi/Chatty/runs/rbmpp6/runs/rpTabA_17168764/sym_tables_A.npz'; SHA_A = 'f98ed0fc6981fd82e30e65aa367c3236de750e4076a4d17ae82c2f898ee69801'
TAJ = '/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/results/rbmpp_6x6/runs/table_A.json'
RES = os.path.join(OUT, 'lp6x6.json'); res = dict(table=SYMA)
N = 36; NB = int(SS.NBOND); f64 = jnp.float64


def log(*a):
    s = time.strftime('%Y-%m-%dT%H:%M:%S') + ' PROGRESS ' + ' '.join(str(x) for x in a)
    print(s, flush=True); open(os.path.join(OUT, 'progress.txt'), 'a').write(s + '\n')


def dump(): open(RES, 'w').write(json.dumps(res, indent=1, default=float))


h = hashlib.sha256(open(SYMA, 'rb').read()).hexdigest(); res['table_sha256_ok'] = h == SHA_A
if h != SHA_A: log('TABLE SHA MISMATCH; exit 3'); sys.exit(3)
sec = SS.Sector(ROOT + '/csr', ROOT + '/data/psi0_6x6_table.npz'); sec.v0 = sec.la0 = sec.s0 = sec.p0 = None
D = sec.D; E0 = sec.E0
z = np.load(SYMA); la = jnp.asarray(z['lP'], f64); la = la - jnp.max(la); s = jnp.asarray(np.where(z['sP'] >= 0, 1.0, -1.0)); del z
TA = json.load(open(TAJ))
def dsite(E): return (E - E0) / N
v = sec.vec(la, s); Hv = sec.H(v); Ev = float(v @ Hv)
res['psi_check'] = dict(dE_site=dsite(Ev), ref=TA['E_re'], rel=abs(dsite(Ev) - TA['E_re']) / TA['E_re'])
log(f'psi: E-E0 /site {dsite(Ev):.10e} (table_A.json {TA["E_re"]:.10e}); D {D}; jax {jax.__version__} {jax.devices()}')
if res['psi_check']['rel'] > 1e-8: log('GUIDE MISMATCH; exit 3'); sys.exit(3)

# ------------------------------------------------------------------ neighbour table -> OFF (D, NB) float32, VALID (D, NB) bool
BI = jnp.asarray(SS.BI); BJ = jnp.asarray(SS.BJ); JB = jnp.asarray(SS.JB); MASKS = jnp.asarray(SS.MASKS)
TYPE = np.asarray(SS.JB < 0.75).astype(int); TYPEj = jnp.asarray(TYPE)
SUB = 4096; RCH = 1 << 19


@jax.jit
def nb_block(x, lx, sx, T, reps, la, s):
    valid = (((x[:, None] >> BI[None, :]) ^ (x[:, None] >> BJ[None, :])) & jnp.uint64(1)).astype(bool)
    y = jnp.where(valid, x[:, None] ^ MASKS[None, :], x[:, None])
    iy = SS.canon(T, reps, y.reshape(-1)).reshape(x.shape[0], NB)
    off = jnp.where(valid, 0.5 * JB[None, :] * s[iy] * sx[:, None] * jnp.exp(jnp.clip(la[iy] - lx[:, None], -80, 80)), 0.0)
    return off.astype(jnp.float32), valid


t0 = time.time(); OFF = []; VAL = []
for r0 in range(0, D, RCH):
    r1 = min(D, r0 + RCH); o_ = []; v_ = []
    for i in range(r0, r1, SUB):
        j = min(r1, i + SUB); x = sec.reps[i:j]
        if j - i < SUB: x = jnp.concatenate([x, jnp.repeat(x[:1], SUB - (j - i))])
        idx = jnp.arange(i, i + SUB) if j - i == SUB else jnp.concatenate([jnp.arange(i, j), jnp.full(SUB - (j - i), i)])
        o, vl = nb_block(x, la[idx], s[idx], sec.T, sec.reps, la, s); o_.append(o[:j - i]); v_.append(vl[:j - i])
    OFF.append(jnp.concatenate(o_)); VAL.append(jnp.concatenate(v_)); del o_, v_
    if r0 == 0 or (r0 // RCH) % 5 == 0: jax.block_until_ready(OFF[-1]); log(f'  neighbour table rows {r1}/{D}  {time.time() - t0:.0f}s')
jax.block_until_ready(OFF[-1]); res['sec_table'] = time.time() - t0; log(f'neighbour table done {res["sec_table"]:.0f}s')
CH = [(r0, min(D, r0 + RCH)) for r0 in range(0, D, RCH)]
DG_of = lambda vl: 0.25 * JB[None, :] * jnp.where(vl, -1.0, 1.0)
v_np = None
EL = Hv / v                                                              # sector form: (H v)_r / v_r = E_L(rep r)
p = v * v                                                                # = n psi^2, normalised
# checks: sum_b eps_b vs E_L; e_t
mx = 0.0; wrms = 0.0; et = np.zeros(2)
for k, (r0, r1) in enumerate(CH):
    E_ = DG_of(VAL[k]) + OFF[k].astype(f64); d = jnp.sum(E_, 1) - EL[r0:r1]
    mx = max(mx, float(jnp.max(jnp.abs(d) / jnp.maximum(1.0, jnp.abs(EL[r0:r1])))))
    wrms += float(jnp.sum(p[r0:r1] * d * d)); pe = p[r0:r1] @ E_
    et += np.array([float(jnp.sum(jnp.where(TYPEj == t, pe, 0.0))) for t in (0, 1)])
ET = et / np.array([np.sum(TYPE == 0), np.sum(TYPE == 1)]); ETj = jnp.asarray(ET)
res['check_sum_eps_EL'] = dict(max_rel=mx, psi2_weighted_rms=float(np.sqrt(wrms)), e_J1=ET[0], e_J2=ET[1], sum_e=float(np.sum(ET[TYPE])), E=Ev); dump()
log(f'sum_b eps_b vs E_L: max rel {mx:.2e}, psi^2-weighted rms {np.sqrt(wrms):.2e} (OFF float32); e_J1 {ET[0]:.8f} e_J2 {ET[1]:.8f}; '
    f'sum e = {np.sum(ET[TYPE]):.10f} vs E {Ev:.10f}')
# site incidence (eps_i = 1/2 sum_{b ni i} eps_b) and the shifted-bond map b -> b + e_x
INC = np.zeros((NB, N)); INC[np.arange(NB), SS.BI.astype(int)] += 0.5; INC[np.arange(NB), SS.BJ.astype(int)] += 0.5; INCj = jnp.asarray(INC)
site = lambda xx, yy: (xx % 6) + 6 * (yy % 6); sh = lambda i: site(i % 6 + 1, i // 6)
key = {(min(int(a), int(b)), max(int(a), int(b))): k for k, (a, b) in enumerate(zip(SS.BI, SS.BJ))}
SHIFT = np.array([key[(min(sh(int(a)), sh(int(b))), max(sh(int(a)), sh(int(b))))] for a, b in zip(SS.BI, SS.BJ)])
assert np.all(SS.JB[SHIFT] == SS.JB) and len(set(SHIFT)) == NB; SHIFTj = jnp.asarray(SHIFT)
kb = lambda i, j: key[(min(i, j), max(i, j))]
PLQ = np.zeros((NB, N))
for yy in range(6):
    for xx in range(6):
        q_ = site(xx, yy); a_, b_, c_, d_ = site(xx, yy), site(xx + 1, yy), site(xx, yy + 1), site(xx + 1, yy + 1)
        for i, j in ((a_, b_), (a_, c_), (c_, d_), (b_, d_)): PLQ[kb(i, j), q_] += 0.5
        for i, j in ((a_, d_), (c_, b_)): PLQ[kb(i, j), q_] += 1.0
assert np.allclose(PLQ.sum(1), 1) and np.allclose(INC.sum(1), 1); PLQj = jnp.asarray(PLQ)
_rng = np.random.default_rng(20261026)
PERMS = [jnp.asarray(_rng.permutation(r1 - r0).astype(np.int32)) for (r0, r1) in CH]   # broken: densities of an unrelated rep (same chunk)


@jax.jit
def _fac(lgsg, F, a):
    fac = 1.0 + a[None, :] * F
    return lgsg[0] + jnp.sum(jnp.log(jnp.abs(fac)), 1), lgsg[1] * jnp.prod(jnp.sign(fac), 1)


def build(kind, a, perm=False):
    """(la', s') per rep: kind 'LP' (a = (a_J1, a_J2)) or a group matrix M (NB, groups) with one alpha (LPs: INC, P_plaq: PLQ)."""
    out_l = []; out_s = []
    for k, (r0, r1) in enumerate(CH):
        if perm: E_ = DG_of(VAL[k][PERMS[k]]) + OFF[k][PERMS[k]].astype(f64)
        else: E_ = DG_of(VAL[k]) + OFF[k].astype(f64)
        F = E_ - (ETP if perm else ETj)[TYPEj][None, :]
        if isinstance(kind, str):
            ab = jnp.where(TYPEj == 0, a[0], a[1]).astype(f64)
        else:
            F = F @ kind; ab = jnp.full(kind.shape[1], float(a), f64)
        l_, s_ = _fac((la[r0:r1], s[r0:r1]), F, ab); out_l.append(l_); out_s.append(s_)
    return jnp.concatenate(out_l), jnp.concatenate(out_s)


def unif(a):
    fac = 1.0 + a * (EL - Ev) / NB; return la + NB * jnp.log(jnp.abs(fac)), s * jnp.sign(fac) ** NB


def energy(ls): return sec.energy(sec.vec(*ls))


# permuted (broken) e_t
et = np.zeros(2)
for k, (r0, r1) in enumerate(CH):
    E_ = DG_of(VAL[k][PERMS[k]]) + OFF[k][PERMS[k]].astype(f64); pe = p[r0:r1] @ E_
    et += np.array([float(jnp.sum(jnp.where(TYPEj == t, pe, 0.0))) for t in (0, 1)])
ETP = jnp.asarray(et / np.array([np.sum(TYPE == 0), np.sum(TYPE == 1)]))
NEV = [0]
GRID = np.concatenate([-np.logspace(-0.5, -3.5, 7), [0.0], np.logspace(-3.5, -0.5, 7)])
if os.environ.get('LP_GRID'): GRID = np.array(json.loads(os.environ['LP_GRID']))


def opt2(f, tag):
    t0 = time.time(); best = (np.inf, None)
    for a in GRID:
        for b in GRID:
            e = energy(f(a, b)); best = min(best, (e, (a, b)), key=lambda q: q[0])
    log(f'{tag}: grid ({GRID.size}^2) min {dsite(best[0]):.6e} at {best[1]}  {time.time() - t0:.0f}s')
    z0 = np.array(best[1]); hh = np.maximum(1e-3, 0.3 * np.abs(z0))
    r = so.minimize(lambda zz: energy(f(*zz)), z0, method='Nelder-Mead',
                    options=dict(initial_simplex=np.array([z0, z0 + [hh[0], 0], z0 + [0, hh[1]]]), xatol=1e-6, fatol=1e-12, maxiter=300))
    log(f'{tag}: NM {dsite(r.fun):.8e} at {r.x} ({r.nfev} evals)  {time.time() - t0:.0f}s')
    return r.x


def opt1(f, tag):
    t0 = time.time(); es = [energy(f(a)) for a in GRID]; i = int(np.argmin(es)); lo, hi = GRID[max(i - 1, 0)], GRID[min(i + 1, GRID.size - 1)]
    r = so.minimize_scalar(lambda a: energy(f(a)), bounds=(lo, hi), method='bounded', options=dict(xatol=1e-8))
    log(f'{tag}: grid min {dsite(min(es)):.6e} at {GRID[i]:.3e}; refined {dsite(r.fun):.8e} at {r.x:.6e}  {time.time() - t0:.0f}s')
    return float(r.x)


S = {}; SG = {}


def rec(name, ls, **kw):
    E = energy(ls); efn, _, info = sec.fn_solve(ls[0], ls[1], maxit=400)
    fl = ls[1] != s
    d = dict(dEvmc=dsite(E), dEFN=dsite(efn), sign_flip_weight=float(jnp.sum(jnp.where(fl, p, 0.0))), fn_it=info['lobpcg_it'], fn_res=info['res'],
             fn_sec=info['sec'], **kw)
    S[name] = d; SG[name] = ls[1]; res['states'] = S; dump()
    log(f'{name:14s} E_VMC-E0 {d["dEvmc"]:.6e}  E_FN-E0 {d["dEFN"]:.6e} /site  flip weight(psi^2) {d["sign_flip_weight"]:.3e}  '
        f'(lobpcg {info["lobpcg_it"]}, {info["sec"]:.0f}s) {kw}')
    return d


def sector_ls(w): return jnp.log(jnp.maximum(jnp.abs(w), 1e-300) / sec.sqrt_n), jnp.where(w >= 0, 1.0, -1.0)


rec('psi', (la, s))
u1 = Hv - Ev * v; Hu1 = sec.H(u1); u2 = Hu1 - Ev * u1; mu2 = float(u1 @ u1); mu3 = float(u1 @ u2)
alG = (mu3 - np.sqrt(mu3 ** 2 + 4 * mu2 ** 3)) / (2 * mu2 ** 2); del Hu1, u2
rec('G', sector_ls(v + alG * u1), alpha=alG)
res['G_vs_table_A'] = dict(dEvmc=S['G']['dEvmc'], ref_var=TA['lanczos'][1]['dE_site'], dEFN=S['G']['dEFN'], ref_FN=TA['lanczos'][1]['dE_FN_site'],
                           diff_FN=S['G']['dEFN'] - TA['lanczos'][1]['dE_FN_site']); dump()
log(f'G vs table_A.json: E_FN {S["G"]["dEFN"]:.8e} vs {TA["lanczos"][1]["dE_FN_site"]:.8e} (diff {res["G_vs_table_A"]["diff_FN"]:.2e}); '
    f'E_var {S["G"]["dEvmc"]:.8e} vs {TA["lanczos"][1]["dE_site"]:.8e}')
xa = opt2(lambda a, b: build('LP', (a, b)), 'LP'); rec('LP', build('LP', xa), alpha_J1=float(xa[0]), alpha_J2=float(xa[1]))
ae = opt1(lambda a: build('LP', (a, a)), 'LP alpha_J1=alpha_J2'); rec('LPeq', build('LP', (ae, ae)), alpha=ae)
asi = opt1(lambda a: build(INCj, a), 'LPs'); rec('LPs', build(INCj, asi), alpha=asi)
apl = opt1(lambda a: build(PLQj, a), 'P_plaq'); rec('Pplaq', build(PLQj, apl), alpha=apl)
# crossed sign/amplitude diagnostics
lsG = sector_ls(v + alG * u1); lsL = build('LP', xa)
rec('absG_sgnLP', (lsG[0], lsL[1])); rec('absLP_sgnG', (lsL[0], lsG[1]))
# ------------------------------------------------------------------ controls
C = res['controls'] = {}
l0, s0_ = build('LP', (0.0, 0.0)); lb, sb = build('LP', (1e-6, 1e-6))
C['alpha0'] = dict(la_bitwise=bool(jnp.array_equal(l0, la)), Pplaq0_la_bitwise=bool(jnp.array_equal(build(PLQj, 0.0)[0], la)), s_bitwise=bool(jnp.array_equal(s0_, s)),
                   BROKEN_alpha1e6_la_bitwise=bool(jnp.array_equal(lb, la)), BROKEN_max_abs_dla=float(jnp.max(jnp.abs(lb - la))))
rec('LP_alpha0', (l0, s0_)); C['alpha0']['EFN_minus_psi'] = S['LP_alpha0']['dEFN'] - S['psi']['dEFN']
rec('LP_alpha1e-6_BROKEN', (lb, sb)); C['alpha0']['BROKEN_EFN_minus_psi'] = S['LP_alpha1e-6_BROKEN']['dEFN'] - S['psi']['dEFN']
del l0, s0_, lb, sb
log('CONTROL alpha=0', json.dumps(C['alpha0']))
fp = lambda a, b: build('LP', (a, b), perm=True)
xp = opt2(fp, 'LP-perm (BROKEN: densities of an unrelated configuration)'); rec('LPperm_BROKEN', fp(*xp), alpha_J1=float(xp[0]), alpha_J2=float(xp[1]))
au = opt1(unif, 'LP-unif (no locality)'); rec('LPunif', unif(au), alpha=au)
gG = S['psi']['dEFN'] - S['G']['dEFN']
C['locality'] = {k: dict(gain=S['psi']['dEFN'] - S[k]['dEFN'], gain_over_G=(S['psi']['dEFN'] - S[k]['dEFN']) / gG) for k in ('LP', 'LPeq', 'LPs', 'Pplaq', 'LPperm_BROKEN', 'LPunif')}
log('CONTROL locality (FN gain / G gain)', json.dumps({k: round(q['gain_over_G'], 4) for k, q in C['locality'].items()}))
fo = {}
for a in (1e-3, 5e-4):
    g_ = v + a * u1; g_ = g_ / jnp.linalg.norm(g_)
    fo[str(a)] = dict(dist=float(jnp.linalg.norm(sec.vec(*build('LP', (a, a))) - g_)), BROKEN_noJ2_dist=float(jnp.linalg.norm(sec.vec(*build('LP', (a, 0.0))) - g_)))
r1 = fo['0.001']['dist'] / fo['0.0005']['dist']; r2 = fo['0.001']['BROKEN_noJ2_dist'] / fo['0.0005']['BROKEN_noJ2_dist']
C['first_order'] = dict(points=fo, ratio_halving=r1, BROKEN_ratio_halving=r2); dump()
log(f'CONTROL first order: |n(LP(a,a)) - n(G(a))| a=1e-3, 5e-4: {fo["0.001"]["dist"]:.3e}, {fo["0.0005"]["dist"]:.3e} ratio {r1:.3f} (O(a^2): 4); '
    f'BROKEN (no J2 factors) {fo["0.001"]["BROKEN_noJ2_dist"]:.3e}, {fo["0.0005"]["BROKEN_noJ2_dist"]:.3e} ratio {r2:.3f} (O(a): 2)')
# ------------------------------------------------------------------ N0 near-node diagnostic
D0 = jnp.concatenate([0.25 * jnp.sum(JB) - 0.5 * jnp.sum(jnp.where(VAL[k], JB[None, :], 0.0), 1) for k in range(len(CH))])
av = jnp.abs(v); den = sec.H(av) - D0 * av; c = jnp.abs(Hv - D0 * v) / jnp.maximum(den, 1e-300)
nn_ = sec.n; diag = {}
fG = SG['G'] != s; fL = SG['LP'] != s
for thr in (0.1, 0.05):
    m = c < thr; dd = {}
    for wname, w in (('n', nn_), ('psi2', p)):
        W = lambda q: float(jnp.sum(jnp.where(q, w, 0.0)))
        tot = W(m)
        dd[wname] = dict(near_node_share=tot / float(jnp.sum(w)), flipped_by_G=W(m & fG) / tot, flipped_by_LP=W(m & fL) / tot,
                         both=W(m & fG & fL) / tot, jaccard=W(m & fG & fL) / max(W(m & (fG | fL)), 1e-300),
                         share_of_G_flips_near_node=W(m & fG) / max(W(fG), 1e-300), share_of_LP_flips_near_node=W(m & fL) / max(W(fL), 1e-300))
    diag[str(thr)] = dd
res['N0_diag'] = diag
vL = sec.vec(*lsL); r = jnp.abs(vL) / jnp.maximum(jnp.abs(v), 1e-300); im = int(jnp.argmax(r))
res['amplification'] = dict(max_ratio=float(r[im]), at_abs_psi_over_max=float(jnp.exp(la[im] - jnp.max(la))), at_c=float(c[im]), at_psi2=float(p[im]),
                            LP_weight_ratio_gt10=float(jnp.sum(jnp.where(r > 10, vL * vL, 0.0))), median_c=float(jnp.median(c)),
                            LP_weight_c_lt_0p1=float(jnp.sum(jnp.where(c < 0.1, vL * vL, 0.0))), psi_weight_c_lt_0p1=float(jnp.sum(jnp.where(c < 0.1, p, 0.0))))
log('AMPLIFICATION |n(LP)/n(psi)|', json.dumps(res['amplification'])); dump()
for thr, dd in diag.items():
    q = dd['psi2']; qn = dd['n']
    log(f'N0 c<{thr}: near-node share (psi^2) {q["near_node_share"]:.3e}; flipped by G {q["flipped_by_G"]:.3e} / by LP {q["flipped_by_LP"]:.3e}, '
        f'both {q["both"]:.3e}, Jaccard {q["jaccard"]:.3f}  [orbit-count weights: G {qn["flipped_by_G"]:.3e} LP {qn["flipped_by_LP"]:.3e} Jaccard {qn["jaccard"]:.3f}]')
# ------------------------------------------------------------------ verdict (PREREG)
best = min(S['LP']['dEFN'], S['LPs']['dEFN'])   # PREREG: LP or LPs (LPeq, Pplaq reported); q = best / S['G']['dEFN']
verd = 'PASS' if q <= 1.0 else ('KILL' if q > 1.5 else 'MARGINAL')
res['verdict'] = dict(verdict=verd, ratio_bestLP_over_G=q, gain_ratio_LP_over_G=(S['psi']['dEFN'] - best) / gG); res['sec'] = time.time() - T00; dump()
log(f'VERDICT 6x6 RBM+PP A: {verd}  E_FN-E0 /site: psi {S["psi"]["dEFN"]:.4e} G {S["G"]["dEFN"]:.4e} LP {S["LP"]["dEFN"]:.4e} LPeq {S["LPeq"]["dEFN"]:.4e} LPs {S["LPs"]["dEFN"]:.4e} Pplaq {S["Pplaq"]["dEFN"]:.4e}  '
    f'(best LP)/G {q:.3f}; FN-gain ratio {res["verdict"]["gain_ratio_LP_over_G"]:.3f}; {res["sec"]:.0f}s')
