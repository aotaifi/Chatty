"""gfmc_b2_test.py: tests of the branch-2 grading providers (gfmc_b2.py) against M1's own provider (m1_code/gfmc_fn_lib.RBMPPGuideG, rb8_prod, consts.json).  CPU.
usage: python gfmc_b2_test.py OUTDIR {bare|p1|timing} [N]
Configurations: N (default 200) REAL beta = 1 draws of rb8_prod (pool_x/prod out_14..out_17, rows of the beta-1 rung, every 50th): walker-like configurations.
The guide ratio rr_b = G(y_b)/G(x) of every anti-aligned bond is captured at GuideBase.finish (both providers, same (walker, bond) order), plus rows' rate and d_FN.
(a) bare: BareM1Sign(B = rb8_prod).  're' convention (Re[w B], = M1's p = 0 guide): rr equal to RBMPPGuideG(alpha = 0) to <= 1e-10 relative, 0 sign differences.
    'abs' convention (sgn(M1)|B|, the branch-2 guide): 0 sign differences; |rr_abs / rr_M1| - 1 MEASURED (= |cos phi_x| / |cos phi_y| - 1, phi = arg(w B)), and the max
    differences of rate and d_FN.  Broken: 'nosign' (s(y) := s(x)) must show sign differences.
(b) p1: P1Guide(B = rb8_prod, M1's alpha, E, w) vs RBMPPGuideG(alpha, E, w): rr to <= 1e-10 relative, 0 sign differences, rate and d_FN.  Broken: alpha = 0.
(c) timing: rows() seconds per walker at M = 64 (second call; first compiles) for BareM1Sign, P1Guide and M1's two providers; a short GFMC run of BareM1Sign (M 64)."""
import os, sys, json, time, hashlib
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gfmc_b2 as G2                                   # imports gfmc_aug -> M1 walker code first on sys.path
import rbmpp_lib as R
from gfmc_fn_lib import RBMPPGuideG, GuideBase, GFMC
import jax, jax.numpy as jnp
import pp2_lemma as PL, b2_prep as B2

OUT = sys.argv[1]; WHAT = sys.argv[2]; NC = int(sys.argv[3]) if len(sys.argv) > 3 else 200; os.makedirs(OUT, exist_ok=True); T0 = time.time()
PX = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_8x8_r1/pool_x/prod'
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
assert os.path.basename(os.path.dirname(R.__file__)) == 'm1_code', R.__file__


def log(m):
    s = f'[{time.time() - T0:7.0f}s] {m}'; print(s, flush=True); open(f'{OUT}/test.log', 'a').write(s + '\n')


def configs(n):
    W = []
    for f in (14, 15, 16, 17):
        z = np.load(f'{PX}/out_{f}.npz'); W.append(z['words'][z['beta'] == 1.0][::50])
    W = np.concatenate(W)[:n]; assert len(W) == n; return np.asarray(B2.PF.unpack64(W)).astype(np.int8)


def m1_guide(alpha):
    c = json.load(open(B2.M1CONSTS)); al, E, w = c['alpha'], c['E'], B2.load_w()
    lat = R.Lattice(8, 0.5, 'translations'); z = np.load(B2.PROD8)
    return RBMPPGuideG({k: z[k] for k in ('f', 'bre', 'bim', 'Wre', 'Wim')}, lat, alpha if alpha is not None else al, E, w=w, chunk=256)


def capture(g):
    rec = {}
    def fin(X, anti, wi, bi, rr):
        rec['wi'], rec['bi'], rec['rr'] = wi.copy(), bi.copy(), np.asarray(rr, float).copy(); return GuideBase.finish(g, X, anti, wi, bi, rr)
    g.finish = fin; return rec


def rows_rr(g, X, chunk=64):
    """rows of X in chunks (as GFMC.refresh would see), with rr captured; returns rate, dFN, rr (concatenated in (walker, bond) order)."""
    rec = capture(g); RT, D, RR = [], [], []
    for i in range(0, len(X), chunk):
        r, d = g.rows(X[i:i + chunk]); RT.append(r); D.append(d); RR.append(rec['rr'])
    return np.concatenate(RT), np.concatenate(D), np.concatenate(RR)


def compare(a, b, tag):
    ra, da, rra = a; rb, db, rrb = b
    rel = np.abs(rra / rrb - 1.0)
    return dict(n_bonds=int(rra.size), sign_diff=int(np.sum(np.sign(rra) != np.sign(rrb))), max_rel_rr=float(rel.max()), median_rel_rr=float(np.median(rel)),
                max_abs_rate=float(np.max(np.abs(ra - rb))), max_abs_dFN=float(np.max(np.abs(da - db))), max_abs_EL=float(np.max(np.abs((da - ra.sum(1)) - (db - rb.sum(1))))))


def t_bare():
    X = configs(NC); p = B2.prod_params(); w = B2.load_w(); res = dict(n_configs=NC)
    m1 = rows_rr(m1_guide(0.0), X); log('M1 p=0 rows done')
    gre = G2.BareM1Sign(p, w, convention='re'); res['re_vs_M1p0'] = compare(rows_rr(gre, X), m1, 're')
    gab = G2.BareM1Sign(p, w, convention='abs'); ab = rows_rr(gab, X); res['abs_vs_M1p0'] = compare(ab, m1, 'abs')
    # the documented convention: rr_abs / rr_re = |cos phi_x| / |cos phi_y|  (phi = arg(w B)); check it holds and size it
    lB, r = PL.one_swap_ratios(p, X.astype(np.float64), return_logB=True); lB = np.asarray(lB); r = np.asarray(r)
    anti = X[:, gab.BI] != X[:, gab.BJ]; wi, bi = np.nonzero(anti); phx = np.angle(w * np.exp(1j * np.imag(lB))); phy = phx[wi] + np.angle(r[wi, bi])
    pred = np.abs(np.cos(phx[wi])) / np.abs(np.cos(phy)); res['convention'] = dict(
        statement='sgn(M1)|B| vs Re[w B]: rr_abs / rr_re = |cos phi(x)| / |cos phi(y)|, phi = arg(w B); identical signs',
        max_abs_ratio_minus_1=float(np.max(np.abs(ab[2] / m1[2] - 1))), max_abs_pred_minus_1=float(np.max(np.abs(pred - 1))),
        max_dev_from_prediction=float(np.max(np.abs(ab[2] / m1[2] - pred))), max_abs_sin_phi_x=float(np.max(np.abs(np.sin(phx)))), max_abs_sin_phi_y=float(np.max(np.abs(np.sin(phy)))))
    gbr = G2.BareM1Sign(p, w, convention='abs', broken='nosign'); res['broken_nosign_vs_M1p0'] = compare(rows_rr(gbr, X), m1, 'broken')
    # separate sign network (rounds k >= 1): a copy of rb8_prod as B, the sign from the original object -> identical rows to the same-object path
    gsep = G2.BareM1Sign(B2.prod_params(), w, sign_params=p, convention='abs'); res['abs_separate_sign_net_vs_abs'] = compare(rows_rr(gsep, X), ab, 'sep')
    res['PASS'] = bool(res['re_vs_M1p0']['max_rel_rr'] <= 1e-10 and res['re_vs_M1p0']['sign_diff'] == 0 and res['abs_vs_M1p0']['sign_diff'] == 0
                       and res['convention']['max_dev_from_prediction'] <= 1e-10 and res['abs_separate_sign_net_vs_abs']['max_rel_rr'] <= 1e-12)
    res['BROKEN_FAILS'] = bool(res['broken_nosign_vs_M1p0']['sign_diff'] > 0); res['min_rel_margin'] = gab.min_margin
    res['code_sha256'] = dict(gfmc_b2=sha(os.path.join(HERE, 'gfmc_b2.py')), gfmc_fn_lib=sha(sys.modules['gfmc_fn_lib'].__file__), m1_rbmpp_lib=sha(R.__file__), pp2_lemma=sha(PL.__file__))
    json.dump(res, open(f'{OUT}/t3a_bare.json', 'w'), indent=1, default=float); log(json.dumps(res, default=float))


def t_p1():
    X = configs(NC); p = B2.prod_params(); w = B2.load_w(); c = json.load(open(B2.M1CONSTS)); res = dict(n_configs=NC, alpha=c['alpha'], E=c['E'], w=[w.real, w.imag])
    t = time.time(); m1 = rows_rr(m1_guide(None), X); res['sec_m1'] = time.time() - t; log(f'M1 p=1 rows done {res["sec_m1"]:.0f}s')
    g = G2.P1Guide(p, c['alpha'], c['E'], w); t = time.time(); mine = rows_rr(g, X); res['sec_p1'] = time.time() - t; res['p1_vs_M1p1'] = compare(mine, m1, 'p1')
    res['max_im_M1'] = None; res['max_im_p1'] = g.max_im
    nbk = 20; nb_ = int((X[:nbk, g.BI] != X[:nbk, g.BJ]).sum())             # broken version on the first 20 configurations (enough to fail; saves CPU)
    gb = G2.P1Guide(p, c['alpha'], c['E'], w, broken='alpha_zero'); res['broken_alpha0_vs_M1p1'] = compare(rows_rr(gb, X[:nbk]), (m1[0][:nbk], m1[1][:nbk], m1[2][:nb_]), 'broken')
    res['broken_n_configs'] = nbk
    res['PASS'] = bool(res['p1_vs_M1p1']['max_rel_rr'] <= 1e-10 and res['p1_vs_M1p1']['sign_diff'] == 0)
    res['BROKEN_FAILS'] = bool(res['broken_alpha0_vs_M1p1']['max_rel_rr'] > 1e-6)
    res['code_sha256'] = dict(gfmc_b2=sha(os.path.join(HERE, 'gfmc_b2.py')), gfmc_fn_lib=sha(sys.modules['gfmc_fn_lib'].__file__), m1_rbmpp_lib=sha(R.__file__), pp2_lemma=sha(PL.__file__))
    json.dump(res, open(f'{OUT}/t3b_p1.json', 'w'), indent=1, default=float); log(json.dumps(res, default=float))


def t_timing():
    M = 64; X = configs(2 * M); p = B2.prod_params(); w = B2.load_w(); c = json.load(open(B2.M1CONSTS)); res = dict(M=M, host=os.uname().nodename, threads=os.environ.get('OMP_NUM_THREADS'))
    for nm, mk in (('BareM1Sign_abs', lambda: G2.BareM1Sign(p, w, convention='abs', walker_chunk=M)), ('M1_RBMPPGuideG_p0', lambda: m1_guide(0.0)),
                   ('P1Guide', lambda: G2.P1Guide(p, c['alpha'], c['E'], w)), ('M1_RBMPPGuideG_p1', lambda: m1_guide(None))):
        g = mk(); t = time.time(); g.rows(X[:M]); t1 = time.time() - t; t = time.time(); g.rows(X[M:]); t2 = time.time() - t
        res[nm] = dict(first_call_sec=t1, sec_per_walker_step=t2 / M); log(f'{nm}: {res[nm]}')
    # a short GFMC run with BareM1Sign at M = 64: hops per walker-tau and seconds per walker-tau on this CPU
    g = G2.BareM1Sign(p, w, convention='abs', walker_chunk=M); gf = GFMC(g, X[:M], 0.1, 1); t = time.time(); hops = 0; nb = 20; E = []
    for _ in range(nb): r = gf.block(); hops += r['hops']; E.append(r['emix'] / 64)
    s = time.time() - t; res['gfmc_bare'] = dict(blocks=nb, dtau=0.1, sec=s, hops=hops, hops_per_walker_tau=hops / (M * nb * 0.1), sec_per_walker_tau=s / (M * nb * 0.1),
                                                 E_mix_last=E[-5:], note='20 blocks from beta-1 draws, no burn: timing only, not an energy')
    res['P1Guide_derived_sec_per_walker_tau'] = res['P1Guide']['sec_per_walker_step'] * res['gfmc_bare']['hops_per_walker_tau']
    json.dump(res, open(f'{OUT}/t3c_timing.json', 'w'), indent=1, default=float); log(json.dumps(res, default=float))


{'bare': t_bare, 'p1': t_p1, 'timing': t_timing}[WHAT]()
