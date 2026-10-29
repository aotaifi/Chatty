"""test_w.py: pre-registered Test W (PLAN.md "TEST W"), CPU only, numpy. usage: python3 test_w.py OUTDIR
Can one exact FN round's gain survive labels estimated from FINITE ideal samples?  Sign s1 (Test Z, pfaffian_z/s1.npy).
  g    = s1|A|, A = pfaffian_c/runs/c1_cplx_s1/round_1/la_table.npy (= delta0.A0), normalised sum_x n_x g_x^2 = 1
  phi'' = FN ground state of H_FN[g, s1] (as test_z.py (b)), normalised sum n phi''^2 = 1
  samples: N orbit indices ~ Multinomial(N, n g phi''/Z_mix) (mixed) and N ~ Multinomial(N, n g^2) (reference), seeded
  h(x) = c_mix(x)/c_ref(x) * Z_mix on c_mix > 0 (estimates phi''/g; Z_mix = sum n g phi'' exact, the normalisation);
       c_mix > 0 but c_ref = 0: (c_mix/N)/(n_x g_x^2) * Z_mix (exact reference probability); c_mix = 0: h = 1 (keep g)
  phi_hat = g h.  Arm (a): E_FN[s1|phi_hat|].  Arm (b): phi_hat e^{c f_J[phi_hat]}, c by the test_z.py line search on [0, 6];
  f_J as test_z.py (c) with A -> phi_hat and E' = E_FN[phi_hat].
  retained = (E_FN[g] - E_FN[guide]) / (E_FN[g] - E_FN[phi'']).
Controls: exact h (-> retained 1); shuffled h (h values permuted among c_mix > 0 orbits at N = 1e6, seed 1) -> retained <= ~0.
Band split (arm a, seed 1, N = 1e6, 1e8): splice exact h into phi''^2 < 1e-11 and into phi''^2 >= 1e-11 (per config), 2 solves.
Everything cached in OUTDIR/w_result.json (resumable).
"""
import os, sys, json, time, resource
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bond_split as BS
import delta0 as D0M
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); T0 = time.time()
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576
def log(m): print(f'[{time.time() - T0:7.0f}s] {m}  (maxrss {rss():.1f} GB)', flush=True)
RJ = f'{OUT}/w_result.json'
res = json.load(open(RJ)) if os.path.exists(RJ) else {}
res['host'] = os.uname().nodename
def dump(): json.dump(res, open(RJ + '.tmp', 'w'), indent=1, default=float); os.replace(RJ + '.tmp', RJ)
ZDIR = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_z'
REF_G, REF_P = 2.5033772674080506e-05, 1.607666205987673e-05          # z_result.json b.E_FN_A / b.E_FN_phi2 dE_site
n = np.load(f'{BS.CSRD}/norb.npy').astype(np.float64); D = len(n); assert n.sum() == 9075135300
HW = BS.HW; JW = BS.JW; NS = BS.NS
t1 = np.load(f'{BS.TAB}/nb_J1.npy', mmap_mode='r'); t2 = np.load(f'{BS.TAB}/nb_J2.npy', mmap_mode='r')
s1 = np.load(f'{ZDIR}/s1.npy').astype(np.float64); assert s1.shape == (D,) and set(np.unique(s1)) <= {-1.0, 1.0}
def norm_la(la): la = la - la.max(); return la - 0.5 * np.log((n * np.exp(2 * la)).sum())
lA = np.load(D0M.A0).astype(np.float64); lgn = norm_la(lA)
K, Vm, D0 = BS.build(n, s1); log('built K,Vm for s1')

def efn(la, tag):
    t = time.time(); r, v = D0M.efn_vec(K, Vm, D0, n, la); r['sec'] = time.time() - t
    log(f'{tag}: dE/site {r["dE_site"]:.9e} it {r["iters"]} res {r["res"]:.1e} {r["sec"]:.0f}s'); return r, v

# ---- base: E_FN[g], phi'', E_FN[phi'']
LP = f'{OUT}/lphi2.npy'
if 'base' not in res or not os.path.exists(LP):
    rg, v = efn(lA, 'E_FN[g]'); chi = np.abs(v) / np.sqrt(n); del v
    np.save(LP, np.log(np.maximum(chi, 1e-300))); del chi
    rp, _ = efn(np.load(LP), "E_FN[phi'']")
    res['base'] = dict(E_FN_g=rg, E_FN_phi2=rp, Delta_site=rg['dE_site'] - rp['dE_site'],
                       rel_vs_testZ_g=rg['dE_site'] / REF_G - 1, rel_vs_testZ_phi2=rp['dE_site'] / REF_P - 1); dump()
lpn = norm_la(np.load(LP))
EG = res['base']['E_FN_g']['dE_site']; EP2 = res['base']['E_FN_phi2']['dE_site']; DEL = EG - EP2
assert abs(res['base']['rel_vs_testZ_g']) < 1e-5 and abs(res['base']['rel_vs_testZ_phi2']) < 1e-5, 'base does not reproduce Test Z'
Zmix = float((n * np.exp(lgn + lpn)).sum()); lhex = lpn - lgn                       # exact ln h = ln(phi''/g)
p2 = np.exp(2 * lpn); LOW = p2 < 1e-11
res['meta'] = dict(Zmix=Zmix, g_min=float(np.exp(lgn.min())), n_low_band_orbits=int(LOW.sum()),
                   phi2_weight_low_band=float((n * p2)[LOW].sum()), g2_weight_low_band=float((n * np.exp(2 * lgn))[LOW].sum())); dump()
log(f'meta {res["meta"]}')
def ret(e): return (EG - e) / DEL

def fJ_of(la, EPr):
    """test_z.py (c) with A -> exp(la), E' = EPr (total energy)"""
    f = np.empty(D); den_min = np.inf; lm = la - la.max()
    for i0 in range(0, D, 65536):
        i1 = min(D, i0 + 65536); t = np.concatenate([t1[i0:i1], t2[i0:i1]], 1); valid = t >= 0; tt = np.where(valid, t, 0)
        ss = s1[i0:i1, None] * s1[tt]; blk = valid & (ss > 0); kpt = valid & (ss < 0)
        ratio = np.exp(np.clip(lm[tt] - lm[i0:i1, None], -700, 700))
        VA = (blk * HW[None, :] * ratio).sum(1); SK = (kpt * HW[None, :] * ratio).sum(1)
        Hxx = 27.0 - 0.5 * (valid * JW[None, :]).sum(1); den = Hxx + VA - EPr
        den_min = min(den_min, float(den.min()))
        with np.errstate(divide='ignore', invalid='ignore'): f[i0:i1] = np.log(SK) - np.log(den)
    bad = np.nonzero(~np.isfinite(f))[0]
    assert len(bad) <= 3 and den_min > 0, (len(bad), den_min)
    f[bad] = 0.0
    return f, dict(den_min=den_min, bad_rows=[int(i) for i in bad], fJ_min=float(f.min()), fJ_max=float(f.max()))

def line_search(la, rec, key):
    """test_z.py line search: c in {0 (= arm a), 2,3,4,5,6} then parabola refinement on [0, 6]."""
    t = time.time(); fJ, info = fJ_of(la, rec['a']['E']); rec[key + '_fJ'] = dict(info, sec=time.time() - t); dump()
    sc = rec.setdefault(key, {}); cache = {float(k): v['dE_site'] for k, v in sc.items()}
    cache[0.0] = rec['a']['dE_site']
    def solve(c):
        c = round(float(c), 6)
        if c in cache: return cache[c]
        r, _ = efn(la + c * fJ, f'  {key} c={c:.3f}'); cache[c] = r['dE_site']; sc[str(c)] = r; dump(); return r['dE_site']
    for c in (2.0, 3.0, 4.0, 5.0, 6.0): solve(c)
    status = 'DONE'
    for it in range(8):
        b = min(cache, key=cache.get)
        if b >= 6.0: status = 'MIN AT UPPER EDGE'; break
        cs = sorted(cache); i = cs.index(b); trio = cs[max(i - 1, 0):i + 2] if i > 0 else cs[:3]
        p = np.polyfit(trio, [cache[c] for c in trio], 2)
        if p[0] <= 0: status = 'NOT CONVEX'; break
        cv = float(np.clip(-p[1] / (2 * p[0]), 0.0, 6.0))
        if min(abs(cv - c) for c in cache) < 0.05: break
        c2 = round(cv / 0.05) * 0.05; solve(c2 if min(abs(c2 - c) for c in cache) > 1e-9 else cv)
    cstar = min(cache, key=cache.get)
    rec['b'] = dict(cstar=cstar, dE_site=cache[cstar], retained=ret(cache[cstar]), status=status, nsolves=len(sc)); dump()
    log(f'  arm b: c* {cstar} dE/site {cache[cstar]:.6e} retained {ret(cache[cstar]):.4f} {status}')

# ---- control 1: exact h
if 'ctrl_exact' not in res:
    r, _ = efn(lgn + lhex, 'ctrl exact h'); res['ctrl_exact'] = dict(dE_site=r['dE_site'], retained=ret(r['dE_site']), iters=r['iters'], res=r['res']); dump()

def estimate(N, seed):
    pm = n * np.exp(lgn + lpn); pm /= pm.sum(); pr = n * np.exp(2 * lgn); pr /= pr.sum()
    rm = np.random.default_rng(np.random.SeedSequence([20261029, int(round(np.log10(N))), seed, 0]))
    rr = np.random.default_rng(np.random.SeedSequence([20261029, int(round(np.log10(N))), seed, 1]))
    cm = rm.multinomial(N, pm); cr = rr.multinomial(N, pr); del pm
    vis = cm > 0; mixonly = vis & (cr == 0); refonly = (cm == 0) & (cr > 0)
    h = np.ones(D)
    both = vis & (cr > 0); h[both] = cm[both] / cr[both] * Zmix
    h[mixonly] = (cm[mixonly] / N) / pr[mixonly] * Zmix
    st = dict(N=int(N), seed=seed, sum_mix=int(cm.sum()), sum_ref=int(cr.sum()),
              frac_orbits_visited_mix=float(vis.mean()), frac_orbits_visited_any=float((vis | (cr > 0)).mean()),
              phi2_weight_visited_mix=float((n * p2)[vis].sum()), phi2_weight_visited_any=float((n * p2)[vis | (cr > 0)].sum()),
              n_mix_only=int(mixonly.sum()), n_ref_only_set_h1=int(refonly.sum()), n_visited_mix=int(vis.sum()),
              phi2_weight_visited_mix_lowband=float((n * p2)[vis & LOW].sum()),
              lnh_err_rms_visited_phi2w=float(np.sqrt(((n * p2) * (np.log(h) - lhex) ** 2)[vis].sum() / (n * p2)[vis].sum())))
    st['h_max'] = float(h.max()); st['h_min'] = float(h.min()); st['h_max_mixonly'] = float(h[mixonly].max()) if mixonly.any() else None
    st['n_h_gt_10'] = int((h > 10).sum()); st['n_h_gt_10_mixonly'] = int((h[mixonly] > 10).sum())
    return h, vis, st, mixonly

def run_N(N, seed, bands=False, shuffle=False, armb=True):
    key = f'N{int(round(np.log10(N)))}_s{seed}'; rec = res.setdefault('runs', {}).setdefault(key, {})
    t = time.time(); h, vis, st, mixonly = estimate(N, seed); st['sample_sec'] = time.time() - t; rec['stats'] = st; dump(); log(f'{key} stats {st}')
    la = lgn + np.log(h)
    if 'a' not in rec:
        r, _ = efn(la, f'{key} arm a'); rec['a'] = dict(r, retained=ret(r['dE_site'])); dump()
    log(f'{key} arm a retained {rec["a"]["retained"]:.4f}')
    if bands and 'diag_mixonly_h1' not in rec:      # diagnostic (not in the spec): mixed-only orbits get h = 1 instead of the exact-g^2 estimate
        r, _ = efn(np.where(mixonly, lgn, la), f'{key} diag mix-only h=1'); rec['diag_mixonly_h1'] = dict(dE_site=r['dE_site'], retained=ret(r['dE_site'])); dump()
    if bands and 'bands' not in rec:
        ea = rec['a']['dE_site']; bd = {}
        for nm, msk in (('low_lt_1e-11', LOW), ('high_ge_1e-11', ~LOW)):
            lh = np.where(msk, lhex, np.log(h)); r, _ = efn(lgn + lh, f'{key} splice exact h in {nm}')
            bd[nm] = dict(dE_site=r['dE_site'], share_of_lost=(ea - r['dE_site']) / (ea - EP2), retained=ret(r['dE_site']))
        bd['sum_shares'] = bd['low_lt_1e-11']['share_of_lost'] + bd['high_ge_1e-11']['share_of_lost']; rec['bands'] = bd; dump(); log(f'{key} bands {bd}')
    if shuffle and 'ctrl_shuffled' not in res:
        hs = h.copy(); idx = np.nonzero(vis)[0]; hs[idx] = np.random.default_rng(777).permutation(h[idx])
        r, _ = efn(lgn + np.log(hs), f'{key} ctrl shuffled h'); res['ctrl_shuffled'] = dict(on=key, dE_site=r['dE_site'], retained=ret(r['dE_site'])); dump()
    if armb and 'b' not in rec: line_search(la, rec, 'scan_b')

# time budget (solves ~150-280 s on spiky guides, > 3 h for the full grid): arm (b) only at seed 1, N = 1e6 and 1e8; arm (a) everywhere
PLANR = [(1e6, 1, True, True, True), (1e8, 1, True, False, True), (1e5, 1, False, False, False), (1e7, 1, False, False, False),
         (1e6, 2, False, False, False), (1e8, 2, False, False, False), (1e5, 2, False, False, False), (1e7, 2, False, False, False)]
for N, sd, bnd, shf, ab in PLANR: run_N(N, sd, bands=bnd, shuffle=shf, armb=ab)
res['status'] = 'DONE'; res['maxrss_GB'] = rss(); res['sec_total_this_invocation'] = time.time() - T0; dump(); log('DONE')
