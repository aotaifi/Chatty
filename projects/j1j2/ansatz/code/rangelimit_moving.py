"""rangelimit_moving.py: TEACHER CHECK A1 (STORAGE_PROBLEM.md). The explicit round map with MOVING signs and an energy-searched c, full 6x6 table, CPU, no training.
usage: python3 rangelimit_moving.py OUTDIR [--mode moving|fixed|broken] [--nrounds N] [--workers 6]
State of round k: log-amplitude table lA_k (A_0 = pfaffian_c c1_cplx_s1 round_1 la_table) and sign S_k (S_0 = s2). Everything of round k (E_L, E_var, the guide,
every FN solve) uses the operators K, Vm built from S_k (rebuilt per round; D0 does not depend on the sign).
  d_k = E_L[S_k A_k] - E_k;  alpha_k = 1-D search of E_FN[S_k A_k |1 + alpha d_k|] (grid of cstar_lanczos.py/rangelimit_rounds.py, parabola refinement);
  g_k = A_k max(|1 + alpha_k d_k|, e^-60) (Lanczos guide, sign S_k);  p_k = FN state of S_k g_k (amplitude |v|/sqrt(n));
  S_{k+1} = S_k sgn(1 + alpha_k d_k)  (sgn 0 -> +1);  A_{k+1}(c) = g_k e^{c f_J[g_k]}  (f_J with the S_k operator and E' = E_FN[S_k g_k]);
  c_k = 1-D search of E_FN[S_{k+1} A_{k+1}(c)] over c in 0.5..3.0 step 0.25 (11 solves) plus parabola refinement (vertex clipped to the grid range, edge flagged).
  c is never fitted to p.
Modes: moving = the written check (3 rounds). fixed = sign update OFF and c = 1.307 fixed (reproduces rounds23: E_FN[s2 A_1] 6.568e-6, E_var 7.938e-6, alpha_1 -0.10).
  broken = round 0 as moving, then at c_0 the round-1 energy with the sign S_1 replaced by: noflip (S_0), complement (flip where the true update does NOT flip: = -S_1,
  a global sign, so identical in exact arithmetic), rand_count (same number of rows flipped, uniformly random), rand_weight (same number, drawn with prob ~ n A_0^2).
Grid solves run in parallel (fork, read-only operators); E_FN solves = delta0.efn_vec (LOBPCG tol 1e-6, maxit 60) as rangelimit_rounds.py.
Energies are dE per site vs E0 = (E - E0)/36."""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[_v] = '1'
import sys, json, time, gc, argparse, resource, multiprocessing as mp
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bond_split as BS
import delta0 as D0M
ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--mode', default='moving', choices=['moving', 'fixed', 'broken'])
ap.add_argument('--nrounds', type=int, default=None); ap.add_argument('--workers', type=int, default=6); ap.add_argument('--cfixed', type=float, default=1.307)
ap.add_argument('--RL', default='/scratch/a/A.Otaifi/ansatz_runs/rangelimit_6x6')
args = ap.parse_args(); OUT = args.out; MODE = args.mode; os.makedirs(OUT, exist_ok=True); T0 = time.time()
NR = args.nrounds or {'moving': 3, 'fixed': 2, 'broken': 1}[MODE]; SUF = '' if MODE == 'moving' else '_' + MODE
def log(m): print(f'[{time.time() - T0:7.0f}s] {m}', flush=True)
A0 = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_c/runs/c1_cplx_s1/round_1/la_table.npy'
n = np.load(f'{BS.CSRD}/norb.npy').astype(np.float64); S0 = np.load(BS.S2).astype(np.float64); D = len(n); assert n.sum() == 9075135300; sn = np.sqrt(n)
lA0 = np.load(A0).astype(np.float64)
EPG0 = json.load(open(f'{args.RL}/rangelimit_A.json'))['step1']['E_FN']
res = dict(host=os.uname().nodename, mode=MODE, nrounds=NR, A0=A0, c_fixed=args.cfixed if MODE == 'fixed' else None, rounds={}, floors=dict(), argv=sys.argv)
RJ = f'{OUT}/moving{SUF}.json'
def dump(): res['sec'] = time.time() - T0; json.dump(res, open(RJ, 'w'), indent=1, default=float)
def site(E): return (E - BS.E0) / BS.NS
t1 = np.load(f'{BS.TAB}/nb_J1.npy', mmap_mode='r'); t2 = np.load(f'{BS.TAB}/nb_J2.npy', mmap_mode='r'); HW = BS.HW; JW = BS.JW; RS = 32768
K = Vm = D0 = None
def set_ops(S):
    global K, Vm, D0
    K = Vm = None; gc.collect(); K, Vm, D0 = BS.build(n, S); log(f'built operators (K nnz {K.nnz})')
def moments(la):
    a = np.exp(la - la.max()); EL = (D0 * a + Vm @ a - (K @ (sn * a)) / sn) / a; w = n * a * a; E = float((w * EL).sum() / w.sum())
    assert np.isfinite(EL).all() and np.isfinite(E); return E, EL - E, w / w.sum()
def lanc(la, d, al):
    m = np.abs(1.0 + al * d); nfl = int((m < np.exp(-60.0)).sum()); return la + np.log(np.maximum(m, np.exp(-60.0))), nfl
def guide_floor(la):
    a = np.exp(la - la.max()); a = a / np.sqrt((n * a * a).sum()); return int((a < 1e-15).sum())
def fJ(lg, Ep, S):
    fj = np.empty(D); nbad = 0
    for i0 in range(0, D, RS):
        i1 = min(D, i0 + RS); t = np.concatenate([t1[i0:i1], t2[i0:i1]], 1); va = t >= 0; tt = np.where(va, t, 0)
        ss = S[i0:i1, None] * S[tt]; blk = va & (ss > 0); kpt = va & (ss < 0); Hxx = 27.0 - 0.5 * (va * JW[None, :]).sum(1)
        ratio = np.exp(np.clip(lg[tt] - lg[i0:i1, None], -700, 700)); VA = (blk * HW * ratio).sum(1); SK = (kpt * HW * ratio).sum(1)
        den = Hxx + VA - Ep; nbad += int((den <= 0).sum() + (SK <= 0).sum())
        with np.errstate(divide='ignore', invalid='ignore'): fj[i0:i1] = np.log(SK) - np.log(den)
    return fj, nbad
# ---- worker side: G is set by the parent before the fork
G = dict(lA=None, d=None, lG=None, fJG=None)
def solve_one(spec):
    kind, x, wv = spec
    la = lanc(G['lA'], G['d'], x)[0] if kind == 'a' else G['lG'] + x * G['fJG']
    t = time.time(); r, v = D0M.efn_vec(K, Vm, D0, n, la); r['sec'] = time.time() - t; r['guide_floor'] = guide_floor(la); r['finite'] = bool(np.isfinite(la).all() and np.isfinite(r['E']))
    return spec, r, (v if wv else None)
def many(specs):
    out = {}
    def rep(sp, r): log(f'  {sp[0]} {sp[1]:+.4f}: dE/site {r["dE_site"]:.6e} iters {r["iters"]} res {r["res"]:.1e} {r["sec"]:.0f}s guide-floor rows {r["guide_floor"]}')
    if args.workers > 1 and len(specs) > 1:
        with mp.get_context('fork').Pool(min(args.workers, len(specs)), maxtasksperchild=1) as pool:
            for sp, r, v in pool.imap_unordered(solve_one, specs): out[sp] = (r, v); rep(sp, r)
    else:
        for sp in specs: _, r, v = solve_one(sp); out[sp] = (r, v); rep(sp, r)
    return out
def search(kind, grid, ext, lo, hi, wv, rec, tag):
    """grid in parallel, optional extension (alpha only), then parabola vertex iterations exactly as rangelimit_rounds.py (best = best evaluated point)."""
    cache = {}; vecs = {}
    def ev(xs):
        xs = [round(float(x), 6) for x in xs if round(float(x), 6) not in cache]
        for sp, (r, v) in many([(kind, x, wv) for x in xs]).items(): cache[sp[1]] = r; vecs[sp[1]] = v
        dump()
    ev(grid); bst = min(cache, key=lambda a_: cache[a_]['dE_site'])
    if ext is not None and bst == ext[0]: ev(ext[1])
    vtx = None; note = None
    for it in range(8):
        cs = sorted(cache); bst = min(cs, key=lambda a_: cache[a_]['dE_site']); near = sorted(sorted(cs, key=lambda c: abs(c - bst))[:3])
        pp = np.polyfit(near, [cache[c]['dE_site'] for c in near], 2)
        if pp[0] <= 0: note = 'NOT CONVEX'; break
        vtx = float(np.clip(-pp[1] / (2 * pp[0]), lo, hi)); log(f'  {tag} iter {it}: points {near} vertex {vtx:.4f}')
        if min(abs(vtx - c) for c in cache) < 0.01: break
        ev([vtx])
    best = min(cache, key=lambda a_: cache[a_]['dE_site'])
    rec[f'{kind}_search'] = dict(best=best, vertex=vtx, note=note, edge=bool(kind == 'c' and best in (lo, hi)), grid={str(x): cache[x]['dE_site'] for x in sorted(cache)},
                                 guide_floor_max=max(cache[x]['guide_floor'] for x in cache), nonfinite=int(sum(not cache[x]['finite'] for x in cache)))
    return best, cache, vecs
AG = [0.0, -0.1, -0.2, -0.3, -0.45, -0.6]; AEXT = (-0.6, [-0.8, -1.0, -1.2]); CG = [0.5 + 0.25 * i for i in range(11)]
# ------------------------------------------------------------------------------------------------------------------ rounds
S = S0.copy(); lA = lA0.copy(); set_ops(S); E, d, w_A = moments(lA); Evar_cur = site(E); fl = res['floors']; fl.update(lanc_e60=0, guide_1e15=0, p_1e300=0, fJ_bad=0, nonfinite=0)
EFN_next_prev = None
for k in range(NR):
    rec = dict(k=k, sign_nneg_S_k=int((S < 0).sum())); res['rounds'][str(k)] = rec; rec['Evar_SkAk_site'] = Evar_cur; log(f'round {k}: E_var[S_k A_k]/site {Evar_cur:.6e}')
    G.update(lA=lA, d=d); ak, cache, vecs = search('a', AG, AEXT, -1.2, 0.0, True, rec, f'round{k} alpha')
    rec.update(alpha=ak, alpha_vertex=rec['a_search']['vertex'], E_FN_SkAk_site=cache[0.0]['dE_site'], E_FN_g_site=cache[ak]['dE_site'])
    if EFN_next_prev is not None: rec['E_FN_SkAk_recheck_vs_prev_cstep'] = cache[0.0]['dE_site'] - EFN_next_prev
    if k == 0 and ak == -0.1: rec['E_FN_g0_minus_rangelimit_A_step1'] = cache[ak]['dE_site'] - site(EPG0)
    log(f"round {k}: alpha {ak} (vertex {rec['alpha_vertex']}) E_FN[S_k A_k] {rec['E_FN_SkAk_site']:.6e} E_FN[g_k] {rec['E_FN_g_site']:.6e}"); dump()
    lG, nfl = lanc(lA, d, ak); rec['floor_lanc_e60'] = nfl; EG = cache[ak]['E']
    v = vecs[ak] * np.sign(vecs[ak].sum()); rec['p_n_neg'] = int((v < 0).sum()); p = np.abs(v) / sn; p = p / np.sqrt((n * p * p).sum()); rec['p_floor_1e300'] = int((p < 1e-300).sum())
    lp = np.log(np.maximum(p, 1e-300)); np.save(f'{OUT}/p_{k}{SUF}.npy', p); del vecs, v
    t = time.time(); rp, _ = D0M.efn_vec(K, Vm, D0, n, lp); rec['E_FN_p_site'] = rp['dE_site']; log(f"round {k}: E_FN[p_k] {rp['dE_site']:.6e} ({time.time() - t:.0f}s)")
    rec['gain_site'] = rec['E_FN_SkAk_site'] - rec['E_FN_p_site']
    fjG, nbad = fJ(lG, EG, S); rec['fJ_bad'] = nbad; rec['fJ_nonfinite'] = int((~np.isfinite(fjG)).sum()); rec['fJ_range'] = [float(np.nanmin(fjG)), float(np.nanmax(fjG))]; dump()
    log(f"round {k}: f_J[g_k] bad {nbad} nonfinite {rec['fJ_nonfinite']} range {rec['fJ_range']}")
    if rec['fJ_nonfinite'] or nbad: rec['status'] = 'NONFINITE'; dump(); log('STOP: nonfinite f_J'); sys.exit(3)
    f1 = 1.0 + ak * d; zero = int((f1 == 0).sum()); sg = np.where(f1 < 0, -1.0, 1.0)
    if MODE == 'fixed': sg = np.ones(D)
    flip = sg < 0; S_next = S * sg; w_p = n * p * p / (n * p * p).sum()
    rec['flips'] = dict(rows=int(flip.sum()), configs_n_weighted=float(n[flip].sum()), config_frac=float(n[flip].sum() / n.sum()), weight_A_k=float(w_A[flip].sum()),
                        weight_p_k=float(w_p[flip].sum()), zero_f1_rows=zero, sign_update='off' if MODE == 'fixed' else 'on')
    log(f"round {k}: sign flips S_k+1 vs S_k: rows {rec['flips']['rows']} (of {D}), config frac {rec['flips']['config_frac']:.3e}, |A_k|^2 weight {rec['flips']['weight_A_k']:.3e}, |p_k|^2 weight {rec['flips']['weight_p_k']:.3e}, zeros {zero}"); dump()
    if flip.any(): set_ops(S_next)
    G.update(lG=lG, fJG=fjG)
    if MODE == 'fixed':
        sp, (r, _) = next(iter(many([('c', args.cfixed, False)]).items())); cache_c = {round(args.cfixed, 6): r}; ck = round(args.cfixed, 6)
        rec['c_search'] = dict(best=ck, vertex=None, note='fixed', grid={str(ck): r['dE_site']}, nonfinite=int(not r['finite']), guide_floor_max=r['guide_floor'], edge=False)
    else:
        ck, cache_c, _ = search('c', CG, None, 0.5, 3.0, False, rec, f'round{k} c')
    rec.update(c=ck, c_vertex=rec['c_search']['vertex'], E_FN_next_site=cache_c[ck]['dE_site']); lA_next = lG + ck * fjG
    rec['next_nonfinite'] = int((~np.isfinite(lA_next)).sum())
    if rec['next_nonfinite']: rec['status'] = 'NONFINITE'; dump(); log('STOP: nonfinite A_k+1'); sys.exit(3)
    E_n, d_n, w_n = moments(lA_next); rec['Evar_next_site'] = site(E_n); EFN_next_prev = rec['E_FN_next_site']
    rec['retention'] = (rec['E_FN_SkAk_site'] - rec['E_FN_next_site']) / (rec['E_FN_SkAk_site'] - rec['E_FN_p_site'])
    rec['certificate_margin_site'] = rec['E_FN_g_site'] - rec['Evar_next_site']
    log(f"round {k}: c {ck} (vertex {rec['c_vertex']}) E_FN[S_k+1 A_k+1] {rec['E_FN_next_site']:.6e} E_var[S_k+1 A_k+1] {rec['Evar_next_site']:.6e} retention {rec['retention']:.4f} margin {rec['certificate_margin_site']:.4e}")
    fl['lanc_e60'] += nfl; fl['guide_1e15'] += max(rec['a_search']['guide_floor_max'], rec['c_search']['guide_floor_max']); fl['p_1e300'] += rec['p_floor_1e300']; fl['fJ_bad'] += nbad
    fl['nonfinite'] += rec['fJ_nonfinite'] + rec['next_nonfinite'] + rec['a_search']['nonfinite'] + rec['c_search']['nonfinite']
    np.save(f'{OUT}/lA_{k + 1}{SUF}.npy', lA_next); np.save(f'{OUT}/S_{k + 1}{SUF}.npy', S_next.astype(np.int8)); dump()
    if MODE == 'broken' and k == NR - 1:
        rng = np.random.default_rng(777); nf = int(flip.sum()); var = {}
        var['complement'] = np.where(flip, 1.0, -1.0)
        mk = np.zeros(D, bool); mk[rng.choice(D, nf, replace=False)] = True; var['rand_count'] = np.where(mk, -1.0, 1.0)
        kk = np.argpartition(-(np.log(np.maximum(w_A, 1e-300)) + rng.gumbel(size=D)), nf - 1)[:nf]; mk = np.zeros(D, bool); mk[kk] = True; var['rand_weight'] = np.where(mk, -1.0, 1.0)
        var['noflip'] = np.ones(D); rec['broken'] = {}
        for nm, sgv in var.items():
            Sb = S * sgv; fb = sgv < 0
            set_ops(Sb); Eb, _, _ = moments(lA_next)
            t = time.time(); rb, _ = D0M.efn_vec(K, Vm, D0, n, lA_next)
            rec['broken'][nm] = dict(rows_flipped_vs_S0=int(fb.sum()), rows_differ_from_true_S1=int((Sb != S_next).sum()), weight_A_k_flipped=float(w_A[fb].sum()), E_FN_site=rb['dE_site'], Evar_site=site(Eb), iters=rb['iters'], res=rb['res'])
            log(f"broken {nm}: flipped rows vs S_0 {int(fb.sum())} (differ from true S_1: {int((Sb != S_next).sum())}) E_FN[S' A_1(c_0)] {rb['dE_site']:.6e} E_var {site(Eb):.6e} ({time.time() - t:.0f}s)"); dump()
    S = S_next; lA = lA_next; d = d_n; w_A = w_n; Evar_cur = rec['Evar_next_site']
res['status'] = 'DONE'; res['maxrss_self_MB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024; res['maxrss_children_MB'] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024; dump()
log('TABLE (dE/site)')
for k, r in res['rounds'].items():
    log(f"k={k} alpha {r['alpha']} c {r['c']} flips rows {r['flips']['rows']} w_A {r['flips']['weight_A_k']:.3e} w_p {r['flips']['weight_p_k']:.3e} | Evar[SkAk] {r['Evar_SkAk_site']:.4e} EFN[SkAk] {r['E_FN_SkAk_site']:.4e} EFN[g] {r['E_FN_g_site']:.4e} EFN[p] {r['E_FN_p_site']:.4e} | EFN[next] {r['E_FN_next_site']:.4e} Evar[next] {r['Evar_next_site']:.4e} ret {r['retention']:.4f} margin {r['certificate_margin_site']:.3e}")
log(f"floors {res['floors']} maxrss self {res['maxrss_self_MB']:.0f} MB children {res['maxrss_children_MB']:.0f} MB"); log('DONE')
