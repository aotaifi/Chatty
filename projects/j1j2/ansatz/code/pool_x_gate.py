"""pool_x_gate.py: registered gate for the 8x8 round-1 x-draw pool (PLAN_pfaffian.md "8x8 R1 pool, x-draws").
usage: python3 pool_x_gate.py RUND --split K [--targets 330000,450000,320000] [--break KIND] [--json OUT]
RUND/out_T.npz, T = 1..NT; seed half A = tasks T <= K, half B = tasks T > K.  Checks (all registered):
 (1) per-decade counts per rung within 20% of decades8/sizing/sizing.json occupancy, decades -2..-9 (decades relative to the median 2lnB of the pool's beta-1 draws, as in sizing);
     cells whose sizing count n_ref < 100 cannot meet a 20% rule (the reference itself is that uncertain): they are tested at 3 sigma combined (reference Poisson + the pool's chain scatter),
     and a zero-count reference cell must have fraction <= 3 / (reference draws of that rung).  The literal 20% failures are printed as well.
 (2) halves A vs B: z = (fA - fB)/se over the decade fractions -2..-9 per rung (se from the scatter of per-chain fractions, floored at binomial), max |z| <= 3.
 (3) median 2lnB of the beta-1 draws within 0.3 of -67.03 (sizing.json: -67.0321).
 (4) every word has 32 bits set (Sz = 0, N = 64).
 (5) lnB of 200 random draws recomputed with pp2_lemma.logB (fresh PP2 from the guide file) agrees to 1e-10.
 plus bookkeeping: per-rung draw counts >= targets, seeds distinct, params sha 0520c5be, chain ids per rung.
--break KIND corrupts the loaded data in memory to show the gate fails: counts (beta .5 draws relabelled .25), halves (half B beta .5 lnB += 0.8), median (beta-1 lnB += 0.5),
 sz (one bit flipped in 50 words), recompute (lnB of 5 draws += 1e-8)."""
import sys, os, glob, json, math, argparse, hashlib
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
LN10 = math.log(10.0)
SIZING = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_8x8build/decades8/sizing/sizing.json'
PROD8 = '/scratch/a/A.Otaifi/rbmpp8/rb8_prod/params_final.npz'
ap = argparse.ArgumentParser(); ap.add_argument('rund'); ap.add_argument('--split', type=int, required=True); ap.add_argument('--targets', default='330000,450000,320000')
ap.add_argument('--break', dest='brk', default=None); ap.add_argument('--json', default=None); ap.add_argument('--nrec', type=int, default=200); ap.add_argument('--files', default='out_*.npz'); a = ap.parse_args()
TARG = [int(x) for x in a.targets.split(',')]; BETAS = [1.0, 0.5, 0.25]; DECS = list(range(-2, -10, -1))
fs = sorted(glob.glob(os.path.join(a.rund, a.files)), key=lambda f: int(os.path.basename(f).split('_')[1].split('.')[0]))
Z = [np.load(f) for f in fs]; assert Z, 'no files'
W = np.concatenate([z['words'] for z in Z]); L = np.concatenate([z['lnB'] for z in Z]); O = np.concatenate([z['orb'] for z in Z]); B = np.concatenate([z['beta'] for z in Z]); C = np.concatenate([z['chain'] for z in Z])
TK = np.concatenate([np.full(len(z['words']), int(z['task'])) for z in Z]); seeds = [int(z['seed']) for z in Z]; psha = set(str(z['params_sha']) for z in Z)
half = TK <= a.split                                                         # True = A
rng = np.random.default_rng(12345)
if a.brk == 'counts': m = B == 0.5; B[m] = 0.25
if a.brk == 'halves': m = (~half) & (B == 0.5); L[m] += 0.8
if a.brk == 'median': L[B == 1.0] += 0.5
if a.brk == 'sz': idx = rng.choice(len(W), 50, replace=False); W[idx] ^= np.uint64(1)
if a.brk == 'recompute': idx = rng.choice(len(W), 5, replace=False); L[idx] += 1e-8
res = dict(rund=a.rund, files=len(fs), halfA_tasks=int(len(set(TK[half].tolist()))), halfB_tasks=int(len(set(TK[~half].tolist()))), broken=a.brk)
# ---- bookkeeping
n_rung = [int((B == b).sum()) for b in BETAS]; book = dict(n_per_rung=n_rung, targets=TARG, ge_targets=all(n >= t for n, t in zip(n_rung, TARG)), seeds_distinct=len(set(seeds)) == len(seeds),
        params_sha_ok=(len(psha) == 1 and list(psha)[0].startswith('0520c5be')), chains_per_rung=[int(len(set(C[B == b].tolist()))) for b in BETAS], n_other_beta=int((~np.isin(B, BETAS)).sum()))
book['ok'] = bool(book['ge_targets'] and book['seeds_distinct'] and book['params_sha_ok'] and book['n_other_beta'] == 0); res['bookkeeping'] = book
# ---- (3) median
x2 = 2.0 * L; m1 = x2[B == 1.0]; med = float(np.median(m1)); medA = float(np.median(x2[(B == 1.0) & half])); medB = float(np.median(x2[(B == 1.0) & ~half]))
ref = json.load(open(SIZING)); res['median'] = dict(median=med, halfA=medA, halfB=medB, ref=-67.03, sizing_med=ref['med2lnB'], diff=med + 67.03, ok=bool(abs(med + 67.03) <= 0.3))
# ---- (4) Sz
def popc(w):
    c = np.zeros(len(w), np.int64)
    for s in range(64): c += ((w >> np.uint64(s)) & np.uint64(1)).astype(np.int64)
    return c
pc = popc(W); res['Sz'] = dict(n_words=int(len(W)), n_bad=int((pc != 32).sum()), ok=bool(np.all(pc == 32)))
# ---- decade bins
dec = np.floor((x2 - med) / LN10)
def decile_counts(mask): return np.array([[int(((dec == d) & mask).sum()) for d in DECS]], np.int64)[0]
# ---- (1) counts vs sizing
cells = []; strict_fail = 0; lit_fail = 0; fb_used = 0; ok1 = True; nrefrung = {}
for b in BETAS:
    fe = ref['frac_eps'][str(b)]; nr = [fe[str(d)][2] / fe[str(d)][0] for d in DECS if fe[str(d)][0] > 0]; nrefrung[b] = float(np.median(nr))
    mb = B == b; N = int(mb.sum()); ch = np.unique(C[mb]); nch = len(ch)
    for d in DECS:
        f, eps, nref = fe[str(d)]; E = f * N; k = int(((dec == d) & mb).sum())
        per = np.array([int(((dec == d) & mb & (C == c)).sum()) for c in ch]); vobs = max(float(per.var(ddof=1)) * nch, float(k)) if nch > 1 else float(k)
        lit = (E > 0 and abs(k / E - 1) <= 0.2)
        if nref >= 100: kind = 'strict'; ok = lit
        elif nref >= 1: kind = 'poisson'; sd = math.sqrt(E * E / nref + vobs); ok = abs(k - E) <= 3 * sd
        else: kind = 'zeroref'; ok = (k / N) <= 3.0 / nrefrung[b]; sd = None
        lit_fail += (not lit)
        if kind != 'strict': fb_used += 1
        ok1 &= bool(ok); cells.append(dict(beta=b, dec=d, n_ref=nref, expected=E, observed=k, ratio=(k / E if E > 0 else None), kind=kind, literal20=bool(lit), ok=bool(ok)))
res['counts'] = dict(cells=cells, n_cells=len(cells), n_fallback_cells=fb_used, literal20_failures=int(lit_fail), ok=bool(ok1),
                     strict_max_dev=max(abs(c['ratio'] - 1) for c in cells if c['kind'] == 'strict'))
# ---- (2) halves
h = []; ok2 = True; mz = {}
for b in BETAS:
    mb = B == b; zs = []
    for d in DECS:
        fr = {}
        for nm, hm in (('A', half), ('B', ~half)):
            mm = mb & hm; ch = np.unique(C[mm]); n = int(mm.sum())
            pf = np.array([((dec == d) & mm & (C == c)).sum() / max(1, (mm & (C == c)).sum()) for c in ch]); f = float(((dec == d) & mm).sum() / n)
            v = max(float(pf.var(ddof=1)) / len(ch), f * (1 - f) / n, 1.0 / n ** 2); fr[nm] = (f, v)
        z = (fr['A'][0] - fr['B'][0]) / math.sqrt(fr['A'][1] + fr['B'][1]); zs.append(z); h.append(dict(beta=b, dec=d, fA=fr['A'][0], fB=fr['B'][0], z=float(z)))
    mz[b] = float(np.max(np.abs(zs)))
mzall = max(mz.values()); res['halves'] = dict(cells=h, max_abs_z_per_rung={str(k): v for k, v in mz.items()}, max_abs_z=mzall, ok=bool(mzall <= 3.0))
# ---- (5) recompute
import pp2_lemma as PL
p = PL.PP2.from_rbmpp_npz(PROD8); idx = np.sort(np.random.default_rng(777).choice(len(W), a.nrec, replace=False))
sig = (((W[idx][:, None] >> np.arange(64, dtype=np.uint64)[None]) & np.uint64(1)).astype(np.float64) * 2 - 1)
lb = np.real(np.asarray(PL.logB(p, sig, dtype='f64'))); dd = np.abs(lb - L[idx]); res['recompute'] = dict(n=int(a.nrec), max_abs_diff=float(dd.max()), ok=bool(dd.max() <= 1e-10), rungs_covered=sorted(set(B[idx].tolist())))
oks = dict(counts=res['counts']['ok'], halves=res['halves']['ok'], median=res['median']['ok'], Sz=res['Sz']['ok'], recompute=res['recompute']['ok'], bookkeeping=book['ok'])
res['checks'] = oks; res['PASS'] = bool(all(oks.values()))
if a.json: json.dump(res, open(a.json, 'w'), indent=1, default=float)
print('draws per rung', n_rung, 'targets', TARG, ' chains per rung', book['chains_per_rung'], ' halves tasks A/B', res['halfA_tasks'], res['halfB_tasks'], ' seeds distinct', book['seeds_distinct'], ' params sha ok', book['params_sha_ok'])
print(f"(1) counts: strict-cell max |ratio-1| {res['counts']['strict_max_dev']:.3f} (limit 0.20), fallback cells {fb_used}, literal-20% failures over all {len(cells)} cells {lit_fail}; failing cells {[(c['beta'], c['dec'], c['kind'], c['expected'], c['observed']) for c in cells if not c['ok']]}")
print(f"(2) halves: max|z| per rung {[round(mz[b], 2) for b in BETAS]} (limit 3), worst {max(h, key=lambda r: abs(r['z']))}")
print(f"(3) median 2lnB beta 1: {med:.3f} (A {medA:.3f}, B {medB:.3f}) vs -67.03, diff {med + 67.03:+.3f} (limit 0.3)")
print(f"(4) Sz: {res['Sz']['n_bad']} of {len(W)} words not at 32 bits set")
print(f"(5) recompute of {a.nrec} draws (rungs {res['recompute']['rungs_covered']}): max |dlnB| {dd.max():.2e} (limit 1e-10)")
print(f"VERDICT pool_x gate{' [BROKEN ' + a.brk + ']' if a.brk else ''}: {'PASS' if res['PASS'] else 'FAIL'} | draws {n_rung} | counts {'ok' if oks['counts'] else 'FAIL'} (strict max dev {res['counts']['strict_max_dev']:.3f}) | halves max|z| {mzall:.2f} {'ok' if oks['halves'] else 'FAIL'} | median {med:.3f} {'ok' if oks['median'] else 'FAIL'} | Sz bad {res['Sz']['n_bad']} {'ok' if oks['Sz'] else 'FAIL'} | recompute max {dd.max():.1e} {'ok' if oks['recompute'] else 'FAIL'} | bookkeeping {'ok' if book['ok'] else 'FAIL'}")
