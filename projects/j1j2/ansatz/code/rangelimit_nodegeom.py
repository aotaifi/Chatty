"""rangelimit_nodegeom.py: NODE-GEOMETRY CHECK (STORAGE_PROBLEM.md, rung 2, CPU, exact least squares, no training).
usage: python3 rangelimit_nodegeom.py RL_DIR OUTDIR
Target v = u - l on round 1 of rangelimit_6x6 (u = rstar - m, l = ln|1 + alpha (E_L - E)| with alpha = -0.10 = column 0 of RL_DIR/features/feat.npy),
metric = the blocked-pair D-norm with p as weights of rangelimit_features.py: w = n_x 0.5 HW_k p_x p_y over blocked pairs (s2_x s2_y > 0),
every unordered pair from both ends, D(a,b) = sum w (a_x - a_y)(b_x - b_y).
Features (sign s2 ONLY, no amplitudes), per orbit rep x with the neighbour tables nb_J1/nb_J2 (entry >= 0 = an off-diagonal element, i.e. an
antiparallel bond; blocked = element with s_x s_y > 0, as in bond_split.build / rangelimit_features):
  nb1, nb2 = number of blocked J1 / J2 elements at x;  c1, c2 = number of off-diagonal J1 / J2 elements (connectivity)
  mnb1, mnb2, mc1, mc2 = the same four counts averaged (unweighted) over the off-diagonal neighbours y of x (kept and blocked, J1 and J2)
Polynomial columns: the 8 base features (standardised with the p^2 n weights), their squares and pairwise products (8 + 36 = 44); constant is
irrelevant in D (it cancels in every difference).  Wrong-sign controls: the same 44 columns from s_wrong = s2 flipped on a random half of the orbits.
Fits are exact: least squares on the pair Gram matrix (accumulated once, split by (fold of x, fold of y)); a direct pass recomputes D(res,res).
CV: 5 folds = 5 contiguous orbit-index blocks of row x (as rangelimit_features.py): train Gram = total - Gram of pairs whose row x is in the fold,
test = pairs whose row x is in the fold ('cv'; the reverse edge of such a pair has its row in another fold, so reverse edges cross folds).
'cv_strict': train on pairs with both ends outside the fold, test on pairs with both ends inside it (no pair or reverse edge shared)."""
import os, sys, json, time, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bond_split as BS
RL, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True); T0 = time.time()
def log(m): print(f'[{time.time() - T0:7.0f}s] {m}', flush=True)
RES = dict(host=os.uname().nodename, RL=RL, script=os.path.abspath(__file__))
RJ = f'{OUT}/node_geom.json'
def dump(): RES['sec'] = time.time() - T0; json.dump(RES, open(RJ, 'w'), indent=1, default=float)
RA = json.load(open(f'{RL}/rangelimit_A.json')); m = RA['m']
n = np.load(f'{BS.CSRD}/norb.npy').astype(np.float64); s = np.load(BS.S2).astype(np.int8); D = len(n); assert n.sum() == 9075135300
p = np.load(f'{RL}/p.npy'); lp = np.log(p); assert abs((n * p * p).sum() - 1) < 1e-9
rs = np.load(f'{RL}/rstar.npy'); u = rs - m
F5 = np.load(f'{RL}/features/feat.npy', mmap_mode='r'); l = np.array(F5[:, 0]); fJG = np.array(F5[:, 2]); del F5
v = u - l; RES['m'] = m; log(f'loaded D={D}; s values {np.unique(s).tolist()}')
t1 = np.load(f'{BS.TAB}/nb_J1.npy', mmap_mode='r'); t2 = np.load(f'{BS.TAB}/nb_J2.npy', mmap_mode='r'); HW = BS.HW; RS = 16384
rng = np.random.default_rng(20261117); flip = rng.permutation(D) < D // 2; sw = np.where(flip, -s, s).astype(np.int8)
RES['wrong_sign'] = dict(seed=20261117, n_flipped=int(flip.sum()), D=D)
def tab(i0, i1):
    t = np.concatenate([t1[i0:i1], t2[i0:i1]], 1); va = t >= 0; return va, np.where(va, t, 0)
# ---- pass 1: counts at x (cnt[:, 0:4] = nb1, nb2, c1, c2) for the right and the wrong sign
cnt = [np.zeros((D, 4), np.int16), np.zeros((D, 4), np.int16)]
for i0 in range(0, D, RS):
    i1 = min(D, i0 + RS); va, tt = tab(i0, i1)
    for q, sg in enumerate((s, sw)):
        blk = va & (sg[i0:i1, None].astype(np.int16) * sg[tt] > 0)
        c = cnt[q]; c[i0:i1, 0] = blk[:, :72].sum(1); c[i0:i1, 1] = blk[:, 72:].sum(1); c[i0:i1, 2] = va[:, :72].sum(1); c[i0:i1, 3] = va[:, 72:].sum(1)
log('pass 1 done')
# ---- pass 2: neighbour means (over all off-diagonal neighbours)
FB = [np.zeros((D, 8)), np.zeros((D, 8))]; nzero = 0
for i0 in range(0, D, RS):
    i1 = min(D, i0 + RS); va, tt = tab(i0, i1); nv = va.sum(1); nzero += int((nv == 0).sum()); den = np.maximum(nv, 1)
    for q in (0, 1):
        FB[q][i0:i1, :4] = cnt[q][i0:i1]
        for j in range(4): FB[q][i0:i1, 4 + j] = (va * cnt[q][tt, j]).sum(1) / den
log(f'pass 2 done; rows with no off-diagonal neighbour {nzero}')
RES['rows_no_neighbour'] = nzero
# ---- brute-force check of the counts and neighbour means on random orbits (explicit loops over the table rows)
rb = np.random.default_rng(1); sample = rb.choice(D, 1500, replace=False); bad = 0; badm = 0
for q, sg in enumerate((s, sw)):
    for i in sample:
        row = np.concatenate([t1[i], t2[i]]); k = np.zeros(4)
        for j, y in enumerate(row):
            if y >= 0:
                if int(sg[i]) * int(sg[y]) > 0: k[0 if j < 72 else 1] += 1
                k[2 if j < 72 else 3] += 1
        bad += int(not np.array_equal(k, FB[q][i, :4]))
    for i in sample[:200]:
        row = np.concatenate([t1[i], t2[i]]); ys = row[row >= 0]; mm = np.zeros(4)
        for y in ys:
            r = np.concatenate([t1[y], t2[y]]); mm += [sum(1 for j in range(72) if r[j] >= 0 and int(sg[y]) * int(sg[r[j]]) > 0), sum(1 for j in range(72, 144) if r[j] >= 0 and int(sg[y]) * int(sg[r[j]]) > 0),
                                                        (r[:72] >= 0).sum(), (r[72:] >= 0).sum()]
        badm += int(not np.allclose(mm / max(len(ys), 1), FB[q][i, 4:]))
RES['brute_check'] = dict(n_rows=len(sample), n_bad_counts=bad, n_bad_nbrmeans=badm, n_nbr_rows=200); log(f"brute-force check {RES['brute_check']}")
assert bad == 0 and badm == 0
RES['changed_rows_wrong_nb'] = int((cnt[0][:, :2] != cnt[1][:, :2]).any(1).sum())
# standardise with the p^2 weights (conditioning only)
wv = n * p * p; NAMES = ['nb1', 'nb2', 'c1', 'c2', 'mnb1', 'mnb2', 'mc1', 'mc2']; Zs = []; STD = []
for q in (0, 1):
    mu = (wv[:, None] * FB[q]).sum(0); sd = np.sqrt((wv[:, None] * (FB[q] - mu) ** 2).sum(0)); assert (sd > 0).all()
    Zs.append((FB[q] - mu) / sd); STD.append(dict(mu=mu.tolist(), sd=sd.tolist()))
RES['standardise'] = dict(right=STD[0], wrong=STD[1], names=NAMES)
MONO = [(j,) for j in range(8)] + list(itertools.combinations_with_replacement(range(8), 2)); NMO = len(MONO); assert NMO == 44
IR, IW = 0, NMO; IFJ = 2 * NMO; IV = IFJ + 1; IU = IFJ + 2; IXI = IFJ + 3; IR8 = IFJ + 4; NC = IR8 + 8
xi = np.random.default_rng(4242).standard_normal(D); r8 = np.random.default_rng(777).standard_normal((D, 8))
C = np.empty((D, NC))
for i0 in range(0, D, 1 << 20):
    i1 = min(D, i0 + (1 << 20))
    for q in (0, 1):
        for j, mo in enumerate(MONO): C[i0:i1, q * NMO + j] = np.prod(Zs[q][i0:i1][:, list(mo)], 1)
    C[i0:i1, IFJ] = fJG[i0:i1]; C[i0:i1, IV] = v[i0:i1]; C[i0:i1, IU] = u[i0:i1]; C[i0:i1, IXI] = xi[i0:i1]; C[i0:i1, IR8:] = r8[i0:i1]
del r8, Zs
log(f'column matrix {C.shape} built')
NF = 5; fold_of = np.minimum((np.arange(D) * NF) // D, NF - 1).astype(np.int8)
GF = f'{OUT}/gram_nodegeom.npz'; Gkk = np.zeros((NF, NF, NC, NC)); Wtot = 0.0; npair = 0
CACHED = os.path.exists(GF)
if CACHED: z = np.load(GF); Gkk = z['Gkk']; Wtot = float(z['Wtot']); npair = int(z['npair']); log('Gram loaded from cache (first run crashed after saving it, in the fit table)')
for i0 in range(0, 0 if CACHED else D, RS):
    i1 = min(D, i0 + RS); va, tt = tab(i0, i1)
    blk = va & (s[i0:i1, None].astype(np.int16) * s[tt] > 0); rr, kk = np.nonzero(blk); x = rr + i0; y = tt[rr, kk]
    w = n[x] * 0.5 * HW[kk] * np.exp(lp[x] + lp[y]); Wtot += w.sum(); npair += w.size
    key = fold_of[x].astype(np.int64) * NF + fold_of[y]; o = np.argsort(key, kind='stable'); key = key[o]; x = x[o]; y = y[o]; w = w[o]
    b = np.searchsorted(key, np.arange(NF * NF + 1))
    for g in range(NF * NF):
        if b[g + 1] > b[g]:
            xs, ys, ws = x[b[g]:b[g + 1]], y[b[g]:b[g + 1]], w[b[g]:b[g + 1]]; A = C[xs] - C[ys]; Gkk[g // NF, g % NF] += A.T @ (ws[:, None] * A)
    if (i0 // RS) % 100 == 0: log(f'gram {i1}/{D}')
if not CACHED: np.savez(GF, Gkk=Gkk, Wtot=Wtot, npair=npair)
G = Gkk.sum((0, 1)); Gk = Gkk.sum(1)
RES['gram'] = dict(Wtot=Wtot, npair=npair, D_vv=G[IV, IV], D_uu=G[IU, IU], D_xixi=G[IXI, IXI], D_vv_over_D_uu=G[IV, IV] / G[IU, IU],
                   prev_features_gram_Wtot=1.5306440446418883, prev_features_gram_npair=512644873, prev_D_uu=0.00183506350542411)
log(f"gram {RES['gram']}"); dump()
assert npair == 512644873 and abs(Wtot - 1.5306440446418883) < 1e-9 and abs(G[IU, IU] / 0.00183506350542411 - 1) < 1e-9, 'Gram does not reproduce rangelimit_features gram.npz'
def fit(Gm, idx, ti):
    Gs = Gm[np.ix_(idx, idx)]; bb = Gm[idx, ti]; sc = np.sqrt(np.diag(Gs)); sc[sc == 0] = 1; M = Gs / np.outer(sc, sc)
    sol = np.linalg.lstsq(M, bb / sc, rcond=1e-12)[0] / sc; return sol, np.linalg.matrix_rank(M, tol=1e-12), np.linalg.cond(M)
def res2(Gm, idx, ti, c): Gs = Gm[np.ix_(idx, idx)]; bb = Gm[idx, ti]; return float(c @ Gs @ c - 2 * c @ bb + Gm[ti, ti])
def sub(q, bases, deg): return [q * NMO + j for j, mo in enumerate(MONO) if len(mo) <= deg and all(i in bases for i in mo)]
X4 = [0, 1, 2, 3]; ALL8 = list(range(8))
FITS = {'ref: fJG alone': [IFJ],
        '(i-b) x blocked counts nb1,nb2 linear': sub(0, [0, 1], 1), '(i-bq) nb1,nb2 + squares, product': sub(0, [0, 1], 2),
        '(i) x-counts linear (nb1,nb2,c1,c2)': sub(0, X4, 1), '(i-q) x-counts quadratic': sub(0, X4, 2),
        '(ii) x + neighbour counts linear (8)': sub(0, ALL8, 1), '(iii) all quadratic (44)': sub(0, ALL8, 2),
        '(ii)+fJG': sub(0, ALL8, 1) + [IFJ], '(iv) (iii)+fJG': sub(0, ALL8, 2) + [IFJ],
        'wrong-sign (i) linear': sub(1, X4, 1), 'wrong-sign (ii) linear': sub(1, ALL8, 1), 'wrong-sign (iii) quadratic': sub(1, ALL8, 2),
        'ctl random xi': [IXI], 'ctl 8 random columns': list(range(IR8, IR8 + 8)), 'ctl R = v': [IV]}
RES['fits'] = {}; COEF = {}
for fn, idx in FITS.items():
    c, rk, cond = fit(G, idx, IV); ein = res2(G, idx, IV, c) / G[IV, IV]; num = den = 0.0; ns = ds = 0.0
    for k in range(NF):
        ck = fit(G - Gk[k], idx, IV)[0]; num += res2(Gk[k], idx, IV, ck); den += Gk[k][IV, IV]
        mk = [a for a in range(NF) if a != k]; Gtr = Gkk[np.ix_(mk, mk)].sum((0, 1)); cs = fit(Gtr, idx, IV)[0]
        ns += res2(Gkk[k, k], idx, IV, cs); ds += Gkk[k, k][IV, IV]
    COEF[fn] = (idx, c); ent = dict(n_par=len(idx), rank=int(rk), cond=float(cond), resid_in=ein, resid_cv=num / den, resid_cv_strict=ns / ds, frac_D_vv_strict=float(ds / G[IV, IV]))
    lin = len(idx) <= 8 and all(j >= IFJ or len(MONO[j % NMO]) == 1 for j in idx) and idx != [IFJ] and idx[0] < IFJ
    if lin:
        q = 0 if idx[0] < NMO else 1; ent['coef_per_raw_count'] = {NAMES[MONO[j - q * NMO][0]]: float(c[a] / STD[q]['sd'][MONO[j - q * NMO][0]]) for a, j in enumerate(idx) if j < IFJ}
        ent['coef_unit'] = 'D(v) minimising coefficient per unit of the raw count (constant not determined by D)'
    elif len(idx) <= 8: ent['coef'] = c.tolist()
    if fn.startswith('(iv)') or fn.startswith('(ii)+fJG'): ent['coef_fJG'] = float(c[idx.index(IFJ)])
    RES['fits'][fn] = ent; log(f"{fn}: in {ein:.5f} cv {num / den:.5f} cv_strict {ns / ds:.5f} rank {rk}/{len(idx)} cond {cond:.2e} {ent.get('coef_per_raw_count', '')}")
RES['coef_quad_Zunits'] = {fn: dict(cols=COEF[fn][0], c=COEF[fn][1].tolist()) for fn in ('(iii) all quadratic (44)', '(iv) (iii)+fJG')}; dump()
# ---- direct pass: D(R - v, R - v) from the fitted row values for the main fits and the controls
DF = ['(i) x-counts linear (nb1,nb2,c1,c2)', '(ii) x + neighbour counts linear (8)', '(iii) all quadratic (44)', '(iv) (iii)+fJG', 'wrong-sign (iii) quadratic', 'ctl random xi', 'ctl R = v']
Rf = np.empty((D, len(DF)))
for a, fn in enumerate(DF): idx, c = COEF[fn]; Rf[:, a] = C[:, idx] @ c
Rf = Rf - v[:, None]; Dee = np.zeros(len(DF)); Dvv = 0.0
for i0 in range(0, D, RS):
    i1 = min(D, i0 + RS); va, tt = tab(i0, i1); blk = va & (s[i0:i1, None].astype(np.int16) * s[tt] > 0); rr, kk = np.nonzero(blk); x = rr + i0; y = tt[rr, kk]
    w = n[x] * 0.5 * HW[kk] * np.exp(lp[x] + lp[y]); dv = v[x] - v[y]; Dvv += (w * dv * dv).sum(); d = Rf[x] - Rf[y]; Dee += (w[:, None] * d * d).sum(0)
RES['direct'] = {fn: float(Dee[a] / Dvv) for a, fn in enumerate(DF)}; RES['direct_D_vv'] = Dvv; log(f"direct {RES['direct']} D_vv {Dvv:.6e}")
RES['status'] = 'DONE'; dump(); log('DONE')
