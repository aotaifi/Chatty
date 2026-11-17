"""rangelimit_features.py: NEXT-METHOD RUNG 2 (STORAGE_PROBLEM.md). Feature-head fits of u = r* - m in the M-R D-norm, 6x6, CPU, existing data.
usage: python3 rangelimit_features.py RL_DIR OUTDIR
Inputs: RL_DIR (rangelimit_6x6): p.npy (FN state of guide s2 G, amplitude per orbit, sum n p^2 = 1), rstar.npy (ln p - ln A_0), m.
Features per orbit (rep x, neighbour orbits y via nb tables, h = HW per bond, Hxx = D0 as BS.build):
  l     = ln|1 + alpha (E_L - E)|, alpha = -0.10, E_L, E of psi = s2 A_0 (Hmag, as cstar_lanczos.py); G = A_0 e^l (floor e^-60 as there)
  fJA   = ln[ sum_kept h A(y)/A(x) ] - ln[ Hxx + sum_blocked h A(y)/A(x) - E'_A ],  E'_A = E_FN[s2 A_0] = -18.13639441554879 (as l_class.py)
  fJG   = same with G and E'_G = E_FN[s2 G] = step-1 E_FN of rangelimit_A.json (-18.136834966752765)
  lnA   = ln A_0 (la_table, unshifted)
  VK    = Vs_G / K_G = sum_blocked h G(y)/G(x) / sum_kept h G(y)/G(x)  (reciprocal of the spec's K/Vs, which is +inf on 4 orbits with Vs = 0)
kept: s2_x s2_y < 0; blocked: s2_x s2_y > 0.
D-norm (mr_residual.py): blocked pairs, w = n_x 0.5 HW_k p_x p_y, every unordered pair from both ends; D(a,b) = sum w (a_x - a_y)(b_x - b_y).
eps^2 = min_c D(R - u, R - u) / D(u, u).  Fits: (i) c fJG; (ii) a l + c fJG + b; (iii) cubic polynomial in the 5 (standardised) features with constant
(56 monomials). Targets: u (unclipped) and clip(u, -4.88, 4.88). 5-fold CV: fold of an entry = fold of its row x, folds = 5 contiguous orbit-index
blocks; fit on 4 folds' Gram, eps^2_CV = sum_k D_k(res_k) / sum_k D_k(u). Break tests: R = u (0), R = 0 (1), R = c xi with xi iid normal (~1);
direct pass recomputes D(R - u) from the e-tables for fit (ii) and the controls. Step 5: E_FN of guide s2 A_0 e^{R_(ii)} (unclipped fit),
tax/gain vs eps^2 with gain = E_FN[s2 A_0] - E_FN[s2 p] (rangelimit_B.json alpha 0 solves)."""
import os, sys, json, time, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bond_split as BS
import delta0 as D0M
RL, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True); T0 = time.time()
def log(m): print(f'[{time.time() - T0:7.0f}s] {m}', flush=True)
A0 = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_c/runs/c1_cplx_s1/round_1/la_table.npy'
ALPHA = -0.1; EPA = -18.13639441554879
RA = json.load(open(f'{RL}/rangelimit_A.json')); EPG = RA['step1']['E_FN']; m = RA['m']
RB = json.load(open(f'{RL}/rangelimit_B.json'))
E_A0 = [x for x in RB['solves'] if x['chi'] == 'chi_0' and x['alpha'] == 0.0][0]['dE_site']
E_P = [x for x in RB['solves'] if x['chi'] == 'phi' and x['alpha'] == 0.0][0]['dE_site']; GAIN = E_A0 - E_P
n = np.load(f'{BS.CSRD}/norb.npy').astype(np.float64); s = np.load(BS.S2).astype(np.float64); D = len(n); assert n.sum() == 9075135300
lA = np.load(A0).astype(np.float64); p = np.load(f'{RL}/p.npy'); rs = np.load(f'{RL}/rstar.npy')
lp = np.log(p); assert abs((n * p * p).sum() - 1) < 1e-9
u = rs - m; uc = np.clip(u, -4.88, 4.88)
res = dict(host=os.uname().nodename, A0=A0, alpha=ALPHA, E_prime_A=EPA, E_prime_G=EPG, m=m, gain_site=GAIN, E_FN_A0_site=E_A0, E_FN_p_site=E_P)
RJ = f'{OUT}/features.json'
def dump(): res['sec'] = time.time() - T0; json.dump(res, open(RJ, 'w'), indent=1, default=float)
K, Vm, D0 = BS.build(n, s); log('built'); sn = np.sqrt(n)
def Hmag(b): return D0 * b + Vm @ b - (K @ (sn * b)) / sn
a = np.exp(lA - lA.max()); EL = Hmag(a) / a; w_ = n * a * a; E = float((w_ * EL).sum() / w_.sum()); x_ = 1.0 + ALPHA * (EL - E)
l = np.log(np.maximum(np.abs(x_), np.exp(-60.0))); lG = lA + l; del a, EL, w_
res['moments_E_site'] = (E - BS.E0) / BS.NS; log(f"E[A_0]/site {res['moments_E_site']:.10e}")
t1 = np.load(f'{BS.TAB}/nb_J1.npy', mmap_mode='r'); t2 = np.load(f'{BS.TAB}/nb_J2.npy', mmap_mode='r'); HW = BS.HW; JW = BS.JW; RS = 32768
FT = f'{OUT}/feat.npy'
if os.path.exists(FT): F5 = np.load(FT); log('features loaded')
else:
    F5 = np.empty((D, 5)); F5[:, 0] = l; F5[:, 3] = lA; nVs0 = 0; nbad = 0
    for i0 in range(0, D, RS):
        i1 = min(D, i0 + RS); t = np.concatenate([t1[i0:i1], t2[i0:i1]], 1); va = t >= 0; tt = np.where(va, t, 0)
        ss = s[i0:i1, None] * s[tt]; blk = va & (ss > 0); kpt = va & (ss < 0); Hxx = 27.0 - 0.5 * (va * JW[None, :]).sum(1)
        for j, (lg, Ep) in ((2, (lG, EPG)), (1, (lA, EPA))):
            ratio = np.exp(np.clip(lg[tt] - lg[i0:i1, None], -700, 700)); VA = (blk * HW * ratio).sum(1); SK = (kpt * HW * ratio).sum(1)
            den = Hxx + VA - Ep; nbad += int((den <= 0).sum() + (SK <= 0).sum())
            with np.errstate(divide='ignore', invalid='ignore'): F5[i0:i1, j] = np.log(SK) - np.log(den)
            if j == 2:
                nVs0 += int((VA == 0).sum())
                F5[i0:i1, 4] = VA / SK   # Vs/K (= 1/(K/Vs)): K/Vs is infinite on the 4 orbits with no blocked bond (first run)
        if (i0 // RS) % 100 == 0: log(f'features {i1}/{D}')
    res['feat_checks'] = dict(n_bad_den_or_sum=nbad, n_Vs_zero=nVs0); np.save(FT, F5)
FL = np.load('/scratch/a/A.Otaifi/ansatz_runs/pfaffian_l/F.npy', mmap_mode='r')
res['fJA_vs_Fnpy12_maxabs'] = float(np.nanmax(np.abs(F5[:, 1] - np.asarray(FL[:, 12]))))
fin = np.isfinite(F5).all(1); res['n_nonfinite_rows'] = int((~fin).sum())
res['feat_range'] = {nm: dict(min=float(np.nanmin(F5[:, j][np.isfinite(F5[:, j])])), max=float(np.nanmax(F5[:, j][np.isfinite(F5[:, j])])),
                              q001=float(np.quantile(F5[:, j][np.isfinite(F5[:, j])], 0.001)), q999=float(np.quantile(F5[:, j][np.isfinite(F5[:, j])], 0.999)))
                     for j, nm in enumerate(['l', 'fJA', 'fJG', 'lnA', 'VsK'])}
log(f"fJA vs F.npy col 12 max |diff| {res['fJA_vs_Fnpy12_maxabs']:.2e}; nonfinite rows {res['n_nonfinite_rows']}; ranges {res['feat_range']}"); dump()
if res['n_nonfinite_rows']: log('non-finite features: STOP (reported)'); res['status'] = 'NONFINITE'; dump(); sys.exit(3)
# standardise with p^2 weights (affects conditioning only)
wv = n * p * p; mu = (wv[:, None] * F5).sum(0); sd = np.sqrt((wv[:, None] * (F5 - mu) ** 2).sum(0)); Z5 = (F5 - mu) / sd
res['standardise'] = dict(mu=mu.tolist(), sd=sd.tolist())
MONO = [()] + [c for d in (1, 2, 3) for c in itertools.combinations_with_replacement(range(5), d)]; NM = len(MONO); assert NM == 56
def poly(Z):
    out = np.empty((Z.shape[0], NM))
    for j, c in enumerate(MONO): out[:, j] = np.prod(Z[:, list(c)], 1) if c else 1.0
    return out
xi = np.random.default_rng(4242).standard_normal(D)
# columns: 56 monomials, then raw l, fJG (for fits i, ii), then targets u, uc, xi
NC = NM + 5; IU, IUC, IXI = NM + 2, NM + 3, NM + 4
def cols(idx): Zi = Z5[idx]; return np.concatenate([poly(Zi), F5[idx][:, [0, 2]], u[idx, None], uc[idx, None], xi[idx, None]], 1)
NF = 5; fold_of = np.minimum((np.arange(D) * NF) // D, NF - 1)
Gk = np.zeros((NF, NC, NC)); Wtot = 0.0; npair = 0
for i0 in range(0, D, RS):
    i1 = min(D, i0 + RS); t = np.concatenate([t1[i0:i1], t2[i0:i1]], 1); va = t >= 0; tt = np.where(va, t, 0)
    blk = va & (s[i0:i1, None] * s[tt] > 0); rr, kk = np.nonzero(blk); x = rr + i0; y = tt[rr, kk]
    w = n[x] * 0.5 * HW[kk] * np.exp(lp[x] + lp[y]); Wtot += w.sum(); npair += w.size
    Cx = cols(np.arange(i0, i1)); dC = Cx[rr] - cols(y); fx = fold_of[i0:i1][rr]
    for f in np.unique(fx):
        mk = fx == f; A = dC[mk]; Gk[f] += A.T @ (w[mk, None] * A)
    if (i0 // RS) % 100 == 0: log(f'gram {i1}/{D}')
np.savez(f'{OUT}/gram.npz', Gk=Gk, Wtot=Wtot, npair=npair); G = Gk.sum(0)
res['gram'] = dict(Wtot=Wtot, npair=npair, D_uu=G[IU, IU], D_ucuc=G[IUC, IUC], D_xixi=G[IXI, IXI]); log(f"gram {res['gram']}"); dump()
def fit(Gm, idx, ti):
    Gs = Gm[np.ix_(idx, idx)]; bb = Gm[idx, ti]; sc = np.sqrt(np.diag(Gs)); sc[sc == 0] = 1
    return np.linalg.lstsq(Gs / np.outer(sc, sc), bb / sc, rcond=1e-12)[0] / sc
def eps2(Gm, idx, ti, c): Gs = Gm[np.ix_(idx, idx)]; bb = Gm[idx, ti]; return float((c @ Gs @ c - 2 * c @ bb + Gm[ti, ti]) / Gm[ti, ti])
FITS = {'(i) c fJG': [NM + 1], '(ii) a l + c fJG + b': [NM, NM + 1, 0], '(iii) cubic, 5 features': list(range(NM)), 'ctl random xi': [IXI], 'ctl R = u': None}
res['fits'] = {}
for tn, ti in (('u', IU), ('u_clip4.88', IUC)):
    for fn, idx in FITS.items():
        if idx is None: idx = [ti]
        c = fit(G, idx, ti); ein = eps2(G, idx, ti, c)
        num = den = 0.0
        for k in range(NF):
            Gtr = G - Gk[k]; ck = fit(Gtr, idx, ti); Gs = Gk[k][np.ix_(idx, idx)]; bb = Gk[k][idx, ti]
            num += ck @ Gs @ ck - 2 * ck @ bb + Gk[k][ti, ti]; den += Gk[k][ti, ti]
        res['fits'][f'{tn} | {fn}'] = dict(eps2_in=ein, eps2_cv=float(num / den), rho_pred_in=1 - ein, coef=c.tolist() if len(c) <= 3 else None, n_par=len(idx))
        log(f"{tn} | {fn}: eps2 in {ein:.5f} cv {num / den:.5f} coef {np.round(c, 5).tolist() if len(c) <= 3 else '(56)'}")
    res['fits'][f'{tn} | ctl R = 0'] = dict(eps2_in=1.0, eps2_cv=1.0)
dump()
# direct pass: D(e,e) from e-tables for fit (ii) (u), R=u, R=0, R = c xi
c2 = np.array(res['fits']['u | (ii) a l + c fJG + b']['coef']); R2 = c2[0] * F5[:, 0] + c2[1] * F5[:, 2] + c2[2]
cx = np.array(res['fits']['u | ctl random xi']['coef']); ecols = {'(ii) R - u': R2 - u, 'R=u: R - u': u - u, 'R=0: R - u': -u, 'R=c xi: R - u': cx[0] * xi - u}
Dee = np.zeros(len(ecols)); Duu = 0.0
for i0 in range(0, D, RS):
    i1 = min(D, i0 + RS); t = np.concatenate([t1[i0:i1], t2[i0:i1]], 1); va = t >= 0; tt = np.where(va, t, 0)
    blk = va & (s[i0:i1, None] * s[tt] > 0); rr, kk = np.nonzero(blk); x = rr + i0; y = tt[rr, kk]
    w = n[x] * 0.5 * HW[kk] * np.exp(lp[x] + lp[y]); du = u[x] - u[y]; Duu += (w * du * du).sum()
    for j, e in enumerate(ecols.values()): d = e[x] - e[y]; Dee[j] += (w * d * d).sum()
res['direct'] = {k: float(Dee[j] / Duu) for j, k in enumerate(ecols)}; res['direct_D_uu'] = Duu; log(f"direct eps2 {res['direct']} D_uu {Duu:.6e}"); dump()
# step 5: M-R law once for fit (ii)
t = time.time(); r, _ = D0M.efn_vec(K, Vm, D0, n, lA + R2); tax = r['dE_site'] - E_P
res['mr_law'] = dict(E_FN_site=r['dE_site'], res=r['res'], iters=r['iters'], tax_site=tax, gain_site=GAIN, tax_over_gain=tax / GAIN,
                     eps2_direct=res['direct']['(ii) R - u'], ratio=tax / GAIN / res['direct']['(ii) R - u'], sec=time.time() - t)
log(f"M-R (ii): {res['mr_law']}"); res['status'] = 'DONE'; dump(); log('DONE')
