"""l_class.py: pre-registered Test L (PLAN_pfaffian.md "Test L"), CPU only.  usage: python3 l_class.py OUTDIR [stageA|all]
R* = ln phi' - ln|A_s1| regressed in the D-norm on nested symmetric feature sets (per orbit rep); exact normal equations over all
blocked pairs (weights as mr_residual.py stage 2); then exact E_FN of A_s1 exp(sum c f) for the best set and of the pure Jacobi copy A_s1 exp(f_J).
Columns of F: 0-8 L1 classes, 9 nbJ1, 10 nbJ2, 11 Hxx, 12 f_J, 13 f_J permuted (broken control), 14 R*.
"""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bond_split as BS

OUT = sys.argv[1]; MODE = sys.argv[2] if len(sys.argv) > 2 else 'all'
os.makedirs(OUT, exist_ok=True); T0 = time.time()
def log(m): print(f'[{time.time() - T0:7.0f}s] {m}', flush=True)

A0 = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_c/runs/c1_cplx_s1/round_1/la_table.npy'
MR = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_mr'
REPS = '/project/theorie/a/A.Otaifi/Chatty/runs/ll6/data/psi0_6x6_table.npz'
HW = BS.HW; JW = BS.JW; RS = 65536
st1 = json.load(open(f'{MR}/stage1.json')); EP = st1['E_FN_A']['E']; E_PHI = st1['E_FN_phi']['dE_site']; GAIN = st1['gain']
assert abs(EP - (-18.13639441554879)) < 1e-12
n = np.load(f'{BS.CSRD}/norb.npy').astype(np.float64); s = np.load(BS.S2).astype(np.float64); D = len(n)
assert n.sum() == 9075135300 and set(np.unique(s)) <= {-1.0, 1.0}
t1 = np.load(f'{BS.TAB}/nb_J1.npy', mmap_mode='r'); t2 = np.load(f'{BS.TAB}/nb_J2.npy', mmap_mode='r')
lA = np.load(A0).astype(np.float64); lphi = np.load(f'{MR}/lphi.npy'); Rstar = lphi - lA
reps = np.load(REPS)['reps']; assert reps.shape == (D,) and np.all(reps[1:] > reps[:-1])
res = dict(host=os.uname().nodename, A0=A0, E_prime=EP, E_PHI_site=E_PHI, gain=GAIN)
def dump(): res['sec'] = time.time() - T0; json.dump(res, open(f'{OUT}/l_result.json', 'w'), indent=1, default=float)

# ---------------- L1 classes
L = 6
CLS = [(1, 0), (1, 1), (2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2), (3, 3)]
site = lambda x, y: (x % L) + L * (y % L)
cls_perms = {c: [] for c in CLS}
for a in range(L):
    for b in range(L):
        if a == b == 0: continue
        ka, kb = min(a, L - a), min(b, L - b); key = (max(ka, kb), min(ka, kb)); assert key in cls_perms
        cls_perms[key].append(np.array([site(x + a, y + b) for y in range(L) for x in range(L)]))
CLS_N = [len(cls_perms[c]) for c in CLS]; log(f'class sizes {CLS_N} (sum {sum(CLS_N)})')
SH = np.arange(36, dtype=np.uint64)
def bits(r): return ((r[:, None] >> SH[None, :]) & np.uint64(1)).astype(np.float32) * 2 - 1
def l1_feats(S):
    out = np.empty((S.shape[0], 9))
    for j, c in enumerate(CLS):
        acc = np.zeros(S.shape[0])
        for p in cls_perms[c]: acc += (S * S[:, p]).sum(1)
        out[:, j] = acc / (36.0 * len(cls_perms[c]))
    return out

FF = f'{OUT}/F.npy'
if os.path.exists(FF) and os.path.exists(f'{OUT}/stageA.json'):
    F = np.load(FF); res['stageA'] = json.load(open(f'{OUT}/stageA.json')); log('F loaded')
else:
    # orbit invariance check of L1 on random reps under all 576 group elements (lat.G from rbmpp_lib convention rebuilt here)
    rng = np.random.default_rng(3); idx = rng.integers(0, D, 40); S0 = bits(reps[idx]); f0 = l1_feats(S0); maxdev = 0.0
    for p in [np.array([site(-y, x) for y in range(L) for x in range(L)]), np.array([site(-x, y) for y in range(L) for x in range(L)]),
              np.array([site(x + 2, y + 1) for y in range(L) for x in range(L)])]:
        maxdev = max(maxdev, float(np.abs(l1_feats(S0[:, p]) - f0).max()))
    maxdev = max(maxdev, float(np.abs(l1_feats(-S0) - f0).max())); log(f'L1 orbit-invariance max dev {maxdev:.2e}'); assert maxdev < 1e-12
    F = np.empty((D, 15)); a_la = lA - lA.max(); den_min = np.inf; den_arg = -1; nbad = 0
    for i0 in range(0, D, RS):
        i1 = min(D, i0 + RS)
        F[i0:i1, :9] = l1_feats(bits(reps[i0:i1]))
        t = np.concatenate([t1[i0:i1], t2[i0:i1]], 1); valid = t >= 0; tt = np.where(valid, t, 0)
        ss = s[i0:i1, None] * s[tt]; blk = valid & (ss > 0); kpt = valid & (ss < 0)
        ratio = np.exp(np.clip(lA[tt] - lA[i0:i1, None], -700, 700))            # |A(y)|/|A(x)|
        VA = (blk * HW[None, :] * ratio).sum(1); SK = (kpt * HW[None, :] * ratio).sum(1)
        Hxx = 27.0 - 0.5 * (valid * JW[None, :]).sum(1)
        den = Hxx + VA - EP
        F[i0:i1, 9] = blk[:, :72].sum(1); F[i0:i1, 10] = blk[:, 72:].sum(1); F[i0:i1, 11] = Hxx
        m = den.min()
        if m < den_min: den_min = float(m); den_arg = int(i0 + den.argmin())
        nbad += int((den <= 0).sum() + (SK <= 0).sum())
        with np.errstate(divide='ignore', invalid='ignore'): F[i0:i1, 12] = np.log(SK) - np.log(den)
        F[i0:i1, 14] = Rstar[i0:i1]
        if (i0 // RS) % 40 == 0: log(f'features {i1}/{D}')
    perm = np.random.default_rng(0).permutation(D); F[:, 13] = F[perm, 12]
    A_ = dict(den_min=den_min, den_min_row=den_arg, n_bad=nbad, orbit_invariance_dev=maxdev, class_sizes=CLS_N,
              fJ_min=float(np.nanmin(F[:, 12])), fJ_max=float(np.nanmax(F[:, 12])), fJ_nonfinite=int((~np.isfinite(F[:, 12])).sum()),
              Rstar_min=float(Rstar.min()), Rstar_max=float(Rstar.max()))
    res['stageA'] = A_; log(f'stageA {A_}')
    if nbad or A_['fJ_nonfinite']: dump(); log('denominator/sum not positive: STOP (reported)'); json.dump(A_, open(f'{OUT}/stageA.json', 'w')); sys.exit(3)
    np.save(FF, F); json.dump(A_, open(f'{OUT}/stageA.json', 'w'), indent=1)
dump()
if MODE == 'stageA': sys.exit(0)

# ---------------- Gram over all blocked pairs
lp = lphi - lphi.max(); lp = lp - 0.5 * np.log((n * np.exp(2 * lp)).sum())
def pairs_pass(ecols, Fm):
    """ecols: list of (D,) vectors e: returns Gram G (15x15) of F columns and Dee for each e."""
    G = np.zeros((15, 15)); Dee = np.zeros(len(ecols)); Wtot = 0.0; npair = 0
    for i0 in range(0, D, RS):
        i1 = min(D, i0 + RS)
        t = np.concatenate([t1[i0:i1], t2[i0:i1]], 1); valid = t >= 0; tt = np.where(valid, t, 0)
        viol = valid & (s[i0:i1, None] * s[tt] > 0); rr, kk = np.nonzero(viol); x = rr + i0; y = tt[rr, kk]
        w = n[x] * 0.5 * HW[kk] * np.exp(lp[x] + lp[y]); Wtot += w.sum(); npair += w.size
        if Fm is not None:
            dF = Fm[x] - Fm[y]; G += dF.T @ (w[:, None] * dF)
        for j, e in enumerate(ecols):
            d = e[x] - e[y]; Dee[j] += (w * d * d).sum()
        if (i0 // RS) % 40 == 0: log(f'pass {i1}/{D}')
    return G, Dee, Wtot, npair

GF = f'{OUT}/gram.npz'
if os.path.exists(GF):
    z = np.load(GF); G = z['G']; res['gram'] = dict(Wtot=float(z['Wtot']), npair=int(z['npair']))
else:
    G, _, Wtot, npair = pairs_pass([], F); np.savez(GF, G=G, Wtot=Wtot, npair=npair); res['gram'] = dict(Wtot=Wtot, npair=npair)
DRR = G[14, 14]; res['D_RR'] = DRR; log(f"D_RR {DRR:.6e} (stage2 1.7615e-4), Wtot {res['gram']['Wtot']:.6f}"); dump()
assert abs(DRR / 1.7614851526084917e-4 - 1) < 1e-6, 'D_RR not reproduced'

SETS = {'L1': list(range(9)), 'L2': list(range(9)) + [9, 10], 'L3': list(range(9)) + [9, 10, 11], 'L4': [12],
        'L5': list(range(9)) + [9, 10, 11, 12], 'ctl_Rstar': [14], 'ctl_perm_fJ': [13]}
coef = {}; ev = {}
for k, idx in SETS.items():
    Gs = G[np.ix_(idx, idx)]; b = G[idx, 14]; sc = np.sqrt(np.diag(Gs)); sc[sc == 0] = 1
    c = np.linalg.lstsq(Gs / np.outer(sc, sc), b / sc, rcond=1e-13)[0] / sc
    eps2 = (c @ Gs @ c - 2 * c @ b + DRR) / DRR
    coef[k] = c; ev[k] = dict(idx=idx, coef=c.tolist(), eps2_gram=float(eps2), cond=float(np.linalg.cond(Gs / np.outer(sc, sc))))
    log(f'{k}: eps2 {eps2:.6f} coef {np.round(c, 5).tolist()}')
res['sets'] = ev; dump()
# direct check of D(e,e) with the e-tables (independent of the Gram algebra)
ecols = {k: F[:, SETS[k]] @ coef[k] - Rstar for k in SETS}
_, Dee, _, _ = pairs_pass(list(ecols.values()), None)
for j, k in enumerate(SETS): ev[k]['eps2_direct'] = float(Dee[j] / DRR); log(f'{k}: eps2 direct {Dee[j] / DRR:.6f}')
res['sets'] = ev; dump()
main = ['L1', 'L2', 'L3', 'L4', 'L5']
best = min(main, key=lambda k: ev[k]['eps2_direct']); res['best'] = best; dump()

# ---------------- exact E_FN
import delta0 as D0M
K, Vm, D0 = BS.build(n, s); log(f'built K nnz {K.nnz}')
def efn(la_new, tag):
    t = time.time(); r, _ = D0M.efn_vec(K, Vm, D0, n, la_new); tax = r['dE_site'] - E_PHI
    out = dict(E_FN=r, tax_site=tax, tax_over_gain=tax / GAIN, sec=time.time() - t); log(f'{tag}: {out}'); return out
res['efn_best'] = dict(set=best, eps2=ev[best]['eps2_direct'], **efn(lA + F[:, SETS[best]] @ coef[best], f'E_FN best {best}'))
res['efn_best']['ratio_tax_over_gain_to_eps2'] = res['efn_best']['tax_over_gain'] / ev[best]['eps2_direct']; dump()
res['efn_jacobi'] = efn(lA + F[:, 12], 'E_FN pure Jacobi A exp(fJ)')
res['status'] = 'DONE'; dump(); log('DONE')
