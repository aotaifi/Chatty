"""test_z.py: pre-registered Test Z (PLAN_pfaffian.md "Test Z"), CPU only, numpy. usage: python3 test_z.py OUTDIR
All numbers: sign s1 (oracle-free) = clipped-Lanczos K1 sign of psi_P, cx_refresh.py l.30-38, reproduced in numpy float64 (same reps/tables).
Stages cached in OUTDIR: s1.npy -> checks vs s2 -> (b) -> (c).
"""
import os, sys, json, time
import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize_scalar
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bond_split as BS
import delta0 as D0M
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); T0 = time.time()
def log(m): print(f'[{time.time() - T0:7.0f}s] {m}', flush=True)
RJ = f'{OUT}/z_result.json'
res = json.load(open(RJ)) if os.path.exists(RJ) else {}
res['host'] = os.uname().nodename
def dump(): json.dump(res, open(RJ, 'w'), indent=1, default=float)
SYM = '/project/theorie/a/A.Otaifi/Chatty/runs/stall6/data/sym_tables.npz'
LASTAR1 = '/project/theorie/a/A.Otaifi/Chatty/runs/compress6/runs/veto16_17356522/la_star.npy'
RUN = '/project/theorie/a/A.Otaifi/Chatty/runs/compress6/runs/refresh_17409419'
A0 = D0M.A0; MR = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_mr'
assert json.load(open(f'{RUN}/spec.json'))['tau'] == 0.1
n = np.load(f'{BS.CSRD}/norb.npy').astype(np.float64); D = len(n); assert n.sum() == 9075135300
HW = BS.HW; JW = BS.JW; NS = BS.NS; E0 = BS.E0
t1 = np.load(f'{BS.TAB}/nb_J1.npy', mmap_mode='r'); t2 = np.load(f'{BS.TAB}/nb_J2.npy', mmap_mode='r')
z = np.load(SYM); laP = z['lP'].astype(np.float64); sP = z['sP'].astype(np.float64); del z
assert laP.shape == (D,) and set(np.unique(sP)) <= {-1.0, 1.0}
def norm_la(la): la = la - la.max(); return la - 0.5 * np.log((n * np.exp(2 * la)).sum())
S1F = f'{OUT}/s1.npy'
if not os.path.exists(S1F):
    # H0: unsigned off-diagonal H (HW per valid bond) + diagonal D0, rep basis; energy of psi = v: sum n v (D0 v + H0 v) / sum n v^2
    ip = np.zeros(D + 1, np.int64); idx = []; dat = []; D0 = np.empty(D)
    for i0 in range(0, D, 131072):
        i1 = min(D, i0 + 131072); t = np.concatenate([t1[i0:i1], t2[i0:i1]], 1); valid = t >= 0; tt = np.where(valid, t, 0)
        D0[i0:i1] = 27.0 - 0.5 * (valid * JW[None, :]).sum(1)
        idx.append(tt[valid].astype(np.int32)); dat.append(np.broadcast_to(HW[None, :], valid.shape)[valid]); ip[i0 + 1:i1 + 1] = valid.sum(1)
    H0 = sp.csr_matrix((np.concatenate(dat), np.concatenate(idx), np.cumsum(ip)), shape=(D, D)); del idx, dat; log(f'H0 built nnz {H0.nnz}')
    def Hv(v): return D0 * v + H0 @ v
    def Hsite(la, s):
        v = s * np.exp(la - la.max()); return (float((n * v * Hv(v)).sum() / (n * v * v).sum()) - E0) / NS
    aP = np.maximum(np.exp(laP - laP.max()), 1e-300); vP = sP * aP
    E = float((n * vP * Hv(vP)).sum() / (n * vP * vP).sum()); EL = Hv(vP) / vP        # E_L(x) = (H psi)_x / psi_x
    log(f'<H>(psi_P,s_P) per site {(E - E0) / NS:.6e}  (cx_lib H_START 1.3193875e-4)'); res['H_psiP_site'] = (E - E0) / NS; res['H_START_recorded'] = 1.3193875041276706e-4
    EPS = 0.3
    sg = lambda a: sP * np.where(1.0 + a * (EL - E) >= 0, 1.0, -1.0)
    lclip = lambda a: laP + np.log(np.maximum(np.abs(1.0 + a * (EL - E)), EPS))
    sd = float(np.sqrt(np.sum(n * vP ** 2 * (EL - E) ** 2) / np.sum(n * vP ** 2)))
    # cx_refresh: sd = sqrt(sum vec^2 (EL-E)^2), vec normalised (sum vec^2 = 1)
    r_ = minimize_scalar(lambda a: Hsite(lclip(a), sg(a)), bounds=(-3.0 / sd, 0.0), method='bounded', options=dict(xatol=1e-4 / sd))
    s1 = sg(float(r_.x)); res['s1_line_search'] = dict(alpha=float(r_.x), sd=sd, alpha_sd=float(r_.x) * sd, H_site_s1_clipped=float(r_.fun), nfev=int(r_.nfev))
    log(f'line search alpha*sd {float(r_.x) * sd:.4f}  <H>(clipped, s1) {r_.fun:.6e} nfev {r_.nfev}')
    np.save(S1F, s1.astype(np.int8)); del H0
    dump()
s1 = np.load(S1F).astype(np.float64)
# ---- check vs s2
s2 = np.load(f'{RUN}/s2.npy').astype(np.float64); ls1 = norm_la(np.load(LASTAR1).astype(np.float64)); laPn = norm_la(laP)
X = np.exp(ls1 - laPn) < 0.1
res['check_s2'] = dict(frac_orbits_s1_ne_s2=float((s1 != s2).mean()), frac_orbits_X=float(X.mean()), n_s1_ne_s2=int((s1 != s2).sum()), n_X=int(X.sum()),
                       mismatch_flipset_vs_X=int(((s1 != s2) != X).sum()), n_weighted_X=float((n * X).sum() / n.sum()),
                       frac_s1_ne_sP=float((s1 != sP).mean()))
log(f'check s2: {res["check_s2"]}'); dump()
K, Vm, D0 = BS.build(n, s1); log('built K,Vm for s1')
if 'psiP_s1' not in res:
    r, _ = D0M.efn_vec(K, Vm, D0, n, laP); res['psiP_s1'] = dict(first_solve_dE_site=r['dE_site'], iters=r['iters'], res=r['res']); log(f'E_FN(|psi_P|, s1) one solve {r["dE_site"]:.6e}'); dump()
# ---- (b)
lA = np.load(A0).astype(np.float64)
if 'b' not in res:
    r1, v = D0M.efn_vec(K, Vm, D0, n, lA); log(f'(b) E_FN[A,s1] dE/site {r1["dE_site"]:.6e} it {r1["iters"]} res {r1["res"]:.1e}')
    chi = np.abs(v) / np.sqrt(n); lc = np.log(np.maximum(chi, 1e-300)); del v
    r2, _ = D0M.efn_vec(K, Vm, D0, n, lc); log(f'(b) E_FN[phi\'\',s1] dE/site {r2["dE_site"]:.6e} it {r2["iters"]} res {r2["res"]:.1e}')
    res['b'] = dict(E_FN_A=r1, E_FN_phi2=r2, gain_site=r1['dE_site'] - r2['dE_site']); dump()
EP = res['b']['E_FN_A']['E']
# ---- (c) f_J with s1's blocked bonds, E' = E_FN[A,s1]
FJ = f'{OUT}/fJ.npy'
if not os.path.exists(FJ):
    fJ = np.empty(D); nbad = 0; den_min = np.inf
    for i0 in range(0, D, 65536):
        i1 = min(D, i0 + 65536); t = np.concatenate([t1[i0:i1], t2[i0:i1]], 1); valid = t >= 0; tt = np.where(valid, t, 0)
        ss = s1[i0:i1, None] * s1[tt]; blk = valid & (ss > 0); kpt = valid & (ss < 0)
        ratio = np.exp(np.clip((lA - lA.max())[tt] - (lA - lA.max())[i0:i1, None], -700, 700))
        VA = (blk * HW[None, :] * ratio).sum(1); SK = (kpt * HW[None, :] * ratio).sum(1)
        Hxx = 27.0 - 0.5 * (valid * JW[None, :]).sum(1); den = Hxx + VA - EP
        den_min = min(den_min, float(den.min())); nbad += int((den <= 0).sum() + (SK <= 0).sum())
        with np.errstate(divide='ignore', invalid='ignore'): fJ[i0:i1] = np.log(SK) - np.log(den)
    res['fJ'] = dict(den_min=den_min, n_bad=nbad, nonfinite=int((~np.isfinite(fJ)).sum()), fJ_min=float(np.nanmin(fJ)), fJ_max=float(np.nanmax(fJ))); dump(); log(f'fJ {res["fJ"]}')
    bad = np.nonzero(~np.isfinite(fJ))[0]; res['fJ']['bad_rows'] = [int(i) for i in bad]; res['fJ']['bad_n_weight'] = [float(n[i]) for i in bad]
    assert len(bad) <= 3 and den_min > 0                      # sum over kept hops empty (SK=0): f_J set to 0 there, flagged
    fJ[bad] = 0.0; res['fJ']['fJ_min_finite'] = float(fJ.min()); dump()
    np.save(FJ, fJ)
fJ = np.load(FJ)
cache = {float(k): v for k, v in res.get('scan', {}).items()}
def solve(c):
    c = round(float(c), 6)
    if c in cache: return cache[c]
    r, _ = D0M.efn_vec(K, Vm, D0, n, lA + c * fJ); cache[c] = r['dE_site']
    res.setdefault('scan', {})[str(c)] = r['dE_site']; res.setdefault('scan_meta', {})[str(c)] = dict(iters=r['iters'], res=r['res']); dump()
    log(f'c={c:.4f} dE/site {r["dE_site"]:.6e} iters {r["iters"]} res {r["res"]:.1e}'); return r['dE_site']
cache[0.0] = res['b']['E_FN_A']['dE_site']; res.setdefault('scan', {})['0.0'] = cache[0.0]
for c in (2.0, 3.0, 4.0, 5.0, 6.0): solve(c)
for it in range(8):
    b = min(cache, key=cache.get)
    if b >= 6.0: res['status'] = 'MIN AT UPPER EDGE'; dump(); break
    cs = sorted(cache); i = cs.index(b); trio = cs[max(i - 1, 0):i + 2] if i > 0 else cs[:3]
    p = np.polyfit(trio, [cache[c] for c in trio], 2)
    if p[0] <= 0: res['status'] = 'NOT CONVEX'; dump(); break
    cv = float(np.clip(-p[1] / (2 * p[0]), 0.0, 6.0)); log(f'iter {it}: trio {trio} vertex {cv:.4f}')
    if min(abs(cv - c) for c in cache) < 0.05: break
    solve(round(cv / 0.05) * 0.05 if min(abs(round(cv / 0.05) * 0.05 - c) for c in cache) > 1e-9 else cv)
cstar = min(cache, key=cache.get); res['cstar'] = cstar; res['dE_cstar_site'] = cache[cstar]; res['status'] = res.get('status', 'DONE'); dump()
log(f'c* {cstar} E_FN(c*) {cache[cstar]:.6e}  DONE')
