"""test_k.py: pre-registered Test K (PLAN_pfaffian.md "Test K"), CPU, numpy, sign s1 (pfaffian_z/s1.npy) throughout. usage: python3 test_k.py OUTDIR
Features f_J[|P|] at s1 for P in {A_s1(c1_cplx_s1), s2, s3, rpT1_pp}, each with E'_P = E_FN[P,s1]; D-norm exact normal equations over all blocked pairs (s1),
D weighted by the target. K1: R*=ln phi''-ln|A_s1|; K2: R*=ln phi_2-ln|A_1|. Stages cached in OUTDIR."""
import os, sys, json, time, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bond_split as BS
import delta0 as D0M
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); T0 = time.time()
def log(m): print(f'[{time.time() - T0:7.0f}s] {m}', flush=True)
RJ = f'{OUT}/k3_result.json'; res = json.load(open(RJ)) if os.path.exists(RJ) else {}
res['host'] = os.uname().nodename
def dump(): json.dump(res, open(RJ, 'w'), indent=1, default=float)
Z = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_z'; SC = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_screen'
CR = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_c/runs/c1_cplx_s{}/round_1/la_table.npy'
PP = '/project/theorie/a/A.Otaifi/Chatty/runs/rbmpp6/runs/rpTabPP_17164048/sym_tables_PP.npz'
PHI2PP = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r1/target_la_s1.npy'
n = np.load(f'{BS.CSRD}/norb.npy').astype(np.float64); D = len(n); assert n.sum() == 9075135300
HW = BS.HW; JW = BS.JW; RS = 65536
t1 = np.load(f'{BS.TAB}/nb_J1.npy', mmap_mode='r'); t2 = np.load(f'{BS.TAB}/nb_J2.npy', mmap_mode='r')
s1 = np.load(f'{Z}/s1.npy').astype(np.float64); assert s1.shape == (D,) and set(np.unique(s1)) <= {-1.0, 1.0}
def norm_la(la): la = la - la.max(); return la - 0.5 * np.log((n * np.exp(2 * la)).sum())
NAMES = ['A_s1', 's2', 's3', 'rpT1_pp']
LA = [np.load(CR.format(1)), np.load(CR.format(2)), np.load(CR.format(3)), np.load(PP)['lP']]
LA = [a.astype(np.float64) for a in LA]
for a in LA: assert a.shape == (D,) and np.isfinite(a).all()
K, Vm, D0 = BS.build(n, s1); log('built K,Vm (s1)')
def efn(la, tag):
    t = time.time(); r, _ = D0M.efn_vec(K, Vm, D0, n, la); log(f'{tag}: dE/site {r["dE_site"]:.6e} it {r["iters"]} res {r["res"]:.1e} {time.time()-t:.0f}s'); return r
def fJ_of(lg, EP):
    fJ = np.empty(D); nbad = 0; den_min = np.inf; l0 = lg - lg.max()
    for i0 in range(0, D, RS):
        i1 = min(D, i0 + RS); t = np.concatenate([t1[i0:i1], t2[i0:i1]], 1); valid = t >= 0; tt = np.where(valid, t, 0)
        ss = s1[i0:i1, None] * s1[tt]; blk = valid & (ss > 0); kpt = valid & (ss < 0)
        ratio = np.exp(np.clip(l0[tt] - l0[i0:i1, None], -700, 700))
        VA = (blk * HW[None, :] * ratio).sum(1); SK = (kpt * HW[None, :] * ratio).sum(1)
        Hxx = 27.0 - 0.5 * (valid * JW[None, :]).sum(1); den = Hxx + VA - EP
        den_min = min(den_min, float(den.min())); nbad += int((den <= 0).sum())
        with np.errstate(divide='ignore', invalid='ignore'): fJ[i0:i1] = np.log(SK) - np.log(den)
    bad = np.nonzero(~np.isfinite(fJ))[0]
    info = dict(den_min=den_min, n_den_le0=nbad, n_nonfinite=len(bad), n_nonfinite_weight=float(n[bad].sum()),
                fJ_max=float(np.nanmax(fJ[np.isfinite(fJ)])), fJ_min_finite=float(fJ[np.isfinite(fJ)].min()))
    assert den_min > 0 and len(bad) <= 50
    fJ[bad] = 0.0; return fJ, info

def pairs_pass(lp, Fm, ecols):
    G = np.zeros((Fm.shape[1], Fm.shape[1])); Dee = np.zeros(len(ecols)); Wt = 0.0; npair = 0
    for i0 in range(0, D, RS):
        i1 = min(D, i0 + RS); t = np.concatenate([t1[i0:i1], t2[i0:i1]], 1); valid = t >= 0; tt = np.where(valid, t, 0)
        viol = valid & (s1[i0:i1, None] * s1[tt] > 0); rr, kk = np.nonzero(viol); x = rr + i0; y = tt[rr, kk]
        w = n[x] * 0.5 * HW[kk] * np.exp(lp[x] + lp[y]); Wt += w.sum(); npair += w.size
        dF = Fm[x] - Fm[y]; G += dF.T @ (w[:, None] * dF)
        for j, e in enumerate(ecols): d = e[x] - e[y]; Dee[j] += (w * d * d).sum()
        if (i0 // RS) % 60 == 0: log(f'pass {i1}/{D}')
    return G, Dee, Wt, npair
def ls(G, idx, b, DRR):
    Gs = G[np.ix_(idx, idx)]; bb = G[idx, b]; sc = np.sqrt(np.diag(Gs)); sc[sc == 0] = 1
    c = np.linalg.lstsq(Gs / np.outer(sc, sc), bb / sc, rcond=1e-13)[0] / sc
    return c, float((c @ Gs @ c - 2 * c @ bb + DRR) / DRR)

FJ4 = np.stack([np.load(f'{OUT}/fJ_{nm}.npy') for nm in NAMES], 1)
fJz = np.load(f'{Z}/fJ.npy'); lA = LA[0]; lA1 = lA + 3.9 * fJz
EP1 = -18.136780889560566
f1, info = fJ_of(lA1, EP1); res['fJ_A1'] = info; np.save(f'{OUT}/k3_fJ_A1.npy', f1); dump(); log(f'fJ[A_1] {info}')
lphi2 = np.load(f'{SC}/ln_phi2.npy').astype(np.float64); R = lphi2 - lA1; lp = norm_la(lphi2)
perm = np.random.default_rng(0).permutation(D)
Fm = np.concatenate([f1[:, None], FJ4[:, [3, 0, 2]], R[:, None], f1[perm][:, None]], 1)   # 0 fJ[A_1], 1 rp, 2 A_s1, 3 s3, 4 R*, 5 perm
G, _, Wt, npair = pairs_pass(lp, Fm, []); np.savez(f'{OUT}/k3_gram.npz', G=G); DRR = G[4, 4]; log(f'D_RR {DRR:.6e} Wtot {Wt:.4f}')
r = res.setdefault('K3', {}); r['D_RR'] = DRR
r['A1_alone'] = dict(zip(['c', 'eps2'], [ls(G, [0], 4, DRR)[0].tolist(), ls(G, [0], 4, DRR)[1]]))
c, e2 = ls(G, [0, 1, 2, 3], 4, DRR); r['A1_plus_best3'] = dict(c=c.tolist(), eps2=e2)
r['ctl_R'] = ls(G, [4], 4, DRR)[1]; r['ctl_perm'] = ls(G, [5], 4, DRR)[1]
sc = np.sqrt(np.diag(G)[:4]); r['cos'] = (G[:4, :4] / np.outer(sc, sc)).tolist()
_, Dee, _, _ = pairs_pass(lp, Fm[:, :1], [f1 * r['A1_alone']['c'][0] - R, Fm[:, :4] @ c - R, Fm[:, 4] - R])
r['direct'] = [float(d / DRR) for d in Dee]; dump(); log(f'K3 {r}')
cache = {}; sc_ = r.setdefault('scan', {})
def solve(c):
    c = round(float(c), 4)
    if c in cache: return
    q = efn(lA1 + c * f1, f'K3 c={c}'); cache[c] = q['dE_site']; sc_[str(c)] = q['dE_site']; dump()
cache[0.0] = json.load(open(f'{SC}/screen_result.json'))['S1']['E_FN_A1']['dE_site']
for c in (1, 2, 3, 4, 5, 6): solve(c)
for it in range(10):
    b = min(cache, key=cache.get); cs = sorted(cache); i = cs.index(b)
    if i == len(cs) - 1: break
    trio = cs[max(i - 1, 0):i + 2] if i > 0 else cs[:3]; p = np.polyfit(trio, [cache[x] for x in trio], 2)
    if p[0] <= 0: break
    cv = round(float(np.clip(-p[1] / (2 * p[0]), 0, 6)) * 10) / 10
    if round(cv, 4) in cache: break
    solve(cv)
for _ in range(10):
    b = min(cache, key=cache.get); new = [round(x, 4) for x in (b - 0.1, b + 0.1) if x >= -1e-9 and round(x, 4) not in cache]
    if not new: break
    for x in new: solve(x)
b = min(cache, key=cache.get); S1r = json.load(open(f'{SC}/screen_result.json'))['S1']
r['scan_min'] = dict(c=b, dE_site=cache[b]); r['tau'] = cache[b] - S1r['E_FN_phi2']['dE_site']; r['tau_over_Delta'] = r['tau'] / S1r['Delta_site']
dump(); log(f'K3 min {r["scan_min"]} tau {r["tau"]:.4e} tau/Delta {r["tau_over_Delta"]:.3f} DONE')
