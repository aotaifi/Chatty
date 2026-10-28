"""mj_split.py: pre-registered Mechanism M-J split (PLAN_pfaffian.md, board 14), CPU only.  usage: python3 mj_split.py OUTDIR
Rebuilds g_1 = hop(a_1) (cx_o2b.py hop(), numpy port: E_L = D0 + Vm a/a - K(a sn)/(a sn); alpha_V fixed), FN ground state phi of H_FN[g_1],
controls (a)-(e), then the split of e = ln|A| - ln phi into (I) + rho_A over the copies, in phi^2 bands."""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bond_split as B
from scipy.sparse.linalg import LinearOperator, lobpcg
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
log = B.log; R = {}
RUN = '/project/theorie/a/A.Otaifi/Chatty/runs/compress6/runs/o2b_17431886'
CP = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_c/runs'
COPIES = {f'c1_cplx_s{k}': f'{CP}/c1_cplx_s{k}/round_1/la_table.npy' for k in (1, 2, 3)}
ALPHA = -0.11787844532393324; E0 = B.E0; NS = 36
n = np.load(f'{B.CSRD}/norb.npy').astype(np.float64); s = np.load(B.S2).astype(np.float64); sn = np.sqrt(n); D = len(n)
K, Vm, D0 = B.build(n, s)


def norm_la(la):
    m = la.max(); a = np.exp(la - m); return la - m - 0.5 * np.log((n * a * a).sum())


# ---- hop (cx_o2b.hop): g = norm_la(lb + log|1 + alpha (E_L - E_b)|)
a1 = np.load(f'{RUN}/round_0/la_table.npy'); lb = norm_la(a1); ab = np.exp(lb)
V = (Vm @ ab) / ab; W = (K @ (ab * sn)) / (ab * sn); EL = D0 + V - W
Eb = float((n * ab * ab * EL).sum()); del V, W
log(f'E_base {Eb!r} (cx_o2b.json: -18.135326930039174)'); R['E_base'] = Eb
g = norm_la(lb + np.log(np.maximum(np.abs(1.0 + ALPHA * (EL - Eb)), 1e-30))); del EL, ab, a1
ag = B.guide(g, n)                                  # floored 1e-15, sum n a^2 = 1 (as fn_solve)
Dg = D0 + (Vm @ ag) / ag


def solve(Dg, tol=1e-11, maxit=400):
    N_ = D
    A = LinearOperator((N_, N_), matvec=lambda x: Dg * np.ravel(x) - K @ np.ravel(x), matmat=lambda X: Dg[:, None] * X - K @ X, dtype=np.float64)
    w = ag * sn; x = (w / np.linalg.norm(w))[:, None]
    th = float(x[:, 0] @ (Dg * x[:, 0] - K @ x[:, 0])); den = 1.0 / np.maximum(Dg - th, 1.0)
    M = LinearOperator((N_, N_), matvec=lambda r: den * np.ravel(r), matmat=lambda Rr: den[:, None] * Rr, dtype=np.float64)
    val, vec, hist = lobpcg(A, x, M=M, largest=False, tol=tol, maxiter=maxit, retResidualNormsHistory=True)
    return float(val[0]), vec[:, 0], len(hist), float(np.ravel(hist[-1])[0])


PH = f'{OUT}/phi_u.npy'
if os.path.exists(PH):
    z = np.load(PH); E = float(z[0]); u = z[1:]; it = res = -1; log('phi loaded from disk')
else:
    E, u, it, res = solve(Dg); np.save(PH, np.concatenate([[E], u]))
u = np.abs(u); phi = u / sn; lphi = np.log(np.maximum(phi, 1e-300)); lphi = norm_la(lphi)   # sum n phi^2 = 1
dE = (E - E0) / NS
R['ctl_a'] = dict(dE_site=dE, ref=1.411883720998159e-05, rel=abs(dE / 1.411883720998159e-05 - 1), it=it, res=res, E=E)
log(f'(a) {R["ctl_a"]}')
tl = norm_la(np.load(f'{RUN}/round_1/target_la.npy')); m = lphi > 0.5 * np.log(1e-20)
d = lphi - tl; w2 = n * np.exp(2 * lphi)
R['ctl_b'] = dict(rms_w=float(np.sqrt((w2 * d * d).sum() / w2.sum())), max_abs=float(np.abs(d[m]).max()), frac_orb_phi2_gt_1e20=float(m.mean()),
                  rms_w_target_floor1e15_note='target_la floored at 1e-15 before its norm')
log(f'(b) {R["ctl_b"]}')

ph = np.exp(lphi); uph = ph * sn
Ku = K @ uph
def rho_of(uA, Dg_, E_, lA):
    with np.errstate(divide='ignore', invalid='ignore'):
        return lA - np.log((K @ uA) / sn / (Dg_ - E_))
rho_phi = rho_of(uph, Dg, E, lphi)
dec = np.floor(2 * lphi / np.log(10)).astype(int)
ok = np.isfinite(rho_phi)
R['n_no_kept'] = int((~ok).sum())
R['ctl_c'] = {int(k): float(np.abs(rho_phi[ok & (dec == k)]).max()) for k in np.unique(dec[ok])}
R['ctl_c_max_all'] = float(np.abs(rho_phi[ok]).max())
log(f'(c) rho_phi max per phi2 decade {R["ctl_c"]}; no-kept orbits {R["n_no_kept"]}')
rho_b = rho_of(uph, D0, E, lphi)
R['ctl_d'] = dict(max_abs=float(np.abs(rho_b[np.isfinite(rho_b)]).max()), rms_w=float(np.sqrt((w2 * np.nan_to_num(rho_b) ** 2).sum() / w2.sum())),
                  median_abs=float(np.median(np.abs(rho_b[np.isfinite(rho_b)]))))
log(f'(d) {R["ctl_d"]}')
json.dump(R, open(f'{OUT}/controls.json', 'w'), indent=1)

# ---- per-copy c_x (blocked pairs, small side = smaller phi), all copies in one pass over the tables
LC = {t: np.load(p) for t, p in COPIES.items()}
refs = [r for r in LC if True]
cx = {t: np.zeros(D) for t in LC}
t1 = np.load(f'{B.TAB}/nb_J1.npy', mmap_mode='r'); t2 = np.load(f'{B.TAB}/nb_J2.npy', mmap_mode='r'); RS = 65536
for i0 in range(0, D, RS):
    i1 = min(D, i0 + RS)
    t = np.concatenate([t1[i0:i1], t2[i0:i1]], 1); valid = t >= 0; tt = np.where(valid, t, 0)
    viol = valid & (s[i0:i1, None] * s[tt] > 0); rr, kk = np.nonzero(viol); x = rr + i0; y = tt[rr, kk]; H = B.HW[kk]
    lpx, lpy = lphi[x], lphi[y]; sw = lpx > lpy
    ls_, ll_ = np.where(sw, lpy, lpx), np.where(sw, lpx, lpy); xs = np.where(sw, y, x)
    lrs = np.clip(ls_ - ll_, -300, 0); rs = np.exp(lrs)
    for tg, la in LC.items():
        cs_ = np.where(sw, la[y], la[x]); cl_ = np.where(sw, la[x], la[y])
        r = np.exp(np.clip(cs_ - cl_, -300, 300))
        c = H * np.exp(2 * ll_) * (r - rs) ** 2 / r * n[x] * 0.5
        cx[tg] += np.bincount(xs, weights=c, minlength=D)
    if (i0 // RS) % 40 == 0: log(f'cost pass {i1}/{D}')
log('cost pass done')

bands = [('<1e-11', -np.inf, 1e-11), ('1e-11..1e-9', 1e-11, 1e-9), ('>=1e-9', 1e-9, np.inf)]
p2 = np.exp(2 * lphi)
out = {}
for tg, la in LC.items():
    lA = norm_la(la); uA = np.exp(lA) * sn
    e = lA - lphi
    with np.errstate(divide='ignore', invalid='ignore'):
        rho = rho_of(uA, Dg, E, lA)
        I = np.log((K @ (uA)) / Ku)                  # (I): ln sum_kept w_xz e^{e_z} with e^{e_z} phi_z = A_z
    ident = e - I - rho; okk = np.isfinite(ident)
    c = cx[tg]; ct = c[okk].sum(); rec = dict(identity_maxabs=float(np.abs(ident[okk]).max()), identity_rms=float(np.sqrt((ident[okk] ** 2).mean())),
                                               n_excluded=int((~okk).sum()), c_total=float(ct), bands={})
    for nm, lo, hi in bands:
        mk = okk & (p2 >= lo) & (p2 < hi); cc = c[mk]; cs = cc.sum()
        rec['bands'][nm] = dict(share_c=float(cs / ct), n_orb=int(mk.sum()), s_II=float((cc * rho[mk] ** 2).sum() / (cc * e[mk] ** 2).sum()),
                                s_I=float((cc * I[mk] ** 2).sum() / (cc * e[mk] ** 2).sum()),
                                mean_e=float((cc * e[mk]).sum() / cs), mean_I=float((cc * I[mk]).sum() / cs), mean_rho=float((cc * rho[mk]).sum() / cs))
    out[tg] = rec; log(f'{tg}: {json.dumps(rec)}')
R['copies'] = out
json.dump(R, open(f'{OUT}/mj_split.json', 'w'), indent=1); log('DONE')
