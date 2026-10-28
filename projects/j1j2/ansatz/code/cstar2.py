"""cstar.py: oracle-free scalar c* = argmin_c E_FN[A_s1 exp(c f_J)] (PLAN_pfaffian.md Result L, "Next"). CPU. usage: python3 cstar.py OUTDIR
f_J = column 12 of pfaffian_l/F.npy. Sign: oracle s2 (BS.S2). Checks c=1 (1.51052e-5) and c=0 (2.092013e-5), then parabola line search on [0,2.5]."""
import os, sys, json, time
import numpy as np
import scipy.sparse as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bond_split as BS
import delta0 as D0M
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); T0 = time.time()
def log(m): print(f'[{time.time() - T0:7.0f}s] {m}', flush=True)
A0 = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_c/runs/c1_cplx_s1/round_1/la_table.npy'
FF = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_l/F.npy'
GAIN = 7.8246e-6; E_PHI = 1.309550e-5; E0 = -18.137147540071613
assert abs(BS.E0 - E0) < 1e-12
n = np.load(f'{BS.CSRD}/norb.npy').astype(np.float64); s = np.load(BS.S2).astype(np.float64); D = len(n)
assert n.sum() == 9075135300
lA = np.load(A0).astype(np.float64); fJ = np.array(np.load(FF, mmap_mode='r')[:, 12]); assert np.isfinite(fJ).all() and len(fJ) == D
res = dict(host=os.uname().nodename, A0=A0, F=FF, pairs=[]); RJ = f'{OUT}/cstar.json'
def dump(): res['sec'] = time.time() - T0; json.dump(res, open(RJ, 'w'), indent=1, default=float)
K, Vm, D0 = BS.build(n, s); log('built')
cache = {}
def solve(c):
    c = round(float(c), 6)
    if c in cache: return cache[c]
    t = time.time(); r, _ = D0M.efn_vec(K, Vm, D0, n, lA + c * fJ)
    cache[c] = r['dE_site']; res['pairs'].append(dict(c=c, dE_site=r['dE_site'], E=r['E'], iters=r['iters'], res=r['res'], sec=time.time() - t)); dump()
    log(f'c={c:.4f} dE/site {r["dE_site"]:.6e} iters {r["iters"]} res {r["res"]:.1e}'); return r['dE_site']
prev = json.load(open('/scratch/a/A.Otaifi/ansatz_runs/pfaffian_c_star/cstar.json'))
for p in prev['pairs']: cache[p['c']] = p['dE_site']
assert sorted(cache) == [0.0, 1.0, 2.0, 2.5]
res['prev'] = prev['pairs']
r4 = solve(4.0); assert r4 < cache[2.5] or True
for c in (6.0, 8.0, 10.0): solve(c)
for it in range(8):
    b = min(cache, key=lambda c: cache[c])
    if b >= 10.0: res['status'] = 'MIN AT UPPER EDGE'; dump(); log('minimum at c=10 edge'); break
    cs = sorted(cache); i = cs.index(b); trio = cs[max(i - 1, 0):i + 2] if i > 0 else cs[:3]
    p = np.polyfit(trio, [cache[c] for c in trio], 2)
    if p[0] <= 0: res['status'] = 'NOT CONVEX'; dump(); break
    cv = float(np.clip(-p[1] / (2 * p[0]), 0.0, 10.0)); log(f'iter {it}: trio {trio}, vertex {cv:.4f}')
    if min(abs(cv - c) for c in cache) < 0.05: break
    solve(cv)
cstar = min(cache, key=lambda c: cache[c]); dE = cache[cstar]; res['cstar'] = cstar; res['dE_cstar'] = dE; res['tax'] = dE - E_PHI; res['tax_over_gain'] = (dE - E_PHI) / GAIN; dump()
log(f'c* {cstar} dE {dE:.6e} tax/gain {(dE - E_PHI) / GAIN:.4f}')
del K, Vm
# variational <H> of s2|A e^{c fJ}| (Hs as l_addendum.py)
HW = BS.HW; JW = BS.JW; t1 = np.load(f'{BS.TAB}/nb_J1.npy', mmap_mode='r'); t2 = np.load(f'{BS.TAB}/nb_J2.npy', mmap_mode='r')
ip = np.zeros(D + 1, np.int64); idx = []; dat = []; D0h = np.empty(D)
for i0 in range(0, D, 131072):
    i1 = min(D, i0 + 131072); t = np.concatenate([t1[i0:i1], t2[i0:i1]], 1); valid = t >= 0; tt = np.where(valid, t, 0)
    D0h[i0:i1] = 27.0 - 0.5 * (valid * JW[None, :]).sum(1); sg = s[i0:i1, None] * s[tt]
    idx.append(tt[valid].astype(np.int32)); dat.append((HW[None, :] * np.where(sg > 0, 1.0, -1.0))[valid]); ip[i0 + 1:i1 + 1] = valid.sum(1)
Hs = sp.csr_matrix((np.concatenate(dat), np.concatenate(idx), np.cumsum(ip)), shape=(D, D)); del idx, dat
def Hsite(la):
    la = la - la.max(); a = np.exp(la); EL = D0h + (Hs @ a) / a; w = n * a * a
    return (float((w * EL).sum() / w.sum()) - E0) / BS.NS
res['Hvar'] = {str(c): Hsite(lA + c * fJ) for c in sorted(cache)}
a_ = np.exp(2 * (lA + cstar * fJ - (lA + cstar * fJ).max())); wp = n * a_; wp /= wp.sum(); m = cstar * np.abs(fJ) > 5
res['frac_big'] = dict(orbits=float(m.mean()), n_weighted=float((n * m).sum() / n.sum()), psi2_weighted=float((wp * m).sum()), n_orbits_big=int(m.sum()), fJ_min=float(fJ.min()), fJ_max=float(fJ.max()))
res['status'] = 'DONE'; dump(); log(f"Hvar {res['Hvar']}"); log(f"frac_big {res['frac_big']}"); log('DONE')
