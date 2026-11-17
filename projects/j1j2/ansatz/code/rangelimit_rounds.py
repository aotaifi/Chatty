"""rangelimit_rounds.py: ROUNDS 2-3 RUNG 2 (STORAGE_PROBLEM.md). Iterate the explicit round map on the full 6x6 table, CPU, no training.
usage: python3 rangelimit_rounds.py RL_DIR OUTDIR
Round k = 1, 2 (round 0 = rangelimit_6x6: A_0 = s1 copy, alpha_0 = -0.10, G_0, p_0):
  ln A_k = ln G_{k-1} + 1.307 f_J[G_{k-1}]       (ln G = ln A + ln max(|1 + alpha (E_L - E)|, e^-60), as cstar_lanczos.py; sign s2)
  alpha_k: line search of cstar_lanczos.py / rangelimit.py (grid 0, -0.1, -0.2, -0.3, -0.45, -0.6 (+ -0.8, -1, -1.2 if min at -0.6), parabola
           iterations, stop when the vertex is within 0.01 of an evaluated point); alpha_k = best evaluated point, vertex reported.
  G_k = A_k |1 + alpha_k (E_L[A_k] - E[A_k])|; p_k = FN state of s2 G_k (amplitude = |v|/sqrt(n), sum n p^2 = 1); u_k = ln(p_k/A_k) - m_k.
  Features and fits exactly as rangelimit_features.py (f_J[A_k] with E' = E_FN[s2 A_k], f_J[G_k] with E' = E_FN[s2 G_k], Vs/K of G_k),
  D-norm with p_k. FN check of fit (ii): tax/gain with gain = E_FN[s2 A_k] - E_FN[s2 p_k], tax = E_FN[s2 A_k e^{R_(ii)}] - E_FN[s2 p_k]."""
import os, sys, json, time, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bond_split as BS
import delta0 as D0M
RL, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True); T0 = time.time()
def log(m): print(f'[{time.time() - T0:7.0f}s] {m}', flush=True)
A0 = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_c/runs/c1_cplx_s1/round_1/la_table.npy'
CJ = 1.307; ALPHA0 = -0.1; EPG0 = json.load(open(f'{RL}/rangelimit_A.json'))['step1']['E_FN']
n = np.load(f'{BS.CSRD}/norb.npy').astype(np.float64); s = np.load(BS.S2).astype(np.float64); D = len(n); assert n.sum() == 9075135300
lA0 = np.load(A0).astype(np.float64)
res = dict(host=os.uname().nodename, A0=A0, c_J=CJ, rounds={}); RJ = f'{OUT}/rounds.json'
def dump(): res['sec'] = time.time() - T0; json.dump(res, open(RJ, 'w'), indent=1, default=float)
K, Vm, D0 = BS.build(n, s); log('built'); sn = np.sqrt(n)
t1 = np.load(f'{BS.TAB}/nb_J1.npy', mmap_mode='r'); t2 = np.load(f'{BS.TAB}/nb_J2.npy', mmap_mode='r'); HW = BS.HW; JW = BS.JW; RS = 32768
def Hmag(b): return D0 * b + Vm @ b - (K @ (sn * b)) / sn
def moments(la):
    a = np.exp(la - la.max()); EL = Hmag(a) / a; w = n * a * a; E = float((w * EL).sum() / w.sum()); assert np.isfinite(EL).all()
    return E, EL - E
def lanc(la, d, al): return la + np.log(np.maximum(np.abs(1.0 + al * d), np.exp(-60.0))) if al != 0 else la.copy()
def efn(la, tag, rec):
    t = time.time(); r, v = D0M.efn_vec(K, Vm, D0, n, la); r['sec'] = time.time() - t; rec.setdefault('solves', []).append(dict(tag=tag, **r)); dump()
    log(f'{tag}: dE/site {r["dE_site"]:.6e} iters {r["iters"]} res {r["res"]:.1e} {r["sec"]:.0f}s'); return r, v
def fJ_and_VsK(lg, Ep):
    fj = np.empty(D); vk = np.empty(D); nbad = 0
    for i0 in range(0, D, RS):
        i1 = min(D, i0 + RS); t = np.concatenate([t1[i0:i1], t2[i0:i1]], 1); va = t >= 0; tt = np.where(va, t, 0)
        ss = s[i0:i1, None] * s[tt]; blk = va & (ss > 0); kpt = va & (ss < 0); Hxx = 27.0 - 0.5 * (va * JW[None, :]).sum(1)
        ratio = np.exp(np.clip(lg[tt] - lg[i0:i1, None], -700, 700)); VA = (blk * HW * ratio).sum(1); SK = (kpt * HW * ratio).sum(1)
        den = Hxx + VA - Ep; nbad += int((den <= 0).sum() + (SK <= 0).sum())
        with np.errstate(divide='ignore', invalid='ignore'): fj[i0:i1] = np.log(SK) - np.log(den); vk[i0:i1] = VA / SK
    return fj, vk, nbad
MONO = [()] + [c for d in (1, 2, 3) for c in itertools.combinations_with_replacement(range(5), d)]; NM = len(MONO)
def poly(Z):
    out = np.empty((Z.shape[0], NM))
    for j, c in enumerate(MONO): out[:, j] = np.prod(Z[:, list(c)], 1) if c else 1.0
    return out
xi = np.random.default_rng(4242).standard_normal(D)
def fits(F5, u, lp, rec):
    wv = np.exp(2 * lp) * n; mu = (wv[:, None] * F5).sum(0); sd = np.sqrt((wv[:, None] * (F5 - mu) ** 2).sum(0)); Z5 = (F5 - mu) / sd
    NC = NM + 4; IU, IXI = NM + 2, NM + 3; NF = 5; fold_of = np.minimum((np.arange(D) * NF) // D, NF - 1)
    def cols(idx): return np.concatenate([poly(Z5[idx]), F5[idx][:, [0, 2]], u[idx, None], xi[idx, None]], 1)
    Gk = np.zeros((NF, NC, NC)); Wtot = 0.0
    for i0 in range(0, D, RS):
        i1 = min(D, i0 + RS); t = np.concatenate([t1[i0:i1], t2[i0:i1]], 1); va = t >= 0; tt = np.where(va, t, 0)
        blk = va & (s[i0:i1, None] * s[tt] > 0); rr, kk = np.nonzero(blk); x = rr + i0; y = tt[rr, kk]
        w = n[x] * 0.5 * HW[kk] * np.exp(lp[x] + lp[y]); Wtot += w.sum()
        dC = cols(np.arange(i0, i1))[rr] - cols(y); fx = fold_of[i0:i1][rr]
        for f in np.unique(fx): mk = fx == f; A = dC[mk]; Gk[f] += A.T @ (w[mk, None] * A)
    G = Gk.sum(0); rec['gram'] = dict(Wtot=Wtot, D_uu=G[IU, IU]); rec['standardise'] = dict(mu=mu.tolist(), sd=sd.tolist())
    def fit(Gm, idx, ti):
        Gs = Gm[np.ix_(idx, idx)]; bb = Gm[idx, ti]; sc = np.sqrt(np.diag(Gs)); sc[sc == 0] = 1
        return np.linalg.lstsq(Gs / np.outer(sc, sc), bb / sc, rcond=1e-12)[0] / sc
    def e2(Gm, idx, ti, c): return float((c @ Gm[np.ix_(idx, idx)] @ c - 2 * c @ Gm[idx, ti] + Gm[ti, ti]) / Gm[ti, ti])
    FITS = {'(i) c fJG': [NM + 1], '(ii) a l + c fJG + b': [NM, NM + 1, 0], '(iii) cubic, 5 features': list(range(NM)), 'ctl random xi': [IXI], 'ctl R = u': [IU]}
    out = {}
    for fn, idx in FITS.items():
        c = fit(G, idx, IU); num = den = 0.0
        for k in range(NF):
            ck = fit(G - Gk[k], idx, IU); num += e2(Gk[k], idx, IU, ck) * Gk[k][IU, IU]; den += Gk[k][IU, IU]
        out[fn] = dict(eps2_in=e2(G, idx, IU, c), eps2_cv=float(num / den), coef=c.tolist() if len(c) <= 3 else None)
        log(f"  {fn}: eps2 in {out[fn]['eps2_in']:.5f} cv {out[fn]['eps2_cv']:.5f} coef {np.round(c, 5).tolist() if len(c) <= 3 else '(56)'}")
    np.savez(f'{OUT}/gram_round{rec["k"]}.npz', Gk=Gk)
    c2 = np.array(out['(ii) a l + c fJG + b']['coef']); R2 = c2[0] * F5[:, 0] + c2[1] * F5[:, 2] + c2[2]; cx = out['ctl random xi']['coef'][0]
    ec = {'(ii)': R2 - u, 'R=u': u - u, 'R=0': -u, 'R=c xi': cx * xi - u}; Dee = np.zeros(len(ec)); Duu = 0.0
    for i0 in range(0, D, RS):
        i1 = min(D, i0 + RS); t = np.concatenate([t1[i0:i1], t2[i0:i1]], 1); va = t >= 0; tt = np.where(va, t, 0)
        blk = va & (s[i0:i1, None] * s[tt] > 0); rr, kk = np.nonzero(blk); x = rr + i0; y = tt[rr, kk]
        w = n[x] * 0.5 * HW[kk] * np.exp(lp[x] + lp[y]); du = u[x] - u[y]; Duu += (w * du * du).sum()
        for j, e in enumerate(ec.values()): dd = e[x] - e[y]; Dee[j] += (w * dd * dd).sum()
    out['direct'] = {k: float(Dee[j] / Duu) for j, k in enumerate(ec)}; log(f"  direct {out['direct']}")
    return out, R2
lG_prev, l_prev, fJG_prev = None, None, None
# round 0 quantities needed for round 1: G_0 and f_J[G_0]
E0m, d0 = moments(lA0); lG_prev = lanc(lA0, d0, ALPHA0)
F0 = np.load(f'{RL}/features/feat.npy', mmap_mode='r'); fJG_prev = np.array(F0[:, 2])
chk, _, _ = fJ_and_VsK(lG_prev, EPG0); res['round0_fJG_recompute_maxabs'] = float(np.abs(chk - fJG_prev).max()); log(f"round-0 fJG recompute max|diff| {res['round0_fJG_recompute_maxabs']:.2e}")
for k in (1, 2):
    rec = dict(k=k); res['rounds'][str(k)] = rec
    lA = lG_prev + CJ * fJG_prev; assert np.isfinite(lA).all(); np.save(f'{OUT}/lA_{k}.npy', lA)
    E, d = moments(lA); rec['Evar_site'] = (E - BS.E0) / BS.NS; log(f'round {k}: E_var[s2 A_{k}]/site {rec["Evar_site"]:.6e}')
    cache = {}; vecs = {}
    def f(al):
        al = round(float(al), 6)
        if al not in cache: r, v = efn(lanc(lA, d, al), f'round{k} alpha={al}', rec); cache[al] = r; vecs[al] = v
        return cache[al]['dE_site']
    for al in (0.0, -0.1, -0.2, -0.3, -0.45, -0.6): f(al)
    if min(cache, key=lambda a_: cache[a_]['dE_site']) == -0.6:
        for al in (-0.8, -1.0, -1.2): f(al)
    vtx = None
    for it in range(8):
        cs = sorted(cache); bst = min(cs, key=lambda a_: cache[a_]['dE_site']); near = sorted(sorted(cs, key=lambda c: abs(c - bst))[:3])
        pp = np.polyfit(near, [cache[c]['dE_site'] for c in near], 2)
        if pp[0] <= 0: rec['line_search'] = 'NOT CONVEX'; break
        vtx = float(np.clip(-pp[1] / (2 * pp[0]), min(cs), 0.0)); log(f'  iter {it}: points {near} vertex {vtx:.4f}')
        if min(abs(vtx - c) for c in cache) < 0.01: break
        f(vtx)
    ak = min(cache, key=lambda a_: cache[a_]['dE_site']); rec.update(alpha=ak, alpha_vertex=vtx, E_FN_A_site=cache[0.0]['dE_site'], E_FN_G_site=cache[ak]['dE_site'],
                                                        grid={str(a_): cache[a_]['dE_site'] for a_ in sorted(cache)})
    log(f"round {k}: alpha {ak} (vertex {vtx}) E_FN[A] {rec['E_FN_A_site']:.6e} E_FN[G] {rec['E_FN_G_site']:.6e}"); dump()
    lG = lanc(lA, d, ak); v = vecs[ak] * np.sign(vecs[ak].sum()); rec['p_n_neg'] = int((v < 0).sum())
    p = np.abs(v) / sn; p = p / np.sqrt((n * p * p).sum()); lp = np.log(np.maximum(p, 1e-300)); np.save(f'{OUT}/p_{k}.npy', p); del vecs
    rp, _ = efn(lp, f'round{k} E_FN[s2 p_{k}]', rec); rec['E_FN_p_site'] = rp['dE_site']; rec['gain_site'] = rec['E_FN_A_site'] - rp['dE_site']
    rs = lp - lA; w2 = n * p * p; m = float((w2 * rs).sum()); u = rs - m; rec['m'] = m
    EA, EG = cache[0.0]['E'], cache[ak]['E']
    fJA, _, nb1 = fJ_and_VsK(lA, EA); fJG, VsK, nb2 = fJ_and_VsK(lG, EG); rec['feat_n_bad'] = nb1 + nb2
    F5 = np.stack([lG - lA, fJA, fJG, lA, VsK], 1); fin = np.isfinite(F5).all(1); rec['n_nonfinite'] = int((~fin).sum())
    rec['feat_range'] = {nm: [float(F5[:, j].min()), float(F5[:, j].max())] for j, nm in enumerate(['l', 'fJA', 'fJG', 'lnA', 'VsK'])}; dump()
    log(f"round {k}: features nonfinite {rec['n_nonfinite']} bad {rec['feat_n_bad']} ranges {rec['feat_range']}")
    if rec['n_nonfinite']: rec['status'] = 'NONFINITE'; dump(); log('STOP'); sys.exit(3)
    rec['fits'], R2 = fits(F5, u, lp, rec); dump()
    r2, _ = efn(lA + R2, f'round{k} E_FN[s2 A_{k} e^R(ii)]', rec); tax = r2['dE_site'] - rp['dE_site']
    rec['fn_check_ii'] = dict(E_FN_site=r2['dE_site'], tax_site=tax, tax_over_gain=tax / rec['gain_site'], eps2_direct=rec['fits']['direct']['(ii)'])
    log(f"round {k}: FN check (ii) {rec['fn_check_ii']}"); dump()
    lG_prev, fJG_prev = lG, fJG
res['status'] = 'DONE'; dump(); log('DONE')
