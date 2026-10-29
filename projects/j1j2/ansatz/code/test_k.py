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
RJ = f'{OUT}/k_result.json'; res = json.load(open(RJ)) if os.path.exists(RJ) else {}
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
# ---- features
for j, nm in enumerate(NAMES):
    f = f'{OUT}/fJ_{nm}.npy'
    if f'E_{nm}' not in res: res[f'E_{nm}'] = efn(LA[j], f'E_FN[{nm}, s1]'); dump()
    if not os.path.exists(f):
        fj, info = fJ_of(LA[j], res[f'E_{nm}']['E']); res[f'fJ_{nm}'] = info; np.save(f, fj); dump(); log(f'fJ[{nm}] {info}')
FJ = np.stack([np.load(f'{OUT}/fJ_{nm}.npy') for nm in NAMES], 1)          # (D,4)
# the Z fJ file must reproduce f_J[A_s1]
fJz = np.load(f'{Z}/fJ.npy'); res['check_fJ_A_vs_Z_maxabs'] = float(np.abs(FJ[:, 0] - fJz).max()); log(f'fJ[A_s1] vs Z/fJ.npy max|diff| {res["check_fJ_A_vs_Z_maxabs"]:.2e}'); dump()
# ---- targets
lA = LA[0]
lA1 = lA + 3.9 * fJz
TG = {'K1': dict(lphi=np.load(PHI2PP).astype(np.float64), base=lA), 'K2': dict(lphi=np.load(f'{SC}/ln_phi2.npy').astype(np.float64), base=lA1)}
perm = np.random.default_rng(0).permutation(D)
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
for tg, T in TG.items():
    R = T['lphi'] - T['base']; lp = norm_la(T['lphi'])
    Fm = np.concatenate([FJ, R[:, None], FJ[perm][:, :1]], 1)         # cols 0-3 feats, 4 R*, 5 permuted f_J[A_s1] (broken control)
    GF = f'{OUT}/gram_{tg}.npz'
    if os.path.exists(GF): z = np.load(GF); G = z['G']; Wt = float(z['Wt']); npair = int(z['npair'])
    else: G, _, Wt, npair = pairs_pass(lp, Fm, []); np.savez(GF, G=G, Wt=Wt, npair=npair)
    DRR = G[4, 4]; r = res.setdefault(tg, {}); r['D_RR'] = DRR; r['Wtot'] = Wt; r['npair'] = npair; log(f'{tg}: D_RR {DRR:.6e} Wtot {Wt:.6f} npair {npair}')
    r['R_std_n'] = float(np.sqrt((n * (R - (n * R).sum() / n.sum()) ** 2).sum() / n.sum()))
    r['single'] = {}; 
    for j, nm in enumerate(NAMES): c, e2 = ls(G, [j], 4, DRR); r['single'][nm] = dict(c=c.tolist(), eps2=e2)
    r['ctl_Rstar'] = ls(G, [4], 4, DRR)[1]; r['ctl_perm_fJ_A'] = ls(G, [5], 4, DRR)[1]
    sc = np.sqrt(np.diag(G)[:4]); r['cos'] = (G[:4, :4] / np.outer(sc, sc)).tolist()
    sub = {}
    for k in (2, 3):
        for comb in itertools.combinations(range(4), k): c, e2 = ls(G, list(comb), 4, DRR); sub['+'.join(NAMES[i] for i in comb)] = dict(idx=list(comb), c=c.tolist(), eps2=e2)
    r['subsets'] = sub
    # greedy forward
    chosen = []; gr = []
    for step in range(3):
        best = min((j for j in range(4) if j not in chosen), key=lambda j: ls(G, chosen + [j], 4, DRR)[1]); chosen.append(best)
        c, e2 = ls(G, chosen, 4, DRR); gr.append(dict(set=[NAMES[i] for i in chosen], idx=list(chosen), c=c.tolist(), eps2=e2))
    r['greedy'] = gr; dump(); log(f'{tg}: single {[(k, round(v["eps2"], 4)) for k, v in r["single"].items()]}; greedy {[(g["set"], round(g["eps2"], 4)) for g in gr]}; ctlR {r["ctl_Rstar"]:.2e} ctlperm {r["ctl_perm_fJ_A"]:.4f}')
    # direct check of D(e,e) for greedy sets + R* control (e-tables, independent of Gram algebra)
    ecols = [FJ[:, g['idx']] @ np.array(g['c']) - R for g in gr] + [Fm[:, 4] - R]
    _, Dee, _, _ = pairs_pass(lp, Fm[:, :1], ecols)
    for g, d in zip(gr + [dict()], Dee): g['eps2_direct'] = float(d / DRR)
    r['ctl_Rstar_direct'] = float(Dee[-1] / DRR); r['greedy'] = gr; dump(); log(f'{tg}: direct eps2 {[round(g["eps2_direct"], 5) for g in gr]} ctl_R_direct {Dee[-1]/DRR:.2e}')
    # ---- exact E_FN of base + s*sum c f for greedy-3
    g3 = gr[2]; corr = FJ[:, g3['idx']] @ np.array(g3['c']); base = T['base']
    cache = {}; sc_ = r.setdefault('scan', {}); 
    if tg == 'K1': cache[0.0] = json.load(open(f'{Z}/z_result.json'))['b']['E_FN_A']['dE_site']
    else: cache[0.0] = json.load(open(f'{SC}/screen_result.json'))['S1']['E_FN_A1']['dE_site']
    sc_['0.0'] = cache[0.0]
    for k, v in sc_.items(): cache[float(k)] = v
    def solve(s):
        s = round(float(s), 4)
        if s in cache: return
        rr_ = efn(base + s * corr, f'{tg} s={s}'); cache[s] = rr_['dE_site']; sc_[str(s)] = rr_['dE_site']; r.setdefault('scan_meta', {})[str(s)] = dict(iters=rr_['iters'], res=rr_['res']); dump()
    for s in (0.5, 1.0, 1.5, 2.0): solve(s)
    for it in range(10):
        b = min(cache, key=cache.get); cs = sorted(cache); i = cs.index(b)
        if i == len(cs) - 1 and b < 6: solve(b + 0.5); continue
        trio = cs[max(i - 1, 0):i + 2] if i > 0 else cs[:3]; p = np.polyfit(trio, [cache[c] for c in trio], 2)
        if p[0] <= 0: r['scan_status'] = 'NOT CONVEX'; break
        cv = round(float(np.clip(-p[1] / (2 * p[0]), 0, 6)) * 10) / 10
        if round(cv, 4) in cache: break
        solve(cv)
    for _ in range(10):
        b = min(cache, key=cache.get); new = [round(c, 4) for c in (b - 0.1, b + 0.1) if c >= -1e-9 and round(c, 4) not in cache]
        if not new: break
        for c in new: solve(c)
    b = min(cache, key=cache.get); r['scan_min'] = dict(s=b, dE_site=cache[b]); r['scan_all'] = {str(k): cache[k] for k in sorted(cache)}
    if tg == 'K2':
        S1r = json.load(open(f'{SC}/screen_result.json'))['S1']; dphi2 = S1r['E_FN_phi2']['dE_site']; Dl = S1r['Delta_site']
        r['tau'] = cache[b] - dphi2; r['Delta'] = Dl; r['tau_over_Delta'] = r['tau'] / Dl; r['dE_FN_phi2'] = dphi2
    dump(); log(f'{tg}: scan min {r["scan_min"]} ' + (f'tau {r["tau"]:.4e} tau/Delta {r["tau_over_Delta"]:.3f}' if tg == 'K2' else ''))
res['status'] = 'DONE'; dump(); log('DONE')
