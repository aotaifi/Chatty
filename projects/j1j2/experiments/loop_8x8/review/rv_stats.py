"""Review T3/T4 (REVIEW_lanczos8x8.md): statistics of the existing 8x8 Lanczos data, numpy only (no ViT evaluations).
  python rv_stats.py DATA_DIR OUT.json
DATA_DIR holds l8ref*/ref.npz (referee: psi_P samples + two-hop data), dira/direct.npz (direct psi_1 samples),
vmc/vmc.npz (plain VMC of psi_P).  Everything per site = total / 64.
"""
import sys, glob, json, os
import numpy as np

D_ = sys.argv[1]; OUT = sys.argv[2]
N = 64; L = 8; AL = -0.160951; ER = -31.92677568; RBMPP = -0.498886
res = {}


def chain_se(v, ch):
    ids, inv = np.unique(ch, return_inverse=True)
    m = np.bincount(inv, weights=v) / np.bincount(inv)
    return float(m.std(ddof=1) / np.sqrt(len(ids)))


def hill(x, frac=0.02):
    """Hill estimator of the tail index k (P(X > t) ~ t^-k) from the top frac of |x|."""
    a = np.sort(np.abs(x))[::-1]; k = max(int(frac * len(a)), 10)
    return float(1.0 / np.mean(np.log(a[:k] / a[k])))


def top_share(t, q):
    """fraction of sum(t) carried by the top q of samples ranked by |t|."""
    o = np.argsort(-np.abs(t)); k = max(int(q * len(t)), 1)
    return float(t[o[:k]].sum() / t.sum())


# ------------------------------------------------------------------ independent bond list (from coordinates)
def my_bonds():
    s = lambda x, y: (x % L) + L * (y % L)
    b = set()
    for y in range(L):
        for x in range(L):
            for dx, dy, J in ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                              (1, 1, 0.5), (-1, -1, 0.5), (1, -1, 0.5), (-1, 1, 0.5)):
                i, j = s(x, y), s(x + dx, y + dy)
                b.add((min(i, j), max(i, j), J))
    return sorted(b)


BONDS = my_bonds()
res['bonds'] = dict(n=len(BONDS), nJ1=sum(1 for b in BONDS if b[2] == 1.0), nJ2=sum(1 for b in BONDS if b[2] == 0.5))

# ------------------------------------------------------------------ referee data
files = sorted(glob.glob(os.path.join(D_, 'l8ref*', 'ref.npz')))
Z = {}; offx = 0; offc = 0; fid = []
for k_, p in enumerate(files):
    z = dict(np.load(p)); z['own'] = z['own'] + offx; z['chain'] = z['chain'] + offc
    offx += len(z['X']); offc += int(z['chain'].max()) + 1; fid.append(np.full(len(z['X']), k_))
    for k, v in z.items():
        if np.ndim(v) == 0: continue
        Z.setdefault(k, []).append(v)
Z = {k: np.concatenate(v) for k, v in Z.items()}; fid = np.concatenate(fid)
R = len(Z['X']); own = Z['own']; ch = Z['chain']
res['referee'] = dict(files=[os.path.basename(os.path.dirname(p)) for p in files], R=R, edges=len(own))

# Sz = 0 check
pc = np.array([bin(int(s)).count('1') for s in np.concatenate([Z['X'], Z['Y'][:200000]])])
res['sz0'] = dict(min_up=int(pc.min()), max_up=int(pc.max()))

# independent E_L from stored neighbour amplitudes with my own bonds and diagonal (T2 partial, no ViT)
ymap = {}
for i in range(len(own)):
    ymap[(int(own[i]), int(Z['Y'][i]))] = i
X = Z['X']; ELmine = np.empty(R); miss = 0
for r in range(R):
    s = int(X[r]); e = 0.0
    for i, j, J in BONDS:
        si = (s >> i) & 1; sj = (s >> j) & 1
        if si == sj: e += 0.25 * J
        else:
            e -= 0.25 * J
            y = s ^ (1 << i) ^ (1 << j); k = ymap.get((r, y))
            if k is None: miss += 1; continue
            e += 0.5 * J * Z['x_s'][r] * Z['y_s'][k] * np.exp(Z['y_la'][k] - Z['x_la'][r])
    ELmine[r] = e
res['EL_independent'] = dict(missing_neighbours=miss, max_abs_diff=float(np.max(np.abs(ELmine - Z['x_ELa']))))
print('EL independent', res['EL_independent'], flush=True)

# T3: single-valuedness: configurations evaluated more than once (as neighbours of different samples / batches)
allS = np.concatenate([Z['X'], Z['Y']]); alla = np.concatenate([Z['x_la'], Z['y_la']]); alls = np.concatenate([Z['x_s'], Z['y_s']])
o = np.argsort(allS, kind='stable'); S_ = allS[o]; a_ = alla[o]; s_ = alls[o]
same = S_[1:] == S_[:-1]
res['single_valued'] = dict(repeated_pairs=int(same.sum()), sign_disagreements=int(np.sum(same & (s_[1:] != s_[:-1]))),
                            max_abs_dloga=float(np.max(np.abs(a_[1:] - a_[:-1])[same])) if same.any() else None)
print('single valued', res['single_valued'], flush=True)

# ------------------------------------------------------------------ moments, tails
JBe = np.array([0.0])
sxy = Z['x_s'][own] * Z['y_s']; rxy = np.exp(np.clip(Z['y_la'] - Z['x_la'][own], -60, 60))
# J of each stored edge from my bond list via the flipped bit pair
diff = Z['X'][own] ^ Z['Y']
Jmap = {(1 << i) | (1 << j): J for i, j, J in BONDS}
Je = np.array([Jmap[int(d)] for d in diff])
h = 0.5 * Je * sxy * rxy
EL = Z['x_ELa']; Dg = Z['x_D']
EL2 = Dg * EL + np.bincount(own, weights=h * Z['y_ELa'], minlength=R)
h1 = EL.mean(); el = EL - h1
t2 = el ** 2; t3 = el * (EL2 - 2 * h1 * EL + h1 * h1)
m2 = t2.mean(); m3 = t3.mean(); m2b = EL2.mean() - h1 ** 2


def ritz_c(m2, m3):
    c = m3 / (2 * m2); return c - np.sqrt(c * c + m2)


f = 1 + AL * (EL - ER); w = f ** 2
EL1 = ((1 - AL * ER) * EL + AL * EL2) / f
E1rw = np.sum(w * EL1) / np.sum(w)
res['moments'] = dict(h1_site=h1 / N, m2=m2, m2_alt_EL2=m2b, m3=m3, c=m3 / (2 * m2),
                      Delta_ritz_site=ritz_c(m2, m3) / N,
                      Delta_ritz_with_m2alt_offdiag_site=None,
                      E1_reweighted_site=E1rw / N, Delta_reweighted_site=(E1rw - h1) / N,
                      E1_plain_mean_on_psiP_site=float(np.mean(EL1) / N),
                      E1_median_on_psiP_site=float(np.median(EL1) / N))
res['tails'] = dict(hill_EL=hill(el), hill_t2=hill(t2), hill_t3=hill(t3), hill_f2=hill(w), hill_EL2c=hill(EL2 - np.mean(EL2)),
                    top001_share_m2=top_share(t2, 1e-3), top01_share_m2=top_share(t2, 1e-2),
                    top001_share_m3=top_share(t3, 1e-3), top01_share_m3=top_share(t3, 1e-2),
                    frac_t3_negative=float(np.mean(t3 < 0)),
                    f_min=float(f.min()), f_max=float(f.max()), f_q=[float(q) for q in np.quantile(f, [1e-4, 1e-3, 0.01, 0.5, 0.99, 0.999, 0.9999])],
                    EL_site_q=[float(q / N) for q in np.quantile(EL, [1e-4, 1e-3, 0.5, 0.999, 0.9999])])
# per-file moment estimates (5 independent batches): spread of Delta vs jackknife SE
per = []
for k_ in range(len(files)):
    i_ = fid == k_; e_ = EL[i_]; hh = e_.mean(); a2 = np.mean((e_ - hh) ** 2); a3 = np.mean((e_ - hh) * (EL2[i_] - 2 * hh * e_ + hh * hh))
    ff = 1 + AL * (e_ - ER); ww = ff ** 2; E1k = np.sum(ww * EL1[i_]) / ww.sum()
    per.append(dict(file=res['referee']['files'][k_], h1_site=hh / N, Delta_ritz_site=ritz_c(a2, a3) / N, Delta_rw_site=(E1k - hh) / N,
                    m2=a2, m3=a3))
res['per_file'] = per
dr = np.array([p['Delta_ritz_site'] for p in per])
res['per_file_summary'] = dict(Delta_ritz_mean=float(dr.mean()), Delta_ritz_sd=float(dr.std(ddof=1)), Delta_ritz_se=float(dr.std(ddof=1) / np.sqrt(len(dr))))

# bootstrap over chains for the Ritz Delta (vs the jackknife 1.25e-5)
rg = np.random.default_rng(0)
ids, inv = np.unique(ch, return_inverse=True); nC = len(ids)
idx_by_c = [[] for _ in range(nC)]
for i_, c_ in enumerate(inv): idx_by_c[c_].append(i_)
idx_by_c = [np.array(v) for v in idx_by_c]
bs = []; bsr = []
for b in range(400):
    pick = np.concatenate([idx_by_c[c_] for c_ in rg.integers(nC, size=nC)])
    e_ = EL[pick]; hh = e_.mean(); a2 = np.mean((e_ - hh) ** 2); a3 = np.mean((e_ - hh) * (EL2[pick] - 2 * hh * e_ + hh * hh))
    bs.append(ritz_c(a2, a3) / N)
    ww = w[pick]; bsr.append((np.sum(ww * EL1[pick]) / ww.sum() - hh) / N)
bs = np.array(bs); bsr = np.array(bsr)
res['bootstrap'] = dict(Delta_ritz_sd=float(bs.std()), Delta_ritz_q=[float(q) for q in np.quantile(bs, [0.025, 0.5, 0.975])],
                        Delta_rw_sd=float(bsr.std()), Delta_rw_q=[float(q) for q in np.quantile(bsr, [0.025, 0.5, 0.975])])

# optimisation bias: alpha (Ritz optimum) chosen on half A, reweighted energy at that alpha on half B
ob = []
for rep in range(50):
    perm = rg.permutation(nC); A = np.isin(inv, perm[:nC // 2]); B = ~A
    for S1, S2 in ((A, B), (B, A)):
        e_ = EL[S1]; hh = e_.mean(); a2 = np.mean((e_ - hh) ** 2); a3 = np.mean((e_ - hh) * (EL2[S1] - 2 * hh * e_ + hh * hh))
        lam = ritz_c(a2, a3); beta = lam / a2                         # coefficient of (H - hh) psi
        al_ = beta; er_ = hh                                          # psi(1 + beta (E_L - hh))
        ff = 1 + al_ * (EL[S2] - er_); ww = ff ** 2
        e1 = ((1 - al_ * er_) * EL[S2] + al_ * EL2[S2]) / ff
        Eb = np.sum(ww * e1) / ww.sum(); hb = EL[S2].mean()
        eS2 = EL[S2]; hb2 = eS2.mean(); b2 = np.mean((eS2 - hb2) ** 2); b3 = np.mean((eS2 - hb2) * (EL2[S2] - 2 * hb2 * eS2 + hb2 * hb2))
        ob.append(((Eb - hb) / N, ritz_c(b2, b3) / N, lam / N))
ob = np.array(ob)
res['optimisation_bias'] = dict(Delta_outofsample_mean=float(ob[:, 0].mean()), Delta_samesample_ritz_mean=float(ob[:, 1].mean()),
                                Delta_insample_A_mean=float(ob[:, 2].mean()),
                                bias_estimate=float(ob[:, 1].mean() - ob[:, 0].mean()), sd_of_diff=float((ob[:, 1] - ob[:, 0]).std()))
# variance of psi_1 from moments (needs <(H^2 psi)^2>: EL2^2 term) -> expected per-sample sd of E_L1 under |psi_1|^2
num = np.mean(((1 - AL * ER) * EL + AL * EL2) ** 2); nrm = np.mean(w)
H2_1 = num / nrm; var1 = H2_1 - E1rw ** 2
res['psi1_variance_from_moments'] = dict(var_total=float(var1), sd_site=float(np.sqrt(max(var1, 0)) / N), sd_psiP_site=float(EL.std() / N))
print(json.dumps({k: res[k] for k in ('moments', 'tails', 'per_file_summary', 'bootstrap', 'optimisation_bias', 'psi1_variance_from_moments')}, indent=1), flush=True)

# ------------------------------------------------------------------ direct run a
d = np.load(os.path.join(D_, 'dira', 'direct.npz')); dEL = d['EL']; dEL2 = d['EL2']; dch = d['chain']
fd = 1 + AL * (dEL - ER); dEL1 = ((1 - AL * ER) * dEL + AL * dEL2) / fd
nch = len(np.unique(dch)); tt = np.repeat(np.arange(len(dEL) // nch), nch)
E1d = dEL1.mean()
byt = [dict(t=int(k), E1_site=float(dEL1[tt == k].mean() / N), se=float(dEL1[tt == k].std(ddof=1) / np.sqrt(nch) / N),
            mean_eps=float((dEL - ER)[tt == k].mean())) for k in np.unique(tt)]
res['direct_a'] = dict(R=len(dEL), E1_site=float(E1d / N), se_chain=chain_se(dEL1, dch) / N,
                       se_naive=float(dEL1.std(ddof=1) / np.sqrt(len(dEL1)) / N), sd_EL1_site=float(dEL1.std() / N),
                       hill_EL1=hill(dEL1 - E1d), hill_EL1_5pc=hill(dEL1 - E1d, 0.05), f_min=float(fd.min()), f_max=float(fd.max()),
                       f_q=[float(q) for q in np.quantile(fd, [0.001, 0.01, 0.5, 0.99, 0.999])],
                       by_time=byt, mean_eps_direct=float((dEL - ER).mean()),
                       mean_eps_expected_psi1=float(np.sum(w * (EL - ER)) / w.sum()), mean_eps_psiP=float(np.mean(EL - ER)),
                       EL1_top5_site=[float(x / N) for x in np.sort(dEL1)[:5]], EL1_bottom5_site=[float(x / N) for x in np.sort(dEL1)[-5:]],
                       median_site=float(np.median(dEL1) / N),
                       lag_corr_t=float(np.corrcoef(dEL1[tt == 0], dEL1[tt == 1])[0, 1]))
print(json.dumps(res['direct_a'], indent=1), flush=True)

# ------------------------------------------------------------------ plain VMC of psi_P
v = np.load(os.path.join(D_, 'vmc', 'vmc.npz')); vEL = v['EL']; vch = v['chain']; nv = len(np.unique(vch)); T = len(vEL) // nv
vt = np.repeat(np.arange(T), nv)
first = vt < T // 2
blocks = []
for b in range(4):
    sel = (vt >= b * T // 4) & (vt < (b + 1) * T // 4)
    blocks.append(dict(sweeps=f'{b * T // 4}-{(b + 1) * T // 4 - 1}', E_site=float(vEL[sel].mean() / N), se=chain_se(vEL[sel], vch[sel]) / N))
# lag autocorrelation of E_L along chains
M = vEL.reshape(T, nv)
ac = [float(np.mean([(np.corrcoef(M[:-k, c], M[k:, c])[0, 1]) for c in range(0, nv, 8)])) for k in (1, 2, 4)]
res['vmc_psiP'] = dict(n=len(vEL), E_site=float(vEL.mean() / N), se=chain_se(vEL, vch) / N, quarters=blocks, lag_autocorr_1_2_4=ac,
                       hill_EL=hill(vEL - vEL.mean()))
print(json.dumps(res['vmc_psiP'], indent=1), flush=True)
json.dump(res, open(OUT, 'w'), indent=1, default=float)
