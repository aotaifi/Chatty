"""Review T5 evaluation (REVIEW_lanczos8x8.md): pool direct VMC of psi_1 (run a + new runs), apply the pre-registered
criteria (a)-(d) and amendment (d').
  python rv_pool.py OUT.json DIRECT_A.npz NEW1_DIR [NEW2_DIR ...]
"""
import sys, os, json
import numpy as np
from scipy import stats

N = 64; AL = -0.160951; ER = -31.92677568; RBMPP = -0.498886; COMB, COMB_SE = -0.4989530, 1.64e-5
OUT = sys.argv[1]


def el1(EL, EL2):
    return ((1 - AL * ER) * EL + AL * EL2) / (1 + AL * (EL - ER))


def chain_means(v, ch):
    ids, inv = np.unique(ch, return_inverse=True)
    return np.bincount(inv, weights=v) / np.bincount(inv)


def hill(x, frac=0.02):
    a = np.sort(np.abs(x))[::-1]; k = max(int(frac * len(a)), 10)
    return float(1.0 / np.mean(np.log(a[:k] / a[k])))


batches = []
d = np.load(sys.argv[2]); batches.append(dict(name='run_a', E=el1(d['EL'], d['EL2']) / N, ch=d['chain'], t=None, f=1 + AL * (d['EL'] - ER)))
for p in sys.argv[3:]:
    z = np.load(os.path.join(p, 'direct2.npz')); ok = np.isfinite(z['EL2'])
    batches.append(dict(name=os.path.basename(p.rstrip('/')), E=el1(z['EL'][ok], z['EL2'][ok]) / N, ch=z['chain'][ok], t=z['sweep'][ok],
                        f=1 + AL * (z['EL'][ok] - ER), complete=bool(ok.all()),
                        trace=json.load(open(os.path.join(p, 'trace.json')))['trace']))
res = dict(batches=[])
allm = []; groups = []
for b in batches:
    cm = chain_means(b['E'], b['ch']); se = cm.std(ddof=1) / np.sqrt(len(cm))
    row = dict(name=b['name'], R=len(b['E']), nchains=len(cm), E1_site=float(b['E'].mean()), se=float(se),
               median=float(np.median(b['E'])), robust_sd=float((np.quantile(b['E'], .75) - np.quantile(b['E'], .25)) / 1.349),
               sd=float(b['E'].std()), min_factor=float(b['f'].min()), hill2=hill(b['E'] - b['E'].mean()))
    if b['t'] is not None:
        ts = np.unique(b['t']); row['by_sweep'] = {int(s): (float(b['E'][b['t'] == s].mean()),
                                                         float(b['E'][b['t'] == s].std(ddof=1) / np.sqrt((b['t'] == s).sum()))) for s in ts}
        tr = b['trace']; me = np.array([x['mean_eps'] for x in tr]); md = np.array([x['median_eps'] for x in tr])
        row['trace_mean_eps'] = [round(float(x), 4) for x in me]; row['trace_median_eps'] = [round(float(x), 5) for x in md]
        row['complete'] = b['complete']
    res['batches'].append(row)
    # 16 chain groups (or 8 for run a)
    ids = np.unique(b['ch']); ng = 16 if len(ids) >= 1024 else 8
    for gi in np.array_split(np.random.default_rng(0).permutation(ids), ng):
        m = np.isin(b['ch'], gi); cmg = chain_means(b['E'][m], b['ch'][m])
        groups.append((b['E'][m].mean(), cmg.std(ddof=1) / np.sqrt(len(cmg)), b['name']))
    allm.append((b['E'].mean(), se, len(b['E'])))
E = np.concatenate([b['E'] for b in batches]); CH = np.concatenate([b['ch'] + 100000 * k for k, b in enumerate(batches)])
cm = chain_means(E, CH); se_chain = float(cm.std(ddof=1) / np.sqrt(len(cm)))
# between-batch SE: inverse-variance... use sample-weighted mean and spread of the batch means
bm = np.array([a[0] for a in allm]); bs = np.array([a[1] for a in allm]); bn = np.array([a[2] for a in allm])
wmean = float(np.sum(bm * bn) / bn.sum())
chi2_b = float(np.sum((bm - E.mean()) ** 2 / bs ** 2)); p_b = float(stats.chi2.sf(chi2_b, len(bm) - 1))
se_between = float(np.sqrt(np.sum(bn ** 2 * (bm - wmean) ** 2) / bn.sum() ** 2 * len(bm) / (len(bm) - 1)))
gm = np.array([g[0] for g in groups]); gs = np.array([g[1] for g in groups])
chi2_g = float(np.sum((gm - E.mean()) ** 2 / gs ** 2)); p_g = float(stats.chi2.sf(chi2_g, len(gm) - 1))
E1 = float(E.mean()); se_use = max(se_chain, se_between)
new = [b for b in batches if b['name'] != 'run_a']
En = np.concatenate([b['E'] for b in new]); CHn = np.concatenate([b['ch'] + 100000 * k for k, b in enumerate(new)])
cmn = chain_means(En, CHn); se_new = float(cmn.std(ddof=1) / np.sqrt(len(cmn)))
# sweep split (new runs): first vs second measurement time
t_all = np.concatenate([b['t'] - b['t'].min() for b in new])
e0 = En[t_all == 0]; e1 = En[t_all > 0]
drift = float(e1.mean() - e0.mean()); drift_se = float(np.sqrt(e0.var(ddof=1) / len(e0) + e1.var(ddof=1) / len(e1)))
# trimmed / tail sensitivity
srt = np.sort(E)
trims = {k: float(srt[k:len(srt) - k].mean()) for k in (1, 3, 10, 30)}
res.update(pooled=dict(R=len(E), nchains=len(cm), E1_site=E1, se_chain=se_chain, se_between_batches=se_between, se_used=se_use,
                       chi2_batches=chi2_b, p_batches=p_b, chi2_groups=chi2_g, n_groups=len(gm), p_groups=p_g,
                       hill2=hill(E - E1), hill1=hill(E - E1, 0.01), hill5=hill(E - E1, 0.05),
                       median=float(np.median(E)), robust_sd=float((np.quantile(E, .75) - np.quantile(E, .25)) / 1.349), sd=float(E.std()),
                       symmetric_trimmed_means=trims, below_RBMPP_sigma=(RBMPP - E1) / se_use),
           new_only=dict(R=len(En), E1_site=float(En.mean()), se=se_new, drift_t1_minus_t0=drift, drift_se=drift_se,
                         z_vs_run_a=float((En.mean() - batches[0]['E'].mean()) / np.hypot(se_new, res['batches'][0]['se'])),
                         z_vs_combined=float((En.mean() - COMB) / np.hypot(se_new, COMB_SE))))
crit = dict(
    a_new_E_plus_2se_below_RBMPP=bool(En.mean() + 2 * se_new < RBMPP),
    b_consistent_with_run_a=bool(abs(res['new_only']['z_vs_run_a']) < 2), b_consistent_with_combined=bool(abs(res['new_only']['z_vs_combined']) < 2),
    c_no_drift=bool(abs(drift) < 2 * drift_se),
    d_hill_gt_2p5=bool(res['pooled']['hill2'] > 2.5),
    dprime_i_chi2_groups_p_gt_0p01=bool(p_g > 0.01),
    dprime_ii_pooled_E_plus_2se_used_below_RBMPP=bool(E1 + 2 * se_use < RBMPP))
res['criteria'] = crit
json.dump(res, open(OUT, 'w'), indent=1, default=float)
print(json.dumps(res, indent=1, default=float))
