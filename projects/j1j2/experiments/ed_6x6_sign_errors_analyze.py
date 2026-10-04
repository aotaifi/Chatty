"""Combine per-sample outputs of ed_6x6_sign_errors.py and compute true wrong-sign probabilities.

usage: python ed_6x6_sign_errors_analyze.py OUTJSON glob_k1 glob_k2
w_s = P_{x~|psi0|^2}[s(x) != sign psi0(x)] with global sign per rule maximising agreement; delete-block
jackknife (nb blocks) errors.  Energies from e_ViT (VMC) + paired differences of k1_repeat_summary.json.
"""
import sys, glob, json
import numpy as np

out_json = sys.argv[1]
def load(pattern):
    fs = sorted(f for pt in pattern.split(',') for f in glob.glob(pt)); parts = [np.load(f) for f in fs]
    # samples files: first-set indices are 0..1e5-1; the extra file has its own index space -> offset by 1e6 for 'extra' files
    off = [1_000_000 if 'extra' in f else 0 for f in fs]
    idx = np.concatenate([p['idx'] + o_ for p, o_ in zip(parts, off)]); o = np.argsort(idx)
    d = {k: np.concatenate([p[k] for p in parts])[o] for k in parts[0].files if parts[0][k].ndim == 1 and len(parts[0][k]) == len(parts[0]['idx'])}
    d['idx'] = idx[o]
    assert len(np.unique(d['idx'])) == len(d['idx'])
    d['T'] = parts[0]['T']; d['PHI'] = float(parts[0]['PHI']); d['nfiles'] = len(fs)
    d['neval'] = int(sum(int(p['neval']) for p in parts))
    return d

def jk(ind, nb=100):
    n = len(ind); nb = min(nb, n); blocks = np.array_split(np.arange(n), nb)
    tot = ind.sum(); vals = np.array([(tot - ind[b].sum()) / (n - len(b)) for b in blocks])
    return float(ind.mean()), float(np.sqrt((nb - 1) / nb * np.sum((vals - vals.mean()) ** 2)))

def wrong(s, ref):
    """returns (w, err, global_sign): global sign maximising agreement."""
    mis = (s != ref).astype(float)
    if mis.mean() > 0.5: mis = 1 - mis; g = -1
    else: g = +1
    w, e = jk(mis); return w, e, g

def rules(d, kmax):
    ref = np.sign(d['psi0'])
    R = {'Marshall': d['s0'], 'ViT': d['h']}
    for k in range(1, kmax + 1): R[f'K{k}'] = d[f's{k}']
    return ref, R

# Energies (per site).  Coordinator correction 2026-10-04: use matched estimate of results/fn_bound_check_6x6/bound_check_final.json
E0N = -0.5038096538908783
eV, eVse = -0.5036542608124052, 2.131044877124918e-05       # E_ViT_sampled (same 49k samples as the matched K1 difference)
dK1, dK1se = 0.0008133415846492942, 4.544673067290982e-05    # Delta<H>(K1(T=-14.985799779) - ViT)
S = json.load(open('/Users/aliotaifi/Chatty-organize/projects/j1j2/results/k1_repeat_fixed_amp_6x6_3576208/k1_repeat_summary.json'))
rows = {r['step']: r for r in S['rows']}
res = {'E0_per_site': E0N, 'e_ViT': eV, 'e_ViT_se': eVse, 'sets': {}}
def eps(e): return (e - E0N) / abs(E0N)
def entry_(e, se, note): return dict(e=e, e_se=se, eps=eps(e), eps_se=se / abs(E0N), note=note)
res['eps'] = {'ViT': entry_(eV, eVse, 'VMC sampled, bound_check_final.json'),
  'K1(T=-14.9858)': entry_(eV + dK1, float(np.hypot(dK1se, eVse)), 'matched paired Delta=+8.133(45)e-4/site on 49k samples + E_ViT'),
  'Marshall': entry_(eV + rows[0]['dE_per_site_vs_vit_full'], float(np.hypot(rows[0]['se'], eVse)), 'OLDER 256-sample paired estimator (+5.30(1.6)e-3 added to e_ViT); biased low'),
  'K2(T0=-15.024,T1=-1.750)': entry_(eV + rows[2]['dE_per_site_vs_vit_full'], float(np.hypot(rows[2]['se'], eVse)), 'OLDER 256-sample paired estimator (+3.5(2.3)e-4); known biased low (misses rare sign defects) -> lower bound only')}

for label, pat, kmax in (('K1T', sys.argv[2], 1), ('K2', sys.argv[3], 2)):
    d = load(pat); n = len(d['idx']); ref, R = rules(d, kmax)
    entry = {'n': n, 'nfiles': d['nfiles'], 'neval': d['neval'], 'T': d['T'].tolist(), 'w': {}}
    for name, s in R.items():
        w, e, g = wrong(s, ref); entry['w'][name] = dict(w=w, err=e, global_sign=g, nwrong=int(round(w * n)))
    # paired changes
    for a, b in (('Marshall', 'K1'), ('K1', 'K2'), ('Marshall', 'ViT')):
        if a in R and b in R:
            ga, gb = entry['w'][a]['global_sign'], entry['w'][b]['global_sign']
            dif = ((R[b] * gb != ref) .astype(float) - (R[a] * ga != ref).astype(float))
            nb = min(100, n); blocks = np.array_split(np.arange(n), nb); tot = dif.sum()
            vals = np.array([(tot - dif[bl].sum()) / (n - len(bl)) for bl in blocks])
            entry[f'paired_{b}_minus_{a}'] = dict(mean=float(dif.mean()), err=float(np.sqrt((nb - 1) / nb * np.sum((vals - vals.mean()) ** 2))))
    # K2-only-differs counts
    if kmax >= 2:
        entry['K2_changed_vs_K1'] = int(np.sum(R['K2'] != R['K1'])); entry['K1_changed_vs_Marshall'] = int(np.sum(R['K1'] != R['Marshall']))
    # ViT amplitude vs exact amplitude
    la = d['la']; lp = np.log(np.abs(d['psi0']))
    dl = la - lp
    entry['amp_check'] = dict(pearson_log=float(np.corrcoef(la, lp)[0, 1]), std_dlog_after_shift=float(np.std(dl)),
                              mean_dlog=float(np.mean(dl)))
    from scipy.stats import spearmanr
    entry['amp_check']['spearman_log'] = float(spearmanr(la, lp)[0])
    # Marshall-vs-ED consistency (psi0 sign relative to Marshall) is the same quantity as w_Marshall
    res['sets'][label] = entry
    # same-subset table for k1 on the k2 subset
json.dump(res, open(out_json, 'w'), indent=1)
print(json.dumps(res, indent=1))
