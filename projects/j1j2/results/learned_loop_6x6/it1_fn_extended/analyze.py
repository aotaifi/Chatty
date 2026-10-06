"""Combine all it1b FN populations (old 16 + extension) -> summary.json.
Usage: python analyze.py DIR_WITH_fn_it1b*.json ...   (globs fn_it1b*.json recursively under given dirs)"""
import sys, glob, os, json
import numpy as np
E0 = -0.5038096538908783
VIT, VIT_ERR = -0.503654, 21e-6
files = sorted(f for d in sys.argv[1:] for f in glob.glob(os.path.join(d, '**', 'fn_it1b*.json'), recursive=True))
seeds = {}
src = {}
for f in files:
    d = json.load(open(f))
    for r in d['reps']:
        assert r['seed'] not in seeds, r['seed']
        seeds[r['seed']] = r['Emean_site']; src[r['seed']] = d['tag']
sd = sorted(seeds); e = np.array([seeds[s] for s in sd]); n = len(e)
old = np.array([seeds[s] for s in sd if s <= 42016]); new = np.array([seeds[s] for s in sd if s > 42016])
def mse(x): return x.mean(), x.std(ddof=1) / np.sqrt(len(x))
rng = np.random.default_rng(0)
B = 20000
idx = rng.integers(0, n, (B, n))
bs = e[idx]
from scipy.stats import trim_mean
def trimmed(x, p=0.1): return trim_mean(x, p)
med = np.median(e)
out = dict(n=n, n_old=len(old), n_new=len(new), E0=E0, ViT=VIT, ViT_err=VIT_ERR)
m, se = mse(e)
out['mean'] = m; out['SE'] = se; out['SD'] = e.std(ddof=1)
out['mean_old16'], out['SE_old16'] = mse(old)
if len(new) > 1: out['mean_new'], out['SE_new'] = mse(new)
out['median'] = med
out['median_boot_SE'] = float(np.median(bs, axis=1).std())
t = trimmed(e); out['trimmed10_mean'] = t; out['trimmed10_boot_SE'] = float(trim_mean(bs, 0.1, axis=1).std())
out['n_below_E0'] = int((e < E0).sum()); out['seeds_below_E0'] = [int(s) for s in sd if seeds[s] < E0]
out['frac_below_E0'] = out['n_below_E0'] / n
out['min'] = float(e.min()); out['max'] = float(e.max())
for k, (v, s) in dict(mean=(m, se), median=(med, out['median_boot_SE']), trimmed10=(t, out['trimmed10_boot_SE'])).items():
    out['dE_vs_ViT_' + k] = v - VIT; out['dE_vs_ViT_' + k + '_err'] = float(np.hypot(s, VIT_ERR))
    out['sigma_vs_ViT_' + k] = (v - VIT) / float(np.hypot(s, VIT_ERR))
# skew / outliers
from scipy.stats import skew, shapiro
out['skew'] = float(skew(e)); out['shapiro_p'] = float(shapiro(e).pvalue)
out['eps_rel_E0_mean'] = (m - E0) / abs(E0)
out['per_seed'] = {int(s): seeds[s] for s in sd}
json.dump(out, open('summary.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != 'per_seed'}, indent=1))
