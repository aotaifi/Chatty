"""vmc_an.py OUT : Lanczos moments + jackknife (over walkers, W groups) from OUT/samples.npz.  Prints alpha, E, var and errors; identity check <vt h2'> vs <h1'^2>."""
import sys, json, numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from vmc10 import from_sums
d = np.load(sys.argv[1] + '/samples.npz'); E = float(d['E_ref']); vt, h1, h2, wid = d['vt'], d['h1'], d['h2'], d['wid']
H1 = h1 - E * vt; H2 = h2 - 2 * E * h1 + E ** 2 * vt
def est(m):
    s = (np.mean(vt[m] ** 2), np.mean(vt[m] * H1[m]), np.mean(H1[m] ** 2), np.mean(H1[m] * H2[m]), np.mean(H2[m] ** 2)); r = from_sums(s, E); r['chk_vtH2'] = np.mean(vt[m] * H2[m]); r['chk_H1sq'] = s[2]; return r
allm = np.ones(len(vt), bool); r0 = est(allm); ws = np.unique(wid); keys = ('E', 'var', 'alpha', 'E1', 'mu3')
J = [est(wid != w) for w in ws]; out = dict(n=int(len(vt)), W=int(len(ws)), E_ref=E)
for k in keys:
    v = np.array([j[k] for j in J]); out[k] = float(r0[k]); out[k + '_err'] = float(np.sqrt((len(ws) - 1) / len(ws) * np.sum((v - v.mean()) ** 2)))
out['chk_vtH2'] = float(r0['chk_vtH2']); out['chk_H1sq'] = float(r0['chk_H1sq'])
# time-order halves (rounds) as a drift/independence look
rn = d['rnd']; hi = rn >= np.median(rn); out['alpha_firsthalf'] = float(est(~hi)['alpha']); out['alpha_secondhalf'] = float(est(hi)['alpha'])
print(json.dumps(out, indent=1)); json.dump(out, open(sys.argv[1] + '/vmc_result.json', 'w'), indent=1)
