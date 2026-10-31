"""r2g_cfill.py ARM(g1|g2): extra c solves for r2g_readout.py; appends res['c_fill'] to readout/ARM/readout.json."""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bond_split as BS
import delta0 as D0M
ARM = sys.argv[1]; T0 = time.time(); R2 = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r2g'; RJ = f'{R2}/readout/{ARM}/readout.json'
res = json.load(open(RJ)); assert res['done']
CS = {'g1': [3.5, 4.5], 'g2': [3.5, 4.5]}[ARM]
n = np.load(f'{BS.CSRD}/norb.npy').astype(np.float64)
s = np.load(res['s1']).astype(np.float64); lF = np.load(res['fit']).astype(np.float64); fJ = np.load(f'{R2}/readout/{ARM}/fJ_fit.npy').astype(np.float64)
K, Vm, D0 = BS.build(n, s); print('built', time.time() - T0, flush=True)
cf = {}
for c in CS:
    r, _ = D0M.efn_vec(K, Vm, D0, n, lF + c * fJ); cf[str(c)] = r['dE_site']; print(c, r['dE_site'], r['iters'], r['res'], flush=True)
    res['c_fill'] = dict(cf, host=os.uname().nodename); json.dump(res, open(RJ, 'w'), indent=1, default=float)
pts = {float(k): v for k, v in res['scan'].items()}; pts.update({float(k): v for k, v in cf.items() if k != 'host'})
low = sorted(sorted(pts, key=pts.get)[:3]); p = np.polyfit(low, [pts[c] for c in low], 2)
res['c_fill'].update(parabola_pts=low, c_star_parab=float(-p[1] / (2 * p[0])), e_parab_min=float(np.polyval(p, -p[1] / (2 * p[0]))), solved_min_c=min(pts, key=pts.get), solved_min_e=min(pts.values()), all_pts={str(k): v for k, v in sorted(pts.items())}, sec=time.time() - T0)
json.dump(res, open(RJ, 'w'), indent=1, default=float); print(res['c_fill'], flush=True)
