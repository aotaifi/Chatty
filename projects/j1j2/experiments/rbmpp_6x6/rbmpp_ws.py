"""w_s (wrong-sign weight against the exact ground state) of saved loop signs: w_s = (1 - |sum_r p0_r s_r s0_r|)/2,
p0 = exact ground-state probability per orbit (st6_sector.score). Reads s_<k>.npy (int8) of cx_chop2 runs and the
guide table's sP.
  python rbmpp_ws.py OUT.json GUIDE_TABLE.npz RUN_DIR [RUN_DIR ...]
"""
import json, os, sys
import numpy as np
ROOT = '/project/theorie/a/A.Otaifi/Chatty/runs/stall6'
z = np.load(os.environ.get('ST6_TABLE', ROOT + '/data/psi0_6x6_table.npz'))
v0 = z['amp'] * np.sqrt(z['orbit'].astype(np.float64)); v0 /= np.linalg.norm(v0)
p0 = v0 ** 2; s0 = np.where(v0 >= 0, 1.0, -1.0)


def ws(s):
    ov = float(np.sum(p0 * s * s0))
    return max(0.0, (1 - abs(ov)) / 2)


out = dict(guide=ws(np.load(sys.argv[2])['sP'].astype(np.float64)), runs={})
for d in sys.argv[3:]:
    r = []
    k = 1
    while os.path.exists(os.path.join(d, f's_{k}.npy')):
        r.append(dict(it=k, w_s=ws(np.load(os.path.join(d, f's_{k}.npy')).astype(np.float64)))); k += 1
    out['runs'][os.path.basename(d.rstrip('/'))] = r
json.dump(out, open(sys.argv[1], 'w'), indent=1)
print(json.dumps(out, indent=1))
