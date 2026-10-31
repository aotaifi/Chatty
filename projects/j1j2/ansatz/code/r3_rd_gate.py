"""r3_rd_gate.py READOUTDIR: prints the round-3 readout summary (bare e, drop, stall rule, scan, c* by the parabola through the 3 lowest points, e_aug_min).  One VERDICT line."""
import sys, json, numpy as np
r = json.load(open(f'{sys.argv[1]}/readout.json')); sc = {float(k): v for k, v in r.get('scan', {}).items()}
ok = bool(r.get('done')) and 'e_aug_min_site' in r
print('VERDICT: ' + ('READOUT OK' if ok else 'READOUT NOT OK (not done)'))
print(f'bare e {r.get("e_fit_site")} drop {r.get("drop")} ({r.get("kill")}); vs e(1200) 1.313344e-5: diff {r.get("e_fit_site", 0) - 1.313343631083165e-05:.4e} (limit 0.05e-6)')
print('scan ' + ' '.join(f'{c}:{v:.6e}' for c, v in sorted(sc.items())))
if len(sc) >= 3:
    low = sorted(sorted(sc, key=sc.get)[:3]); p = np.polyfit(low, [sc[c] for c in low], 2); cp = -p[1] / (2 * p[0])
    print(f'c* (solved min) {min(sc, key=sc.get)} e_aug_min {min(sc.values()):.6e}; parabola through {low}: c* {cp:.3f} e {np.polyval(p, cp):.6e}; status {r.get("status")}')
