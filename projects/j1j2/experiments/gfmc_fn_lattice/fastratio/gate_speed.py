"""gate for the fastratio speed job: python3 gate_speed.py RUN   -> prints GATE lines and a final `VERDICT` summary"""
import json, sys, os
import numpy as np
R = sys.argv[1]; out = {}
b = json.load(open(f'{R}/bench.json'))
for s in ('8x8', '10x10'):
    if 'speedup_median' not in b.get(s, {}):
        out[s] = dict(speedup=float('nan'), error=str(b.get(s))[:300]); print('GATE kernel', s, 'MISSING', out[s]['error']); continue
    out[s] = dict(ref_ms=b[s]['ref']['ms_per_call_median'], fast_ms=b[s]['fast']['ms_per_call_median'], speedup=b[s]['speedup_median'],
                  ref_temp_GiB=b[s]['ref']['temp_GiB'], fast_temp_GiB=b[s]['fast']['temp_GiB'], EL_rel_err=b[s]['EL_rel_err_max'], fallback_fast=b[s]['fast']['n_fallback'])
    print('GATE kernel', s, json.dumps(out[s]))
rec = np.array([-31.91460454, -31.92512456, -31.92364442])      # launched code, gpu_profile/equiv.txt (job 4075217, H100)
try:
    e = {m: np.load(f'{R}/eq_{m}/blocks.npz') for m in ('ref', 'fast')}
    d = np.abs(e['fast']['emix'] - e['ref']['emix']) / np.abs(e['ref']['emix']); out['eq_relerr_per_block'] = d.tolist()
    out['eq_ref_vs_launched_abs'] = np.abs(e['ref']['emix'] - rec).tolist()
    for m in ('ref', 'fast'):
        sec = e[m]['sec']; out[f'eq_{m}_sec_per_block'] = sec.tolist(); out[f'eq_{m}_walker_tau_per_s'] = float(64 * 0.1 / np.median(sec))
    out['eq_speedup_walker_tau'] = out['eq_fast_walker_tau_per_s'] / out['eq_ref_walker_tau_per_s']
    print('GATE equivalence rel err per block', d.tolist(), '| ref vs launched (abs, 8 printed digits)', out['eq_ref_vs_launched_abs'], '| walker-tau/s ref %.3f fast %.3f' % (out['eq_ref_walker_tau_per_s'], out['eq_fast_walker_tau_per_s']))
    eq_ok = bool((d <= 1e-10).all())
except Exception as ex:
    print('GATE equivalence MISSING', ex); eq_ok = False
for m in ('ref', 'fast'):
    p = f'{R}/g10_{m}/blocks.npz'
    if os.path.exists(p):
        z = np.load(p); out[f'g10_{m}_walker_tau_per_s'] = float(32 * 0.1 / np.median(z['sec'])); out[f'g10_{m}_emix'] = z['emix'].tolist()
if 'g10_ref_walker_tau_per_s' in out and 'g10_fast_walker_tau_per_s' in out:
    out['g10_speedup_walker_tau'] = out['g10_fast_walker_tau_per_s'] / out['g10_ref_walker_tau_per_s']; print('GATE g10 walker-tau/s speedup', out['g10_speedup_walker_tau'])
json.dump(out, open(f'{R}/gate_speed.json', 'w'), indent=1)
print('VERDICT speedup kernel 8x8 %.2fx 10x10 %.2fx | equivalence <=1e-10 per block: %s | gate >= 2x at 10x10: %s' % (out['8x8']['speedup'], out['10x10']['speedup'], eq_ok, 'PASS' if out['10x10']['speedup'] >= 2 else 'MISS'))
