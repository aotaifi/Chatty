"""Test L8 decision gate (README compress_6x6 "Test L8"): E_FN(g_1) vs E_FN(g_0), both p = 1, two chains each, from gfmc8_analyze.py's
analysis_final.json of each root.
  python l8_decide.py G1ROOT [--g0 /scratch/a/A.Otaifi/gfmc_m13b/prod/rb8]
Runs gfmc8_analyze.py on G1ROOT (chains p1_s1, p1_s2; quotable.py verdict), reads G0's analysis_final.json, and prints one line
  L8_VERDICT GAIN|NO_GAIN|LOSS|UNDECIDED|NOT_QUOTABLE  dE = E_FN(g1) - E_FN(g0) = ... +- sigma (z)
Rule: dE < -3 sigma GAIN; |dE| <= 2 sigma NO_GAIN; dE > +3 sigma LOSS; 2-3 sigma UNDECIDED (pre-registration names no branch: lead decides).
A g_1 pair that is NOT QUOTABLE or whose chains differ by > 3 sigma gives NOT_QUOTABLE (continue the chains, no verdict).
"""
import sys, os, json, subprocess
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
g1 = sys.argv[1]
g0 = sys.argv[sys.argv.index('--g0') + 1] if '--g0' in sys.argv else '/scratch/a/A.Otaifi/gfmc_m13b/prod/rb8'
r = subprocess.run(['python3', os.path.join(HERE, '..', 'gfmc8_analyze.py'), g1], capture_output=True, text=True)
print(r.stdout.strip()); print(r.stderr.strip()[-2000:])
F1 = json.load(open(os.path.join(g1, 'analysis_final.json')))['p1']; F0 = json.load(open(os.path.join(g0, 'analysis_final.json')))['p1']
d = F1['E_FN_site'] - F0['E_FN_site']; s = float(np.hypot(F1['err'], F0['err'])); z = d / s
q = F1.get('quotable', ''); okq = ('QUOTABLE' in q and 'NOT QUOTABLE' not in q) and abs(F1.get('chain_diff_sigma', 99)) <= 3
if not okq: v = 'NOT_QUOTABLE'
elif z < -3: v = 'GAIN'
elif abs(z) <= 2: v = 'NO_GAIN'
elif z > 3: v = 'LOSS'
else: v = 'UNDECIDED'
out = dict(verdict=v, dE=d, sigma=s, z=z, E_g1=F1, E_g0=F0)
json.dump(out, open(os.path.join(g1, 'l8_decision.json'), 'w'), indent=1, default=float)
print(f'L8_VERDICT {v} dE = E_FN(g1) - E_FN(g0) = {d:+.2e} +- {s:.1e} per site ({z:+.1f} sigma); E_FN(g1) {F1["E_FN_site"]:.7f} +- {F1["err"]:.1e} '
      f'[{q[:80]}; chain diff {F1.get("chain_diff_sigma", float("nan")):+.1f} sigma, {F1["gpu_h"]:.2f} H100-h]; E_FN(g0) {F0["E_FN_site"]:.7f} +- {F0["err"]:.1e}')
