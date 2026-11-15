"""analyze_fn10.py --e0 DIR1 DIR2 --e1 DIR1 DIR2 --out OUTDIR : quotable.py series per arm (E_mix.npy = post-burn block energies/site), then g_1 = e_0 - e_1.
READ (T-BAD10): round USEFUL iff g_1 > 2 sigma(g_1) and both arms QUOTABLE; sigma(g) = hypot of the arms' quotable errors.  KILL of 'the loop improves the sign every round at 10x10': g_1 not beyond 2 sigma with data adequate -> diagnosis + Astra."""
import sys, os, re, json, argparse, subprocess
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('--e0', nargs=2, required=True); ap.add_argument('--e1', nargs=2, required=True); ap.add_argument('--out', required=True); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
Q = '/project/theorie/a/A.Otaifi/tools/quotable.py'; res = {}
for tag, dirs in (('e0', a.e0), ('e1', a.e1)):
    files = [d + '/E_mix.npy' for d in dirs]
    for f in files: x = np.load(f); res[f] = dict(n=int(len(x)), mean=float(x.mean()), sd_block=float(x.std(ddof=1)))
    q = subprocess.run(['python3', Q, 'series', *files, '--name', tag, '--plot', f'{a.out}/quotable_{tag}.png'], capture_output=True, text=True); print(q.stdout)
    res[tag + '_quotable'] = q.stdout; res[tag + '_rc'] = q.returncode
    m = re.search(r'^QUOTABLE: .*= ([-+0-9.eE]+) \+- ([0-9.eE+-]+) \(', q.stdout, re.M)
    if m: res[tag + '_value'] = float(m.group(1)); res[tag + '_err'] = float(m.group(2))
if all(k in res for k in ('e0_value', 'e1_value')):
    g = res['e0_value'] - res['e1_value']; sg = float(np.hypot(res['e0_err'], res['e1_err'])); res['g1'] = g; res['sigma_g1'] = sg
    print(f'READ g_1 = e_0 - e_1 = {g:.6g} +- {sg:.2g} per site; both arms quotable; USEFUL (g_1 > 2 sigma): {g > 2 * sg}; sigma <= 1e-4 per arm: {res["e0_err"] <= 1e-4 and res["e1_err"] <= 1e-4}')
else: print('READ: at least one arm NOT QUOTABLE -> g_1 not quoted (plot only); per-chain means in fn_summary.json')
json.dump(res, open(a.out + '/fn_summary.json', 'w'), indent=1)
