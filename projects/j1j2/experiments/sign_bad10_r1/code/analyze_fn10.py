"""analyze_fn10.py --e0 DIR1 DIR2 --e1 DIR1 DIR2 --out OUTDIR : quotable.py series per arm (E_mix.npy = post-burn block energies/site), then g_1 = e_0 - e_1.
READ (T-BAD10): round USEFUL iff g_1 > 2 sigma(g_1) and both arms QUOTABLE; sigma(g) = hypot of the arms' quotable errors.  KILL of 'the loop improves the sign every round at 10x10': g_1 not beyond 2 sigma with data adequate -> diagnosis + Astra."""
import sys, os, re, json, argparse, subprocess
import numpy as np
sys.path.insert(0, '/project/theorie/a/A.Otaifi/tools'); import quotable as QT
ap = argparse.ArgumentParser(); ap.add_argument('--e0', nargs=2, required=True); ap.add_argument('--e1', nargs=2, required=True); ap.add_argument('--out', required=True); ap.add_argument('--bias_json', default=None); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
Q = '/project/theorie/a/A.Otaifi/tools/quotable.py'; res = {}; halves_bad = []; zs = []
for tag, dirs in (('e0', a.e0), ('e1', a.e1)):
    files = [d + '/E_mix.npy' for d in dirs]
    for f in files:
        x = np.load(f); res[f] = dict(n=int(len(x)), mean=float(x.mean()), sd_block=float(x.std(ddof=1)))
        h = len(x) // 2; A, B = x[:h], x[h:2 * h]; tau = QT.series_chain(x).get('tau_int', 0.5); tau = 0.5 if not np.isfinite(tau) else max(tau, 0.5)
        se = lambda v: v.std(ddof=1) / np.sqrt(len(v)) * np.sqrt(2 * tau); z = (B.mean() - A.mean()) / np.hypot(se(A), se(B))
        res[f]['halves'] = dict(first=float(A.mean()), second=float(B.mean()), se_first=float(se(A)), se_second=float(se(B)), z=float(z), tau_int=float(tau))
        print(f'HALVES {tag} {f}: first half {A.mean():.6f} +- {se(A):.1e}, second half {B.mean():.6f} +- {se(B):.1e}, z = {z:+.2f}')
        zs.append(z)
        if abs(z) > 3: halves_bad.append(f)
    q = subprocess.run(['python3', Q, 'series', *files, '--name', tag, '--plot', f'{a.out}/quotable_{tag}.png'], capture_output=True, text=True); print(q.stdout)
    res[tag + '_quotable'] = q.stdout; res[tag + '_rc'] = q.returncode
    m = re.search(r'^QUOTABLE: .*= ([-+0-9.eE]+) \+- ([0-9.eE+-]+) \(', q.stdout, re.M)
    if m: res[tag + '_value'] = float(m.group(1)); res[tag + '_err'] = float(m.group(2))
from scipy import stats as _st
if zs:
    chi2 = float(np.sum(np.square(zs))); p = float(_st.chi2.sf(chi2, len(zs))); res['halves_chi2'] = dict(chi2=chi2, dof=len(zs), p=p); print(f'HALVES chi2 over all {len(zs)} chains of both arms: {chi2:.2f}, p = {p:.3g}')
    if p < 0.01 and not halves_bad: halves_bad.append(f'chi2 of halves z over all chains, p = {p:.2g}')
if halves_bad:
    print('NOT QUOTABLE: halves differ (any chain |z| > 3, or chi2 p < 0.01 over all chains; drift or long correlation, not noise): ' + '; '.join(halves_bad)); res['halves_fail'] = halves_bad
    for k in ('e0_value', 'e1_value'): res.pop(k, None)
if all(k in res for k in ('e0_value', 'e1_value')):
    g = res['e0_value'] - res['e1_value']; sg = float(np.hypot(res['e0_err'], res['e1_err'])); res['g1'] = g; res['sigma_g1'] = sg
    if a.bias_json:
        bj = json.load(open(a.bias_json)); B64 = (abs(bj['diff']) + 2 * bj['sigma_d']) / 3; res['B64'] = B64; res['bias_json'] = a.bias_json
        br = 'KILL INPUT (g_1 not beyond 2 sigma)' if not g > 2 * sg else ('RESOLVED (g_1 - B64 > 2 sigma)' if g - B64 > 2 * sg else 'INCONCLUSIVE (g_1 beyond 2 sigma, only the bias bound fails)')
        res['branch'] = br; print(f'BIAS B64 = (|dM| + 2 sigma_d)/3 = {B64:.3g} (dM {bj["diff"]:.3g}, sigma_d {bj["sigma_d"]:.3g} from {a.bias_json}); g_1 - B64 = {g - B64:.4g}; BRANCH: {br}')
    print(f'READ g_1 = e_0 - e_1 = {g:.6g} +- {sg:.2g} per site; both arms quotable; USEFUL (g_1 > 2 sigma): {g > 2 * sg}; sigma <= 1e-4 per arm: {res["e0_err"] <= 1e-4 and res["e1_err"] <= 1e-4}')
else: print('READ: at least one arm NOT QUOTABLE -> g_1 not quoted (plot only); per-chain means in fn_summary.json')
json.dump(res, open(a.out + '/fn_summary.json', 'w'), indent=1)
