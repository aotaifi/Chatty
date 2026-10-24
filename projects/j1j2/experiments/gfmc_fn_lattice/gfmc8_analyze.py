"""Analysis of the M1.3b 8x8 GFMC chains (gfmc_run.py output) and the VMC p = 0 / p = 1 estimator (vmc_moments.py).

  python gfmc8_analyze.py RUNROOT [--burn_tau 10] [--ref -0.498886] [--out DIR]
RUNROOT holds one directory per chain, named p<P>_s<seed> (blocks.npz, result.json, progress.txt), and optionally vmc/ (vmc_s*.json,
vmc_pooled.json, E0_s*.npy, E1_s*.npy).  Per chain: mixed and growth estimator (energies per site, absolute), blocking error (plateau of
the last 4 levels), integrated autocorrelation time (blocks and tau), drift over fifths (qmc-statistics: relaxation vs trendless scatter),
ESS, and the cost (GPU-seconds = sum of block seconds).  Per guide p: the two independent chains (own init and RNG) are combined; the
series (burn removed) are written to OUT/E_p<P>_s<seed>.npy for tools/quotable.py, which is run on the pair and its verdict line printed.
Final table: E_FN(p), VMC E(p), gap E_VMC - E_FN, GPU-h per number, against the published -0.498886 (RBM+PP, 8x8), -0.49906 (extrapolated)
and the DMRG -0.4992.  Bias scans (dtau, M) are chains of the same p with other names p<P>_s<seed>_M<M>_dt<dtau> and are fitted if present.
"""
import sys, os, json, glob, re, subprocess
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gfmc_fn_lib import blocked_error, iat

root = sys.argv[1]
burn_tau = float(sys.argv[sys.argv.index('--burn_tau') + 1]) if '--burn_tau' in sys.argv else 10.0
out = sys.argv[sys.argv.index('--out') + 1] if '--out' in sys.argv else root
os.makedirs(out, exist_ok=True)
REF = dict(published_RBMPP=-0.498886, published_extrap=-0.49906, DMRG=-0.4992)
QUOT = '/project/theorie/a/A.Otaifi/tools/quotable.py'
rows = []
for d in sorted(glob.glob(os.path.join(root, 'p[01]_*'))):
    f = os.path.join(d, 'blocks.npz')
    if not os.path.exists(f): continue
    z = np.load(f); dt = float(z['dtau']); M = int(z['M']); N = int(z['N']); p = int(z['p'])
    nb0 = max(int(round(burn_tau / dt)), int(z['burn']))
    name = os.path.basename(d)
    seed = int(re.search(r'_s(\d+)', name).group(1)) if re.search(r'_s(\d+)', name) else -1
    r = dict(name=name, p=p, seed=seed, M=M, dtau=dt, nblocks=len(z['emix']), burn_blocks=nb0, gpu_s=float(np.sum(z['sec'])))
    for key in ('emix', 'egrow'):
        x = z[key][nb0:] / N
        if len(x) < 32: continue
        m, errs, tb = blocked_error(x)
        fifths = [float(np.mean(c)) for c in np.array_split(x, 5)]
        r[key] = dict(mean=m, err=max(errs[-4:]), err0=errs[0], iat_blocks=iat(x), iat_tau=iat(x) * dt, fifths=fifths,
                      drift_sigma=(fifths[-1] - fifths[0]) / (max(max(errs[-4:]), 1e-30) * np.sqrt(10.0)))   # error of a fifth = err*sqrt(5); of a difference of two fifths = err*sqrt(10)
    r['ess_over_M'] = float(np.mean(z['ess'][nb0:]) / M); r['elvar_total'] = float(np.mean(z['elvar'][nb0:]))
    r['sec_per_walker_tau'] = float(np.median(z['sec']) / (M * dt)); r['sigma_walker_site'] = float(np.sqrt(np.mean(z['elvar'][nb0:])) / N)
    rows.append(r)
    if 'emix' in r: np.save(os.path.join(out, f'E_{name}.npy'), z['emix'][nb0:] / N)
for r in rows:
    if 'emix' not in r: print(r['name'], 'too few blocks'); continue
    e = r['emix']
    print(f"{r['name']:24s} p={r['p']} M={r['M']:5d} dtau={r['dtau']:.2f} nb={r['nblocks']:6d} E_mix/N {e['mean']:.7f} +- {e['err']:.1e} (iat {e['iat_blocks']:.1f} blk = {e['iat_tau']:.2f} tau; "
          f"drift {e['drift_sigma']:+.1f} sig)  grow {r['egrow']['mean']:.7f} +- {r['egrow']['err']:.1e}  sigma_walker/N {r['sigma_walker_site']:.2e}  "
          f"{r['sec_per_walker_tau']:.3g} s/walker-tau  ESS/M {r['ess_over_M']:.2f}  GPU {r['gpu_s'] / 3600:.2f} h")
json.dump(rows, open(os.path.join(out, 'analysis_chains.json'), 'w'), indent=1, default=float)
final = {}
for P in (0, 1):
    ch = [r for r in rows if r['p'] == P and 'emix' in r and re.fullmatch(rf'p{P}_s\d+', r['name'])]
    if not ch: continue
    v = np.array([r['emix']['mean'] for r in ch]); s = np.array([r['emix']['err'] for r in ch]); w = 1 / s ** 2
    mean = float(np.sum(w * v) / w.sum()); err = float(1 / np.sqrt(w.sum()))
    gpu_h = sum(r['gpu_s'] for r in ch) / 3600
    entry = dict(E_FN_site=mean, err=err, chains=[r['name'] for r in ch], gpu_h=gpu_h)
    if len(ch) >= 2:
        entry['chain_diff_sigma'] = float((v[0] - v[1]) / np.hypot(s[0], s[1]))
        try:
            q = subprocess.run(['python3', QUOT, 'series'] + [os.path.join(out, f'E_{r["name"]}.npy') for r in ch[:2]] +
                               ['--name', f'E_FN/N 8x8 p={P}', '--plot', os.path.join(out, f'quotable_p{P}.png')], capture_output=True, text=True)
            entry['quotable'] = (q.stdout.strip().splitlines() or [q.stderr.strip()])[0]
        except Exception as ex:
            entry['quotable'] = f'quotable.py failed: {ex}'
    final[f'p{P}'] = entry
    print(f'FINAL GFMC p={P}: E_FN/N = {mean:.7f} +- {err:.1e} from {len(ch)} chains, {gpu_h:.2f} H100-h' +
          (f'; chain difference {entry["chain_diff_sigma"]:+.1f} sigma; {entry["quotable"]}' if 'quotable' in entry else ''))
for tag, sub in (('vmc_pooled.json', 'V'), ('vmc_pooled.json', 'vmc'), ('vmc_s1.json', 'V'), ('vmc_s1.json', 'vmc')):
    f = os.path.join(root, sub, tag)
    if os.path.exists(f):
        v = json.load(open(f)); ps = v['per_site']
        final['vmc'] = dict(E0=ps['E'], E1=ps['E1'], var0=ps['var'], var1=ps['var1'], alpha=v['alpha']['value'], file=f)
        print(f'VMC ({tag}): p=0 E/N {ps["E"]["value"]:.7f} +- {ps["E"]["err"]:.1e}  p=1 E_1/N {ps["E1"]["value"]:.7f} +- {ps["E1"]["err"]:.1e}  alpha {v["alpha"]["value"]:.4f}')
        break
print('references (per site): ' + ', '.join(f'{k} {x}' for k, x in REF.items()))
for P in (0, 1):
    if f'p{P}' in final and 'vmc' in final:
        ev = final['vmc']['E0' if P == 0 else 'E1']['value']; ef = final[f'p{P}']['E_FN_site']
        print(f'p={P}: E_VMC - E_FN = {ev - ef:+.3e} per site;  E_FN - DMRG(-0.4992) = {ef - REF["DMRG"]:+.3e};  E_FN - published(-0.498886) = {ef - REF["published_RBMPP"]:+.3e}')
json.dump(final, open(os.path.join(out, 'analysis_final.json'), 'w'), indent=1, default=float)
