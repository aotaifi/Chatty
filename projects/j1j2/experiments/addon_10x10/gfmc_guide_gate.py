"""A0.4 item 2 gate (pre-registered in /scratch/a/A.Otaifi/vit10_runs/a04/ANALYSIS_BRIEF.md).
  python3 gfmc_guide_gate.py RUN      RUN/{s1,s2}/E_mix.npy (two independent FN p=0 chains), RUN/broken/E_mix.npy (drop V_sf)
Reference: exact lattice FN energy of the 6x6 RBM+PP table-A guide, table_A.json guide.dE_FN_site (st6_sector.fn_solve), overridden by
RUN/../fnexact/fn_exact.json when present (recomputed here).  Pass: |dE_GFMC - dE_FN| < 2 sigma (sigma from quotable.py, two chains, series)
and the broken run misses by > 5 sigma_broken.  Last line 'A04_GATE ...'."""
import json, os, re, subprocess, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, 'm1_code'))
from gfmc_fn_lib import blocked_error
RUN = sys.argv[1]; E0 = -0.503809653890878                       # exact 6x6 ground state per site (table_A: E_site - dE_site)
ref = 3.524943767886491e-05; src = 'table_A.json guide.dE_FN_site'
fx = os.path.join(os.path.dirname(RUN.rstrip('/')), 'fnexact', 'fn_exact.json')
if os.path.exists(fx):
    r2 = json.load(open(fx))['dE_FN_site']; print(f'recomputed exact dE_FN_site {r2:.10e} (table_A {ref:.10e}, diff {r2 - ref:.1e})')
    ref, src = r2, 'fn_exact.json (recomputed)'
files = []
for c in ('s1', 's2'):
    e = np.load(os.path.join(RUN, c, 'E_mix.npy')) - E0; f = os.path.join(RUN, f'dE_{c}.npy'); np.save(f, e); files.append(f)
q = subprocess.run([sys.executable, '/project/theorie/a/A.Otaifi/tools/quotable.py', 'series'] + files +
                   ['--name', 'dE_FN/N 6x6 p=0 (GFMC, new driver)', '--plot', os.path.join(RUN, 'quotable_item2.png')], capture_output=True, text=True)
print(q.stdout.strip()); qline = q.stdout.splitlines()[0] if q.stdout else 'no quotable output'
m = re.search(r'=\s*([-+0-9.eE]+)\s*\+-\s*([0-9.eE+-]+)', qline)
if m:
    val, err = float(m.group(1)), float(m.group(2))
else:                                             # NOT QUOTABLE: report the inverse-variance mean of the chains (exact gate then FAILS)
    st = [blocked_error(np.load(f)) for f in files]; w = np.array([1 / max(s[1][-4:]) ** 2 for s in st])
    val = float(np.sum(w * np.array([s[0] for s in st])) / w.sum()); err = float(1 / np.sqrt(w.sum()))
z = (val - ref) / err
eb = np.load(os.path.join(RUN, 'broken', 'E_mix.npy')) - E0; mb, errs, _ = blocked_error(eb); sb = max(errs[-4:])
zb = (mb - ref) / sb
ok1 = qline.startswith('QUOTABLE') and abs(z) < 2; ok2 = abs(zb) > 5
out = dict(quotable=qline, value=val, err=err, ref=ref, ref_source=src, z=z, broken_mean=mb, broken_err=sb, broken_z=zb, PASS_exact=ok1, PASS_broken=ok2)
json.dump(out, open(os.path.join(RUN, 'gate_item2.json'), 'w'), indent=1)
print(f"A04_GATE item2: exact {'PASS' if ok1 else 'FAIL'} (dE_FN/N GFMC {val:.4e} +- {err:.1e} vs exact {ref:.4e}: {z:+.2f} sigma) | "
      f"broken drop_vsf {'PASS' if ok2 else 'FAIL'} (dE {mb:.4e} +- {sb:.1e}: {zb:+.0f} sigma) | {qline}")
