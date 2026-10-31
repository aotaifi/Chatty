"""r3_gate.py DIR: gate of the round-3 GPU jobs (fit and table).  No Monte Carlo (deterministic fit + exact table), quotable.py does not apply.
DIR = pfaffian_r3/fit : r1_gate.py checks plus steps == 1500 and fit/round_1/theta_step{900,1200}.npy present, finite, size 2740.
DIR = pfaffian_r3/tabs: r1_gate.py checks on tab_900 and tab_1200, each with steps 0 and la_table.npy present.
The physics gate is the CPU readout (r3_readout_run.sh), run by the lead.  Prints one VERDICT line."""
import sys, os, json, subprocess
import numpy as np
D = sys.argv[1].rstrip('/'); here = os.path.dirname(os.path.abspath(__file__)); why = []; lines = []
runs = [D] if os.path.basename(D).endswith('fit') else [f'{D}/tab_900', f'{D}/tab_1200']
for r in runs:
    o = subprocess.run([sys.executable, f'{here}/r1_gate.py', r], capture_output=True, text=True); lines += [f'[{os.path.basename(r)}] ' + x for x in o.stdout.splitlines()]
    if o.returncode or 'VERDICT: FIT OK' not in o.stdout: why.append(f'{os.path.basename(r)}: r1_gate not OK')
    j = json.load(open(f'{r}/cx_r1.json')); st = j.get('fit', {}).get('steps')
    want = 1500 if os.path.basename(r).endswith('fit') else 0
    if st != want: why.append(f'{os.path.basename(r)}: steps {st} != {want}')
    if want == 1500:
        for N in (900, 1200):
            p = f'{r}/round_1/theta_step{N}.npy'
            if not os.path.exists(p): why.append(f'theta_step{N}.npy missing')
            else:
                a = np.load(p); lines.append(f'theta_step{N}: shape {a.shape} finite {bool(np.isfinite(a).all())}')
                if a.shape != (2740,) or not np.isfinite(a).all(): why.append(f'theta_step{N} bad')
print('VERDICT: ' + ('R3 OK' if not why else 'R3 NOT OK: ' + '; '.join(why)))
print('\n'.join(lines))
