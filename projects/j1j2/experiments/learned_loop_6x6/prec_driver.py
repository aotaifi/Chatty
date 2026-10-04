import subprocess, sys
for v in ['float64', 'float32', 'float32+ln2', 'float32+tf32']:
    r = subprocess.run([sys.executable, 'prec_test.py', v], capture_output=True, text=True)
    print([l for l in r.stdout.splitlines() if 'PREC' in l] or r.stderr[-2000:], flush=True)
