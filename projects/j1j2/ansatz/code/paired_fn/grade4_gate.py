#!/usr/bin/env python3
"""grade4_gate.py RUN : extract per-chain Delta/site series (measured blocks) to RUN/gate/delta_<id>.npy and run quotable.py series on them (the gate for g_4 = -mean Delta/site)."""
import sys, glob, os, subprocess, numpy as np
run = sys.argv[1]; os.makedirs(f'{run}/gate', exist_ok=True); fs = []
for f in sorted(glob.glob(f'{run}/chain_*.npz')):
    if f.endswith('.tmp.npz'): continue
    z = np.load(f); m = ~z['burn']; d = (z['emix_B'][m] - z['emix_A'][m]) / 64; o = f'{run}/gate/delta_{os.path.basename(f)[6:-4]}.npy'; np.save(o, d); fs.append(o); print(o, len(d), d.mean())
sys.exit(subprocess.call(['python3', '/project/theorie/a/A.Otaifi/tools/quotable.py', 'series', *fs, '--name', 'Delta_site_n4_minus_n3', '--plot', f'{run}/gate/delta_quotable.png']))
