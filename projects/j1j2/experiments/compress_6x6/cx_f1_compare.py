"""F1 control 2: window 3 (ctl1) vs window 5 with the outer ring frozen (ctl2) at equal steps.  usage: python3 cx_f1_compare.py RUN_A RUN_B
RUN_x = a run dir holding cx_f1.json and round_1/la_table.npy (both files on Paderborn: run it there with the venv python).
Prints the tax relative difference and the max |difference| of the log-amplitudes (after the common offset) on 4096 orbits (fixed seed) and on all orbits."""
import sys, json, numpy as np
a, b = sys.argv[1:3]
ja = json.load(open(f'{a}/cx_f1.json')); jb = json.load(open(f'{b}/cx_f1.json'))
la = np.load(f'{a}/round_1/la_table.npy'); lb = np.load(f'{b}/round_1/la_table.npy')
idx = np.random.default_rng(0).choice(la.size, 4096, replace=False)
d4 = la[idx] - lb[idx]; dall = la - lb
print('tax A', ja.get('tax'), 'tax B', jb.get('tax'), 'rel diff', abs(ja['tax'] / jb['tax'] - 1) if ja.get('tax') and jb.get('tax') else None)
print('steps', ja['fit']['steps'], jb['fit']['steps'], 'L_ho step0', ja['fit']['L_ho_step0'], jb['fit']['L_ho_step0'])
print('max|dlogA| 4096 orbits', float(np.max(np.abs(d4))), ' all orbits', float(np.max(np.abs(dall))), ' (tables are raw, same offset convention)')
