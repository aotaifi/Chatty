"""pool_x_pilot_cost.py RUND [--cap 280 --ntasks 40 --E 4]: projected production core-h from the pilot tasks' own thin and s/sweep/walker (shared-PT design: draws of all 3 rungs come
from the same ensembles, so the sweeps are set by the largest rung target S_.5).  Prints a verdict line."""
import sys, glob, json, math, argparse
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('rund'); ap.add_argument('--cap', type=float, default=280.0); ap.add_argument('--ntasks', type=int, default=40); ap.add_argument('--E', type=int, default=4)
ap.add_argument('--smax', type=float, default=4.5e5); a = ap.parse_args()
fs = sorted(glob.glob(a.rund + '/out_*.npz')); assert fs, 'no pilot outputs'
thin = []; sw = []; tau = []; burnpil = None
for f in fs:
    z = np.load(f); thin.append(int(z['thin'])); sw.append(float(z['s_per_sweep_per_walker_draw'])); tau.append([max(t['tau_lnB2'], t['tau_dec']) for t in json.loads(str(z['tau']))]); burnpil = int(z['burn']) + int(z['pilot'])
print('thin per pilot task', thin, ' s/sweep/walker', np.round(sw, 3).tolist(), ' max tau per rung (task rows)', np.round(tau, 2).tolist())
def cost(th, s_w, E=a.E, nt=a.ntasks):
    draw = math.ceil(a.smax / nt / E) * th * 3 * E; start = burnpil * 3 * E          # walker-sweeps per task
    return nt * (draw + start) * s_w / 3600.0, nt * start * s_w / 3600.0
sw_m = float(np.median(sw)); rows = []
for th in sorted(set(thin + [max(thin)])):
    c, o = cost(th, sw_m); rows.append((th, c, o)); print(f'thin {th}: projected {c:.0f} core-h (of which burn+pilot overhead {o:.0f}) at {sw_m:.3f} s/sweep/walker')
cmed = cost(int(np.median(thin)), sw_m)[0]; cmax = cost(max(thin), sw_m)[0]
print(f'VERDICT pool_x pilot: thin median {int(np.median(thin))} max {max(thin)}; projected {cmed:.0f} core-h (median thin) / {cmax:.0f} (max thin) vs cap {a.cap:.0f}: ' + ('WITHIN CAP' if cmax <= a.cap else 'EXCEEDS CAP, stop and report'))
