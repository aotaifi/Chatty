"""Collect FN populations (fn_*.json from several jobs) per guide tag into mean +- SE per site.
Usage: python ll6_summary.py RUNS_DIR   (prints a JSON table; groups by the guide-tag prefix before the last letter)"""
import sys, json, glob, os, re
import numpy as np
E0 = -0.5038096538908783
groups = {}
for f in glob.glob(os.path.join(sys.argv[1], '*', 'fn_*.json')):
    d = json.load(open(f))
    g = re.sub(r'[a-z]$', '', d['tag'])
    for r in d['reps']:
        groups.setdefault(g, {})[r['seed']] = r['Emean_site']
out = {}
for g, reps in sorted(groups.items()):
    e = np.array(list(reps.values()))
    se = e.std(ddof=1) / np.sqrt(len(e)) if len(e) > 1 else float('nan')
    out[g] = dict(n=len(e), E_FN_site=float(e.mean()), SE=float(se), SD=float(e.std(ddof=1)) if len(e) > 1 else None,
                  eps=float((e.mean() - E0) / abs(E0)), median=float(np.median(e)))
print(json.dumps(out, indent=1))
