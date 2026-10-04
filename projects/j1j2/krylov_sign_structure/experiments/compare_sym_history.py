#!/usr/bin/env python3
"""Compare a symmetric-sector loop history against a stored full-basis history, iteration by iteration."""
import json, sys
import numpy as np
new = json.load(open(sys.argv[1])); old = json.load(open(sys.argv[2]))
hn, ho = new["history"], old["history"]
n = min(len(hn), len(ho))
print("lengths new/old", len(hn), len(ho), "E0 new/old", new["E0"], old["E0"])
print("summary new", new["summary"]); print("summary old", old["summary"])
for key in ["w_s", "eps", "E_FN", "F_amp", "amp_delta_l2", "n_hist", "r_groups"]:
    dev = []; first_bad = None
    for k in range(n):
        a, b = hn[k].get(key), ho[k].get(key)
        if a is None or b is None: continue
        d = abs(a - b) / (abs(b) + (1e-12 if key in ("w_s",) else 1e-300))
        d = abs(a - b) if key in ("w_s", "n_hist", "r_groups") else abs(a - b) / max(abs(b), 1e-14)
        dev.append(d)
        if first_bad is None and d > (1e-6 if key not in ("n_hist", "r_groups", "w_s") else 1e-9): first_bad = k
    print(f"{key:14s} max dev {np.max(dev):.3e}  first dev>tol at it {first_bad}")
