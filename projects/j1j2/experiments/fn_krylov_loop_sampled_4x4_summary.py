#!/usr/bin/env python3
"""Summarize results/fn_krylov_loop_sampled_4x4/*.json: w_s and eps at it=10, it=30 and plateau
(per-seed mean over the last 10 iterations), mean +- std over seeds, grouped by (sampler, N/M, rule)."""
import json, glob, re
from collections import defaultdict
import numpy as np

R = "/Users/aliotaifi/Chatty-organize/projects/j1j2/results/fn_krylov_loop_sampled_4x4"
groups = defaultdict(list)
for fn in sorted(glob.glob(R + "/*.json")):
    if "summary" in fn or "ess" in fn:
        continue
    d = json.load(open(fn))
    name = re.sub(r"_seed\d+\.json$", "", fn.split("/")[-1])
    groups[name].append(d["history"])

out = {}
for name, hs in groups.items():
    nit = min(len(h) for h in hs) - 1
    rec = dict(n_seeds=len(hs), n_iter=nit)
    for key in ("w_s", "eps", "eps_sampled_signs_exact_phi"):
        if key not in hs[0][-1]:
            continue
        for lab, sel in (("it10", lambda h: h[10][key]), ("it30", lambda h: h[30][key] if len(h) > 30 else np.nan),
                         ("plateau", lambda h: np.mean([r[key] for r in h[-10:]]))):
            v = np.array([sel(h) for h in hs], float)
            rec[f"{key}_{lab}"] = [float(np.nanmean(v)), float(np.nanstd(v))]
    if "N_eff" in hs[0][-1]:
        rec["N_eff_median"] = float(np.median([r["N_eff"] for h in hs for r in h[1:]]))
    out[name] = rec
    f = lambda k: ("%.2e+-%.1e" % tuple(rec[k])) if k in rec else "-"
    print(f"{name:32s} seeds={len(hs)} it={nit} | w_s: {f('w_s_it10')} {f('w_s_it30')} {f('w_s_plateau')} | "
          f"eps: {f('eps_it10')} {f('eps_it30')} {f('eps_plateau')} | eps(exact phi): {f('eps_sampled_signs_exact_phi_plateau')}"
          + (f" | Neff~{rec['N_eff_median']:.2g}" if "N_eff_median" in rec else ""))
json.dump(out, open(R + "/summary.json", "w"), indent=1)
