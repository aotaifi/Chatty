"""Aggregate fn_run_*.json (FN curves) and guide_sym_*.json / guide_vmc_s*.json (variational energies).
Usage: python aggregate_bound_check.py RESULTS_DIR"""
import sys, glob, json, os
import numpy as np
R = sys.argv[1]; N = 36
out = {}
# ---------------- FN
groups = {"T_FN_M128_b1.2_tau.025": ["repro", "b12a", "b12b"], "T_FN_M128_b2.4_tau.025": ["b24a", "b24b"],
          "T_FN_M128_b1.2_tau.0125": ["taua", "taub"], "T_old_M128_b1.2_tau.025": ["told_a", "told_b"]}
files = {os.path.basename(f)[7:-5]: f for f in glob.glob(f"{R}/fn/*/fn_run_*.json")}
fn = {}
for g, tags in groups.items():
    reps = []
    for t in tags:
        if t in files:
            d = json.load(open(files[t])); reps += d["reps"]; beta = d["beta_target"]; burn = d["burn_beta"]
    if not reps: continue
    em = np.array([r["Emean"] for r in reps]) / N
    entry = dict(nrep=len(reps), seeds=[r["seed"] for r in reps], E_site_reps=em.tolist(), E_site=float(em.mean()),
                 SE_site=float(em.std(ddof=1) / np.sqrt(len(em))) if len(em) > 1 else None, beta=beta, burn=burn)
    # curve: bin E_mixed(beta) over reps, width 0.2
    edges = np.arange(0, beta + 1e-9, 0.2); cb = []; ce = []; cse = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        vals = []
        for r in reps:
            b = np.array(r["curve_beta"]); e = np.array(r["curve_E"]) / N
            m = (b >= lo) & (b < hi)
            if m.any(): vals.append(e[m].mean())
        if vals: cb.append(float((lo + hi) / 2)); ce.append(float(np.mean(vals))); cse.append(float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else None)
    entry["curve_beta_bins"] = cb; entry["curve_E_site"] = ce; entry["curve_SE"] = cse
    entry["E_beta0_site_mean"] = float(np.mean([r["E_beta0"] for r in reps]) / N)
    fn[g] = entry
out["FN"] = fn
# ---------------- variational energy (symmetrised estimator)
sym = []
for f in sorted(glob.glob(f"{R}/guide_sym_*.json")):
    d = json.load(open(f)); sym.append(d)
vm = {}
if sym:
    for n in sym[0]["T"]:
        # combine chains across runs: weight by number of chains
        arrs = []
        for f in sorted(glob.glob(f"{R}/guide_sym_*.npz")):
            z = np.load(f); arrs.append((z["ViT"].mean(0), z["dE_" + n.replace('+', 'p')].mean(0)))
        V = np.concatenate([a[0] for a in arrs]); D = np.concatenate([a[1] for a in arrs])
        vm[n] = dict(nchains=len(V), dE_site=float(D.mean() / N), dSE_site=float(D.std(ddof=1) / np.sqrt(len(D)) / N),
                     E_ViT_sampled_site=float(V.mean() / N), E_ViT_SE_site=float(V.std(ddof=1) / np.sqrt(len(V)) / N))
out["variational_sym"] = vm
json.dump(out, open(f"{R}/bound_check_summary.json", "w"), indent=1)
print(json.dumps({k: {kk: vv for kk, vv in v.items() if not kk.startswith('curve')} for k, v in fn.items()}, indent=1))
for g, e in fn.items():
    print(g, "curve", [round(x, 5) for x in e["curve_E_site"]])
print(json.dumps(vm.get("T_FN"), indent=1)); print(json.dumps(vm.get("T_old"), indent=1))
