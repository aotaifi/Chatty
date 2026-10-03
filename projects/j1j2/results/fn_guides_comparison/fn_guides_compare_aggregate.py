"""Aggregate fn_guides_compare outputs (theorie jobs 16809572-16809586) with reused
matched-setting FN runs into results/fn_guides_comparison/fn_guides_comparison.json."""
import json, glob, os
import numpy as np

R = "/Users/aliotaifi/Chatty-organize/projects/j1j2/results"
D = f"{R}/fn_guides_comparison"
TJ = f"{D}/theorie_jobs"


def stat(vals, N):
    v = np.asarray(vals, float)
    m = float(v.mean()); se = float(v.std(ddof=1) / np.sqrt(len(v)))
    return dict(E=m, SE=se, E_site=m / N, SE_site=se / N, rep_values=v.tolist(),
                rep_diff=float(v[0] - v[1]) if len(v) == 2 else None)


out = {}
for L in (6, 8):
    N = L * L; res = {}
    # ---- E_ViT: pool all production VMC jobs (exclude pilot 16809550)
    js = sorted(glob.glob(f"{TJ}/fnguides_*/vmc_vit_{L}x{L}.json"))
    js = [j for j in js if "16809550" not in j]
    cm_all = {"ViT": [], "ViTbinary": []}; per_job = []
    for j in js:
        meta = json.load(open(j)); z = np.load(j.replace(".json", "_samples.npz"))
        cm_all["ViT"].append(z["eA"].mean(axis=0)); cm_all["ViTbinary"].append(z["eB"].mean(axis=0))
        per_job.append(dict(job=os.path.basename(os.path.dirname(j)).split("_")[1], seed=meta["seed"],
                            E=meta["ViT"]["E"], SE=meta["ViT"]["SE"], phi=meta["phi"], leak=meta["leak"]))
    for k in cm_all:
        cm = np.concatenate(cm_all[k]); m = float(cm.mean()); se = float(cm.std(ddof=1) / np.sqrt(len(cm)))
        res[f"E_{k}_VMC"] = dict(E=m, SE=se, E_site=m / N, SE_site=se / N, nchains_total=int(len(cm)),
                                 nsamples_total=int(sum(len(c) for c in cm_all[k]) * 12),
                                 jobs=[p["job"] for p in per_job])
    jE = np.array([p["E"] for p in per_job])
    res["E_ViT_VMC"]["between_job_SE_site"] = float(jE.std(ddof=1) / np.sqrt(len(jE)) / N)
    res["E_ViT_VMC"]["per_job"] = per_job
    # ---- FN
    fn = {}
    for g in ("vit", "marshall"):
        f = glob.glob(f"{TJ}/fnguides_*/fn_{g}_{L}x{L}_M128_summary.json")
        f = [x for x in f if "16809551" not in x]  # 8x8 Marshall single-seed reproduction check
        if f:
            s = json.load(open(f[0]))
            fn[g] = dict(stat([r["Emean"] for r in s["reps"]], N), seeds=[r["seed"] for r in s["reps"]],
                         tail8=[r["tail8"] for r in s["reps"]], M=128, beta_target=1.2, burn_beta=0.4,
                         tau_max=0.025, phi=s["phi"], source="NEW theorie job " + f[0].split("fnguides_")[1].split("/")[0])
    if L == 6:
        z = np.load(os.path.expanduser("~/j1j2_vit_bench/gr_6x6_population_scaling_M128.npz"))
        rows = z["rows"]
        fn["krylov"] = dict(stat(rows[:, 1], N), seeds=rows[:, 0].astype(int).tolist(), tail8=rows[:, 2].tolist(),
                            M=128, beta_target=1.2, burn_beta=0.4, tau_max=0.025, T=-14.985799779143964,
                            source="REUSED Mac durable job 20260928-102846-50383, gfmc_6x6_gr_population_size.py 128 "
                                   "(~/j1j2_vit_bench/gr_6x6_population_scaling_M128.npz)")
        fn["marshall_M32_8rep_reference"] = dict(E=-17.97267437, SE=0.02954654, E_site=-17.97267437 / N,
                                                 SE_site=0.02954654 / N, M=32,
                                                 source="REUSED krylov_fn_6x6_matched_verdict_2026-09-28.md (NOT matched: M=32)")
    else:
        fn["krylov"] = dict(stat([-31.887015719100066, -31.883832500002562], N), seeds=[10501, 10502],
                            M=128, beta_target=1.2, burn_beta=0.4, tau_max=0.025, T=-28.37107876288694,
                            source="REUSED Paderborn job 3471544 (results/8x8_krylov_3471544)")
        fn["marshall"] = dict(stat([-31.703826476888693, -31.651127258473807], N), seeds=[11501, 11502],
                              M=128, beta_target=1.2, burn_beta=0.4, tau_max=0.025,
                              source="REUSED Paderborn job 3471545 (results/8x8_marshall_3471545); seed 11501 "
                                     "reproduced to 1e-14 by new script in theorie job 16809551")
    res["FN"] = fn
    ev = res["E_ViT_VMC"]["E"]
    res["FN_minus_EViT_total"] = {g: fn[g]["E"] - ev for g in ("marshall", "krylov", "vit")}
    out[f"{L}x{L}"] = res

out["notes"] = [
    "All matched FN: guide amplitude |psi_ViT| (same checkpoint), M=128 walkers, beta_target=1.2, burn_beta=0.4, "
    "tau_max=0.025, discrete-time lattice FN-GFMC with systematic resampling each step, mixed estimator averaged "
    "over beta in [0.4,1.2]; two independent replicas; SE = between-replica SE (2 replicas: crude).",
    "ViT sign for FN: h(x)=sgn cos(Im log psi(x) - phi), phi global phase; phase leakage <sin^2> ~ 5e-7 (8x8), 2e-7 (6x6), "
    "binarized-ViT variational energy equals complex-ViT energy to <1e-5 per site.",
    "E_ViT: VMC with |psi|^2 Metropolis bond-exchange chains (burn 60 sweeps, 12 samples spaced 3 sweeps); SE from per-chain means.",
    "Previous 6x6 E_ViT -0.50345(16)/site (256 samples, a1_node_audit/energy_krylov_vs_vit_6x6_indep) is consistent.",
    "Caveat: FN[ViT] - E_ViT = +0.0033(23) total (6x6) and +0.0027(80) total (8x8): no measurable FN gain over the ViT itself; "
    "exact FN must give <= E_ViT, so any residual is population-control/incomplete-projection bias at M=128, beta=1.2 "
    "(<~1e-4 per site). Krylov-FN and Marshall-FN lie ABOVE E_ViT (by 0.022/0.146 total at 6x6, 0.039/0.247 at 8x8).",
]
with open(f"{D}/fn_guides_comparison.json", "w") as f:
    json.dump(out, f, indent=1)
for L in ("6x6", "8x8"):
    r = out[L]
    print(L, "E_ViT", r["E_ViT_VMC"]["E_site"], r["E_ViT_VMC"]["SE_site"], "between-job", r["E_ViT_VMC"]["between_job_SE_site"],
          "ViTbin", r["E_ViTbinary_VMC"]["E_site"])
    for g in ("marshall", "krylov", "vit"):
        x = r["FN"][g]; print("  FN", g, x["E"], x["SE"], "site", x["E_site"], x["SE_site"], x["seeds"], "| FN-EViT", r["FN_minus_EViT_total"][g])
