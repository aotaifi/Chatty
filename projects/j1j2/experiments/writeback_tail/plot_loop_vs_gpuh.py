"""6x6 J2/J1=0.5: energy above E0 vs GPU-hours — FN/Krylov loop (stored tables; callable networks) vs continued ViT VMC training."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
R = Path(__file__).resolve().parents[2] / "results"
E0 = -0.50380965
L = json.load(open(R / "writeback_tail/runs/wtLOOP8_17090119.json"))
tab_x = np.cumsum([0] + [r["sec"] / 3600 for r in L["iters"]]); tab_y = [1e5 * y for y in [L["iters"][0]["guide_H"]] + [r["H_kry"] for r in L["iters"]]]
cal = []
for s in ("s0", "s1"):
    C = json.load(open(R / f"compress_6x6/runs/cx_chop_inc_{s}.json"))
    cal.append((np.cumsum([0] + [r["sec"] / 3600 for r in C["iters"]]), [1e5 * y for y in [C["iters"][0]["guide_H"]] + [r["H"] for r in C["iters"]]]))
V = [json.loads(l) for l in open(R / "vmc_continue_6x6/lr1e-3/eval.jsonl")]
vx = [v["train_gpu_h"] for v in V]; vy = [(v["E_site"] - E0) * 1e5 for v in V]; ve = [v["err_site"] * 1e5 for v in V]
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(7.0, 4.4))
ax.axhline(0, color="k", lw=1.2); ax.text(4.55, 0.2, "exact ground state (ED)", ha="right", va="bottom", fontsize=9)
ax.axhline(4.47, color="#009E73", lw=1.2, ls=":"); ax.text(4.55, 3.5, "RBM+PP (best published network)", color="#009E73", ha="right", va="top", fontsize=9)
ax.errorbar(vx, vy, yerr=ve, fmt="s-", color="#CC79A7", lw=1, ms=5, capsize=2, label="ViT trained further (VMC + minSR)")
ax.plot(tab_x, tab_y, "o-", color="#0072B2", lw=1, ms=6, label="FN/Krylov loop, stored tables (6x6 only)")
for k, (x, y) in enumerate(cal):
    ax.plot(x, y, "D-", color="#D55E00", lw=1, ms=5, mfc="white" if k else "#D55E00", label="FN/Krylov loop, callable networks" + (" (seed 2)" if k else ""))
ax.set_xlabel("GPU-hours after the starting network (A40 or A40 slice)")
ax.set_ylabel(r"energy above exact $E_0$ per site  [$10^{-5}$]")
ax.set_xlim(-0.1, 4.6); ax.set_ylim(-0.8, 21)
ax.set_title("6×6, J2/J1 = 0.5. Loop starts from the symmetrised ViT; VMC from the plain ViT", fontsize=9.5, loc="left")
ax.legend(frameon=False, fontsize=8.5, loc="center right", bbox_to_anchor=(1.0, 0.52))
fig.tight_layout(); fig.savefig(R / "writeback_tail/fig_loop_vs_gpuh_6x6.png", dpi=200); fig.savefig(R / "writeback_tail/fig_loop_vs_gpuh_6x6.pdf")
print("ok", tab_x[-1], [c[0][-1] for c in cal])
