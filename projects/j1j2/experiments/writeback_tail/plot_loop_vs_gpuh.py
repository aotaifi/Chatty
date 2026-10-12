"""6x6 J2/J1=0.5: FN/Krylov loop vs continued ViT training vs GPU-hours, with Lanczos polish. (a) E - E0 per site, (b) relative error."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
R = Path(__file__).resolve().parents[2] / "results"
E0 = -0.50380965
L = json.load(open(R / "writeback_tail/runs/wtLOOP8_17090119.json"))
LA = json.load(open(R / "writeback_tail/runs/wt_val_a_17092072.json"))
tx = np.cumsum([0] + [r["sec"] / 3600 for r in L["iters"]])
ty = np.array([L["iters"][0]["guide_H"]] + [r["H_kry"] for r in L["iters"]])
tly = np.array([2.55e-5] + [r["lanczos_H"] for r in LA["iters"]])
C = [json.load(open(R / f"compress_6x6/runs/cx_chop_inc_{s}.json")) for s in ("s0", "s1")]
cx = np.mean([np.cumsum([0] + [r["sec"] / 3600 for r in c["iters"]]) for c in C], 0)
cy = np.mean([[c["iters"][0]["guide_H"]] + [r["H"] for r in c["iters"]] for c in C], 0)
cly = np.mean([[2.55e-5] + [r["lanczos_p1"] for r in c["iters"]] for c in C], 0)
V = [json.loads(l) for l in open(R / "vmc_continue_6x6/lr1e-3/eval.jsonl")]
vx = np.array([v["train_gpu_h"] for v in V]); vy = np.array([v["E_site"] - E0 for v in V]); ve = np.array([v["err_site"] for v in V])
RBM, VL1 = 4.47e-5, 2.55e-5
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.3))
for ax, s in ((a, 1e5), (b, 1.0 / abs(E0))):
    ax.errorbar(vx, vy * s, yerr=ve * s, fmt="s-", color="#CC79A7", lw=1, ms=5, capsize=2, label="ViT + more VMC")
    ax.plot(tx, ty * s, "o-", color="#0072B2", lw=1, ms=6, label="loop (tables)")
    ax.plot(cx, cy * s, "D-", color="#D55E00", lw=1, ms=5, label="loop (networks)")
    ax.plot(tx, tly * s, "o--", color="#0072B2", lw=1, ms=5, mfc="white", label="loop (tables) + Lanczos")
    ax.plot(cx, cly * s, "D--", color="#D55E00", lw=1, ms=5, mfc="white", label="loop (networks) + Lanczos")
    ax.axhline(VL1 * s, color="0.4", lw=1, ls="-.", label="ViT + Lanczos")
    ax.axhline(RBM * s, color="#009E73", lw=1.2, ls=":", label="RBM+PP")
    ax.set_xlabel("GPU-hours"); ax.set_xlim(-0.1, 4.6)
a.axhline(0, color="k", lw=1.2)
a.text(4.5, 0.35, r"exact: $E_{\rm exact}/N = -0.503810$", ha="right", fontsize=10)
a.set_ylabel(r"$(E - E_{\rm exact})/N$  [$10^{-5}$]"); a.set_ylim(-0.8, 21)
b.set_yscale("log"); b.set_ylim(1.2e-5, 5e-4)
b.set_yticks([2e-5, 5e-5, 1e-4, 2e-4, 4e-4]); b.set_yticklabels([r"$2\times10^{-5}$", r"$5\times10^{-5}$", r"$10^{-4}$", r"$2\times10^{-4}$", r"$4\times10^{-4}$"]); b.minorticks_off()
b.set_ylabel(r"$\epsilon = (E - E_{\rm exact})/|E_{\rm exact}|$")
b.legend(frameon=False, fontsize=7.8, loc="lower right", ncol=2, handlelength=1.6, columnspacing=0.8, bbox_to_anchor=(1.02, -0.01))
for ax, t in ((a, "(a)"), (b, "(b)")):
    ax.text(-0.13, 1.02, t, transform=ax.transAxes, fontsize=12, fontweight="bold")
fig.tight_layout(); fig.savefig(R / "writeback_tail/fig_loop_vs_gpuh_6x6.png", dpi=200); fig.savefig(R / "writeback_tail/fig_loop_vs_gpuh_6x6.pdf")
print("ok")
