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
WS = json.load(open(R / "writeback_tail/runs/wtWS_17092071.json"))
wy = np.array([WS["iters"][0]["w_s_guide"]] + [r["w_s_kry"] for r in WS["iters"]])
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.3))
s = 1.0 / abs(E0)
a.errorbar(vx, vy * s, yerr=ve * s, fmt="s-", color="#CC79A7", lw=1, ms=5, capsize=2, label="ViT + more VMC")
a.plot(tx, ty * s, "o-", color="#0072B2", lw=1, ms=6, label="loop (tables)")
a.plot(cx, cy * s, "D-", color="#D55E00", lw=1, ms=5, label="loop (networks)")
a.plot(tx, tly * s, "o--", color="#0072B2", lw=1, ms=5, mfc="white", label="loop (tables) + Lanczos")
a.plot(cx, cly * s, "D--", color="#D55E00", lw=1, ms=5, mfc="white", label="loop (networks) + Lanczos")
a.axhline(VL1 * s, color="0.4", lw=1, ls="-.", label="ViT + Lanczos")
a.axhline(RBM * s, color="#009E73", lw=1.2, ls=":", label="RBM+PP")
a.set_yscale("log"); a.set_ylim(1.2e-5, 5e-4)
a.set_yticks([2e-5, 5e-5, 1e-4, 2e-4, 4e-4]); a.set_yticklabels([r"$2\times10^{-5}$", r"$5\times10^{-5}$", r"$10^{-4}$", r"$2\times10^{-4}$", r"$4\times10^{-4}$"]); a.minorticks_off()
a.set_ylabel(r"$\epsilon = (E - E_{\rm exact})/|E_{\rm exact}|$")
a.text(0.97, 0.6, r"$E_{\rm exact}/N = -0.503810$", transform=a.transAxes, ha="right", fontsize=10)
a.legend(frameon=False, fontsize=7.8, loc="lower right", ncol=2, handlelength=1.6, columnspacing=0.8, bbox_to_anchor=(1.02, -0.01))
b.plot(tx, wy, "o-", color="#0072B2", lw=1, ms=6, label="loop (tables)")
b.axhline(1.3e-4, color="#CC79A7", lw=1.2, ls="--", label="ViT")
b.set_yscale("log"); b.set_ylim(2e-6, 3e-4)
b.set_ylabel(r"wrong-sign weight $w_s$")
b.legend(frameon=False, fontsize=9, loc="upper right")
for ax, t in ((a, "(a)"), (b, "(b)")):
    ax.set_xlabel("GPU-hours"); ax.set_xlim(-0.1, 4.6 if ax is a else 1.5)
    ax.text(-0.13, 1.02, t, transform=ax.transAxes, fontsize=12, fontweight="bold")
fig.tight_layout(); fig.savefig(R / "writeback_tail/fig_loop_vs_gpuh_6x6.png", dpi=200); fig.savefig(R / "writeback_tail/fig_loop_vs_gpuh_6x6.pdf")
print("ok")
