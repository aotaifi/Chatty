"""6x6 J2/J1=0.5: FN/Krylov loop vs continued ViT training vs GPU-hours, with Lanczos polish.
Relative energy error. GPU-hours = wall-clock of the single GPU job (4 CPU cores attached,
no separate CPU jobs); loop times include the exact measurements. Network loop = round 4 (cx_chop2, 3 seeds averaged,
run on A40-24GB slices, slice-hours counted 1:1 as GPU-hours)."""
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
C = [json.load(open(R / f"compress_6x6/runs/cx_chop2_{s}.json")) for s in ("s0", "s1", "s2")]
cx = np.mean([np.cumsum([0] + [r["sec"] / 3600 for r in c["iters"]]) for c in C], 0)
cy = np.mean([[c["iters"][0]["guide_H"]] + [r["H"] for r in c["iters"]] for c in C], 0)
cly = np.mean([[2.55e-5] + [r["lanczos_p1"] for r in c["iters"]] for c in C], 0)
V = [json.loads(l) for l in open(R / "vmc_continue_6x6/lr1e-3/eval.jsonl")]
vx = np.array([v["train_gpu_h"] for v in V]); vy = np.array([v["E_site"] - E0 for v in V]); ve = np.array([v["err_site"] for v in V])
RBM, VL1 = 4.47e-5, 2.55e-5
WS = json.load(open(R / "writeback_tail/runs/wtWS_17092071.json"))
wy = np.array([WS["iters"][0]["w_s_guide"]] + [r["w_s_kry"] for r in WS["iters"]])
CW = json.load(open(R / "compress_6x6/runs/cx_chop2_ws.json"))
cwy = np.mean([[CW["psiP"]] + CW[k] for k in ("s0", "s1", "s2")], 0)
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
fig, a = plt.subplots(figsize=(6.6, 4.4))
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
a.text(0.03, 0.04, r"$E_{\rm exact}/N = -0.503810$", transform=a.transAxes, ha="left", fontsize=10)
a.legend(frameon=False, fontsize=7.8, loc="lower right", ncol=2, handlelength=1.6, columnspacing=0.8, bbox_to_anchor=(1.02, -0.01))
a.set_xlabel("GPU-hours"); a.set_xlim(-0.05, 2.1)
fig.tight_layout(); fig.savefig(R / "writeback_tail/fig_loop_vs_gpuh_6x6.png", dpi=200); fig.savefig(R / "writeback_tail/fig_loop_vs_gpuh_6x6.pdf")
print("ok")
