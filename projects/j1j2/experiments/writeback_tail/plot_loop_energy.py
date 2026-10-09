"""Energy error per site vs loop iteration: realistic FN/Krylov loop (wtLOOP8) vs ideal exact loop, 6x6 J2/J1=0.5."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
R = Path(__file__).resolve().parents[2] / "results"
L = json.load(open(R / "writeback_tail/runs/wtLOOP8_17090119.json"))
I = json.load(open(R / "stall_6x6/exact_loop_from_projvit.json"))
it = [0] + [r["it"] for r in L["iters"]]
H = [L["iters"][0]["guide_H"]] + [r["H_kry"] for r in L["iters"]]
EF = [L["iters"][0]["guide_E_FN"]] + [r["E_FN_next"] for r in L["iters"]]
iti = [r["it"] for r in I]; Hi = [r["H_dE_site"] for r in I]
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.plot(iti, Hi, "--", color="0.55", lw=1, marker="o", ms=4, mfc="white", label=r"ideal loop $\langle H\rangle$ (exact FN amplitude)")
ax.plot(it, H, "-", color="#0072B2", lw=1, marker="o", ms=6, label=r"our loop $\langle H\rangle$ (network write-back)")
ax.plot(it, EF, "-", color="#D55E00", lw=1, marker="s", ms=5, label=r"our loop $E_{\rm FN}$")
ax.axhline(L["rbm_pp"], color="#009E73", lw=1.2, ls=":")
ax.text(12.2, L["rbm_pp"] * 1.06, "RBM+PP (best published)", color="#009E73", ha="right", fontsize=9)
ax.set_yscale("log"); ax.set_xlim(-0.4, 12.4); ax.set_ylim(6e-6, 1.6e-4)
ax.set_yticks([1e-5, 2e-5, 5e-5, 1e-4]); ax.set_yticklabels([r"$10^{-5}$", r"$2\times10^{-5}$", r"$5\times10^{-5}$", r"$10^{-4}$"])
ax.set_xlabel("loop iteration"); ax.set_ylabel(r"energy above exact $E_0$, per site")
ax.set_title("6×6, J2/J1 = 0.5, start: symmetrised ViT", fontsize=10, loc="left")
ax.legend(frameon=False, fontsize=8.5, loc="upper right")
fig.tight_layout(); fig.savefig(R / "writeback_tail/fig_loop_energy_6x6.png", dpi=200); fig.savefig(R / "writeback_tail/fig_loop_energy_6x6.pdf")
