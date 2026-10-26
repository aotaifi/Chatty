import json, sys
sys.path.insert(0, "/project/theorie/a/A.Otaifi/tools")
import paperstyle as ps
ps.apply()
cv = json.load(open("/scratch/a/A.Otaifi/l8/prod/F/refit.json"))["curve"]
d8 = json.load(open("/scratch/a/A.Otaifi/l8/diag/3973532/diag.json"))["a"]
d4 = json.load(open("/scratch/a/A.Otaifi/l8/diag4x4/diag.json"))["a"]
sh = ["0.1", "1.0", "10.0"]
x = [float(s) for s in sh]
fig, (ax, bx) = ps.panels(2)
it = [c["it"] for c in cv]
ps.series(ax, it, [c["R_train"] for c in cv], [c["R_train_err"] for c in cv], 0, label="train", line=False)
ps.series(ax, it, [c["R_held"] for c in cv], [c["R_held_err"] for c in cv], 1, label="held-out", line=False)
ax.axhline(1, color=ps.GRAY, lw=0.7, ls="--", zorder=0)
ax.set_xlabel("refit step"); ax.set_ylabel("$R$")
for d, i, lab in ((d8, 2, r"$8\times8$"), (d4, 3, r"$4\times4$")):
    ps.series(bx, x, [d[s]["cross"]["cos"][0] for s in sh], [d[s]["cross"]["cos"][1] for s in sh], i, label=lab)
bx.axhline(0, color=ps.GRAY, lw=0.7, ls="--", zorder=0)
bx.set_xscale("log"); bx.set_xticks(x); bx.set_xticklabels(["0.1", "1", "10"])
bx.set_xlabel("metric shift"); bx.set_ylabel("cross-chain cosine")
ps.tag(ax, "a"); ps.tag(bx, "b", y=0.95)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = bx.get_legend_handles_labels()
ps.legend_top(fig, ax, handles=h1 + h2, labels=l1 + l2)
ps.save(fig, __file__.replace(".py", ".pdf"))
for n, d in (("8x8", d8), ("4x4", d4)):
    print(n, [(round(d[s]["cross"]["cos"][0], 4), round(d[s]["cross"]["cos"][1], 4)) for s in sh])
