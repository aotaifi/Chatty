import sys; sys.path.insert(0, "/project/theorie/a/A.Otaifi/tools"); import paperstyle as ps
import matplotlib.pyplot as plt, numpy as np
ps.apply()
ref, eref = -0.498754, 2.3e-6
e = np.array([-0.495522, -0.498158, -0.498658, -0.498709]); de = np.array([23e-6, 17e-6, 9e-6, 10e-6])
gap = e - ref; dg = np.sqrt(de**2 + eref**2)
fig, ax = plt.subplots(figsize=(3.4, 2.3))
ps.series(ax, [0, 1, 2, 3], gap, dg, 0, label=r"$\mathrm{stored\ sign\ rounds}$")
yc = -0.497651 - ref
ax.axhline(yc, color=ps.GRAY, lw=0.7, ls="--", zorder=0); ax.text(1.1, yc * 1.15, "SR, same network class, equal compute", fontsize=6, color=ps.GRAY)
ax.set_yscale("log"); ax.set_xticks(range(4)); ax.set_xticklabels([r"$\mathrm{Marshall}$", "1", "2", "3"])
ax.set_xlabel(r"$\mathrm{round}\ k$"); ax.set_ylabel(r"$(E_k - E[\mathrm{sgn}\,\mathrm{M1}])/N$")
ps.save(fig, "fig_sign.pdf")
