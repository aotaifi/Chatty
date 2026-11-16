# 8x8 strong network (s306), one loop round, bare FN-GFMC (no Lanczos step).
# Source: docs/reports/2026-11-15_r1prime_grade_8x8.md (Paderborn 4271870, 4271928; 8 chains per arm, chain-scatter errors)
import sys; sys.path.insert(0, "/project/theorie/a/A.Otaifi/tools"); import paperstyle as ps
import matplotlib.pyplot as plt
ps.apply()
off = -0.49888
E = {"base": (-0.4988795, 2.7e-6), "warm": (-0.4988799, 1.6e-6), "r1": (-0.4988919, 1.6e-6)}
y = lambda k: (E[k][0] - off) * 1e5
d = lambda k: E[k][1] * 1e5
fig, ax = plt.subplots(figsize=(3.4, 2.3))
w = y("warm")
ax.axhspan(w - 8.3, w - 4.6, color="0.88", lw=0, zorder=0)
ps.series(ax, [0], [y("base")], [d("base")], 0, line=False, open_=True)
ps.series(ax, [1, 2], [w, y("r1")], [d("warm"), d("r1")], 0, line=False)
ax.set_xticks([0, 1, 2]); ax.set_xticklabels([r"$\mathrm{base}$", r"$\mathrm{warm\ start}$", r"$\mathrm{round\ 1}$"])
ax.set_xlim(-0.5, 2.5); ax.set_ylim(-9.2, 1.2)
ax.set_ylabel(r"$(E_{\rm FN}/N + 0.49888)/10^{-5}$")
ps.save(fig, "fig_r1p.pdf")
