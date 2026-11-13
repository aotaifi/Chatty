import sys; sys.path.insert(0, "/project/theorie/a/A.Otaifi/tools"); import paperstyle as ps
import matplotlib.pyplot as plt
ps.apply()
s = lambda e: (e + 0.4989) / 1e-5          # E/N + 0.4989 in 1e-5
u = lambda d: d / 1e-5
NI, VB = s(-0.498886), s(-0.498963)
def refs(ax, x):
    ps.vline  # noqa
    for y, t, dy in [(NI, "NI", 0.5), (VB, "VB", 0.5)]:
        ax.axhline(y, color=ps.GRAY, lw=0.7, ls="--", zorder=0); ax.text(x, y + dy, t, fontsize=6, color=ps.GRAY)
# (a) 8x8 ladder; (b) zoom on FN p=1
fig, (ax, bx) = plt.subplots(1, 2, figsize=(3.4, 2.3), gridspec_kw=dict(wspace=0.45, width_ratios=[1.6, 1]))
ps.series(ax, [0, 1, 2, 3], [s(-0.498674), s(-0.4987565), s(-0.498923), s(-0.498946)],
          [u(12e-6), u(2.8e-6), u(17e-6), u(5e-6)], 0, label=r"$\mathrm{M1}$")
ps.series(ax, [0], [s(-0.4989092)], [u(1.4e-6)], 2, label=r"$\mathrm{s306}$", line=False)
ax.errorbar([2], [s(-0.498954)], fmt="^", color=ps.COLORS[2], mfc="white", ms=3.5, mew=0.7)
ax.set_xticks(range(4)); ax.set_xticklabels([r"$\mathrm{VMC}$", r"$\mathrm{FN}_0$", r"$\mathrm{VMC}_1$", r"$\mathrm{FN}_1$"])
ax.set_ylabel(r"$E/N + 0.4989\ \ (10^{-5})$"); ax.set_xlim(-0.4, 3.4); refs(ax, 0.5)
ps.series(bx, [0, 1], [s(-0.498946), s(-0.4989390)], [u(5e-6), u(4.5e-6)], 0, line=False)
ps.series(bx, [2], [s(-0.4989545)], [u(3.4e-6)], 1, label=r"$B_1$", line=False)
bx.set_xticks(range(3)); bx.set_xticklabels([r"$\mathrm{M1}$", r"$B_0$", r"$B_1$"]); bx.set_xlim(-0.5, 2.5)
bx.set_ylim(-7.5, 0.5); bx.axhline(VB, color=ps.GRAY, lw=0.7, ls="--", zorder=0); bx.text(1.6, VB + 0.15, "VB", fontsize=6, color=ps.GRAY)
ps.tag(ax, "a", x=0.96, ha="right"); ps.tag(bx, "b", x=0.94, ha="right")
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = bx.get_legend_handles_labels()
ps.legend_top(fig, ax, handles=h1 + h2, labels=l1 + l2)
ps.save(fig, "fig_8x8.pdf")
# 6x6 rounds
fig, ax = plt.subplots(figsize=(3.4, 2.3))
ps.series(ax, [1, 2, 3, 4, 5], [6.07, 5.29, 4.64, 4.37, 4.14], None, 0, label=r"$\mathrm{one\ PP}$")
ps.series(ax, [5, 6], [3.275, 2.81], None, 1, label=r"$\mathrm{two\ PP}$")
ps.series(ax, [6], [2.546], None, 3, label=r"$\mathrm{three\ PP}$", line=False)
ps.series(ax, [5, 6], [4.12, 3.70], None, 2, label=r"$\mathrm{two\ PP},\ S$", open_=True)
for y, t in [(5.574, "RBM+PP + Lanczos + FN"), (2.654, "CNN-MPS 2611.04333"), (2.284, "CNN-MPS 2603.14425")]:
    ax.axhline(y, color=ps.GRAY, lw=0.7, ls="--", zorder=0); ax.text(1.0, y + 0.07, t, fontsize=6, color=ps.GRAY)
ax.set_xlabel(r"$\mathrm{round}\ k$"); ax.set_ylabel(r"$(E_{\rm FN}-E_0)/N\ \ (10^{-6})$")
ax.set_xticks(range(1, 7)); ax.set_ylim(2.0, 6.6)
ps.legend_top(fig, ax)
ps.save(fig, "fig_6x6.pdf")
