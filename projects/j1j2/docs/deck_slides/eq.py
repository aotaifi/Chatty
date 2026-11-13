import subprocess, sys, re, os
INK = "15233D"
EQ = {
 "ham": r"H = J_1\sum_{\langle ij\rangle}\mathbf S_i\cdot\mathbf S_j + J_2\sum_{\langle\langle ij\rangle\rangle}\mathbf S_i\cdot\mathbf S_j,\qquad J_2/J_1 = 0.5",
 "guide": r"\psi_k(x) = S_k(x)\,|B_k(x)|",
 "hfn": r"H_{\rm FN}[\psi] = H^{\rm kept} + \mathrm{diag}\,V_\psi,\qquad V_\psi(x)=\!\!\sum_{y\ \mathrm{blocked}}\!\!|H_{xy}|\,\Big|\frac{\psi(y)}{\psi(x)}\Big|",
 "bound": r"E_{\rm FN}[\psi] = \lambda_{\min}\big(H_{\rm FN}[\psi]\big)\ \ge\ E_0",
 "fj": r"f_J(x) = \ln\frac{\sum_{z\ \mathrm{kept}}|H_{xz}|\,|B(z)|}{\big(H_{xx}+V_B(x)-E'\big)\,|B(x)|}",
 "target": r"T_k = |B_k|\,e^{\,c\,f_J[B_k]}",
 "fit": r"B_{k+1} = \arg\min_\theta\,\mathcal L\big(B_\theta, T_k\big)",
 "sign": r"S_{k+1} = \mathrm{sgn}\big[(1+\alpha(H-E))\,\psi_k\big]",
 "readout": r"\mathrm{FN\!-\!GFMC\ with\ guide}\ (1+\alpha(H-E))\,\psi_k",
 "share": r"r_{k+1} = \frac{e(B_k)-e(B_{k+1})}{e(B_k)-e(T_k)},\qquad e=\frac{E_{\rm FN}-E_0}{N}",
 "cost": r"\mathrm{cost}\ \propto\ d^{\,p+1},\qquad d\approx 2N",
 "tangent": r"C = \frac{b^\dagger G^{+} b}{D_{RR}} = 0\text{--}0.04",
}
tpl = r"""\documentclass{article}\usepackage{amsmath,amssymb,xcolor}\definecolor{ink}{HTML}{%s}
\newsavebox\eqbox\begin{document}\sbox\eqbox{\color{ink}$\displaystyle %s$}
\pdfpagewidth=\dimexpr\wd\eqbox+2pt\relax\pdfpageheight=\dimexpr\ht\eqbox+\dp\eqbox+2pt\relax
\hoffset=-1in\voffset=-1in\shipout\hbox{\kern1pt\vbox{\kern1pt\hbox{\usebox\eqbox}\kern1pt}\kern1pt}\end{document}"""
for k, tex in EQ.items():
    open(k + ".tex", "w").write(tpl % (INK, tex))
    subprocess.run(["pdflatex", "-interaction=batchmode", k + ".tex"], check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["pdftocairo", "-svg", k + ".pdf", k + ".svg"], check=True)
    print(k, os.path.getsize(k + ".svg"))
