from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
    KeepTogether, HRFlowable
)
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon
from reportlab.lib.colors import HexColor
from pathlib import Path

OUT = Path("/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure/KRYLOV_SIGN_STRUCTURE_NOTE_2026-09-28.pdf")
GITHUB = "https://github.com/aotaifi/Chatty/tree/main/projects/j1j2/krylov_sign_structure"

NAVY = HexColor("#183B56")
BLUE = HexColor("#2A6FBB")
PALE = HexColor("#EEF5FB")
PALE2 = HexColor("#F5F7F9")
INK = HexColor("#1A1D21")
MUTED = HexColor("#5E6973")
GREEN = HexColor("#247A52")
RED = HexColor("#9B3A3A")
LINE = HexColor("#D5DCE2")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    name="PaperTitle", parent=styles["Title"], fontName="Times-Bold",
    fontSize=19, leading=22, textColor=NAVY, alignment=TA_LEFT,
    spaceAfter=6
))
styles.add(ParagraphStyle(
    name="Subtitle", parent=styles["Normal"], fontName="Helvetica",
    fontSize=9.5, leading=13, textColor=MUTED, spaceAfter=10
))
styles.add(ParagraphStyle(
    name="H1x", parent=styles["Heading1"], fontName="Helvetica-Bold",
    fontSize=11.5, leading=14, textColor=NAVY, spaceBefore=9, spaceAfter=5
))
styles.add(ParagraphStyle(
    name="Bodyx", parent=styles["BodyText"], fontName="Times-Roman",
    fontSize=9.2, leading=12.3, textColor=INK, spaceAfter=5
))
styles.add(ParagraphStyle(
    name="Smallx", parent=styles["BodyText"], fontName="Helvetica",
    fontSize=7.8, leading=10.2, textColor=MUTED, spaceAfter=3
))
styles.add(ParagraphStyle(
    name="BoxHead", parent=styles["BodyText"], fontName="Helvetica-Bold",
    fontSize=9.2, leading=11, textColor=NAVY, spaceAfter=3
))
styles.add(ParagraphStyle(
    name="BoxBody", parent=styles["BodyText"], fontName="Times-Roman",
    fontSize=8.9, leading=11.7, textColor=INK, spaceAfter=0
))
styles.add(ParagraphStyle(
    name="Eq", parent=styles["BodyText"], fontName="Courier",
    fontSize=8.6, leading=11.5, textColor=INK, leftIndent=8, rightIndent=8,
    spaceBefore=3, spaceAfter=5
))
styles.add(ParagraphStyle(
    name="Caption", parent=styles["BodyText"], fontName="Helvetica-Oblique",
    fontSize=7.4, leading=9.3, textColor=MUTED, spaceAfter=5
))

def link(label, url):
    return f'<link href="{url}" color="#2A6FBB"><u>{label}</u></link>'

def box(title, body, bg=PALE):
    t = Table([
        [Paragraph(title, styles["BoxHead"])],
        [Paragraph(body, styles["BoxBody"])]
    ], colWidths=[178*mm])
    t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),bg),
        ("BOX",(0,0),(-1,-1),0.7,LINE),
        ("LEFTPADDING",(0,0),(-1,-1),7),
        ("RIGHTPADDING",(0,0),(-1,-1),7),
        ("TOPPADDING",(0,0),(-1,-1),6),
        ("BOTTOMPADDING",(0,0),(-1,-1),6),
    ]))
    return t

def flow_diagram():
    d = Drawing(510, 74)
    xs = [8, 132, 256, 380]
    labels = [
        ("signs s_k", "fixed amplitudes a"),
        ("local coordinate", "r_k=(H a s_k)/(a s_k)"),
        ("choose threshold t_k", "minimize fixed-amplitude E"),
        ("new signs s_{k+1}", "restore the same amplitudes a"),
    ]
    for i, x in enumerate(xs):
        d.add(Rect(x, 17, 108, 40, rx=4, ry=4, fillColor=PALE, strokeColor=LINE, strokeWidth=0.8))
        d.add(String(x+54, 42, labels[i][0], fontName="Helvetica-Bold", fontSize=7.4, textAnchor="middle", fillColor=NAVY))
        d.add(String(x+54, 29, labels[i][1], fontName="Helvetica", fontSize=6.5, textAnchor="middle", fillColor=MUTED))
        if i < len(xs)-1:
            x1=x+108; x2=xs[i+1]
            d.add(Line(x1+4,37,x2-6,37,strokeColor=BLUE,strokeWidth=1.2))
            d.add(Polygon(points=[x2-6,37,x2-12,41,x2-12,33],fillColor=BLUE,strokeColor=BLUE))
    return d

def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.4)
    canvas.line(17*mm, 12.5*mm, 193*mm, 12.5*mm)
    canvas.setFont("Helvetica", 6.8)
    canvas.setFillColor(MUTED)
    canvas.drawString(17*mm, 8.2*mm, "Krylov sign structure - internal research note - 28 Sep 2026")
    canvas.drawRightString(193*mm, 8.2*mm, f"{doc.page}")
    canvas.restoreState()

doc = SimpleDocTemplate(
    str(OUT), pagesize=A4,
    rightMargin=16*mm, leftMargin=16*mm, topMargin=15*mm, bottomMargin=17*mm,
    title="Projected Krylov Sign Descent and Sign-Basin Structure",
    author="Aly Otaifi"
)
story = []
story += [
    Paragraph("Projected Krylov Sign Descent and the Structure of Ground-State Sign Basins", styles["PaperTitle"]),
    Paragraph(
        "A compact mechanism note for the square-lattice J1-J2 and triangular Heisenberg tests. "
        + link("Chatty subproject", GITHUB)
        + " &nbsp; | &nbsp; current repository head at preparation: <font face=\"Courier\">328bc37</font>",
        styles["Subtitle"]
    ),
    HRFlowable(width="100%", thickness=0.7, color=LINE, spaceBefore=1, spaceAfter=7),
    box(
        "Abstract",
        "A one-step Krylov/local-energy rule was observed to reconstruct the frustrated J1-J2 ground-state signs with unexpectedly high accuracy. "
        "Exact finite-cluster tests show that this is the first strong contraction step of a nonlinear projected-Krylov sign descent. "
        "At fixed exact amplitudes, repeated label-free threshold updates reach the exact square-lattice signs even from poor Marshall starts. "
        "The ground-state attraction basin is empirically large under random sign corruption. "
        "The triangular model does not invalidate local contraction: instead, structured gauges can occupy competing basins. "
        "One such obstruction is exact and analytic - the update preserves symmetry character whenever the amplitude modulus is symmetry invariant.",
        PALE
    ),
    Spacer(1, 6),
    Paragraph("1. The object being iterated", styles["H1x"]),
    Paragraph(
        "Write the wavefunction as a positive modulus times a binary sign field, "
        "<i>psi_s(x)=a(x)s(x)</i>, with <i>a(x)&gt;=0</i> and <i>s(x)=+/-1</i>. "
        "For the mechanism tests, <i>a(x)</i> is held fixed to the exact ground-state modulus.",
        styles["Bodyx"]
    ),
    Paragraph("r_s(x) = [H psi_s](x) / psi_s(x)", styles["Eq"]),
    Paragraph(
        "A single Krylov vector obeys <i>[(t-H)psi_s](x)=[t-r_s(x)]psi_s(x)</i>. "
        "Therefore its sign is exactly the threshold rule",
        styles["Bodyx"]
    ),
    Paragraph("T_t[s](x) = s(x) sign[t - r_s(x)].", styles["Eq"]),
    flow_diagram(),
    Paragraph(
        "At every iteration we choose <i>t</i> without using hidden ground-state signs: among all realizable thresholds of <i>r_s</i>, "
        "we select the one with the lowest fixed-amplitude variational energy and then project back to the same modulus <i>a(x)</i>.",
        styles["Caption"]
    ),
    box(
        "Key exact fact: monotone projected descent",
        "The threshold family always contains the current sign pattern: choosing t above max_x r_s(x) leaves every sign unchanged. "
        "Hence the energy-minimizing update satisfies <b>E[T(s)] <= E[s]</b>. "
        "With the exact ground-state modulus, the exact signs are a global minimum and a fixed point because r_*(x)=E_0 for every nonzero component.",
        PALE2
    ),
]
story += [
    Paragraph("2. What the square-lattice tests establish", styles["H1x"]),
    Paragraph(
        "On the periodic 4x4 square-lattice spin-1/2 J1-J2 model in S^z=0, exact diagonalization provides both the modulus and reference signs. "
        "The first step is spectacular near J2/J1=1/2, but the stronger result is that repeated updates converge all the way to the exact signs.",
        styles["Bodyx"]
    ),
]

traj_data = [
    ["J2/J1", "start", "sign-overlap trajectory O_S", "exact at"],
    ["0.5", "Marshall", "0.97454 -> 0.999319 -> 0.999998 -> 1", "3 updates"],
    ["0.6", "Marshall", "0.79557 -> 0.95713 -> 0.99127 -> 0.99944 -> 1", "4"],
    ["0.8", "Marshall", "0.12366 -> 0.06197 -> ... -> 0.96288 -> 0.99120 -> 0.99985 -> 1", "10"],
    ["1.0", "Marshall", "0.06421 -> 0.17135 -> 0.29020 -> 0.33926 -> 0.91039 -> ... -> 1", "9"],
    ["1.0", "stripe", "0.95582 -> 0.99488 -> 0.999692 -> 0.9999994 -> 1", "4"],
]
tbl = Table(traj_data, colWidths=[18*mm, 24*mm, 112*mm, 22*mm], repeatRows=1)
tbl.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(-1,0),NAVY),
    ("TEXTCOLOR",(0,0),(-1,0),colors.white),
    ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
    ("FONTNAME",(0,1),(-1,-1),"Helvetica"),
    ("FONTSIZE",(0,0),(-1,-1),7.0),
    ("LEADING",(0,0),(-1,-1),8.6),
    ("GRID",(0,0),(-1,-1),0.35,LINE),
    ("VALIGN",(0,0),(-1,-1),"TOP"),
    ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,PALE2]),
    ("LEFTPADDING",(0,0),(-1,-1),4),
    ("RIGHTPADDING",(0,0),(-1,-1),4),
    ("TOPPADDING",(0,0),(-1,-1),4),
    ("BOTTOMPADDING",(0,0),(-1,-1),4),
]))
story += [
    tbl,
    Paragraph(
        "<b>Interpretation.</b> A good gauge is not required for eventual convergence on this tested square cluster. "
        "It mainly makes the first step exceptionally efficient. Even Marshall at J2/J1=0.8 and 1.0, where the first step is poor, eventually reaches the exact signs.",
        styles["Bodyx"]
    ),
    box(
        "Why the first step looked like a miracle",
        "At J2/J1=0.5, the Marshall state is already positioned so that the scalar coordinate r_0 orders nearly all physically important residual sign defects. "
        "The first global threshold therefore removes almost all of them at once. Recomputing r exposes progressively smaller residual defects. "
        "The observed sequence 0.97454 -> 0.999319 -> 0.999998 -> 1 is the signature of rapid basin contraction, not a one-off accident.",
        PALE
    ),
    Paragraph("3. The attraction basin is large, but distance is not the whole story", styles["H1x"]),
    Paragraph(
        "To quantify 'start close', the exact sign pattern was randomly corrupted until a chosen fraction q of the ground-state probability weight carried the wrong sign. "
        "The same label-free iteration was then applied. Across J2/J1=0.5, 0.6, and 1.0, every one of 32 random starts recovered for q <= 0.42. "
        "Since O_initial is approximately |1-2q|, this includes starts with sign overlap only about 0.16.",
        styles["Bodyx"]
    ),
    Paragraph(
        "Above this range the outcome becomes strongly pattern- and model-dependent. The correct statement is therefore not a rigorous spherical 'basin radius'. "
        "Rather, the ground-state basin occupies a large region under random physical-weight sign corruption, while structured sign patterns may sit in qualitatively different basins.",
        styles["Bodyx"]
    ),
]
story += [
    PageBreak(),
    Paragraph("4. Triangular lattice: competing basins, not failure of contraction", styles["H1x"]),
    Paragraph(
        "For the nearest-neighbor triangular Heisenberg antiferromagnet on a periodic 6x3 torus (18 spins, S^z=0), "
        "three simple two-color gauges do not approach the exact sign structure under repeated updates. "
        "They flow to wrong threshold-stable fixed points with essentially zero physical sign overlap.",
        styles["Bodyx"]
    ),
]

tri_data = [
    ["initial structured gauge", "terminal behavior", "ground-state sign overlap"],
    ["maxcut_x", "wrong fixed point after 13 updates", "~ 2 x 10^-16"],
    ["parity_y", "wrong fixed point after 18 updates", "~ 1 x 10^-15"],
    ["parity_xy", "wrong fixed point after 18 updates", "~ 2 x 10^-15"],
]
tt = Table(tri_data, colWidths=[48*mm, 80*mm, 48*mm], repeatRows=1)
tt.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(-1,0),NAVY),
    ("TEXTCOLOR",(0,0),(-1,0),colors.white),
    ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
    ("FONTNAME",(0,1),(-1,-1),"Helvetica"),
    ("FONTSIZE",(0,0),(-1,-1),7.3),
    ("GRID",(0,0),(-1,-1),0.35,LINE),
    ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,PALE2]),
    ("VALIGN",(0,0),(-1,-1),"TOP"),
    ("LEFTPADDING",(0,0),(-1,-1),4),
    ("RIGHTPADDING",(0,0),(-1,-1),4),
    ("TOPPADDING",(0,0),(-1,-1),4),
    ("BOTTOMPADDING",(0,0),(-1,-1),4),
]))
story += [
    tt,
    Paragraph(
        "However, this is not because the triangular ground-state signs are locally unstable under the map. "
        "Starting from randomly corrupted versions of the exact signs, all 8/8 trials recovered through q=0.45, and 4/8 recovered even at q=0.49. "
        "Thus the exact sign state is also a strong attractor on this cluster; the structured gauges simply begin in other basins.",
        styles["Bodyx"]
    ),
    box(
        "Exact basin invariant: symmetry character",
        "Let P be a symmetry with [P,H]=0. If the fixed modulus is invariant, P a=a, and the current fixed-amplitude wavefunction has character "
        "P psi_s = chi psi_s, then r_s(Px)=r_s(x). Any threshold mask sign(t-r_s) is therefore P-invariant, so every projected-Krylov update preserves chi. "
        "<b>A start in the wrong exact symmetry sector can never reach the target sector.</b>",
        PALE
    ),
    Paragraph(
        "The numerical translation audit makes this concrete. On the triangular cluster the exact ground state has T_x=+1 and T_y=+1. "
        "The maxcut_x and parity_xy gauges have T_x=-1 (residuals of order 10^-12), so they are rigorously excluded from the ground-state basin. "
        "By contrast, on the square 4x4 cluster the exact state, Marshall gauge, and stripe gauge all have T_x=T_y=+1, consistent with eventual recovery even when their initial overlap is poor.",
        styles["Bodyx"]
    ),
    Paragraph(
        "The parity_y triangular fixed point is not explained by this simple translation-character theorem. "
        "It is the clean remaining example of a competing basin inside, or across a mixture of, compatible symmetry content.",
        styles["Bodyx"]
    ),
]
story += [
    Paragraph("5. What is established, and what is not", styles["H1x"]),
    box(
        "Established on the tested exact clusters",
        "<b>(i)</b> The update is exactly the sign of a Krylov vector followed by projection back to fixed amplitudes. "
        "<b>(ii)</b> Energy-minimizing threshold selection is monotone in the fixed-amplitude energy. "
        "<b>(iii)</b> The square-lattice ground-state signs are reached exactly from every tested physical start, including poor Marshall starts at large J2. "
        "<b>(iv)</b> The ground-state basin is empirically large under random corruption. "
        "<b>(v)</b> Exact symmetry characters are invariant and can split sign space into disconnected basins.",
        PALE2
    ),
    Spacer(1,5),
    box(
        "Still open",
        "This is not yet a scalable ground-state algorithm. The controlled tests use the exact ground-state modulus and evaluate the threshold objective over the finite Hilbert space. "
        "The immediate theory problem is to explain wrong fixed points not already symmetry-forbidden (notably triangular parity_y), likely through weighted sign flux/frustration or another invariant. "
        "The immediate algorithmic problem is to determine how much of the contraction survives with sampled/approximate amplitudes and stochastic threshold-energy estimates.",
        HexColor("#FFF7E8")
    ),
    Paragraph("6. Reproducibility and live project links", styles["H1x"]),
    Paragraph(
        link("Basin dynamics verdict", GITHUB + "/BASIN_DYNAMICS_VERDICT_2026-09-28.md")
        + " &nbsp; | &nbsp; "
        + link("Symmetry-sector invariant", GITHUB + "/SYMMETRY_BASIN_INVARIANT_2026-09-28.md")
        + " &nbsp; | &nbsp; "
        + link("Current STATUS", GITHUB + "/STATUS.md"),
        styles["Bodyx"]
    ),
    Paragraph(
        "Primary scripts: "
        + link("iterated square test", GITHUB + "/experiments/iterated_krylov_exact.py")
        + ", "
        + link("wrong-gauge iteration", GITHUB + "/experiments/wrong_gauge_iterated.py")
        + ", "
        + link("basin scan", GITHUB + "/experiments/basin_radius_refined.py")
        + ", "
        + link("triangular iteration", GITHUB + "/experiments/triangular_iterated_exact.py")
        + ", "
        + link("symmetry audit", GITHUB + "/experiments/symmetry_sector_audit.py")
        + ".",
        styles["Bodyx"]
    ),
    Spacer(1,3),
    HRFlowable(width="100%", thickness=0.5, color=LINE, spaceBefore=2, spaceAfter=5),
    Paragraph(
        "<b>Working interpretation.</b> The one-step J1-J2 result is not a mysterious shortcut to the ground state. "
        "It is the exceptionally strong first move of an energy-decreasing projected-Krylov sign dynamics whose exact target is an attractor. "
        "What determines success is not sign overlap alone, but basin membership; exact symmetry sectors provide the first rigorous invariant separating those basins.",
        styles["Bodyx"]
    ),
    Paragraph(
        "This note is an internal snapshot of the falsification-driven subproject, not a publication claim. "
        "Finite-size and exact-amplitude assumptions are deliberately kept explicit.",
        styles["Smallx"]
    ),
]

doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(OUT)
