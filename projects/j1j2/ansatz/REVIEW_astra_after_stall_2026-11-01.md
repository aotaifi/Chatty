I would test **(d), two jointly trained complex PPs with one Jacobi level**, next. An active RBM is a credible smaller intervention, but the residual tests make a large gain unlikely. Ordinary window backflow and a second explicit Jacobi level fail the requested cost budget with the algorithms below. This is a referee recommendation; the decision stays with the lead.

**M = measured in the records; D = derived, unvalidated timing extrapolation; P = prediction/judgment.** Energies below are per site above E0, in units of **10^-6**, at fixed s_G unless stated otherwise.

One important correction: **36.5x is backflow augmentation divided by bare backflow, not bare plain PP.** The raw T2 file gives 45.724/1.252 = 36.5, whereas plain PP takes 0.01609 s. The ratio of those recorded medians is **2,842x plain PP**. Backflow used two timing walkers versus twenty for plain PP, so this is an implementation comparison, not a matched performance benchmark. T2's operation counts give **804x** on the common plain-PP denominator. The plan/report compressed away this distinction. [Raw T2](/scratch/a/A.Otaifi/ansatz_runs/pfaffian_t2/t2_result.json)

The failure equation is

\[
 K_B(x)=\sum_{y\in\mathcal K_x}|H_{xy}|\frac{B_y}{B_x},\quad
 V_B(x)=\sum_{y\in\mathcal V_x}|H_{xy}|\frac{B_y}{B_x},\quad
 f_J[B](x)=\log\frac{K_B(x)}{H_{xx}+V_B(x)-E_B},
 \tag{1}
\]
\[
 T_k=B_k e^{c_k f_J[B_k]},\qquad
 r_{k+1}=\frac{e(B_k)-e(B_{k+1})}{e(B_k)-e(T_k)}. \tag{2}
\]

Here B denotes the positive magnitude, E_B the total FN energy, and kept/blocked edges are fixed by the external sign. The refit changes the representable log-amplitude directions in (2); augmentation changes (1)'s neighbour ratios. At round 4, **M:** bare 12.1845, augmented 4.3671, available gap 7.8174. Assuming an augmented/bare improvement ratio of 0.5, **D:** crossing 4 requires **r > 0.094**; ratios 0.4–0.6 give r > 0.117–0.078. Round 4 itself had ratio 0.82, which would require only r > 0.057 if sustained. I use 0.5 conservatively below; this transfer law is unestablished. P16 concerns the tangent at B_1 under s1, not a nonlinear impossibility theorem at B_4 under s_G.

| Rank | Option | P(below 4 AND <=20x at 10x10) | Claim on which the ranking rests |
|---|---|---:|---|
| 1 | (d) Two trainable PPs + one Jacobi level | **35%** | A second independently shaped pairing matrix supplies useful directions outside the old tangent, at roughly twice its evaluation cost. |
| 2 | (a) Active RBM + PP + one Jacobi level | **20%** | Joint nonlinear bulk adjustment can do more than the failed residual banks, enough to restore r around 0.1. |
| 3 | (c) Two explicit Jacobi levels | **<1%** | Accuracy is plausible; a further >30-fold algorithmic cost reduction would be needed. |
| 4 | (b) Existing 3x3 backflow + one Jacobi level | **<1%, below (c)** | It needs both useful new storage and a much larger reduction of its measured/common-denominator cost. |

These are subjective, nonexclusive probabilities, not fitted confidence levels.

**(a) Activate a small RBM, jointly train it and the PP.** Use the code's four symmetry-replicated hidden prototypes first: R = sum_(g,a) log cosh(b_a + W_a·z_g), B = |P| exp(R). Nonzero W and b make dR/dW = sum_g z_g tanh(b + W·z_g) nonzero. In (1), every B_y/B_x becomes (|P_y|/|P_x|) exp(R_y−R_x), changing both K and V; recompute the Jacobi factor of the whole product. This can correct moderate bulk-ratio errors while the explicit level carries the difficult correction. It cannot remove exact PP zeros with a finite R.

**P:** one matched averaged refit gives **r = 0.08**, hence bare **11.56**, augmented **4.05** under the 0.5 transfer hypothesis. I would not predict an order-of-magnitude improvement from fixing initialization. P6/P13 and the activated-initialization tangent arm are adverse evidence: small fields remain approximately a two-body Jastrow. Published RBM–PP improvements motivate joint optimization, but do not establish this near-stall FN mechanism ([Nomura et al., abstract](https://arxiv.org/abs/1709.06475)).

**Kill:** verify nonzero derivatives and actual RBM motion; compare active versus inactive arms from B_4, two matched seeds, 1,500 steps and the same tail averaging. If active r <= 0.05 and its augmented improvement over the inactive control is <0.10 in both seeds, reject the proposed bulk rescue. Budget **~8 H100-hours + 8 CPU core-hours**, prospective. A 10x10 ratio benchmark exceeding 20x independently kills deployment.

**(b) Assess the existing complex, linear 3x3 backflow; do not enlarge to 5x5.** Its matrix is

\[
 M_{ij}(z)=F_{ij}+\sum_{\delta\in3\times3}
 [u^{(1)}_{\delta,j-i}z_{i+\delta}+u^{(2)}_{\delta,i-j}z_{j+\delta}]. \tag{3}
\]

This changes d log|P| = Re(dP/P), hence the refit's tangent directions and the ratios in (1), by letting pairing depend on surrounding spins. **P:** if jointly refitted successfully, **r = 0.15**, bare **11.01**, augmented **3.78**. That is a weak forecast: B31's 3x3→5x5 tax improvement was only 5.60→4.58, G4/C11 found no VMC gain, and D13 found a nonlocal residual. A 1x1 version of (3) merely reparametrizes F because its row/column spins are fixed up/down.

**Kill:** cost comes first and already fails for the existing implementation. Budget **<=2 CPU core-hours** for a compiled 10x10 kernel timing over a representative subset of images/moves, including matrix construction; >20x rejects this concrete route before training. If a new implementation passed, the conditional accuracy test would cost **~12 H100-hours + 8 CPU core-hours** for two matched seed pairs; r <= 0.05 with augmented e >= 4.2 in both backflow arms refutes the forecast. Rank alone is not a theorem excluding every structured backflow implementation.

**(c) Add the current augmented guide's own Jacobi factor.** Let G_1 = B exp(c_1 f_J[B]) and G_2 = G_1 exp(c_2 f_J[G_1]). In the second application of (1),

\[
 K_{G_1}(x)=\sum_{y\in\mathcal K_x}|H_{xy}|\frac{B_y}{B_x}
 e^{c_1(f_J[B](y)-f_J[B](x))}, \tag{4}
\]

and V changes identically on blocked edges, with E_B replaced by E_(G_1). This supplies precisely the direction K3 found missing from primitive dictionaries. It improves accuracy without storing anything in B: **r = 0** for this operation. **M:** earlier, under s1, the two-level minimum was 7.219 versus 10.185 for one level; that is not a result at today's guide/sign. **P:** today's minimum is **3.5**, after rescanning c_2 with c_1 initially fixed at 4.

**Kill:** using existing exact tables, scan c_2 including zero; if the minimum remains >=4, it fails the requested accuracy target. Budget **<=16 CPU core-hours**, including construction/readout. Independently, the three-shell cost below rejects deployment without a new evaluator. Products versus equivalent nested notation do not remove that third shell; stochastic truncation would be a different, presently unvalidated guide.

**(d) Preserve low update rank while enlarging the PP: B = |P_1 + lambda P_2|.** Each P_j is a separately parameterized, symmetry-projected complex PP. Jointly refit both and lambda to B_4 exp(4 f_J[B_4]), then compute one Jacobi factor of their sum. Start P_2 at finite separation from P_1: identical copies just reproduce the old tangent. The new direction is

\[
 \partial_\lambda\log B=\operatorname{Re}\frac{P_2}{P_1+\lambda P_2}
 \tag{5}
\]

for a real lambda component. This configuration-dependent interference changes B_y/B_x in (1), without making either pairing matrix depend on the spin environment. It is a concrete way to buy new directions while retaining rank-2/4 updates per component.

**Prior-art collision checked:** P11 killed a dictionary of fixed primitive Jacobi features (eps² >=0.97); equation (5) is not that dictionary, but the same lack of overlap could recur. G11 killed a particular rank-20 vison/BCS construction at 4x6; adding pairing components is therefore not itself evidence of improvement. P16 also cautions against initializing both components in the same tangent neighbourhood.

**P:** **r = 0.16**, bare **10.93**, augmented **3.74**. **Kill:** two matched seed pairs against single-PP continuation, fixed steps/averaging, budget **~12 H100-hours + 8 CPU core-hours**. If both expanded fits have r <= 0.05 and augmented e >=4.2, reject this two-component proposal. A matched 10x10 timing >20x is an independent kill. These kills concern the specified finite-budget proposals, not all nonlinear functions in each ansatz class.

**Walker operation counts, on one common denominator.** A step means all Hamiltonian-neighbour ratios, with T2's per-step inverse/precomputation convention; external sign evaluation is excluded. Set N=L², n=N/2 and S=16N symmetry images. The determinant lemma gives

\[
 \frac{\det(M+UV^T)}{\det M}=\det(I_r+V^TM^{-1}U). \tag{6}
\]

For a static pairing matrix, cached contractions reduce any k-swap endpoint to a 2k determinant, including after symmetry summation. T2 measured d=76 neighbours and q_2=2729.5 unique second-shell endpoints at N=36. For extrapolation use **D:** d=(76/36)N, q_2=2729.5(N/36)²; q_3≈d³/6 counts commuting disjoint triples to leading order, with finite-size corrections unmeasured. Per-image complex multiply-add proxies, directly extending T2's model, are

\[
 C_0=\tfrac73n^3+d(2n+8),\qquad
 C_1=\tfrac{13}3n^3+12d+56q_2, \tag{7}
\]
\[
 C_a=C_1+m[N+\chi(d+q_2)],\quad
 C_c\simeq C_1+130q_3,\quad C_d\simeq2C_1. \tag{8}
\]

Here m=4 hidden prototypes **per symmetry image**, as in cx_r1fit.py, not 4N extra hidden units per image. Cached fields cost O(m) per endpoint; chi represents updates plus log-cosh evaluation, with **chi=8 assumed**, not measured. The 130 is a rounded rank-6 assembly/LU/reduction count. Multiplication by S cancels in each ratio. Scalar edge reductions add overhead but do not restore a cheap second level.

For the generic 3x3 backflow evaluator, define

\[
 L(n,r)=rn^2+r^2n+r^3/3,\quad
 Q(n,r)=9n^2+\min\{n^3/3,L(n,r)\},
\]
\[
 C_b=\tfrac43n^3+dQ(n,13)+q_2Q(n,26). \tag{9}
\]

The 9n² term builds the window correction; the minimum chooses full factorization when the lemma is dearer. Ranks 13/26 approximate one/two-swap structural ranks; at 6x6 the numerical rank saturates at 18. These are feasible generic-kernel counts, not optimal-algorithm lower bounds. Even deleting all contraction/construction costs leaves q_2·26³/(3C_0)≈**392x** at 10x10 for ordinary rank-26 LU.

| Guide | 6x6 / plain PP | 10x10 / plain PP | 14x14 / plain PP |
|---|---:|---:|---:|
| One level, current plain PP | **5.38 M**; 10.56 D | **5.48 D** | **3.78 D** |
| (a) Active RBM, m=4, one level | **15.9 D** | **7.65 D** | **4.92 D** |
| (b) 3x3 backflow, one level | **~805 D**; raw timing ratio ~2842 | **~4341 D** | **~14239 D** |
| (c) Two explicit levels | **~572 D** | **~654 D** | **~677 D** |
| (d) Two PPs, one level | **21.1 D** | **11.0 D** | **7.55 D** |

The measured/model discrepancy at 6x6 prohibits presenting these as stopwatch predictions. For (a), m=16 gives **14.1x at 10x10** if chi=8, but breaches 20x if chi>13.4; RBM work is not free. For (c), deduplicating commuting triples makes the estimate smaller than the plan's 10³–10⁴, but still far above 20. Persistent inverse updates would change the denominator and must be benchmarked for both baseline and candidate.

I read the requested plan sections, report and P1–P20, and searched the ledger for RBM/Jastrow/backflow and the proposed sum. I found no equation-board export; equations (1)–(5) use the plan and source definitions. Raw round-4, K3, calibration and T2 JSONs were checked; no FN solve or new timing experiment was performed. CPU budget estimates use core-hours (the existing round-4 readout used eight cores for 0.526 hours). All training budgets above are prospective; no jobs were submitted.

Sources: [plan](/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/PLAN_pfaffian.md), [loop report](/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/docs/reports/2026-10-31_label_free_loop_6x6.md), [idea ledger](/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/TRIED_IDEAS.md), [T2 evaluator and cost model](/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/ansatz/code/t2_lemma.py:519), [round-4 readout](/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r4/readout/readout.json), [K3 readout](/scratch/a/A.Otaifi/ansatz_runs/pfaffian_k/k3_result.json).
