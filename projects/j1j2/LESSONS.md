# Shared lessons, j1j2 (project lead + angle leads; PI request 2026-11-05, which overrides root CLAUDE.md §8 "no lessons file" for this project)

Read before designing any test; add a row when a lesson cost a run or a wrong number. One line each: lesson | what it cost | record.
Each angle adds its own; nobody deletes another's row. A lesson that becomes a firm rule also goes into CLAUDE.md.

## Objectives and metrics
- Train and grade a loop component on the loop's OWN metric, asked from the lead first, not on a proxy | 8x8 sign: w_T fell 3x, eps^2 0.54 vs sgn(M1) 0.40 (11-05) | PLAN_sign "GATE (a) on ppw_s1"
- Never weight errors per configuration by psi^2; the FN energy and the labels weigh pairs/bonds across the node | amplitude storage, weeks (10-2026); sign 8x8 (11-05) | CLAUDE.md (repo) record; PLAN_sign "T-BOND"
- |H_xy| a_x a_y is a psi^2-type bond weight: it starves the deep near-node bonds where errors cost | T-EDGE worsened the 6x6 tax 1.45 -> 1.65e-6 | PLAN_sign "S2b'/T-EDGE"
- The exact cost of a sign error on bond (x,y) is the shift of h = |H| r between SK and den in f_J; tiny SK next to the node makes wrong unblocking catastrophic | explains 80% of eps^2 on 0.4% of bonds | PLAN_sign "T-BOND"
- Held-out BCE does not track the tax at high accuracy | T-DEEP2: accuracy 0.979 -> 0.990 while tax +0.27e-6 | PLAN_sign "T-DEEP2 RESULT"
- Check a proxy or diagnostic on the known (passing) cases before using it | '61% off-training-set' misled; Gate S added | PLAN_sign "T-BOND"
- A coefficient calibrated at one size or measure does not transfer (17.8 -> 32.6 with the right measure; w -> tax across L unsupported) | factor-2 calibration error | PLAN_sign "T-SCALE REVIEW"
- The loop gains come from the TARGET and the CLASS, not the weighting: label-free A e^{c f_J} stored 37% vs 15% with FN labels; two PPs r 0.49 vs 0.05 | weeks of loss variants (B16-B27, F1/F2/T1s) | PLAN.md "What we understand"; PLAN_pfaffian R5
- Pointwise labels from walker counts are 20-70x noisier than a round's correction (rms 0.006 in ln phi); use exact per-sample targets | Test W | PLAN.md "What we understand"
- Draw equally per |B|^2 decade and size pools by BOND-weight ESS (partner weights rho), not configuration ESS | V deepest band bond ESS 1.3P vs config 23k; 8x8 gate (c) failed | Astra cj-6a20013aa7ba; PLAN_pfaffian gate (c)
- A fitted range or quantile set at 6x6 does not transfer: measure it at the new size (+2..-6 held 0.648 at 8x8; f0 0.62 -> 1.22) | 8x8 pool resized | PLAN_pfaffian decade range, (e)
- f0 is the n-weighted SIGNED q.9999 of c f_J / c, not an |f_J| or |B|^2 quantile; non-finite f_J rows are a node, not 0 | f_J defect audit 11-03 | PLAN.md "Step back 2026-11-03"
- Shares of a frozen-phi BOUND are not shares of the true cost (bound/true 6-18x); use a saturating or exact measure before predicting splits | Test D prediction refuted | PLAN.md "TEST D RESULT"
- Compare against the strongest published network, measured on our own E0, not a competitor we built; re-derive a published gap from the raw energy | CNN-MPS bar 3.2 -> 2.65e-6 (rounded ED) | PLAN.md Queue "Competitor bar"
- Independent energies cannot bound a small difference (two 3e-6 errors -> 4.2e-6 on the tax); use paired / common-random-number estimators | c1 redesign | Astra cj-6a20013aa7ba

- A per-bond (or per-configuration) surrogate for eps^2 misses its cross terms: eps^2 is a Laplacian quadratic form in label errors, errors at one x add inside the logs | T-BOND rejected before running; Gate S had a demonstrated false positive | PLAN_sign "ASTRA on T-BOND"
- Validating a surrogate on 2-3 correlated known cases proves little; test it directionally on controlled perturbations (isolated, cluster flips) | Gate S | ASTRA_eps2.md (3)
- A per-bond diagonal surrogate of eps^2 is wrong: eps^2 is a graph-Laplacian form eta^T L eta with endpoint cross terms; errors at one x add inside the logs and most roots carry several wrong bonds (5085/5724); the x end held only 0.07-0.09 | T-BOND design rejected (lead's derivation), Gate S would have passed falsely | Astra cj-a2cbf28c7311; PLAN_sign "T-BOND"
- Cutting the wrong-sign FRACTION (w_T) can raise the loop's label error: training moved errors into high-amplitude decades (-2..-5: .40-.72 vs sgn(M1) .04-.31), where |H| B_x B_y weighs them most | 12 trained 8x8 signs all worse than the untrained sign (best 0.61/0.67 vs 0.39/0.41) | PLAN_pfaffian gate (a) round 2; gate_a/round2/summary.json

- Check that a diagnostic's target is the quantity the network actually copied (round k vs k+1 numerator) before reading a kill | T-SOFT1 kill and a wrong mechanism reading, withdrawn by Astra | PLAN_sign "ASTRA combined review"

## Training and parametrisation
- Warm start must equal the reference EXACTLY (gate 0); check what each factor of the parent contributes (the RBM factor flips signs) | PP-only start 6.7% off sgn(M1) | PLAN_sign "T-SIGN8"
- A complex perturbation of a real Psi flips signs where |Re Psi| < eps^2 |Psi_2|; the normalised margin Re(Psi)/|Psi| has zero gradient at real Psi. Use a real f with m = Psi / sum|terms| | 2.1% step-0 flips | PLAN_sign "T-SIGN8 GATE 0 FAIL"
- Random init sits on a plateau whose length grows with L and with deeper data; warm-start instead | 8x8 stuck at chance 3000 steps; 6x6 beta-mix stalled | PLAN_sign "T-SCALE 8x8 FIRST ATTEMPT", "T-DEEP RESULT"
- A small gated correction (A/(1+A), lambda ~ 1e-3) is a bilinear saddle: the correction never grows | T-CAP arm 3 (= A10) | PLAN_sign "T-CAP"
- A larger class that contains the smaller one but trains worse means the optimiser, not representability | 2-PP sum 2.76 vs 1-PP 0.85e-6 | PLAN_sign "T-CAP RESULT"
- Full space-group projection generalised far better than translations-only on fresh chains (hypothesis: inductive bias) | 8x8 w_T 5.9e-5 vs no gain | PLAN_sign "8x8 PP-FULL WARM RESULT"
- One constant set (alpha, E, f0) everywhere; label and reference built with different alpha make a target mismatch | sign labels -0.18 vs loop -0.17543 | PLAN_pfaffian "Two branches for R1"
- Judge whether a block is used in FUNCTION space, not parameter space: a large-column block moves few parameters per unit change | M5' declared INVALID and cancelled prematurely, rerun ~12 A40-h | PLAN_pfaffian M5'
- Zero-initialised RBM blocks get zero gradient under cx_r1fit; L-SWO/R1 were plain PP, not RBM x PP | mislabelled runs | PLAN.md "CODE FINDING"
- cx_r1fit step acceptance passed a negated step as noise; fit-to-fit E_FN walk +-0.2e-6 makes single fits unquotable | candidate cause of the walk | delta_today 11-03
- Pick checkpoints on one half of the grading draws and gate on the other (selection bias) | - | PLAN_pfaffian gate (a)

- A fired kill test is not a verdict: diagnose (budget, tweak), try cheap variants, ask Astra, then decide | T-IW crossed d only after ~15k steps; a 3k kill would have killed it | PLAN_sign "T-IW RESULT"; PI 11-08

## Operations
- Paderborn normal QoS can queue for weeks; name --qos=express_hpcalot (20 H100-h/month) in every brief | 8x8 work fell back to A40 for a day | delta_today 11-05
- A watcher on a job that logs only at its end fires false STALLED; set the stall limit from the job's own logging cadence | false STALLED on 17751660 | delta_today 11-05
- A race between machines is won by expected FINISH time, not by which copy starts first | 9x faster H100 copy of V cancelled, ~10 h lost | delta_today 11-04
- Check a peer's units: a parenthesised error off by 10x turns a 1.2-sigma pass into a 12-sigma miss | c3 notation | PLAN_pfaffian c3
- A peer's "being submitted now" is not a submission: arm a wait on the job id or progress file in the same turn | 8x8 R1 fit never submitted, ~2 days lost (11-06 -> 11-08) | delta_today 11-08
- A configuration key must fit the lattice: pp2_lemma dedup keys by one uint64, so it collides above 64 sites (14x14 collision shown) | caught before any 10x10 use | Astra cj-852432069699, arch_tied/ANSWER.md
