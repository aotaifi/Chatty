# Shared lessons, j1j2 (project lead + angle leads; PI request 2026-11-05, which overrides root CLAUDE.md §8 "no lessons file" for this project)

Read before designing any test; add a row when a lesson cost a run or a wrong number. One line each: lesson | what it cost | record.
File it under its topic (no dated blocks); if it repeats a row, sharpen that row instead. Each angle adds its own; nobody deletes
another's row (merging duplicates keeps every number and record). A lesson that becomes a firm rule also goes into CLAUDE.md.

## First: predict from the equations
- Every step is a known function, the optimiser and fit included: derive what it must do before running (tangent-space share = what the network can store, Gauss-Newton spectrum along the target = how fast), find the missing ingredient by asking what the physics requires (the hexagon move), predict, then run; 'the learning side is empirical' is a label, not a cause | weeks of training variants labelled empirical; the one tangent check (Theta v2, 0-4%) found the missing direction and two PPs stored r 0.49 vs 0.05 | PI 2026-11-12; CLAUDE.md point 1
- A fired kill test is not a verdict: diagnose (budget, tweak), try cheap variants, ask Astra, then decide | T-IW crossed d only after ~15k steps; a 3k kill would have killed it | PLAN_sign "T-IW RESULT"; PI 11-08

## Objectives, targets and weights
- Train and grade a loop component on the loop's OWN metric, asked from the lead first, not on a proxy | 8x8 sign: w_T fell 3x, eps^2 0.54 vs sgn(M1) 0.40 (11-05) | PLAN_sign "GATE (a) on ppw_s1"
- Never weight errors per configuration by psi^2; the FN energy and the labels weigh pairs/bonds across the node. |H_xy| a_x a_y is the same psi^2-type weight on bonds: it starves the deep near-node bonds where errors cost | amplitude storage, weeks (10-2026); sign 8x8 (11-05); T-EDGE worsened the 6x6 tax 1.45 -> 1.65e-6 | CLAUDE.md (repo) record; PLAN_sign "T-BOND", "S2b'/T-EDGE"
- The exact cost of a sign error on bond (x,y) is the shift of h = |H| r between SK and den in f_J; tiny SK next to the node makes wrong unblocking catastrophic | explains 80% of eps^2 on 0.4% of bonds | PLAN_sign "T-BOND"
- Cutting the wrong-sign FRACTION (w_T) can raise the loop's label error: training moved errors into high-amplitude decades (-2..-5: .40-.72 vs sgn(M1) .04-.31), where |H| B_x B_y weighs them most | 12 trained 8x8 signs all worse than the untrained sign (best 0.61/0.67 vs 0.39/0.41) | PLAN_pfaffian gate (a) round 2; gate_a/round2/summary.json
- The loop gains come from the TARGET and the CLASS, not the weighting: label-free A e^{c f_J} stored 37% vs 15% with FN labels; two PPs r 0.49 vs 0.05 | weeks of loss variants (B16-B27, F1/F2/T1s) | PLAN.md "What we understand"; PLAN_pfaffian R5
- Pointwise labels from walker counts are 20-70x noisier than a round's correction (rms 0.006 in ln phi); use exact per-sample targets | Test W | PLAN.md "What we understand"
- f0 is the n-weighted SIGNED q.9999 of c f_J / c, not an |f_J| or |B|^2 quantile; non-finite f_J rows are a node, not 0 | f_J defect audit 11-03 | PLAN.md "Step back 2026-11-03"
- Compare against the strongest published network, measured on our own E0, not a competitor we built; re-derive a published gap from the raw energy | CNN-MPS bar 3.2 -> 2.65e-6 (rounded ED) | PLAN.md Queue "Competitor bar"

## Diagnostics, surrogates and proxies
- Check a proxy or diagnostic on the known (passing) cases before using it, and then directionally on controlled perturbations (isolated, cluster flips): 2-3 correlated known cases prove little | '61% off-training-set' misled; Gate S added and later shown a false positive | PLAN_sign "T-BOND"; ASTRA_eps2.md (3)
- Check that a diagnostic's target is the quantity the network actually copied (round k vs k+1 numerator) before reading a kill | T-SOFT1 kill and a wrong mechanism reading, withdrawn by Astra | PLAN_sign "ASTRA combined review"
- A per-bond (or per-configuration) diagonal surrogate of eps^2 is wrong: eps^2 is a graph-Laplacian form eta^T L eta with endpoint cross terms; errors at one x add inside the logs and most roots carry several wrong bonds (5085/5724); the x end held only 0.07-0.09 | T-BOND design rejected before running (lead's derivation); Gate S would have passed falsely | Astra cj-a2cbf28c7311; PLAN_sign "T-BOND", "ASTRA on T-BOND"
- Held-out BCE does not track the tax at high accuracy | T-DEEP2: accuracy 0.979 -> 0.990 while tax +0.27e-6 | PLAN_sign "T-DEEP2 RESULT"
- Shares of a frozen-phi BOUND are not shares of the true cost (bound/true 6-18x); use a saturating or exact measure before predicting splits | Test D prediction refuted | PLAN.md "TEST D RESULT"
- A coefficient, range or quantile set at one size or measure does not transfer: re-measure it at the new size (17.8 -> 32.6 with the right measure; w -> tax across L unsupported; +2..-6 held 0.648 at 8x8; f0 0.62 -> 1.22) | factor-2 calibration error; 8x8 pool resized | PLAN_sign "T-SCALE REVIEW"; PLAN_pfaffian decade range, (e)

## Sampling, pools and grading statistics
- Draw equally per |B|^2 decade and size pools by BOND-weight ESS (partner weights rho), not configuration ESS | V deepest band bond ESS 1.3P vs config 23k; 8x8 gate (c) failed | Astra cj-6a20013aa7ba; PLAN_pfaffian gate (c)
- Independent energies cannot bound a small difference (two 3e-6 errors -> 4.2e-6 on the tax); pairing needs correlated sampling: same seeds and starts do NOT pair branching FN runs, populations decorrelate within a few blocks (block corr ~0.04; paired 3.8e-6 vs unpaired 3.6e-6) | c1 redesign; T-NODE | Astra cj-6a20013aa7ba; ASTRA_node.md s1; astra/paired_fn/ASTRA_paired.md
- FN chains started from |B|^2 draws still settle by ~1e-5 after a 50-block (5 tau) burn; use 150 blocks + a halves check, and never compare arms with different burns | base arm 4222088 halves -1.2e-5 in 3/4 chains | PLAN_pfaffian 'R1' GRADE DESIGN'
- Two jobs with the same chain table and seeds are ONE sample; a gate pooling them halves the error | 11-10 headline read 3.3 sigma, true 2.7 (prodB = prodBx byte-identical) | delta_today 11-11; headline_p1/prodB/README_DUPLICATE.md
- Pick checkpoints on one half of the grading draws and gate on the other (selection bias) | - | PLAN_pfaffian gate (a)
- cx_r1fit step acceptance passed a negated step as noise; fit-to-fit E_FN walk +-0.2e-6 makes single fits unquotable | candidate cause of the walk | delta_today 11-03

## Training and parametrisation
- Warm start must equal the reference EXACTLY (gate 0); check what each factor of the parent contributes (the RBM factor flips signs) | PP-only start 6.7% off sgn(M1) | PLAN_sign "T-SIGN8"
- A complex perturbation of a real Psi flips signs where |Re Psi| < eps^2 |Psi_2|; the normalised margin Re(Psi)/|Psi| has zero gradient at real Psi. Use a real f with m = Psi / sum|terms| | 2.1% step-0 flips | PLAN_sign "T-SIGN8 GATE 0 FAIL"
- Random init sits on a plateau whose length grows with L and with deeper data; warm-start instead | 8x8 stuck at chance 3000 steps; 6x6 beta-mix stalled | PLAN_sign "T-SCALE 8x8 FIRST ATTEMPT", "T-DEEP RESULT"
- A small gated correction (A/(1+A), lambda ~ 1e-3) is a bilinear saddle: the correction never grows | T-CAP arm 3 (= A10) | PLAN_sign "T-CAP"
- A larger class that contains the smaller one but trains worse means the optimiser, not representability | 2-PP sum 2.76 vs 1-PP 0.85e-6 | PLAN_sign "T-CAP RESULT"
- Full space-group projection generalised far better than translations-only on fresh chains (hypothesis: inductive bias) | 8x8 w_T 5.9e-5 vs no gain | PLAN_sign "8x8 PP-FULL WARM RESULT"
- Judge whether a block is used in FUNCTION space, not parameter space: a large-column block moves few parameters per unit change | M5' declared INVALID and cancelled prematurely, rerun ~12 A40-h | PLAN_pfaffian M5'
- Zero-initialised RBM blocks get zero gradient under cx_r1fit; L-SWO/R1 were plain PP, not RBM x PP | mislabelled runs | PLAN.md "CODE FINDING"

## Inputs and constants
- One constant set (alpha, E, f0/w) per base, used everywhere and produced once by one job with an install guard; a second producer must not run | sign labels -0.18 vs loop -0.17543 (target mismatch); s306 p=1 constants nearly computed twice (Paderborn + sign angle) | PLAN_pfaffian "Two branches for R1"; delta_today 11-12
- A configuration key must fit the lattice: pp2_lemma dedup keys by one uint64, so it collides above 64 sites (14x14 collision shown) | caught before any 10x10 use | Astra cj-852432069699, arch_tied/ANSWER.md

## Operations
- Placement: take the card that FINISHES first. When a queue blocks, check idle cards (sinfo -p cip,inter -o '%N %G %t') and test-only starts and decide on the spot; slices that start now beat a full A40 or H100 a day out; judge speed from specs and cluster/COMPUTE_SCOUT_2026-11-12.md, no new probes; the cip cap is 3 GPUs per user across all angles | 9x faster H100 copy of V cancelled, ~10 h lost (11-04); constants job 17930563 sat PENDING beside idle slices (11-12) | delta_today 11-04, 11-12; PI 11-12
- Paderborn normal QoS can queue for weeks; name --qos=express_hpcalot (20 H100-h/month) in every brief | 8x8 work fell back to A40 for a day | delta_today 11-05
- Paderborn cont start estimates are days off in both directions; a 'backup' copy of a run must use distinct seeds or be cancelled | 4186183, 4203776 started days early | delta_today 11-11/12
- A watcher on a job that logs only at its end fires false STALLED; set the stall limit from the job's own logging cadence | false STALLED on 17751660 | delta_today 11-05
- A peer's "being submitted now" is not a submission: arm a wait on the job id or progress file in the same turn | 8x8 R1 fit never submitted, ~2 days lost (11-06 -> 11-08) | delta_today 11-08
- Check a peer's units: a parenthesised error off by 10x turns a 1.2-sigma pass into a 12-sigma miss | c3 notation | PLAN_pfaffian c3
