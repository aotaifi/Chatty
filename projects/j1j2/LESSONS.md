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

## Training and parametrisation
- Warm start must equal the reference EXACTLY (gate 0); check what each factor of the parent contributes (the RBM factor flips signs) | PP-only start 6.7% off sgn(M1) | PLAN_sign "T-SIGN8"
- A complex perturbation of a real Psi flips signs where |Re Psi| < eps^2 |Psi_2|; the normalised margin Re(Psi)/|Psi| has zero gradient at real Psi. Use a real f with m = Psi / sum|terms| | 2.1% step-0 flips | PLAN_sign "T-SIGN8 GATE 0 FAIL"
- Random init sits on a plateau whose length grows with L and with deeper data; warm-start instead | 8x8 stuck at chance 3000 steps; 6x6 beta-mix stalled | PLAN_sign "T-SCALE 8x8 FIRST ATTEMPT", "T-DEEP RESULT"
- A small gated correction (A/(1+A), lambda ~ 1e-3) is a bilinear saddle: the correction never grows | T-CAP arm 3 (= A10) | PLAN_sign "T-CAP"
- A larger class that contains the smaller one but trains worse means the optimiser, not representability | 2-PP sum 2.76 vs 1-PP 0.85e-6 | PLAN_sign "T-CAP RESULT"
- Full space-group projection generalised far better than translations-only on fresh chains (hypothesis: inductive bias) | 8x8 w_T 5.9e-5 vs no gain | PLAN_sign "8x8 PP-FULL WARM RESULT"
- One constant set (alpha, E, f0) everywhere; label and reference built with different alpha make a target mismatch | sign labels -0.18 vs loop -0.17543 | PLAN_pfaffian "Two branches for R1"

## Operations
- Paderborn normal QoS can queue for weeks; name --qos=express_hpcalot (20 H100-h/month) in every brief | 8x8 work fell back to A40 for a day | delta_today 11-05
- A watcher on a job that logs only at its end fires false STALLED; set the stall limit from the job's own logging cadence | false STALLED on 17751660 | delta_today 11-05
