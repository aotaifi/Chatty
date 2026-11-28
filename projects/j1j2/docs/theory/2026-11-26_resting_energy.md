# Resting energy of the complete loop (FN-lead, 2026-11-26; paper, rung 1; account for PLAN.md 'LEAD AND OPEN QUESTION')
Notation: guide g = S A; e(S,A) = E_FN[S A] - E_0 per site; chamber floor c(S) = inf_A e(S,A); amplitude excess x(S,A) = e - c >= 0.
One full round = sign step T[S,A] = sgn(1 + alpha(E_L - E)) then amplitude round(s) at the new sign, each stored by a copy.

(1) [exact] e = c + x, and a copy (S^, A^) of a target (S', A') costs  e(S^,A^) - e(S',A') = [e(S^,A^) - e(S',A^)] + [e(S',A^) - e(S',A')] = sigma + tau_A
    (telescoping; sigma = sign copy tax graded at the stored amplitude, tau_A = amplitude copy tax at the target sign; board 24 and board 5 bound them).
(2) [cond: linear regime near the rest point] amplitude round at fixed sign: x' = rho_A x + tau_A, with tau_A = (1 - q) G + f_A, G = (1 - rho_A) x the exact
    round gain, q the kept share of the correction (M-R: tax/gain = eps^2) and f_A the part of the copy error that does not scale with the correction
    (reach floor, NNd7/NNp1). Hence x' = [1 - q(1 - rho_A)] x + f_A,  x* = f_A / (q (1 - rho_A)).
    Consequence [exact given (2)]: a constant kept share q < 1 alone never makes a floor, it only slows the chain; the floor is f_A. The measured fall of q_k
    after round 6 is f_A meeting a shrinking G, as the amplitude lead diagnosed.
(3) [conj, 60%; closure, form to be checked against signdepth/result.json when the job ends] sign step: c' = rho_S c + kappa x + sigma.
    rho_S = contraction of the exact step at exact amplitude; kappa x = mis-flips caused by amplitude ratio errors in E_L.
    From the UNFINISHED depth test's log lines (job 18227574, s2 start, graded at |psi0|, c(s2) taken as E_FN[s2 |psi0|] 1.494e-6; not yet re-read from result.json):
      exact amplitude: c' 3.6e-9  -> rho_S ~ 0.0024
      A_0: x 1.94e-5 -> c' 2.685e-6 -> kappa 0.138;   A_2: x 1.83e-6 -> c' 3.70e-7 -> kappa 0.20
    Prediction that can fail (third point, same file): E_FN[s2 A_6] - 1.494e-6 = x(A_6) in 0.43-0.62e-6 (from c' 8.6e-8), i.e. E_FN[s2 A_6] 1.9-2.1e-6.
    Not valid far from the exact sign: at the Marshall start |psi0| is a poor amplitude for the chamber (E_FN[M |psi0|] 4.75e-3 > E_FN[M A_2] 1.54e-3).
(4) [exact given (2),(3)] rest point:  x* = f_A/(q(1 - rho_A)),  c* = (kappa x* + sigma)/(1 - rho_S),
      E_rest - E_0 = sigma/(1 - rho_S) + [1 + kappa/(1 - rho_S)] x*  ~  sigma + (1 + kappa) x*      (rho_S << 1).
Readings:
 A. The sign copy tax enters ONCE, unamplified: an exact sign step forgets the previous sign's errors (rho_S ~ 0), so sign copy errors do not accumulate over
    rounds. The only sign number that matters is the terminal sigma measured in FN energy; a per-round kept-share bar on the sign copy is not needed.
 B. The stall of the chained sign copy at round 3 (G_4 ~ R_3; round 4 adds -6.4(6.2)e-6, PLAN_sign STEP 17) is this rest point, c* = sigma, not a new failure:
    more rounds cannot lower it, a smaller sigma does, one for one.
 C. Amplitude depth matters for the sign step only through kappa x: a surcharge of ~15-20% on the amplitude excess. Shallow rounds are enough for the sign step
    whenever x is already small against sigma.
 D. Bar at 8x8: with sigma 6.1(1.6)e-5 (CF vs sgn(M1)) and x* of order 1e-5, the rest point is sigma-dominated. To end a margin m below the baseline the copy
    needs sigma <= e_base - c_target - (1 + kappa) x* - m: a factor of ~3 below today's sigma, not orders of magnitude. Open numbers at 8x8: c_target (value of the
    exact sign-step target against sgn(M1)) and x*.
Not covered: the jump of x when the sign changes (the old amplitude is not adapted to the new chamber; TEST X), which affects the path, not the rest point;
kappa's constancy beyond two points; rho_A, q, f_A at 8x8 (measured at 6x6 only).
Limit on (3) (sign lead, 11-26, PLAN_sign 'FN-LEAD QUESTIONS'): two exact deep sign steps raised E_FN by ~2 sigma at 8x8 (G(-5) +6.4(3.5)e-6, T-NODE +8.0(4.1)e-6);
rho_S ~ 0 is claimed only at good amplitude and not for deep rows. Nothing in the sign records grew over rounds (gaps 5.96e-4, 9.6e-5, 6.7e-5, round 4 -0.6(0.6)e-5).

## Depth test read against the account (FN-lead 11-26 20:00; runs/j1j2/consol_6x6/signdepth/result.json, job 18227574, numbers re-read by me)
PI's rule: NOT met (Marshall: T_2 vs T_6 gain 4.177e-3 vs 4.160e-3 at |psi0|, energy-weighted agreement 0.983; s2: 1.123e-6 vs 1.407e-6, agreement 0.823).
(5) [measured, corrects the picture behind (3)] a sign step is best graded, and best used, with the amplitude it was computed from:
    s2, graded at A_2^S (E_FN[s2 A_2] 3.320e-6): step from A_2 gains 0.515e-6, from A_6 0.493e-6, from chi_6 0.514e-6, from the exact |psi0| only 0.145e-6.
    Marshall, graded at |psi0|: step from A_2 gains 4.18e-3, from A_6 4.16e-3, from the exact |psi0| only 1.88e-3 (precision against the oracle flips 0.50, gain 2.2x).
    So the oracle amplitude is not the best input for the sign step. Mechanism [conj, 70%]: rounds at a fixed sign suppress the amplitude on wrong-sign rows
    (|psi0|^2-weighted dip -0.37 -> -1.10 -> -4.22 at s2 for rounds 0, 2, 6; -0.05 -> -3.17 -> -5.53 from Marshall), E_L ~ 1/A grows there, and the step flips exactly
    those rows. The amplitude round is the detector of wrong signs and the sign step the actuator. The chamber wall is what the loop uses, not what stops it.
(6) Closure (3) near s2: kappa 0.138 (A_0), 0.20 (A_2) stand as read from result.json. Third point (E_FN[s2 A_6] predicted 1.9-2.1e-6) NOT in the file: asked of the
    amplitude lead, with E_FN[s2 chi_6] and E_FN[s2 h1]. Closure (3) fails far from the exact sign, as flagged: from Marshall the exact amplitude is the worst input.
(7) Regimes of E_rest = sigma + (1 + kappa) x*: at 6x6 near s2 the amplitude excess is the larger term (E_FN[s2 A_2] 3.32e-6 against 1.49e-6 at the exact amplitude,
    sign copies 0.42-0.88e-6); at 8x8 sigma 6.1e-5 is. Both leads' terms are live; which dominates depends on the size.
PREDICTION registered now, for the first complete exact cycle with a stored amplitude (rung 2, to be asked for no earlier than next turn): start chi_6 at s2,
    sign step T = T[s2, chi_6], then exact amplitude rounds at T from chi_6. E_FN[T |psi0|] = 1.52e-7 bounds the new chamber floor, so the rounds should pass
    below the s2 chain's rest (N* ~1.05e-6, floor of the cheap copies 1.07e-6) within two rounds: N after round 2 at T <= 0.8e-6 (60%). Kill of the coupling idea
    at 6x6: N stays >= 1.0e-6 after three rounds at T (then the s2 rest was not the sign's doing).

## PI idea 11-26: copy from the walkers instead of from the ratios (recorded, not yet derived)
Two separate things. (i) Walkers as LABELS (phi/psi_T by walker weights or forward walking): Test W, 6x6, pointwise labels 20-70x noisier than one round's correction
(PLAN.md 'TEST W RESULT', pfaffian_w/w_result.json); closed for pointwise labels only. (ii) Walkers as the TRAINING ROWS, labels still exact from the ratio formula:
not tried for either copy. [conj, 55%] It changes the term my pushback named: the fit weight would be the walker residence weight, where E_FN responds, instead
of Born weight (sign proxy realised 0.30 of its first-order gain, PLAN_sign STEP-BACK 8; amplitude target weight 2e-5 'never drawn from |B|^2', PLAN.md task card).
To derive before any test: E_FN's first-order response to a copy error as a sum over the FN mixed distribution (Hellmann-Feynman, board 10: weights chi*^2, not phi^2),
and the ratio of that weight to the Born weight per decade, from the walker histogram already on disk (sign_bad/whist/hist.json, STEP 16: ratio ~1 per decade there).

## Third point: closure (3) REFUTED as written (FN-lead 11-26 20:45; amp lead's values from existing grade files, paths in delta_today)
E_FN[s2 A_k], alpha 0, 1e-6/site: A_2 3.320, A_3 2.156, A_4 1.636, A_5 1.315, A_6 1.117; chi_6 1.761; h1 2.327. Predicted E_FN[s2 A_6] 1.9-2.1: MISSED by 0.8.
Failing term: I measured the amplitude excess x from E_FN[s2 |psi0|] = 1.494e-6 as if it were the chamber floor c(s2). It is not: A_6 is already below it.
The floor is reached by wall-adapted amplitudes, not by |psi0|. Aitken on the chain (differences 1.164, 0.520, 0.321, 0.198; ratio 0.617 twice) gives
c(s2) ~ 0.80e-6 [extrapolated, unvalidated; record: s_G-type floor <= 0.83e-6, compress_6x6 0.9e-6].
With c(s2) 0.80: kappa = c'/x = 0.134 (A_0), 0.147 (A_2), 0.27 (A_6), 0.158 (chi_6), 0.077 (h1): not a constant, and h1 (higher E_FN than chi_6) gives the better sign.
So 'mis-flips proportional to the FN excess' is wrong; reading C's '15-20% surcharge' is WITHDRAWN. The step is the sign of the Lanczos factor, and the copies
order correctly by the Lanczos-graded N (h1 1.098 < chi_6 1.350; residuals 1.18e-7 < 1.52e-7): the right variable for the sign step is the quality of E_L on the
flip rows, not E_FN. [open, to derive: residual after the step as a function of the E_L error on rows with |1 + alpha(E_L - E)| small.]
What stands: (1), (2), (4) with the kappa term replaced by an unknown g(amplitude) >= 0 that is small at 6x6 (<= 1.5e-7 for every round-6 input), and readings A, B:
the sign copy tax enters once. What the miss adds: the fixed-s2 chain rests near 0.80e-6 whatever the copy does, so below that only a sign step helps.
The registered cycle prediction (N <= 0.8e-6 within two rounds at T) therefore tests exactly whether the step opens a lower chamber. Asked of the amp lead 11-26 20:45.
Walker rows, sign side (sign lead 11-26): not expected to change R. Walker/Born ~1 per decade (STEP 16), CF misses 45% of flips on its own training rows (input is
missing, not rows), energy-weighted rows gave -1.8(1.6)% (T-GROW O). My 55% -> 15% for the sign copy. Amplitude side still open (target weight 2e-5 under |B|^2).

## Sign-step residual law (paper + exact 6x6, 2026-11-26 23:00)
Files: `/project/theorie/a/A.Otaifi/runs/j1j2/consol_6x6/signdepth_paper/` (`table.py`, `pass2_M.json`, `pass2_s2.json`); numbers re-read by FN-lead.

(8) R_k = E_FN[T[s2,A_k]|psi0|] - E0 ~ R* + kappa D1_k,  D1_k = (1/N) sum_{x in F_k xor F*} e*_x,
    e*_x = -4 T*_x a_x sum_{y!=x} H_xy T*_y a_y / |a|^2  (a=|psi0|, T* = oracle step, F = flip set).
    Measured R vs D1 (1e-6/site): A_0 2.685/3.169, A_2 0.370/0.405, A_6 0.0865/0.0895, chi_6 0.152/0.162, h1 0.118/0.124; kappa 0.85-0.93 [measured, not derived]; Spearman +1.00 (6 inputs). Fails: input FN energy (+0.89), mu2 (+0.37), wall dip (+0.03).
    Meaning: the residual is the energy mass of rows where the E_L error crosses the flip threshold in the damaging direction; error size alone does not count.

(9) Guide S|psi0|rho, rho=r on wrong rows W, 1 elsewhere [exact, uses only H psi0 = E0 psi0]:
    wrong row:   E_L - E0 = (1+1/r) B_x,  B_x = -sum_{y notin W} H_xy psi0_y/psi0_x  -> fixed if B_x > theta r/(1+r)
    correct row: E_L - E0 = (1+r) C_x,    C_x = -sum_{y in W} H_xy psi0_y/psi0_x    -> broken if C_x > theta/(1+r)
    theta = 1/|alpha| + E - E0. Suppressing wrong rows lowers the bar to fix and raises the bar to break.
    Exact r-scan, E_var[T|psi0|]-E0: Marshall (1e-3/site) r=1 5.66, 0.3 0.845, 0.1 0.484, 0.03 0.397, 0.003 0.361; s2 (1e-9/site) 3.79, 0.79, 0.51, 0.48, 0.43.
    Floor of one step: wrong rows with B_x <= 0 (interior of wrong domains).
    Not checked: FN grade of the r-scan (only E_var), non-uniform r.

Consequence [conj 45%]: the amplitude handed to the sign step should be MORE wall-suppressed than the one that is best for the energy. Oracle-free knob: over-relax only the sign step's input, A_gamma = G exp(gamma f_J[G]), gamma > 1.307.
PREDICTION (registered before the run): at 6x6 exact, s2 start, residual E_FN[T[s2,A_gamma]|psi0|] has its minimum at gamma in [2,3]; from A_2 it falls from 0.370e-6 to <= 0.25e-6.
KILL: no gamma in {2,3,4} beats gamma=1.307 by more than 10% from either A_2 or A_6. If killed: the FN dip is not aligned with W (it suppresses correct rows near walls too, raising C-type breaks); the term to change is then the one-sided weighting, not the depth.

## First complete cycle: PASS at exact level (2026-11-26 23:40)
File: `/project/theorie/a/A.Otaifi/runs/j1j2/consol_6x6/cycle1/result.json` (md5 4af5e600), re-read by FN-lead. N at best alpha, 1e-6/site, rounds 0-3:
A (chi_6, T[s2,chi_6], wrong weight w=1.16e-7): 0.951, 0.603, 0.452, 0.373
B (chi_6, s2, w=1.23e-6):                        1.375, 1.098, 1.021, 0.958
C (s2 + random flips, w=2.57e-6):                2.161, 1.860, 1.729, 1.640
D (A_6, T[s2,A_6], w=6.6e-8):                    0.690, 0.452, 0.372, 0.311
Registered line (A <= 0.8e-6 within two rounds): met at round 1. Sub-prediction "A worse than B at alpha=0 in round 1" MISSED (0.807 vs 1.262): the step's sign needs no re-adaptation of the amplitude before it pays.
(10) Three arms from the same start: N_3 ~ 0.31e-6 + 0.52 w  (A 0.37/0.373, B 0.95/0.958, C 1.65/1.640). [3 points; B and C share s2, so only A is independent.] The resting energy after three rounds is linear in the sign's wrong weight w; w is the number the stored sign must deliver.

NEXT, registered before the run: same cycle with stored pieces, start chi_6, three rounds.
 S-amp: exact T, stored amplitude rounds. Prediction: N_3 <= 0.60e-6 (60%).
 S-sign: stored copy of T[s2,chi_6], exact amplitude rounds. By (10) it beats B only if its w < 1.23e-6, i.e. it keeps >= ~90% of the flip weight between s2 and T. Prediction: current CF copy does NOT (55%).
 S-both: pass = N_3 < 0.958e-6 (below the fixed-sign control). Kill for "stored complete loop at 6x6": S-both >= 0.958e-6 with the best available sign copy; diagnosis then goes through w and (8).

## Sign copy: per-bond remedy killed (sign lead STEP 20, commit cf72d68; 2026-11-27 00:00)
Train flip miss 0.24 -> 0.000 with plain BCE; held-out flip miss at chance (0.47-0.53 +-0.08, 36 roots) for all arms; held-out parts R^2 <= 0.1. The failure is generalisation on flip rows, not gradient. Literature remedy 1 withdrawn.
Open, registered before the data: learning curve at 6x6 (held-out flip miss and w vs number of train flip roots, exact enumeration). FN-lead guess: flat at chance (60%) -> only gate / ratio-input routes remain; falling -> targeted flip-row sampler at 8x8.

## Gamma scan RESULT (2026-11-27 00:30; `runs/j1j2/consol_6x6/gammascan/result.json`, md5 762926d4, known cases checked by FN-lead)
R = E_FN[T[s2,A_gamma]|psi0|]-E0, 1e-6/site, gamma = 0 / 1.307 / 2 / 3 / 4 / -2:
k=2: 0.418 / 0.370 / 0.342 / 0.298 / 0.262 / 0.469     k=6: 0.091 / 0.0865 / 0.0860 / 0.0863 / 0.0892 / 0.128
Kill not fired (k=2: -29% at gamma=4, still falling). Prediction "minimum in [2,3]" WRONG at k=2 (monotone to 4), "R(A_2) <= 0.25" missed (0.262). At k=6 flat: a converged fixed-node amplitude is already wall-adapted, so over-relaxation only speeds early rounds; it does not lower the floor.
r-scan under the FN grade: T(r=0.03) from Marshall 2.76e-4 (A_2 arm 5.74e-4), from s2 4.6e-10: eq (9) holds under FN, conjecture confirmed.
Status: side result, off the critical path (the block is the stored sign). Not pursued further now; cheapest follow-up if wanted: gamma 6, 8 at k=2 and gamma=4 used in round 1 of the cycle.

## 8x8: gated sign copy lowers the FN energy (sign lead STEP 18b; 2026-11-27 01:00)
`/project/theorie/a/A.Otaifi/runs/j1j2/sign_bad/branched/cfgate_vs_cf/result.json` (job 18215513), re-read by FN-lead: Delta(cfgate(2,-5) - CF) = -2.57(0.41)e-5/site, 12/12 pairs negative, error from pair scatter (pairs branch from one parent chain). Half-window means -2.40(58) and -2.73(62)e-5: no drift inside the measurement.
sigma 6.1(1.6) -> 3.5(1.7)e-5 [derived, assumes direct subtraction; the two errors are not independent of the shared CF baseline]. Cost 2.35x. Realized 0.31(5) of the first-order 8.22e-5.
Reading: the stored sign is still ABOVE sgn(M1) by 3.5(1.7)e-5, so at 8x8 the stored loop does not yet beat the baseline; (2,-6) is predicted to add only 0.2e-5 for +0.5x cost, so the gate family alone saturates near sigma/2. The missing factor has to come from STEP 22's answer (learnable with more flip rows, or ratio input).

## Step back (2026-11-27 01:30): two sign-copy routes closed (per-bond targets; ungated ratio input, t2/t1 = 57-71 vs bar 5.5, job 18230995)
- Tool that makes it minutes: the 6x6 exact flip learning curve (STEP 22, running).
- Who met this form: sign of a frustrated wavefunction learned from a subset, Westerhout et al. arXiv:1907.08186 [from memory, not re-read]: networks fit the training signs and fail to generalise near J2/J1 = 0.5 until the training fraction is large. Consistent with STEP 20.
- Switch of question, not of method: stop asking the copy to carry the step. Gate with sgn(M1) fallback: exact one-step sign on gated rows, sgn(M1) elsewhere, step computed with the fixed-node-adapted amplitude (eq 9), FN-graded against sgn(M1) itself.
  HYPOTHESIS [45%]: Delta(gate - M1) <= -1.0e-5/site with the adapted amplitude; zero or positive with |M1| (60%), which would explain the two earlier records that raised E_FN (G(-5), T-NODE).
  KILL: adapted arm Delta >= 0 within error. Sent to the sign lead with a records check first.

## 6x6 stored copy of one sign step meets the bar (sign lead STEP 21; 2026-11-27 03:20)
`/project/theorie/a/A.Otaifi/runs/j1j2/consol_6x6/signcopy/result.json`, read by FN-lead: seeds w = 6.61e-7, 7.00e-7; two-seed average 5.74e-7 [sign lead's grade; the average arm not re-read by me] vs bar 1.23e-6 (s2) and 1.16e-7 (exact T). Error mostly false flips (4.2e-7) not missed flips (0.9e-7), all in decades <= -11.
My prediction "the current copy does not beat s2 (55%)" MISSED.
Registered from eq (10) before the cycle arm runs: S-sign N_3 ~ 0.31 + 0.52*0.574 = 0.61e-6 (band 0.55-0.70), below the fixed-sign 0.958e-6.
Not shown: generalisation. The copy was fit with the full 6x6 table available; whether it transfers to 8x8, where only sampled rows exist, is STEP 22's curve.

## S-sign RESULT: eq (10) REFUTED (2026-11-27 11:30; `runs/j1j2/consol_6x6/cycle2_ssign/result.json`, md5 e11fb334, read by FN-lead)
Stored sign (w 5.74e-7), exact amplitude rounds: N = 1.366, 1.056, 0.971, 0.901e-6 (rounds 0-3). Predicted 0.61 (0.55-0.70): missed by +0.29e-6. Passes the 0.958 bar by 6% only; tracks the fixed-sign arm B (ratio 0.94-0.99), best alpha -> 0 from round 2.
Diagnosis (equation): eq (10) used the psi0^2 weight of wrong rows. Eq (8) already said the cost of a wrong row is e*_x = -4 T*_x a_x sum_y H_xy T*_y a_y / |a|^2, linear in a_x times its neighbours, not a_x^2. The copy's false flips sit in decades <= -11: negligible in psi0^2, not in e*. (10) had one independent point and I relied on it; the term to change is the weight in the copy's loss and in the bar: energy cost e*_x, not psi0^2.
Amp lead's registered check (xpsi0 run): x = E_FN[S|psi0|]-E0 as the variable, predicted x(avg2) = 1.36e-6. Mine: agree (65%); if x(avg2) ~ 0.7 the cause is in the amplitude adaptation instead.

## Cost of one wrong row: paper step before the xpsi0 data (2026-11-27 12:00)
Wrong sign on row x turns its bonds (x,y) into sign-violating ones. Standard fixed-node H: the off-diagonal -|H_xy| is removed and V_sf(x) = sum_y |H_xy| a_y/a_x, V_sf(y) += |H_xy| a_x/a_y are added (guide amplitude a).
At fixed amplitude: dE = sum_y [2|H_xy| a_x a_y (lost bonding) + a_x^2 |H_xy| a_y/a_x + a_y^2 |H_xy| a_x/a_y] = 4 sum_y |H_xy| a_x a_y = e*_x (D1).
After the FN state relaxes its weight on x to 0: the x-diagonal term drops, leaving ~3 sum_y |H_xy| a_x a_y. Still linear in a_x.
So with this H_FN I do not get the amp lead's D2 = sum_y |H_xy|(a_x+a_y)^2 or D3 = sum_y |H_xy| a_y^2 (their paper step, STORAGE S-sign block, c120436); D3 would need a penalty at y that does not vanish with a_x.
PREDICTION before the data: x tracks D1 with slope 0.75-0.95, not D3 (55%). If D1(avg2) is far below the measured x, either the code's H_FN differs from the form above or the loss is in the amplitude adaptation; then that is the next thing to read, not a new metric.

## 6x6 learning curve of a stored sign step (sign lead STEP 22; 2026-11-27 12:30)
`/scratch/a/A.Otaifi/ansatz_runs/consol_6x6/flipcurve/curve.md`, read by FN-lead. Held flip miss by weight: 0.934 (n=100), 0.064 (300), 0.038 (1k), 0.025 (10k), 0.022 (100k), 0.017 (all); shuffled control 1.000 at every n. Held false flips by weight stay at 0.32-0.70 and the deployed table has w = 0.02-0.2 (s2: 1.2e-6): the net flips heavy rows it never saw (keeps capped at 20 per flip, none in decades -9..-4).
My guess "(b) flat at chance, 60%" MISSED for the flips that carry weight. Reading: configuration-only learning of the flips works with >= 10k flip roots; the failure is the law of the training rows. 8x8 had 130 flip roots.
NEXT (sign lead): 6x6 with keeps from the Born law over all decades plus flip rows; pass = deployed w < 1.23e-6 with the threshold set on training rows. Then the same law at 8x8 with >= 10k flip roots from a targeted walk, FN-graded. This is the PI's "train on walker rows" idea plus the flip rows.

## Reading: Kairon & Clark arXiv:2609.16308 (read in text by FN-lead, 2026-11-27) and what it changes
Their map: psi_{k+1} = T(psi_k) = ground state of the lattice fixed-node Hamiltonian F(psi_k). Our amplitude round is an approximate T.
Proved (exact T, finite space, unique full-support ground state): (i) E_{k+1} <= E_FN,k <= E_k; (ii) fixed points are eigenstates of H or of a principal submatrix H_SS; (iii) in the ground-state sign chamber every wall is repulsive and the iteration converges to the exact ground state (Thm 1, needs irreducible F and uniformly interior iterates); (iv) in a wrong chamber the iteration drives selected amplitudes to zero ("support collapse"); (v) a one-component wall that attracts from one side repels from the other (kappa_sigma kappa_-sigma = 1).
Numerics there (L<=14 chains, 8-site strip): 48 of 49 collapsed configurations carry the wrong sign relative to the ground state.
What it explains of ours: fixed-s2 chain descends and stalls (i, iv); wall dips deepen with rounds (iv); a sign step works with the adapted amplitude because the wrong rows are the collapsed ones (iv -> our eq 9 with r -> 0); "amplitude round = detector, sign step = actuator" is (iv)+(v).
What it gives the PI's question "inside the right chamber, do we flip right configurations?": by (iii) nothing collapses there, so a collapse-triggered sign update is silent in the right chamber. Guarantee holds for exact T only.
NEW HYPOTHESIS [55%]: the gate needs no network. g(x) = 1 where the amplitude round collapses the amplitude, f_J(x) = ln A_{k+1}(x)/A_k(x) < -c; sign on gated rows from the exact step (or simply flipped, by (v)).
PREDICTION before data (6x6, existing tables: s2, psi0, A_0..A_6, f_J): ranking configurations by accumulated collapse ln A_6/A_0, the e*-weighted AUC for wrong-vs-right sign of s2 is > 0.9 (60%); under T[s2,chi_6] (cycle1 arm A) the collapsed e*-mass is at least 3x smaller than under s2 (60%).
KILL: AUC < 0.7: one-shot approximate rounds do not show the collapse cleanly; then the detector needs more rounds or the exact step's E_L, not f_J.
Not covered by the paper: sampled walkers, approximate projection, stored networks (they state this as open). Wang-He-Lu arXiv:2502.01264 (read in text): plain cross-entropy on |psi_trg|^2 samples, sign accuracy 91.2% at 10x10 p=1; no weighting, no oracle; nothing to adopt beyond the warning that later steps have noisier targets.

## Collapse check RESULT (amp lead, job 18238556, `runs/j1j2/consol_6x6/collapse/result.json` md5 8bf06323; block read by FN-lead in STORAGE_PROBLEM l.1814, result file not re-opened)
One round's f_J[G_6] separates s2-wrong from s2-right rows: AUC 0.99993 (psi0^2 weight), 0.994 (neighbour weight D3); after one early round f_J[G_2] 0.992 / 0.951. Shuffle within decades -> 0.42-0.55: the signal is not "small amplitude". Gate at the cut equal to the wrong weight: precision = recall = 0.893 (psi0^2), 0.645 (D3).
My registered metric was flawed: e*_x is negative on wrong rows, so the e*-weighted AUC is void; on the meaningful weights the line passes and the kill (0.7) is far. Mass ratio s2/T at matched depth: 8.7 (psi0^2), 1.4 (D3): the ">= 3x" claim holds on one weight only.
In-loop test registered by the amp lead (their block "GATE SIGN IN THE CYCLE"): sign = s2 flipped where f_J < -c, no network, no exact step; arms G6 (upper bound) and Gchi (what the stored loop has); kill Gchi N_3 >= 0.90e-6; objective margin N_3 <= 0.665e-6.

## Collapse gate in the loop: RESULT (2026-11-27 14:10)

Source: `runs/j1j2/consol_6x6/cycle3_gate/result.json` md5 75d0899d (jobs 18240405, 18240752, 18240753), read by me.
N_3 (1e-6/site): G6 (gate from the exact chain's f_J) 0.534, margin 0.665 MET, in my band; Gchi (gate from one
round on the stored amplitude) 1.736, kill (>= 0.90) FIRES. References: A 0.373, B 0.958, S 0.901.

Diagnosis (before any verdict on the route). The gate is the cut f_J < -c with c = 0.0088. On the stored amplitude
f_J = (node suppression) + (correction of the copy's own amplitude error). The second term has unweighted spread
0.016 on right-sign rows (0.004 on the exact chain), larger than c. So the cut flips 1.43e6 rows instead of 4.8e5:
recall 0.89 but precision 0.35 by psi0^2 weight, and the sign's wrong weight rises to 2.20e-6 (s2: 1.23e-6).
Failing term: the amplitude-copy error inside f_J, compared against a 0.9% threshold.

The same file holds a rule that does not have this term: R1, flip where the sign-violating neighbour sum exceeds
the sign-keeping one (VA > SK), an O(1) comparison. On the stored amplitude: precision 0.991, recall 0.927
(psi0^2), 0.903 / 0.602 (D3); on the exact chain 0.988 / 0.938. Its sign would have wrong weight about 1.0e-7 [derived:
1.2265e-6 x 0.073 + 1.147e-6 x 0.009], below the exact step's 1.16e-7.

Prediction (registered before the run): fixed sign = s2 flipped by R1 on the stored amplitude, three rounds:
N_3 in 0.42-0.55e-6 (50%). Kill: N_3 >= 0.90e-6. Margin: <= 0.665e-6. Not measured: whether regating every round
helps, and 8x8 (STEP 23 carries R1 there).

## When a sign flip provably lowers E_FN (2026-11-27 14:50, paper step, not yet checked numerically)

Amp lead's point: R1 lowers E_var[S'a], not the ten Haaf E_FN. Sandwich: E_FN[S,a] <= E_var[Sa], equality at the
amplitude fixed point (a = FN ground state of its own H_FN, since <a|H_FN|a> = <a|H|a>). With slack
g = E_var[Sa] - E_FN[S,a] >= 0: E_FN[S',a] <= E_FN[S,a] + g + Delta E_var. A flip set is certified iff
-Delta E_var > g; at g = 0, single-row R1 (B > 0) is the provable rule. Direction only, not size: on a collapsed
row Delta E_var ~ a_x^2 B -> 0. Our three-round arms have g > 0 and sit outside the guarantee.
Prediction for the check on existing tables: arms with -Delta E_var > g end below B; Gchi has Delta E_var > 0.

## Where the loop can stop (2026-11-27 15:20, paper step, exact loop only, not checked numerically)

1. Ten Haaf: H_FN and H have the same local energy on the guide. So at an amplitude fixed point (guide psi = S a
   is the ground state of its own H_FN, energy E): (H psi)_x = E psi_x on every row with psi_x != 0.
2. Hence a fixed point with full support is an exact eigenstate of H. Every other fixed point has collapsed rows
   Z (psi = 0 there): psi is an eigenvector of H restricted to the surviving rows. This is where fixed node alone
   stops, for any imperfect sign (matches Kairon-Clark's support collapse).
3. Reseed a collapsed row x with small amplitude eps and sign sigma: E_var changes by 2 eps sigma (H psi)_x / <psi|psi>
   + O(eps^2). sigma = -sgn (H psi)_x lowers it whenever (H psi)_x != 0; this is R1 in the collapsed limit
   (R1 = give the row the sign its neighbours' sum asks for). By the sandwich E_FN then drops strictly.
4. So the loop "amplitude to fixed point, then R1 on collapsed rows" is stationary only where H psi = E psi on all
   rows: eigenstates of H. Energy is monotone, so from a start below the first excited level of the sector the
   only stopping point is the ground state.
Gaps: (i) exact loop, one certified move at a time; (ii) no rate: the gain per reseed is first order in eps, and the
approach to collapse is only asymptotic, so "stuck" is replaced by "slow"; (iii) with stored networks the copy
error sets a floor the argument does not see; (iv) a monotone sequence converging to a non-stationary point is not
excluded without a continuity argument.
Timing: the certified gain of flipping a wrong row is 4 a_x^2 B(x) and a_x decays every round, so waiting loses
gain; flip as soon as the flag is reliable, every round. Prediction: the arm regated every round ends at or below
the fixed-R1 arm (65%).

## R1 gate in the loop: RESULT (2026-11-27 19:00)

Source: `runs/j1j2/consol_6x6/cycle5_r1/armR1/result_arm.json` md5 55732b2f (job 18244986), read by me. Sign = s2
flipped once by R1 computed from the stored amplitude table (359,588 rows flipped, wrong weight 9.98e-8); no sign
network, no exact-chain input; amplitude rounds exact.
N rounds 0-3 (1e-6/site): 0.946 / 0.594 / 0.448 / 0.368. Exact sign step (arm A): 0.951 / 0.603 / 0.452 / 0.373.
B (no sign step): 0.958 at round 3. Margin 0.665 met; my band 0.42-0.55 missed on the good side (amp lead's
0.38-0.52 also).
Sandwich check at round 0 [derived from the same file]: Delta E_var = 1.664 - 1.995 = -0.331; slack g of s2 =
1.995 - 1.761 = 0.234; -Delta E_var > g, so the flip was certified, and E_FN(alpha=0) fell 1.761 -> 1.419.
Not covered: the gate was evaluated on all rows by enumeration (at 8x8 it is evaluated on walker rows and costs
neighbour evaluations); amplitude rounds were exact, not stored; regated arm still running.

## Stopping indicators over three rounds: prediction missed (2026-11-27 19:25; amp lead's record, not re-read by me)

My prediction (variance and R1-flag weight fall for A and G6, stall for B and Gchi) fails over rounds 0-3: both
fall in every arm (amplitude error dominates and falls under any sign), and in A the variance rises while E_FN
falls. B and S plateau only in rounds 3-6. Open reading [amp lead, 55%]: the plateau height marks a sign's floor;
missing control: a right-sign arm to round 6. Sandwich on existing arms: every certified arm (A, R1) ends below B;
S and G6 are uncertified yet also end below B, so the bound is sufficient, not necessary (as derived).

## R1 regated every round: RESULT (2026-11-27 20:10)

Source: `runs/j1j2/consol_6x6/cycle5_r1/armR1r/result_arm.json` md5 37915cb4 (job 18244987), read by me.
N rounds 0-3 (1e-6/site): 0.946 / 0.594 / 0.441 / 0.359 (fixed R1: 0.368; exact step A: 0.373). The sign's wrong
weight falls round by round: 9.98e-8 -> 9.98e-8 -> 6.01e-8 -> 2.95e-8. Prediction (regated <= fixed, 65%) holds;
the gain in N is small (2.5%) but the sign itself improves 3.4x, as the paper step (flip every round) says.
Exact amplitude rounds; gate on all rows.

## 8x8 one-step sign on gated rows: KILL (2026-11-27 21:20)

Source: `runs/j1j2/sign_bad/branched/step23/B_R1/result.json` md5 a690acb7 (job 18237615), read by me: Delta vs
sgn(M1) = +1.05(0.55)e-6/site, 12 pairs; sign lead's B_M1 +0.59(0.84)e-6, broken control +22(12)e-6 (not re-read).
My registered kill (adapted arm >= 0) fires; my 45% for a gain of 1e-5 was wrong. The two arms' walkers coincide
99.9% of the time, so the gated rows are barely visited.
Hypotheses, not verdicts: (a) 60% the 8x8 network refit loses the suppression signal on low-weight rows (copy
error above signal, the term that killed the f_J gate at 6x6); (b) 25% sgn(M1) is already R1-consistent with its
own amplitude (VMC removes single flips with B > 0), while 6x6 started from s2, which is not; (c) the gated flip
set has -Delta E_var << g, so no test could see it. Decision: sign folded into the stored loop (R1 on each round's
stored amplitude), sign-copy route closed on the sign lead's evidence (STEP 22, 24, CF, tfield); 8x8 loop spend
waits for 6x6 arm C's first round.

## 6x6 loop with nothing exact inside: RESULT (2026-11-28 05:00)

Source: `runs/j1j2/consol_6x6/cycle6f_store/result.json` md5 f5d44545, read by me. Fixed R1 sign, amplitude copied
into the network each round. N rounds 0-3 (1e-6/site): 0.946 / 0.820 / 0.801 / 0.774; exact round from each stored
start: 0.594 / 0.546 / 0.539. Copy loss per round L = 0.226 / 0.256 / 0.235, flat. Between pass (0.665) and kill
(0.90); my band 0.50-0.70 missed. The same copy under s2 lost 0.08 (amp lead's record).
8x8 (sign lead's records, not re-read): the refit amplitude reproduces the round on gated rows with R^2 0.25
(exact 6x6 round ~1); my hypothesis (b) refuted (R1-consistent start still keeps 0.55 of the gain).
Reading: the sign rule is settled at 6x6; the failing term at both sizes is the amplitude copy.
Hypothesis (55%), registered before the splice test: flipped rows and their one-exchange neighbours carry >= 50%
of L, because the round must regrow rows that collapsed under s2 (large positive f_J on low-weight rows). Remedy if
so: compute the regrowth on flagged rows by one local power step, psi_x = -(sum_y H_xy psi_y)/(H_xx - E), instead
of learning it. Kill: loss diffuse (flipped rows < 50% of L).

## Where the copy loses: my hypothesis refuted (2026-11-28; amp lead's record c6_where/result.json 38a54ec1, not re-read)

Flipped rows plus one-exchange neighbours carry 43% of L_0 (bar >= 50%), flipped rows alone 0.4%; graded under s2
the same copy loses 0.187, so the sign does not drive the loss. The regrowth is real (log miss 0.31) but carries
negligible weight; the local power step would recover < 1% of L and is dropped. Failing term: fit error everywhere,
4.3x the earlier fit at equal steps. Next (amp lead): round-0 refit at width 128 and at 3x steps; pass L_0 <= 0.12.
