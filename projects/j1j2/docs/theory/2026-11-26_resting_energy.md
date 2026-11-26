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
