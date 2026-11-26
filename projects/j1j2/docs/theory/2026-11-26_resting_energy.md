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
