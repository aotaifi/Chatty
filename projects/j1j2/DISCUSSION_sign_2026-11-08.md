# Sign discussion with the PI, 2026-11-08 (open points; nothing here is a result)

Status key: OPEN = to discuss; RUNNING = test launched; PARKED = noted for later.

## Facts the discussion rests on
- 8x8 stored sign: best loop score (decades8 eps^2) is the old network's own sign, 0.40; trained signs 0.54-0.72 (bar 0.3). (PLAN_sign "GATE (a)")
- 6x6 exact: one final Lanczos step at c = 0 beats the competitor (5.57e-6): s_G|B_6| 2.65e-6, sgn(psi_P)|B_6| 3.71e-6; fixed own sign + Jacobi head 7.20e-6. (T-B2P1, A47)
- At s_G the step's 7.06 -> 2.65e-6 gain is almost all amplitude (sign change on 1.2e-6 of the weight).
- 6x6: the first sign step carries almost all sign gain; later steps < 0.5e-6 (A41). 8x8 first step weight 28x the 6x6 one.
- Pfaffian angle: under the net's own sign, stranded configurations (f_J < -0.5, few kept bonds) are ~100x rarer than under s_G.

## Points
P1 (OPEN) Double work. FN = projection to convergence inside the node; Lanczos = the same projection, a few steps, allowed to cross nodes.
   We keep only the sign of the Lanczos step and throw away its amplitude gain, which FN then rebuilds. Revisit how the two are combined.
P2 (RUNNING, T-GATE) Keep the old sign where the stored network is unsure, store the new sign elsewhere. Test on existing checkpoints, no training.
P3 (OPEN) Why rare configurations matter: FN and the labels use ratios psi(y)/psi(x), not psi^2; the gains chased are ~1e-6/site;
   eps^2 -> energy at 8x8 not measured.
P4 (OPEN) Is Lanczos the only sign update? Alternatives: energy-based sign optimisation (Marshall trap at J2 = 0.5); node update from the FN state
   itself (self-healing DMC, Reboredo et al. 2009; our 6x6 V > K rule; Astra's cut certificate), which does not nest neighbour shells.
P5 (OPEN, PI) Loop "stored net -> Lanczos -> better amplitude AND sign -> FN with Jacobi -> store -> Lanczos again".
   Angle lead: this is storing psi_1 = (1 + alpha(H - E)) psi as ONE object (sign and amplitude together) = T-JOINT, and it keeps the step's
   amplitude gain (P1). Connects P1, P4, T-JOINT and the own-sign observation.
P6 (OPEN, PI) The amplitude side succeeded with a label-free target A e^{c f_J}, the Jacobi head computed at walker time from neighbours.
   Angle lead: the transferable part is "store the smooth part, compute the hard near-node part exactly at walker time". For the sign: stored (or
   old) sign in the bulk, exact one-step sign s_G computed on the fly only at stranded configurations (small SK), where it matters.
   Cost = (fraction of stranded evaluations) x 2N per evaluation; the stranded fraction is small under the net's own sign. Hypothesis, untested.
P7 (OPEN, PI) Make the ratio explosion numerically stable (logs/exponentials instead of ratios).
   Angle lead: the code already works in logs (log|B|, f_J = ln SK - ln den); the large values are physical, not round-off. Logs avoid overflow but
   not the size of the effect. What can change it is a modelling choice: the f0 cap (already used), or computing those terms exactly (P6).
P8 (PARKED) T-JOINT, one complex network for sign and amplitude: Astra 45%, own-sign observation raises it to ~50%; PI decides.

## Order proposed
P2 runs now (cheap). Then P1 + P5 + P6 together on paper (they are one question: what to store and what to compute at walker time), then P4.
