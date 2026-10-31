# Two label-free loop rounds beat one Lanczos step at 6x6 (same sign)

All energies are exact fixed-node values (6x6, J2 = 0.5), per site above E0. No Monte Carlo.

- **Result:** the augmented guide B_2 e^{c f_J[B_2]}, refit for two label-free rounds under the competitor's signed-Lanczos sign s_G, reaches
  5.207e-6 (seed 1) and 5.290e-6 (seed 2). Network + one signed Lanczos step + FN (competitor, same sign) reaches 5.574e-6.
  Margin 0.28-0.37e-6; seed spread 0.083e-6 (pfaffian_r2g/readout/g{1,2}/readout.json).
- **Path:** round 1 under s1 (L-SWO), 1.018e-5 -> 8.34e-6; switching the Jacobi factor to s_G's walls gives 6.07e-6 untrained
  (pfaffian_lswo2/prep_result.json); round 2 under s_G gives 5.2-5.3e-6. Bare network per round: drop 1.3-1.6e-6.
- **Why it works:** label-free refit to the network's own Jacobi-corrected state stores more than an exact-label fit (R1b 15% vs
  L-SWO 37% of a round's gain); the correction itself is a direction the PP cannot absorb (Test Theta v2, held-out 0-4%),
  so the Jacobi factor stays an explicit layer and the base only refits the bulk.
- **Sign and amplitude interact** (Test X): s_G helps the loop's magnitude by 0.95e-6 and the competitor's by 4.45e-6.
- **Not shown:**
  - equal cost (a Jacobi level costs ~5.4x a plain PP per walker step, the same two-hop class as one Lanczos step;
    training 0.58 H100-h + ~3.5 H100-h + 7.6 A40-h vs the competitor's VMC 14.3 GPU-h);
  - deployable sign (s_G is computed, not stored; sign angle);
  - sampled fits (exact tables used);
  - sizes > 6x6;
  - seeds independent only in the optimiser path.
- **Next:** round 3 (A40) tests whether the gain per round holds; then the scale test.
