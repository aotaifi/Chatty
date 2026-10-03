# K1 vs true K2: amplitude-distortion mechanism — 2026-09-30

## Question
Why does K1 give an excellent 6x6 node while the direct continuation
`psi2=(T2-H)(T1-H)psi0` worsens the fixed-amplitude energy?

## Existing decisive facts
- On the 256 physical 6x6 sample K1 matches the hidden ViT signs exactly.
- True K2 flips only 0.78–1.95% of configurations, and those flips raise the energy by +0.646 to +0.960 total.
- The bad K2 flips are not near the K1 threshold: their K1 margins are typically ~1.6–2.3x the sample median.
- They do not share one simple NN/NNN flippable-bond motif.

## Mechanism test
True K2 uses the full first Krylov vector, whose magnitude is
`|psi1(x)| = |T1-r0(x)| a0(x)`.
Thus the second local-energy coordinate inherits an amplitude multiplier `q(x)=|T1-r0(x)|`.
On physical samples, `q` has CV ~0.18 and estimated magnitude overlap `F(a0,|psi1|)~0.9843`.
The true-K2 coordinate correlates strongly with this multiplier: corr(log q, r1_true)=+0.84 on the targeted 32-state audit.

Restore the original ViT amplitudes while keeping the K1 signs and recompute the second coordinate.
Then corr(log q, r1_projected)=-0.34.
For the five configurations flipped by the loosest true-K2 test, three move from the upper tail to the bottom/middle of the restored-amplitude control distribution; two remain upper-tail anomalies.

## Refined verdict
Amplitude distortion is a major part of the true-K2 failure, but not the whole mechanism. Restoring the original amplitudes removes 3/5 directly observed bad K2 flips; the remaining two are extreme local-energy outliers of the original ViT guide.

A later oracle audit shows that `r1` still contains useful residual sign information only in an ultra-rare tail (~3e-4--6e-4 physical mass). The ordinary unsupervised K2 thresholds flip ~1e-2 mass and therefore operate at high recall but only ~4--8% precision. The complete mechanism is documented in `WHY_ONE_KRYLOV_STEP_VERDICT_2026-09-30.md`.

## Tail generalization test
The tiny residual K2 signal is real but extremely sparse. Choosing an r1 upper-tail cut on train generalizes to validation only for physical-weight tails of about 1e-4 to 7e-4.

Examples:
- train tail 3e-4: overlap 0.9983837 -> 0.9989029; same threshold on validation gives 0.9983968 -> 0.9988927.
- train tail 5e-4: overlap 0.9983837 -> 0.9987768; same threshold on validation gives 0.9983968 -> 0.9989629.
- by tail 1e-3 the gain is marginal/unstable; at 2e-3 and above the update is clearly harmful.

Thus K1 removes almost all sign-defect signal. r1 retains a reproducible residual only in an extreme ~1e-4–1e-3 physical-weight tail. Generic unsupervised two-cluster thresholds target ~1e-2 mass and therefore overshoot by 1–2 orders of magnitude.

Updated mechanism verdict: K1 is a strong sign-denoising step. True K2 is dominated by the amplitude distortion |T1-r0| and only contains a tiny residual sign signal in an extreme tail. Any scalable continuation would need a rare-defect detector / confidence gate, not another global two-cluster threshold.
