# Global sign-model sampling exponent diagnostic — 2026-09-27

Goal: choose q_alpha(x) proportional to |a_ViT(x)|^alpha for boundary-free global sign learning while estimating the physical fixed-amplitude Rayleigh quotient by exact importance weights w(x) proportional to |a(x)|^(2-alpha).

Small diagnostic: 320 samples per alpha, 6x6 public ViT checkpoint.

| alpha | acceptance | ESS / 320 | ESS fraction | unweighted non-Marshall fraction |
|---:|---:|---:|---:|---:|
| 0.1 | 0.905 | 2.87 | 0.009 | 0.559 |
| 0.3 | 0.752 | 7.02 | 0.022 | 0.450 |
| 0.5 | 0.607 | 8.73 | 0.027 | 0.363 |
| 0.8 | 0.417 | 57.96 | 0.181 | 0.247 |
| 1.0 | 0.354 | 87.42 | 0.273 | 0.169 |
| 1.5 | 0.213 | 219.47 | 0.686 | 0.053 |

Interpretation: alpha=0.1 is useful for sign discovery but unusable for direct physical-energy optimization because importance weights collapse. alpha=0.8 is the chosen first compromise: roughly one quarter of sampled states expose non-Marshall structure while retaining ~18% effective sample size. alpha=1.0 is the fallback if weighted-gradient variance remains too high.

Next experiment: translation-invariant periodic CNN correction model, exact hard ±1 forward signs with straight-through gradients, trained on alpha=0.8 samples using the importance-weighted physical Rayleigh quotient. Hidden ViT signs are held out for diagnostics only.
