# One-step Krylov / Marshall-local-energy sign rule — 2026-09-27

## Core observation
Let psi_M(x)=a_ViT(x) s_M(x), where a_ViT is the public 6x6 J1-J2 ViT amplitude and s_M is the Marshall sign. Define the Marshall local energy

r_M(x) = (H psi_M)(x) / psi_M(x).

A one-step Krylov wavefunction (t-H) psi_M has correction sign sign[t-r_M(x)]. Therefore the sign model

s_t(x) = s_M(x) sign[t-r_M(x)]

is globally defined, boundary-free, and costs only one Hamiltonian-neighbor evaluation per configuration.

## Oracle capacity diagnostic
On the fixed alpha=0.8 train/validation samples, the hidden ViT correction states have strongly separated r_M distributions:
- hidden Marshall-sector mean r_M ~ -18.86
- hidden corrected-sector mean r_M ~ -11.90

A threshold fitted using hidden labels only as a diagnostic gives t ~ -15.71 and:
- validation correction accuracy: 95.90%
- physical importance-weighted hidden-sign overlap: 0.99843
- predicted corrected fraction: ~30.5%

This shows a single Krylov/local-energy scalar almost completely captures the hidden 6x6 sign correction.

## Label-free threshold
Crucially, t can be extracted without hidden signs. Two-means clustering of the TRAIN r_M distribution alone gives centers roughly (-18 to -19) and (-10 to -11), with midpoint t ~ -14.5. Applying this threshold unchanged to the independent validation set gives:
- validation accuracy: 96.48%
- physical weighted overlap: 0.998379
- predicted corrected fraction: 28.71%

Otsu thresholding on the same unlabeled TRAIN r_M distribution gives essentially the same result. A robust Otsu trim (10%-90%) gives t=-14.929 and validation accuracy 96.68%, weighted overlap 0.998569.

No hidden sign is used to construct r_M or choose the unsupervised threshold. Hidden ViT signs are used only afterward to score the benchmark.

## Interpretation
This resolves the collective-barrier problem found by local-field descent. Marshall is stable to almost all single hard flips, but a finite Krylov step crosses many coefficients through zero simultaneously. The correction is therefore collective in hard-sign space but one-dimensional in the local-energy/Krylov coordinate.

The rule is compact and polynomial: evaluating r_M requires all Hamiltonian neighbors of x, O(N^2) for the J1-J2 configuration graph, and thresholding is O(1). It has no finite configuration-space boundary and no learned high-capacity classifier.

## 8x8 falsification — PASSED
The identical label-free procedure was applied to the public 8x8 J2/J1=0.5 ViT checkpoint (302,540 parameters), using 1024 training and 512 independent validation configurations sampled with alpha=0.8. Each configuration has about 141 connected J1/J2 exchange neighbors. Hidden ViT signs were used only after threshold selection for diagnostics.

Marshall alone on validation has weighted hidden-sign overlap 0.972927 and about 34% of validation configurations are non-Marshall unweighted.

Unsupervised thresholds inferred only from the TRAIN r_M distribution transfer directly to validation. Representative results:
- two-means (10%-90% robust trim): t=-28.0346, validation accuracy 91.21%, weighted overlap 0.999003;
- Otsu (5%-95% trim): t=-28.1363, validation accuracy 91.41%, weighted overlap 0.999234;
- Otsu (10%-90% trim): t=-28.3832, validation accuracy 91.21%, weighted overlap 0.999224.

The hidden-label oracle threshold is not better: its validation weighted overlap is 0.997709. Thus the label-free one-dimensional Krylov/local-energy coordinate generalizes from 6x6 to 8x8 and actually outperforms the train-fitted oracle threshold on this finite validation sample.

Important caveat: alpha=0.8 importance weighting is substantially noisier at 8x8 than at 6x6. ESS is ~23.6/1024 on train and ~22.0/512 on validation, so the ~0.999 weighted-overlap estimate needs a larger-sample / alpha sensitivity check before being treated as a precise physical-probability number. The unweighted validation accuracy (~91%) and stable threshold across several unlabeled clustering rules are independent supporting evidence.

## Current interpretation
The local free-boundary cluster route is no longer the leading constructive mechanism. A single global, boundary-free Krylov coordinate
`s_t(x)=s_M(x) sign[t-r_M(x)]`
recovers almost all physically weighted hidden ViT sign structure on both 6x6 and 8x8 without sign labels. Evaluation requires one Hamiltonian-neighbor sum per sampled configuration, polynomial in system size. This is now the strongest constructive sign-reconstruction result in the project.

## Next falsification
Before feeding this into fixed node, establish scaling rather than two-size coincidence: (1) increase 8x8 sample size and repeat for alpha=0.8,1.0,1.2 to control importance-weight ESS; (2) test any available larger checkpoint with the identical label-free rule; (3) measure physical fixed-amplitude energy of the reconstructed sign rule versus Marshall and hidden-ViT signs. Then test the rule as the guide-sign update in the fixed-node loop.
