# Deliverable C: costed plan for 8x8 (rest), 10x10, 12x12, 14x14 (Paderborn H100 drafts, NOT submitted)

Cost model (all numbers are evaluations of the guide amplitude; measured where marked "meas."):
- One hop at L x L: n1 ~ 0.57 x 4N valid exchanges (8x8 meas. 143-145). Two hops n2 ~ n1^2/2 (8x8 meas. 1.0e4),
  three hops n3 (8x8 meas. 4.4e5). Estimates: 10x10 n1 ~ 230, n2 ~ 2.6e4; 12x12 n1 ~ 330, n2 ~ 5.4e4; 14x14 n1 ~ 450,
  n2 ~ 1.0e5.
- Projected ViT (64 network passes) throughput: 8x8 A40 slice 5.2e3/s (meas., results/loop_8x8/probe.json); full A40 5.2e3/s,
  RTX 2080 Ti 4.6e3/s, H100 NVL 2.36e4/s (meas., `probes/`; Paderborn H100 SXM expected similar or faster). Per-pass cost scales ~ tokens x d^2: 10x10 HF ViT (25 tokens, d = 72) ~2.3x the
  8x8 net, 12x12 (36 tokens) ~3.2x, 14x14 (49 tokens) ~4.4x.
- Direct VMC of psi_1 per sample: n2 (exact two-hop local energy) + delayed-acceptance moves (~0.3 N n1 per 2 sweeps).
- FN-DMC per population (M = 512, beta 3.2): M x (beta / tau) x p_hop x (n1 + 1) evaluations; 8x8: tau ~ 0.025,
  p_hop ~ 0.55 (estimate; measured in this round, see README).
- Statistics: per-site sd of E_L is heavy-tailed (8x8 psi_1: 1.24e-3, top 1% of samples carry 69% of the variance);
  per-site sd ~ 1/sqrt(N) at fixed state quality. Samples for SE s: n ~ (sd / s)^2.

## 8x8 (rest of the deliverable)
| job | what | cost estimate | draft |
|---|---|---|---|
| p = 2 production | mixture sampling psi_P^2 (f^2 + g^2), exact three-hop E_L of psi_1 and psi_2, paired E_2 - E_1, var_2 | n3 = 4.4e5 per sample (+ 1.4e5 sampling); pilot sd of the paired difference 6.9e-4/site -> SE 5e-6 needs ~1.9e4 samples = 1.1e10 evaluations = **~130 H100-h**; SE 1e-5: ~33 H100-h | `cluster/paderborn/deliverable_C/p2_prod_h100.sbatch` (b1 = 2 alpha, b2 = alpha^2, amendment 3) |
| FN-DMC, psi_1 guide | best-bound candidate: at 6x6 FN(psi_1) removed 26% of psi_1's error | psi_P run: 6.5e6 evaluations per population; psi_1 guide x146 -> 9.5e8 per population = **~11 H100-h per population**, 8 populations (SE ~1e-5) ~90 H100-h; expected gain only ~6e-6 below psi_1 -> low priority | `fn_psi1_h100.sbatch` |

## 10x10 (public checkpoint nqs-models/j1j2_square_10x10_05)
- Status: checkpoint not on any of our machines; download needs the user's OK (3.49 MB weights + 13.1 MB `spins` + small
  python/config files; Apache-2.0). Revision symm_trxy_ising = sum of amplitudes over 32 site permutations
  (symm_indexes.txt) x spin flip = 64 passes, the same construction as our psi_P; the network is the same ViT family as
  our vit_dt.py (L_eff = 25, d_model = 72, heads = 12, 8 layers, b = 2). float64 on the card; we run fp32 with full-precision
  matmuls (to be checked against the card energy in the reproduction gate).
- Steps: (1) reproduction gate, VMC of all three revisions (~1 GPU-h on ws1); (2) alpha from two-hop moments on ~6000 psi^2
  samples (6000 x 2.6e4 = 1.6e8 evaluations); (3) direct VMC of psi_1 with fixed alpha, ~1e4 samples (3.3e8 evaluations).
  Record claim only below -0.4976921(4) with SE <= 3e-6 (needs ~5e4-1e5 samples if the per-site sd is ~7e-4; to be measured
  in step 2). Total 0.5-2e9 evaluations at ~1.1e4/s on an H100 (2.1x the 8x8 cost per evaluation, measured on a 2080 Ti) = **~12-50 H100-h**.
- Expected gain: the HF ViT is 3.9e-5 above the Chen-Heyl ZVE; one Lanczos step removed 73-92% of the error at N <= 64,
  so psi_1 could land at ~-0.49770, i.e. ~1e-5 below the current best variational -0.4976921. This is the one place where
  deliverable C can produce a record at a saturated size; it needs a Paderborn-scale budget to resolve 1e-5.

## 12x12 and 14x14 (no deep-NQS energies with error bars published; 14x14: only Hu VMC p1 -0.49638(1))
Guide options:
| option | expected guide quality | training cost | amplitude cost | notes |
|---|---|---|---|---|
| (a) our ViT, trained per size (SR/minSR, symmetry-projected at the end) | GCNN level (12x12 -0.496769) needs ~1e2-3e2 A100-h by the published costs (GCNN 16x16 ~300 A100-h; 10x10 ViT records 2-4e3 A100-h) | high | 64 passes of a 36-49-token ViT | most expensive, but the "deep NQS" headline |
| (b) PP / RBM+PP, own JAX code (results/rbmpp_6x6, L-generic, tests pass) | at 6x6 the PP alone (6.3e-5) beat our ViT 2x, RBM+PP + 1 Lanczos 7.4e-6 vs ViT + 1 Lanczos 2.55e-5 | 6x6: 0.2-7 RTX-2080-h; 12x12/14x14 to be measured (det of N/2 x N/2 per image) | cheap with rank-1/2 updates for one-hop ratios | best guide per GPU-h; published RBM+PP 12x12 -0.496791(4) is the bar |
Recommended path (cheapest decisive): (b) at 12x12, PP -> RBM+PP with SR, full projection; then one Lanczos step (direct VMC
of psi_1) and FN-DMC with the RBM+PP guide (n1 ~ 330 ratio updates per move). New-result bar: 12x12 below -0.496791(4)
(RBM+PP) by 2 SE; 14x14 any variational number with error bar (Hu p1 -0.49638(1), EXT -0.49722(2)). Then (a) if the
budget allows, as the deep-NQS add-on test (does the ViT + Lanczos beat RBM+PP + Lanczos at equal GPU-h?).
Order-of-magnitude H100 budget (to be firmed up by a 1-hour pilot each): RBM+PP training 12x12 ~20-50 H100-h, 14x14
~40-100; Lanczos direct VMC ~10-30 H100-h per size; FN-DMC with the RBM+PP guide ~10-30 H100-h per size.
Drafts: `cluster/paderborn/deliverable_C/` (12x12/14x14 RBM+PP drafts wait for a 1-hour Paderborn pilot of rbmpp_train.py at L = 12).
