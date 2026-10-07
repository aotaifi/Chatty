# J1-J2 square lattice, J2/J1 = 0.5, PBC: benchmark energies and compute cost

Literature survey, no computation. Compiled 2026-10-07. Energies are E/N in units of J1 (spin-1/2,
H = J1 sum_<ij> S_i.S_j + J2 sum_<<ij>> S_i.S_j, L x L torus, N = L^2) unless stated otherwise.
Numbers were read from the papers themselves (text/tables of the arXiv PDFs or HTML), not from secondary quotes.
Where I could not read a number from the paper, it is marked "not tabulated" or "not verified".

Type codes: **UB** = variational upper bound (stochastic estimate); **UB+L(p)** = variational after p Lanczos steps
(still an upper bound); **EXT** = extrapolated (zero-variance / variance / truncation-error extrapolation; NOT a bound);
**EXACT** = exact diagonalisation; **PROJ** = projector (GFMC) energy with a guiding state, not a plain variational
number; **MPS** = DMRG energy at fixed bond dimension M (variational in the MPS manifold, but see torus caveat).

## 0. Corrections to the reference values we were given

| Item given | Finding |
|---|---|
| 6x6 exact -0.50380965 | Consistent with literature ED -0.503810 (Schulz-Ziman-Poilblanc 1996, quoted in Hu 2013, Lin 2024, Chen 2022). Our own ED -0.5038097 (paper/main.tex) agrees. |
| 10x10 minSR -0.4971633(8) | **Not a 10x10 number.** Chen & Heyl (arXiv:2302.01941) report **10x10: E/N = -0.4976921(4)** (best variational) with zero-variance extrapolation **-0.497715(9)**, and **16x16: E/N = -0.4967163(8)**. The quoted -0.4971633(8) looks like a garbled 16x16 value (-0.4967163(8)). Use -0.4976921(4) for 10x10. |
| 6x6 ViT -0.503654(21), 8x8 ViT -0.498823(35) | These are **our own** ViT numbers (projects/j1j2/paper/main.tex line ~439, experiments/amp_design/t3_summary.py), not published values. I found no published ViT/transformer energies at 6x6 or 8x8 for J1-J2 at 0.5 (see "Where a new result would be"). Relative to the best literature variational numbers: 6x6 is 1.1e-4/site above Nomura RBM+PP (-0.503765(1)); 8x8 is 6e-5/site above Nomura RBM+PP (-0.498886(1)) and above Hu VMC p=2 (-0.49886(1)). |
| arXiv:2507.02644 as "GFMC/FN with NQS" | arXiv:2507.02644 is Gu et al., "Solving the Hubbard model with Neural Quantum States" (transformer NQS for the doped Hubbard model). It is not a J1-J2 or GFMC paper. I found **no published fixed-node/GFMC calculation using a deep-NQS guide for J1-J2 at 0.5**. The only guided-GFMC J1-J2 papers are RBM-GFMC (Lin, He, Lu, Chin. Phys. B 31, 080203 (2022), doi:10.1088/1674-1056/ac615f) and PEPS-GFMC (arXiv:2406.12207). |
| Viteritti-Rende-Becca PRL 130, 236401 (arXiv:2211.05504) | The ViT-wavefunction PRL is for the **1D** J1-J2 chain; it has no 2D square-lattice energies. The 2D square-lattice ViT result is Rende et al. (arXiv:2310.05715), 10x10 only. |

## 1. Per-size tables (J2/J1 = 0.5, periodic boundary conditions)

### 1.1 6x6 (N = 36)

| Method | E/N | Err | Type | Params / config | Compute | Reference |
|---|---|---|---|---|---|---|
| ED | -0.503810 | exact | EXACT | - | - | Schulz et al., J. Phys. I 6, 675 (1996); quoted in arXiv:1304.2630, arXiv:2406.12207 |
| DMRG torus, M = 8192 SU(2) | -0.503805 | n/a | MPS | 8192 SU(2) states | not reported | Gong et al., arXiv:1311.5962, PRL 113, 027201 (Table I) |
| DMRG torus, extrapolated in truncation error | -0.503808 | (1) | EXT | | not reported | same |
| VMC (Gutzwiller+Jastrow-type fermionic) + Lanczos extrapolation (variance) | -0.50382 | (1) | EXT | | not reported | Hu, Becca, Parola, Sorella, arXiv:1304.2630, PRB 88, 060402(R) (Table IV) |
| VMC p=0 / p=1 / p=2 | -0.50117 / -0.50323 / -0.50357 | (1) each (text: p=2 -0.503571(3)) | UB / UB+L(1) / UB+L(2) | few params | not reported | Hu 2013 (Tables I-III) |
| RBM+PP (mVMC), projections | -0.503765 | (1) | UB | 16 hidden units; PP f_ij real, full (no sublattice) | K/Fugaku supercomputers, no hours reported | Nomura & Imada, arXiv:2005.14142, PRX 11, 031034 (Table I, App. B) |
| RBM (symmetrised) p=0 / p=1 / p=2 Lanczos | -0.50364 / -0.50376 / -0.50378 | (2) / (3) / (4) | UB / UB+L | RBM | not reported | Chen, Hendry, Weinberg, Feiguin, arXiv:2206.14307 (Table 2) |
| Shallow CNN (Choo) | -0.50185 | (1) | UB | CNN | | Choo, Neupert, Carleo, arXiv:1903.06713, as quoted in Table II of arXiv:2406.12207 and Table 2 of arXiv:2206.14307 |
| PEPS-guided GFMC, D = 6 (D = 5: -0.5029(1)) | -0.5033 | (4) | PROJ | PEPS D=6 guide | Tianhe-2; hours not reported | Lin, Guo, He, Xie, Lu, arXiv:2406.12207, PRB 109, 235133 (Tables I-II) |
| Power-law extrapolation in D of PEPS-GFMC | -0.50376 | (4) | EXT | | | same, footnote 66 |
| Deep ResNet + MinSR (64 layers, 146,320 params) | not tabulated | | | | | Chen & Heyl, arXiv:2302.01941: only amplitude/infidelity vs ED shown (Extended Data Fig. 2), no energy number in the text |
| **Ours: ViT** | **-0.503654** | (21) | UB | | this project | projects/j1j2/paper/main.tex |

Relative error to ED: Nomura 9e-5; DMRG(8192) 1e-5; ours 3.1e-4.

### 1.2 8x8 (N = 64)

No ED. Reference "ground-state" estimates are extrapolations and disagree at the 1.5e-4 level (Hu -0.49906(1) vs DMRG -0.4992(1)).

| Method | E/N | Err | Type | Params / config | Compute | Reference |
|---|---|---|---|---|---|---|
| DMRG torus, M = 4096 / 6144 / 8192 SU(2) | -0.497598 / -0.497961 / -0.498175 | n/a | MPS | up to 8192 SU(2) states (SI: "more than 32000" states used in the extrapolation) | not reported | Gong et al., arXiv:1311.5962 (Table I) |
| DMRG torus extrapolated in truncation error | -0.4992 | (1) | EXT (unreliable: torus hard to converge, extrapolation overshoots Hu) | | | same |
| VMC p=0 / p=1 / p=2 | -0.49656 / -0.49855 / -0.49886 | (1) each | UB / UB+L(1) / UB+L(2) | few params | not reported | Hu 2013 |
| VMC + variance extrapolation | -0.49906 | (1) | EXT | | | Hu 2013 (Table IV) |
| RBM+PP, no sublattice structure | -0.498886 | (1) | UB | 16 hidden units | K/Fugaku, no hours | Nomura & Imada 2021 (Table I) |
| RBM+PP, 4x4 sublattice PP (used for production) | -0.498460 | (6) | UB | | | same, App. B |
| HQT (Holographic Quantum Transformer) | -0.5001 | (1) | UB claimed (**suspect**) | params not stated | single RTX 5070 Ti; time, params, steps not stated | Guo, Xiao, Liu, Li, arXiv:2607.00398. The value lies ~1e-3 below every extrapolated 8x8 estimate (-0.49906(1), -0.4992(1)) so it violates the variational bound unless the estimator is biased (reported variance 1.4e-3, short chains). Treat as unreliable. |
| Deep NQS (ViT, ResNet, GCNN) | no tabulated energy found | | | | | Chen-Heyl use 8x8 only for the S=1 gap (energies via ZVE, tabulated only in figures); FNQS (arXiv:2502.09488) shows V-scores only |
| **Ours: ViT** | **-0.498823** | (35) | UB | | this project | main.tex |

### 1.3 10x10 (N = 100), the NQS benchmark

Best-estimate ground state energy: ZVE -0.497715(9) (Chen-Heyl). Hu's variance extrapolation (-0.49781(2)) is lower by 1e-4, i.e. the two extrapolations disagree at about 5 sigma. Rel. err. below is vs -0.497715.

| Method | E/N | Err | Type | Params | Compute (reported) | Rel. err | Reference |
|---|---|---|---|---|---|---|---|
| Deep ResNet2 + MinSR | **-0.4976921** | (4) | UB | >1e6 (1.07e6 per Nutakki et al.) | Ns = 1e4 samples; A100 80 GB, up to 4 A100 per training step in the timing test; **total GPU-h not reported** | 4.6e-5 | Chen & Heyl arXiv:2302.01941, Nat. Phys. 20, 1476 (2024), doi:10.1038/s41567-024-02566-1 |
| ZVE of the above (2 ResNet2 sizes + Lanczos step) | -0.497715 | (9) | EXT | | | 0 | same |
| ViT, HF checkpoint `symm_trxy_ising` (T, C4v, spin inversion) | -0.497676335 | no error on card | UB | 434,760 | 40 x A100-64GB for about 4 days (about 3,800 A100-h, my arithmetic) | 7.8e-5 | huggingface.co/nqs-models/j1j2_square_10x10_05 |
| ViT, HF `symm_t` / `main` | -0.49760546 / -0.497505103 | none | UB | 434,760 | same | 1.5e-4 / 4.2e-4 | same |
| Deep ViT + SRt, full symmetry | -0.497634 | (1) | UB | 267,720 | **4 days on 20 A100 (about 1,900 A100-h, arithmetic)**; M = 6000 samples; 5000 (T) + 5000 (rot) + 4000 (refl+parity) steps after symmetry restoration, Fig. 1 axis about 1e4 steps; lr 0.03 cosine; lambda 1e-4; no Marshall prior | 1.6e-4 | Rende, Viteritti, Bardone, Becca, Goldt, arXiv:2310.05715, Commun. Phys. (2024), doi:10.1038/s42005-024-01732-4 (Table I) |
| RBM+PP | -0.497629 | (1) | UB | 13,132 | K/Fugaku, no hours (Roth quotes about 6e4 CPU-h for the 16x16 case) | 1.7e-4 | Nomura & Imada 2021 |
| ResNet1 MinSR, 64 layers | -0.497627 | (1) | UB | 146,320 | not reported (listed in Rende Table I as ref [30]) | 1.8e-4 | Chen & Heyl |
| ConvNext (fViT-like) | -0.497583 | (6) | UB | 2.6e5 | 8 x A100, 4092 samples/iter, 2500 iterations per symmetry stage, final eval 6.4e5 samples; GPU-h not stated | 2.7e-4 | Nutakki, Shokry, Vicentini, arXiv:2505.03466, PRR 7, 043099 (2025) |
| VMC p=2 (Gutzwiller-type) | -0.497549 | (2) | UB+L(2) | 5 | not reported | 3.3e-4 | Hu 2013 |
| VMC p=0 / p=1 | -0.49521 / -0.49718 | (1) | UB / UB+L(1) | | | | Hu 2013 |
| VMC variance extrapolation | -0.49781 | (2) | EXT | | | -1.3e-4 (below) | Hu 2013 |
| Deep CNN (Sunway) | -0.497468 | (1) | UB | 421,953 | Sunway, up to 4e7 cores; per-iteration timings only (about 1.4e2+7e1+8e1 s at 1e7 cores for ~6e6 Markov chains, scaling test, not a total) | 5.0e-4 | Liang et al., arXiv:2204.07816 |
| GCNN | -0.497437 | (7) | UB | 67,548 | 1000 SR steps x 1024 samples + 500 x 4096; single A100; (about 300 GPU-h is quoted for the 16x16 case) | 5.6e-4 | Roth, Szabo, MacDonald, arXiv:2211.07749, PRB 108, 054410 (2023) |
| PITQS (Strang), 143,010 params | -0.49741 | (3) | UB (symmetry level not stated) | 143,010 | 2 x A100/H100; 10,000 iterations x 8192 samples; hours not reported | 6.1e-4 | Yamazaki et al., arXiv:2602.03031 |
| TQS (plain transformer, beta=2 / 4) | -0.49652 / -0.49651 | (9) / (10) | UB | 155,620 / 303,260 | same | | same |
| Deep CNN (Li) | -0.49717 | (1) | UB | 106,529 | | | Li et al., IEEE TPDS 33, 2846 (2022) |
| RBM+Lanczos p=0 / p=1 | -0.49580 / -0.4968 | (2) / (4) | UB / UB+L(1) | RBM | | | Chen-Feiguin, arXiv:2206.14307 |
| DMRG torus M=4096/6144/8192 | -0.495044 / -0.495301 / -0.495530 | n/a | MPS | | not reported; SI says 10x10 torus truncation error 8e-5 even with 32000 states | | Gong et al. (Table I) |
| DMRG torus extrapolated | -0.4988 | no error bar given | EXT (unreliable) | | | | same |
| PEPS-guided GFMC D=4 / D=5 | -0.4954(6) / -0.4957(2) | | PROJ | | Tianhe-2, hours not reported | | Lin et al. arXiv:2406.12207 (Table III) |
| HQT zero-shot transfer from 8x8 | -0.49782 | (3) | UB claimed (**suspect**) | not stated | single RTX 5070 Ti | -1.3e-4 (below ZVE) | Guo et al. arXiv:2607.00398; below the Chen-Heyl ZVE and below Hu's bound; not credible without independent check |
| Shallow CNN / MLP (older) | -0.49516(1) (Choo), -0.4947359(1) (Liang 2018), -0.494757(12) (Szabo-Castelnovo), -0.48941(1) (Ledinauskas MLP) | | UB | | | | Table I of arXiv:2310.05715 |

### 1.4 12x12 (N = 144)

| Method | E/N | Err | Type | Params / config | Compute | Reference |
|---|---|---|---|---|---|---|
| RBM+PP (4x4 sublattice PP) | -0.496791 | (4) | UB | 16 hidden | K/Fugaku | Nomura & Imada, as quoted in Roth Table I |
| GCNN | -0.496769 | (9) | UB | 12 layers x 6 feature maps | A100, 1000 x 1024 + 500 x 4096 | Roth et al. arXiv:2211.07749 (Table I) |
| Deep ResNet2 + MinSR | computed (gap data, ZVE) but E/N not tabulated in text | | | | | Chen & Heyl |
| Transformer (scaling-law study, L = 12) | computed, V-scores only, no tabulated E/N | | UB | | M = 16384 samples | Rende et al. arXiv:2606.02794 |

### 1.5 14x14 (N = 196)

| Method | E/N | Err | Type | Reference |
|---|---|---|---|---|
| VMC p=0 / p=1 | -0.49447 / -0.49638 | (1) | UB / UB+L(1) | Hu 2013 (Tables I-II) |
| VMC variance extrapolation | -0.49722 | (2) | EXT | Hu 2013 (Table IV) |
| anything NQS | none found | | | |

### 1.6 16x16 (N = 256)

| Method | E/N | Err | Type | Params / config | Compute | Reference |
|---|---|---|---|---|---|---|
| Deep ResNet + MinSR | **-0.4967163** | (8) | UB | ResNet2 (size not stated in the text for this lattice) | not reported | Chen & Heyl (Fig. 2c) |
| GCNN | -0.496509 | (6) | UB | 16 layers x 6 maps | **about 300 GPU-h (A100)**, 1000 x 1024 + 500 x 4096 samples/steps | Roth et al. (Table I) |
| RBM+PP | -0.496213 | (3) | UB | 4x4 sublattice | K/Fugaku; Roth quotes about 6e4 CPU-h | Nomura & Imada, as quoted in Roth |

### 1.7 18x18 (N = 324)

| Method | E/N | Err | Type | Params / config | Compute | Reference |
|---|---|---|---|---|---|---|
| Deep CNN (Sunway) | -0.496500 | n/a | UB | CNN1 106,529 (transfer from 6 to 10 to 18) | Sunway, up to 4e7 cores | Liang et al. arXiv:2204.07816 |
| RBM+PP | -0.496275 | (3) | UB | 6x6 sublattice PP | K/Fugaku | Nomura & Imada (text of App. B) |
| VMC p=0 / p=1 | -0.49426 / -0.49611 | (1) | UB / UB+L(1) | | | Hu 2013 |
| VMC variance extrapolation | -0.49717 | (2) | EXT | | | Hu 2013 |

### 1.8 20x20 (N = 400) and thermodynamic limit

| Method | E/N | Err | Type | Params / config | Compute | Reference |
|---|---|---|---|---|---|---|
| ViT with Spatial Attention, C4v symmetry | -0.496732 | (1) | UB | not stated | M = 2^14 = 16384 samples, 5000 + 500 + 100 steps; the paper's total (mostly triangular up to 42x42) was 25,000 GH200 GPU-h; 20x20 share not separated | Viteritti, Rende, Sachdev, Carleo, arXiv:2602.02665 |
| ViT, ZVE | -0.49684 | (1) | EXT | | | same |
| ResNet2 + Lanczos (Chen-Heyl) | shown in Fig. 5 of arXiv:2602.02665 as higher than the translation-symmetric ViT (about -0.4966 to -0.4967 by eye); number not tabulated | | | | | |
| DMRG cylinders (bulk, width up to 12), 2D limit | -0.4968 | n/a | EXT | | | Gong et al. |
| finite PEPS D=8, **open boundaries**, central-bulk extrapolation, L up to 24 | -0.49635 | (5) | EXT (OBC, not comparable size by size) | | cost not reported | Liu et al., arXiv:2009.01821, Sci. Bull. 67, 1034 (2022) |

## 2. Compute-cost summary (what is actually reported)

| Result | Hardware / time | Parameters | Samples x steps | Note |
|---|---|---|---|---|
| 10x10 Deep ViT + SRt (-0.497634(1)) | 4 days on 20 A100, i.e. about 1,900 A100-h (arithmetic) | 267,720 | M = 6000; about 1.4e4 steps (5000+5000+4000 after symmetry restoration, plus earlier unrestored phase, see Fig. 1) | the only explicit full-run cost for a best-class 10x10 transformer result |
| 10x10 HF ViT checkpoint (-0.497676335) | about 4 days on 40 A100-64GB, i.e. about 3,800 A100-h (arithmetic) | 434,760 | not on the card | per-sweep sampling time on one A100-40GB: 41 s (main), 166 s (symm_t), 3317 s (symm_trxy_ising) |
| 10x10 ResNet2 MinSR (-0.4976921(4)) | not reported. Only a per-step benchmark (Extended Data Fig. 1, A100 80G) | >1e6 (ResNet2), 146,320 (64-layer ResNet1) | Ns = 1e4 | no GPU-h or step count in text. I do not attempt an estimate: the step count is not stated |
| 16x16 GCNN (-0.496509(6)) | about 300 GPU-h (A100) | 16 layers x 6 feature maps | 1000 x 1024, then 500 x 4096 | the cheapest strong large-L number; Roth quotes about 6e4 CPU-h for the RBM+PP result at this size (secondary) |
| 10x10 ConvNext (-0.497583(6)) | 8 x A100, hours not given | 2.6e5 | 4092 x 2500 per symmetry stage | |
| 10x10 PITQS (-0.49741(3)) | 2 x A100/H100, hours not given | 143,010 | 8192 x 1e4 | |
| 20x20 ViT-SA (-0.496732(1)) | per-run hours not separated; whole paper 25,000 GH200 h; authors say the earlier GP/RNN-type results can be had in under 1000 GPU-h | | 16384 x 5600 | |
| Scaling-law study, L = 12 to 20, 4 models | about 1e5 A100-h total | | M = 16384 | gives the compute accounting f = C x M x N_opt (forward-pass FLOPs); V-score about A f^-alpha N^beta, alpha = 0.86(3) for square J1-J2 at 0.5 (beta = 0.93(4)); saturation of the power law beyond n_l about 12 layers |
| 10x10 Sunway CNN (-0.497468) | up to 4e7 SW26010Pro cores; no total | 106,529 and 421,953 | ~6e6 Markov chains per iteration | not comparable to GPU-h |
| RBM+PP (all sizes) | K and Fugaku CPU; no hours given | 1.3e4 (10x10) | | |
| DMRG torus (Gong) | CPU, not reported; M = 4096 to 8192 SU(2) kept (more than 32000 states used for the torus extrapolation) | | | 10x10 torus not converged (truncation error 8e-5) |
| finite PEPS (Liu) | not reported; D = 8 (checked 6 to 10) | | | OBC only |
| PEPS-GFMC (Lin) | Tianhe-2; not reported; D <= 7 | | | |

Rule of thumb from these numbers: reaching rel. error about 1.6e-4 at 10x10 with a transformer cost about 2e3 A100-h in 2023; the 2025 checkpoint (0.8e-4) cost about 4e3 A100-h; Chen-Heyl reach 4.6e-5 with >1e6 parameters at unreported cost. At 16x16 a GCNN reaches 2e-3 relative to the Chen-Heyl value at about 300 GPU-h.

## 3. Pretrained NQS checkpoints on the Hugging Face hub (org `nqs-models`)

Contact on the cards: R. Rende and L. L. Viteritti. Created Jan-Feb 2025 (heisenberg_disorder_fnqs Oct 2025). NetKet + JAX + Flax loaders; custom code (`trust_remote_code`).

| Repo | Content | Params | Training | Energy on card |
|---|---|---|---|---|
| `nqs-models/j1j2_square_10x10_05` (Apache-2.0; files: model.safetensors 3.49 MB, `spins` 13.11 MB pre-thermalised samples, transformer.py, attentions.py, vitnqs_model.py, config.json) | ViT NQS, 10x10, J2/J1 = 0.5; 8 layers, d = 72, hidden 288, 12 heads; three revisions | 434,760 (F64) | 40 x A100-64GB, about 4 days | `main` (translation-invariant ViT): -0.497505103, 41 s/sweep; `symm_t`: -0.49760546, 166 s/sweep; `symm_trxy_ising` (T + point group + spin inversion): **-0.497676335**, 3317 s/sweep (A100-40GB). No error bars on the card. The README cites Rende et al. (Commun. Phys. 2024). Params differ from the paper's 267,720 network, so this is a retrained larger model. |
| `nqs-models/j1j2_square_fnqs` | Foundation NQS, 10x10, trained on R = 100 values of J2 in [0.4, 0.6]; 4 layers, d = 72, hidden 288, 12 heads, patch 2x2 | 223,104 | 4 x A100-64GB for several hours; batch 16,000 | none; about 21 s/sweep on A100-40GB; valid only for 0.4 <= J2 <= 0.6 |
| `nqs-models/j1j2j3_square_fnqs` | FNQS for J1-J2-J3, 10x10, J2 in [0,1], J3 in [0,0.6] | 434,904 (8 layers) | 4000 systems, 16,000 samples | none |
| `ising_fnqs`, `ising_disorder_fnqs`, `heisenberg_disorder_fnqs` (arXiv:2507.05073) | other models | | | not J1-J2 square |

All J1-J2 checkpoints are 10x10 only; I found no 6x6 or 8x8 checkpoints on the hub, and no checkpoints from other groups (search of the hub for "j1j2" returned only these three). The factored-attention ViT has L-dependent attention matrices, so these do not transfer to 6x6/8x8 without retraining (my inference, not stated on the cards). Useful as 10x10 test guides: `symm_trxy_ising` (best energy, -0.497676335, gap to ZVE 3.9e-5) or `main` (-0.497505103, 1.1e-4 higher, cheap, translation-invariant only, 41 s/sweep). The pre-thermalised `spins` file saves burn-in.

## 4. Where a new result would be

1. **10x10 is saturated for plain NQS energies.** Best UB -0.4976921(4), HF ViT -0.497676335, ZVE -0.497715(9). A new variational number is only interesting if it beats -0.49769 or if it comes with a decisive cost. The real open issue is the reference itself: Hu's extrapolation (-0.49781(2)) and Chen-Heyl's ZVE (-0.497715(9)) differ by 1e-4. A projector/FN/Krylov estimate with controlled error bar at 10x10 would help decide, and none exists (no FN/GFMC with a deep-NQS guide has been published for this model; RBM- and PEPS-guided GFMC are 1e-3 above the NQS bounds at 10x10).
2. **8x8 has no strong published deep-NQS energy.** The best variational numbers are Nomura RBM+PP -0.498886(1) and Hu VMC p=2 -0.49886(1); our ViT -0.498823(35) is slightly above both. Chen-Heyl computed 8x8 only inside the gap analysis (ZVE, no tabulated E/N), and the FNQS paper shows V-scores. The extrapolated references disagree (-0.49906(1) Hu vs -0.4992(1) DMRG torus). A transformer/ResNet or FN energy with error bar at 8x8 below -0.49889 would be a new result; so would a tight E0 estimate at 8x8 (the cheapest size on which to settle the Hu vs DMRG extrapolation).
3. **6x6: exact is known**, so an energy is not new; it is useful as a validation. Pure deep NQS energies at 6x6 are not tabulated either (Chen-Heyl show only amplitudes). Best published UB are Nomura -0.503765(1) and Chen-Feiguin RBM+L(2) -0.50378(4).
4. **12x12, 14x14 and 18x18 are the sparsest.** 12x12: only Nomura -0.496791(4) and Roth -0.496769(9) tabulated. 14x14: only Hu (p=1 UB -0.49638(1), EXT -0.49722(2)). 18x18: Liang Sunway -0.496500, Nomura -0.496275(3), Hu EXT -0.49717(2). No transformer/ResNet energy with error bars is published for these sizes (the scaling-law and ResNet papers computed them but report V-scores/gaps/figures). A transformer or FN energy at 12x12, 14x14 or 18x18 would fill a visible gap.
5. **16x16 and 20x20 already have strong results** (Chen-Heyl -0.4967163(8) at 16x16; ViT-SA -0.496732(1) at 20x20, ZVE -0.49684(1)); not good targets for energy-only claims.
6. **Method gaps.** (a) DMRG on the torus is unreliable beyond 8x8 and its extrapolations overshoot (10x10 -0.4988 with no error bar); cylinder DMRG and OBC PEPS give only thermodynamic-limit estimates (-0.4968, -0.49635(5)), no PBC PEPS finite-size energies beyond D<=5 PEPS-GFMC (-0.4957(2) at 10x10). (b) The only controlled E0 estimates on PBC clusters above 6x6 are extrapolations. (c) Compute: only one full-run cost is reported for a best-class 10x10 transformer (about 1,900 A100-h, Rende 2023), none for Chen-Heyl's record, and the 2026 scaling-law paper gives a compute axis but only for the whole study (1e5 A100-h). A result matching rel. error 1e-4 at 10x10 (or a given 8x8/12x12 target) with a stated, much smaller GPU-h budget would be a new, citable cost data point.
7. **Caution on recent claims.** HQT (arXiv:2607.00398) quotes 8x8 -0.5001(1) and 10x10 -0.49782(3), both below every extrapolated reference; I would treat them as unverified/probable estimator bias, not as new records.

## 5. Sources and verification notes

- Verified directly in the PDFs (text/tables): arXiv:2310.05715 (Table I, cost sentence), arXiv:2302.01941 v3 (main text, Extended Data captions), arXiv:1304.2630 (Tables I-IV), arXiv:1311.5962 (Table I, SI), arXiv:2211.07749 (Table I, cost sentence), arXiv:2005.14142 (Tables I-II, App. A-B), arXiv:2406.12207 (Tables I-III), arXiv:2206.14307 (Tables 2, 4), arXiv:2009.01821, arXiv:2204.07816, arXiv:2602.02665, arXiv:2606.02794, arXiv:2502.09488, arXiv:2211.05504 (1D only).
- Read from arXiv HTML summaries (numbers quoted by the fetch tool, one pass each, so please re-check against the PDF before citing): arXiv:2602.03031 (PITQS), arXiv:2505.03466 (ConvNext), arXiv:2607.00398 (HQT), arXiv:2606.07825 (PII: 10x10 J1-J2 at 0.5 with ViT/RBM, M = 16384 samples; converges in about 20-30 iterations vs about 100 for SR; no final energy or cost found in the text I could access). PITQS also quotes -0.49725 (267,720 params) and -0.49730 (994,700 params) from its Ref. [25]; these do not match the -0.497634(1) of Rende et al., presumably a different symmetry level; unresolved.
- Hugging Face data: model cards, API listing and file sizes read on 2026-10-07 (nqs-models org).
- Not found / not reported: GPU-hours for Chen-Heyl; CPU-hours for DMRG, PEPS, mVMC; any published 6x6/8x8/12x12/14x14/18x18 transformer energies with error bars; any NQS-guided GFMC for J1-J2 at 0.5.
- Other related 2026 items seen but not used for numbers: arXiv:2608.21291 (review, Rigo et al.), arXiv:2607.02292 (RL view of NQS, 1024-sample/A100 benchmarks at J2 = 0.5, no records), Ferrari & Becca arXiv:2005.12941 (Gutzwiller + Lanczos, finite-size level crossings, not used for energies).
