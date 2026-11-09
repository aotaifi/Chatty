# Released 8x8 J1-J2 (J2/J1=0.5, PBC) wavefunction weights? Read-only survey, 2026-11-09
Answer: NO loadable RBM+PP (-0.498886 or -0.498963) or other released 8x8 weights found. Energies only.

| Candidate | What exists | Class / energy |
|---|---|---|
| Nomura-Imada 2005.14142 | Paper text (pdftotext, /scratch/a/A.Otaifi/downloads/rbmpp8_weights/arxiv2005/p.pdf): only "implemented on mVMC [65]"; no data/code availability statement, no parameter files. Table I energies only | RBM+PP -0.498886(1); 4x4-sublattice PP -0.498460(6) |
| Varbench data (github.com/varbench/varbench, J1J2/square_64_P_0.5.md; Zenodo 13377360 = release v1.1.0) | One markdown table, 7 rows, columns Energy/Sigma/Variance. Row 1 is -127.73465 (Pauli units) = -0.498963 per site, "RBM+PP K=0, spin-parity, A1", links only the PRX paper, no code, no weights | The -0.498963(2) class: energy only |
| Varbench methods (github.com/varbench/methods, scripts/J1J2/square_64_P_0.5/) | Scripts only: dmrg.sh, vmc_gutzwiller.sh (+inputs wf.in, VMC.in, datprep.in, geometry.d), vmc_jastrow.sh, vmc_rbm.sh. NO RBM+PP script (the two best rows are cited from papers) | Gutzwiller-projected BCS Z2 SL: -127.115(4) = -0.4965 (not near our bar); RBM alpha=1 -0.4825 |
| ketworks (/scratch/a/A.Otaifi/ext/ketworks, commit ef98955) | ketboard.py reads data/results/<arXiv-id>.json (LLM-extracted energy tables with source_location), data/benchmarks.json. Energies and provenance strings only, no parameters (grep: weights/zenodo only hit unrelated entries) | -0.498963 and -0.498886 are both listed, as numbers |
| HuggingFace org nqs-models (API listing, 6 models) | j1j2_square_10x10_05 (ViT, safetensors, 434,760 params, Apache-2.0), j1j2_square_fnqs (10x10), j1j2j3_square_fnqs (10x10), ising_fnqs, ising_disorder_fnqs, heisenberg_disorder_fnqs. No 8x8, no PP/BCS. FNQS cards are 10x10 only (ViT attention is L-dependent) | ViT, -0.497676 at 10x10 |
| Zenodo search (keywords RBM/PP/J1-J2) | Nothing relevant; hits: 18334851 (code, latent imag-time transformer), 17518770 (convolutional transformer) - not opened, not RBM+PP or 8x8 as far as titles go | not checked in detail |
| NetKet / jVMC / mVMC example data | Not checked in depth (no web listing opened); mVMC ships input samples (Heisenberg 6x6-type), not trained 8x8 J1-J2 weights - from memory, unverified | |

Best lead (weak): the Varbench Gutzwiller-projected-BCS inputs (github.com/varbench/methods/tree/main/scripts/J1J2/square_64_P_0.5/vmc_gutzwiller_inputs) give a runnable Z2 RVB start (-0.4965), 0.002 above the RBM+PP bars; it is a recipe, not weights.
Only route to the actual -0.498886 / -0.498963 weights: email Y. Nomura (RIKEN) for the mVMC output (zvo_*.dat/ parameter files) at 8x8 J2=0.5; Varbench row 1 is attributed to the same group.
Not checked: whether Zenodo 13377360 archive holds more than the repos; Science supplement of Varbench; mVMC/NetKet/jVMC repos beyond memory.
