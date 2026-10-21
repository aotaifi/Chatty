# PLAN (J1-J2 FN/Krylov loop) - current state only; history in RESEARCH_MAP.md and delta_today.md
Goal: a loop (FN amplitude <-> sign flip, stored in networks) that improves a trained network's ground state at
constant cost per round and beats brute-force NQS in GPU-h at 8x8-14x14.

## Step back 2026-10-21 (trigger: the ViT refit is the 8th failed version of "a network stores the near-node tail")
1. Verdict: stop making networks precise at |psi|^2 < 1e-10. Four networks share the same tail displacement,
   eps 3-6e-7, and give E_FN(s_2) ~0.8 x their own VMC error: ViT, PP, RBM+PP, and a ViT fitted to phi* itself.
2. Thought that changed it: the near-node tail is the FN wall term. Lattice FN has a published tool that removes the
   reported energy's dependence on it: the gamma-extrapolated FN of Sorella-Capriotti (PRB 61, 2599 (2000);
   ten Haaf et al. PRB 51, 13039 (1995)). The project's notes cite it, but no run of ours ever used it. The
   continuum analogue of our loop is self-healing DMC (Reboredo-Hood-Kent PRB 79, 195117 (2009)).
3. Cheapest next test: G0. On exact 6x6 tables, E_FN^gamma(s_2, ratios of the ViT / PP / RBM+PP) at
   gamma = 0, 0.25, 0.5, 1, linearly extrapolated to gamma = -1. No training; <= 0.5 A40-slice-h. Pre-register first.
   - PASS (the reported energy no longer needs a tail network): the extrapolated ViT value is <= 1e-5, or within
     3x of the ceiling 9.15e-7. Then T1 is moot for the energy, and only the sign labels remain open.
   - FAIL: the extrapolation removes < 50% of the ViT gap 1.02e-4 -> ceiling.
   - Caveat: an extrapolated value is an estimate, not a bound.

## Open questions
| question | budget | scope line |
|---|---|---|
| Does tail precision track global accuracy rather than architecture? (T0b) | 0.5 slice-h | cx_t0 bases + B1/B2 checkpoints; passed over: new training |
| Can the reported energy avoid the tail? (G0, above) | 0.5 slice-h | gamma-FN extrapolation; passed over: tail networks |
| Backflow Pfaffian (T1 arm ii), the PI-approved design | <= 4 A40-h | runs only if T0b shows a pairing advantage at equal accuracy, or the PI asks |
| Sign labels without the oracle phi* | - | open; fitted-net labels recall 5e-7 (refresh2) |

## Running
| job | run dir | next step per verdict |
|---|---|---|
| G0 (worker building; launcher read by the lead before sbatch) | chatty_compress6/runs/g0_<id> | PASS -> T1 moot for the energy; work moves to the sign labels. FAIL -> T1(ii) is the remaining tail route (PI's design). Partial -> report the fraction removed. |
| T0-phys 17410698 (submitted by another session, Mac identity) | chatty_compress6/runs/t0phys_17410698 | same-construction one-hop control for T0 (c); read per its README amendment |
| T0b 17410565 | chatty_compress6/runs/t0b_17410565 | eps(B2) <= eps(ViT)/2 -> propose T1(ii) to the PI. Within 1.5x on one eps(dE) curve -> PI kill stands; run G0. Otherwise inconclusive -> G0. |

## Dead ends (number, run)
- ViT refit to phi*(s_2): E_FN 9.16e-5 / 9.22e-5, bar 1e-5 (refresh_17409419, refresh2_17409866).
- Fixed pairing determinant as the tail basis: eps 0.78x the ViT's, kill rule (t0_17410220).
- Constant-cost storage of loop steps: 7 designs (RESEARCH_MAP 2026-10-13).
- Learned one-hop bond field: plateau 1.92e-5, bar 1e-5 (RESEARCH_MAP 2026-10-20).
