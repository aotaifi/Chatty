# PLAN (J1-J2 FN/Krylov loop) - current state only; history in RESEARCH_MAP.md and delta_today.md
Goal: a loop (FN amplitude <-> sign flip, stored in networks) that improves a trained network's ground state at
constant cost per round and beats brute-force NQS in GPU-h at 8x8-14x14.

## Step back 2026-10-22 (trigger: second failed version, T1s then T1s-2, of the tail-trained amplitude)
1. Verdict: switch from training harder to one hop. 4x samples and 3x steps moved r(-11) by only 5-12%. The splice puts the whole gap at |phi*|^2 <= 1e-10.
2. Thought that changed it: a Pfaffian makes neighbour ratios cheap (Woodbury), so a one-hop correction |a| |1 + alpha(E_L - E)| is constant cost. It was the consultant's PfH1 form. On the VMC tables it already gave PP 2.37e-5 and ViT 2.90e-5 (T0, T0-phys).
3. Cheapest next test: G1. Apply the one-hop correction to the tail-trained tables (t1s2 ppbf / pp / vit); no training, <= 0.5 slice-h. Pass: E_FN <= 1e-5 and below psi_P + 1 Lanczos (1.89e-5) at equal hops.

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

## Result 2026-10-22: G1 PASS (6x6, oracle target): a tail-trained PP + 3x3 backflow with one hop gives E_FN 7.06e-6 (bar 1e-5; best equal-hop reference 1.29e-5). Run g1_17426232.

## Next (2026-10-22): remove the oracles from the G1 guide, one at a time (README 'De-oracling'): O1 alpha by VMC energy (minutes) -> O2 the loop with a walker-producible target, stall stop + viability projection (<= 8 A40-h) -> O3 sign labels -> O4 walkers.

## Open questions
| question | budget | scope line |
|---|---|---|
| Tail-trained amplitude to E_FN <= 1e-5 (T1s-2 done: PP+bf 2.42e-5, PP 2.61e-5, ViT 6.89e-5) | spent ~16 slice-h | pairing family (PP, PP+backflow) trained decade-stratified; ViT kept as control |
| T2 design: (i) the same guide trained without oracle phi* (from FN walkers + fast-update neighbour ratios); (ii) cost at 8x8 with Woodbury updates; (iii) edge over plain SR VMC of PP+backflow at equal GPU-h | design first (Opus partner + literature), then pre-register | PI's T2 spec |
| Sign labels without the oracle phi* | - | open; fitted-net labels recall 5e-7 (refresh2) |

## Running
| job | run dir | next step per verdict |
|---|---|---|
| T0-phys 17410698 (submitted by another session, Mac identity) | chatty_compress6/runs/t0phys_17410698 | same-construction one-hop control for T0 (c); read per its README amendment |

## Dead ends (number, run)
- gamma-FN on the tail problem: min <H> over gamma at gamma = 0 (ViT 7.57e-5), 26% of the gap removed, bar 50% (g0_17411369).
- ViT refit to phi*(s_2): E_FN 9.16e-5 / 9.22e-5, bar 1e-5 (refresh_17409419, refresh2_17409866).
- Fixed pairing determinant as the tail basis: eps 0.78x the ViT's, kill rule (t0_17410220).
- Constant-cost storage of loop steps: 7 designs (RESEARCH_MAP 2026-10-13).
- Learned one-hop bond field: plateau 1.92e-5, bar 1e-5 (RESEARCH_MAP 2026-10-20).
