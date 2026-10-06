# Bad starts for the exact FN/Krylov loop (J1-J2, J2/J1 = 0.5)

Question (FUTURE_ANGLES.md, angle 1): can the exact loop
`(a_k, s_k) -> phi_FN = Perron GS of H_FN[a_k, s_k] -> a_{k+1} = phi_FN -> energy-optimal Krylov sign step`
start from a very bad guide (excited state, large excited-state overlap, other symmetry sector, random signs) and still reach the ground state?

Code: `projects/j1j2/experiments/bad_start/` (re-uses `fixed_node_solve`, `projected_krylov_update`, `Anderson` of
`krylov_sign_structure/experiments/closed_fn_krylov_anderson.py`; only the starting guide and the per-guide metrics are new).
Full Sz=0 basis, **no symmetry restriction inside the loop** (4x4: D=12870; 20-site 4x5 tilted torus: D=184756).
Target ED ground state is used only for scoring. Data: `data/*.json` (columnar, per start; `tables.md` = full tables for every run).
Figure: `fig_bad_start.png/.pdf`.

Reported per guide k: `E_FN[a_k,s_k]` (eps_FN = (E_FN-E0)/|E0|), `<H>_guide`, wrong-sign probability w_s (weighted by |phi_0|^2, vs ED GS),
overlaps of the guide with phi_0, phi_1, phi_2 (GS sector) and with the lowest state of each other sector, symmetry expectation values.
"Sector" = (S=0, k=0, point group trivial rep, spin-flip even) for the ground state; excited states phi_1, phi_2 are the 2nd/3rd states of this sector
(4x4: E = -8.45792 | -7.32340, -6.74256; 20 sites: -10.44666 | -9.69096, -9.16476), computed by penalty-method diagonalisation in the sector.
Other sectors (lowest state): `S1` = spin-flip odd, k=0 (S=1, <S^2>=2, checked); `kPiPi` = S=0, translation character (-1,-1); `Brot` (4x4) = C4-odd;
`Iodd` (20 sites) = inversion-odd. Their energies (eps): 4x4 S1 0.280, Brot 0.306, kPiPi 0.342; 20 sites S1 0.202, kPiPi 0.229, Iodd 0.225.

## Exact fixed-point check (`check_fixed_point.py`, `data/fixed_point_check_4x4.json`)
For (|phi_n|, sgn phi_n) of the GS, phi_1, phi_2 and the lowest state of S1, kPiPi, Brot: `max|r(x)-E_n|` = 1e-11..7e-11 (r = H psi / psi, one distinct value),
`E_FN - E_n` ~ 1e-13, `||a_FN - a||` ~ 1e-13, Krylov flips = 0. **Every eigenstate is an exact fixed point of the loop**
(|phi_n| is a positive eigenvector of its own H_FN, hence its Perron vector; r is constant, so the sign step has nothing to do).
Zeros: same-sector states have no zeros (min |phi| = 2e-5 .. 6e-5). Other-sector states have symmetry-enforced exact zeros
(4198 / 3190 / 3630 of 12870 configurations for S1 / kPiPi / Brot). The code clips a at 1e-15, which puts a ~1e15 diagonal on those configurations
and ARPACK then fails to converge; in `bad_start_lib.run_loop` the FN eigenproblem is therefore solved on the configurations with a > 1e-12 and the others
are held at 1e-15 (equivalent to the clip, only used when the starting guide has such zeros). Their signs are arbitrary
(sgn(0) = +1), which is why w_s of such states is not meaningful (it varies between 0.24 and 0.49 for the same eigenstate).

## Results, 4x4 (numbers: iterations; eps = eps_FN; plain loop to eps<1e-9, "stuck" = no change)
| start | initial eps / w_s / GS overlap | plain | Anderson (m=5) |
|---|---|---|---|
| (a) J2=0 amplitude + Marshall | 4.8e-2 / 0.013 / 0.62 | eps<1e-6 at 103, <1e-9 at 477, w_s<1e-8 at 100 | 35 / 59 |
| (b) phi_1 pure | 0.134 / 0.45 / 1e-32 | leaves the fixed point at 174 (eps<1e-2), eps<1e-6 at 284, <1e-9 at 752, w_s=0 at ~225 | stuck (E_FN = E_1 for 300 it) |
| (b) phi_2 pure | 0.203 / 0.40 / 4e-32 | 101 / 201 / 672 | stuck |
| (c) phi_0 weight 0.5 (phi_1), + sign | 5.9e-2 / 0.095 / 0.5 | 107 / 481 | 36 / 53 |
| (c) 0.1 | 0.117 / 0.49 / 0.1 | 113 / 487 | 136 / 162 |
| (c) 0.01 | 0.132 / 0.50 / 0.01 | 114 / 491 | **stuck at E_1 (400 it)** |
| (c) 1e-4, 1e-6, 1e-8 | 0.134 / 0.47 / ... | 128 / 501, 141 / 515, 153 / 526 | stuck (1e-4, 1e-8) |
| (c) 0.5 and 0.01 with phi_1 sign flipped; with phi_2 | | 113/584, 125/594; 81/477, 100/480 | 0.5 converges (33-36 / 51-62), 0.01 stuck |
| (d) S1 pure (other sector) | 0.280 / 0.47 / 1e-32 | **trapped: eps = 0.2804 after 3000 it, E_FN = E_sector** | stuck |
| (d) Brot pure | 0.306 / 0.44 | **trapped, 3000 it** | stuck |
| (d) kPiPi pure | 0.342 / 0.47 | frozen 450 it, then leaks to k=0 by roundoff, ends on a **spurious stationary point** eps = 0.272 (1481 it, not an eigenstate, see below) | stuck |
| (d) S1 / Brot + sector-symmetric noise (0.1, 0.5 x random vector projected on the sector) | 0.283..0.358 | **converge to the sector ground state** (eps = 0.2804 / 0.3060), 3000 it, overlap with phi_0 < 1e-26 | |
| (d) kPiPi + symmetric noise | 0.344 | leaks after ~250 it, spurious stationary point eps = 0.256 | |
| (d) GS + other-sector state, GS weight 0.5 | 7.6e-2..8.5e-2 | S1: eps<1e-6 at 41 (<1e-9 at 202); kPiPi, Brot: 99 / 570 | 18-27 / 45-54 |
| (d) GS weight 0.01 | 0.28..0.32 | S1 51 (eps 1.2e-9 at 800); kPiPi 94 / 470; Brot 100 / 568 | S1, Brot stuck at the sector state (400 it); kPiPi 34 / 49 |
| (d) GS weight 1e-4 | | S1 83 (eps 3.9e-9 at 800), kPiPi 114 / 490, Brot 134 / 508 | |
| (e) exact abs(phi_0), random signs (3 seeds) | 0.45-0.52 / 0.47-0.50 / 6e-5..3e-3 | 56-75 / 465-510 | 22-25 / 49-51 |
| (e) uniform amplitude, random signs (3 seeds) | 0.74 / 0.47-0.50 / 3e-7..7e-5 | 98-123 / 547-568 | 47-56 / 72-80 |
| uniform amplitude + Marshall | 0.145 / 0.013 | 104 / 478 | 38 / 59 |

Escape tests requested by the coordinator (plain loop, 4x4, phi_1 start; "escape" = first iteration with eps_FN < 1e-2):
| perturbation of phi_1 | GS weight | escape | eps<1e-6 | eps<1e-9 | reaches phi_0 |
|---|---|---|---|---|---|
| none (roundoff only, eigsh residual 1e-13) | 1e-32 | 174 | 284 | 752 | yes |
| phi_0 admixture | 1e-8 / 1e-6 / 1e-4 / 1e-2 | 52 / 40 / 27 / 14 | 153 / 141 / 128 / 114 | 526 / 515 / 501 / 491 | yes |
| random vector, 1e-4 / 1e-2 (relative) | 3e-12 / 3e-8 | 77 / 50 | 178 / 159 | 552 / 629 | yes |
| log-amplitude noise sigma = 1e-3 / 1e-2 (signs of phi_1 untouched, no explicit phi_0) | 2e-10 / 2e-8 | 67 / 54 | 166 / 155 | 546 / 529 | yes |
| GS-sector-symmetric random vector, 0.1 | 4e-6 | 34 | 140 | 611 | yes |

20 sites (D=184756; Anderson 250 it, plain 700 it; many starts only run with one method, see `tables.md`): baseline 83 (Anderson, eps<1e-9) / 691 (plain);
phi_1 plain: escape at 207, eps<1e-6 at 284, 2.9e-9 at 700; GS weight 0.5 / 0.1 / 0.01 Anderson 105 / 111 / 123 (plain 0.01: eps<1e-6 at 104, 1e-9 at 700);
random signs + uniform amplitude 106 (Anderson) / 686 (plain); GS + S1 / kPiPi / Iodd state (weight 0.5) 84 / 80 / 74 (Anderson), S1 674 (plain).
phi_1, phi_2 and the lowest S1 / kPiPi / Iodd states: Anderson stuck at the starting energy for 250 it (eps = 0.072, 0.123, 0.202, 0.229, 0.225).
Plain S1: frozen over the 300 iterations observed. Plain S1 + sector-symmetric noise: frozen 200 it, then leaks (flip and translation expectation values
go from (-1, +1) to (+1, -1) around iteration 250-300, then to (+1,+1)) and ends at eps = 0.145, w_s = 0.008, GS overlap 0.52 with 0 flips/it
(support frozen, see below).

## Interpretation
1. **Eigenstates are exact fixed points** (above): the guarantee E0 <= E_FN <= <H>_guide is saturated (E_FN = <H>_guide = E_n), so the loop does *not* project out
   excited states in the sense of imaginary-time evolution: FN is a projection inside the nodal class of the guide, and an eigenstate is its own nodal class.
2. **Within the GS sector the fixed point is unstable.** The GS component of the guide grows geometrically: ln(GS weight) rises by 0.35 per iteration for
   phi_1 (amplitude factor 1.19/it; 0.68 per iteration, 1.40/it for phi_2; 0.31 on 20 sites), independent of how the perturbation is generated (explicit admixture,
   random vector, multiplicative amplitude noise: slopes 0.36-0.45; a sector-symmetric random vector, which also contains phi_2, gives 0.65, the phi_2 rate). Hence escape time ~ ln(1/w)/0.35 +- a few iterations (14 / 27 / 52 / 174 iterations for w = 1e-2 / 1e-4 / 1e-8 / roundoff 1e-32),
   after which the loop follows the same slow power-law tail as the baseline (eps<1e-9 at ~480 + escape delay). A guide with large excited-state weight is therefore harmless;
   even a guide with a 1e-8 ground-state weight only loses ~50 iterations. Random signs (w_s = 0.5, GS overlap 1e-3..1e-7, <H> far above E0) converge as fast as the J2=0 baseline.
3. **Different symmetry sector: trapped in exact arithmetic.** H_FN[a,s] is invariant under every symmetry g with U_g psi_guide = chi_g psi_guide (in the gauge of s the
   FN hopping K_ij = s_i H_ij s_j is invariant, and a is invariant), so its Perron vector, and the Krylov update (grouped by r, which is g-symmetric), keep the character.
   The starting state is the lowest state of its sector, hence an eigenstate and a fixed point; sector-symmetric perturbations converge back to the sector ground state
   (S1, Brot: eps = 0.2804, 0.3060 after 3000 iterations, overlap with phi_0 < 1e-26, w_s of the final state vs phi_0 irrelevant). The loop thus acts as a *sector* ground-state solver.
   Roundoff can break translation (and, on 20 sites, spin-flip) symmetry: the k=(pi,pi) 4x4 state and the 20-site S1+noise state leak after ~250-450 iterations into the
   k=0 / flip-even sector, but end on **spurious stationary points** (4x4 kPiPi: eps = 0.272 / 0.256 for two starts, ||H psi - E psi|| ~ 2, <S^2> ~ 3.3, 24-25% of the configurations frozen at a = 1e-15;
   20 sites: eps = 0.145). The frozen configurations are the exact zeros of the other-sector state: with a(x) = 1e-15, FN puts ~1e15 on the diagonal at x, a stays 1e-15
   (weight 1e-30, so also the Krylov energy never sees the wrong signs there). So exact zeros of a are an absorbing support trap of the amplitude map, in addition to the symmetry trap.
   (S1 and C4-odd 4x4 states did not leak within 3000 iterations.)
4. **Mixing in any ground-state weight removes the trap** also for other-sector starts (GS weight 0.5, 0.01, 1e-4 all reach the GS on 4x4, 0.5 on 20 sites), because the mixture has no exact zeros and no symmetry.
   An NQS (no exact symmetry, no exact zeros) is generically in this situation.
5. **Anderson mixing is dangerous from bad starts.** It is a root finder for g(x)=x and the excited-state fixed points are roots: on 4x4 it stays at E_1 for every GS weight <= 1e-2 (phi_1 and phi_2 mixtures),
   although the plain loop leaves them; the safeguard (reset if E_FN rises) never fires because E_FN is constant. It converges quickly when the start has weight >= 0.1 (4x4) or when
   the sector structure is generic (random signs: 22-80 iterations; 20 sites with GS weight 0.01: 123 iterations). A restart rule (plain steps when the residual is small but w_s or
   the Krylov energy gain suggests a stationary excited state) is needed if Anderson is used with a poorly known guide.

## Verdict
* From a bad guide the exact loop **does reach the ground state** if the guide has any generic ground-state content in its symmetry sector (even 1e-8 or random-sign guides with
  <H> far above E0): the cost is the baseline (~100 iterations to eps<1e-6, ~480 to 1e-9 for the plain loop; 4x4 and 20 sites alike) plus ~ln(1/w)/0.35 iterations for the escape.
* It **cannot** leave an exact eigenstate by itself (fixed point) except through roundoff (~170-250 iterations), and it **cannot leave a symmetry sector**: a pure other-sector guide converges to that sector's ground state
  (and exact zeros of a freeze the support). In practice NQS guides are neither exact eigenstates nor exactly symmetric, so the relevant picture is the delay law of item 2.
* Not completed (machine overloaded, disk full at times): 4x4 GS weight 1e-8 in other-sector mixtures and random-noise perturbations of other-sector states; 20-site random signs with exact |phi_0|.
* Caveat: exact loop only (no sampling noise; sampling noise would help escape). Not tested: genuinely sampled loops, N > 20.

## Files
`experiments/bad_start/`: `bad_start_lib.py` (loop, sectors, projectors), `run_bad_start.py` (starts, runner), `check_fixed_point.py`, `spurious_fixed_point.py`, `finalize.py`,
`summarize.py` (-> `tables.md`), `make_figure.py`, `bad_start_20site.sbatch` (ws1 cip/small CPU partition; jobs 16848827, 16849588, 16849589).
Figure: (a) eps_FN vs iteration, plain loop, 4x4; (b) wrong-sign probability; (c) Anderson loop; (d) escape iteration vs ground-state weight of the start (4x4 and 20 sites).
