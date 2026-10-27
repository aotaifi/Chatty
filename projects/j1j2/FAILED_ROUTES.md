# Failed / blocked routes

Keep this file concise. Each entry must state the obstruction.

- Local stoquasticization / Marshall-type gauges: blocked by nontrivial Z2 loop/sign flux in configuration space.
- CTQMC contour deformation of event times: analytically blocked in tested sectors; sector phase is effectively fixed and does not remove the sign.
- Simple local regrouping / low-order transport rules: failed to close globally without increasingly nonlocal borrowing.
- Simple operator-count / local ordering-statistic compression in continuous time: falsified by explicit negative classes.
- ML sign classification alone: near-random accuracy in tested GNN/transformer experiments; local handcrafted structure performed better.
> CORRECTION 2026-10-27 (index): too broad. Direct sign predictors fail (A26: 8/8 unchecked updates raise E_FN; K12 June: frozen at the Marshall saddle, not a clean representability verdict), but the exact ground-state sign IS learnable (A11, 2026-10-06: N=36, 186k net, rel 6.3e-5; one exact hop removes 96-99% of net error) and GNN rankers behind a verifier work (AP .324). See TRIED_IDEAS.md A11, A26, A29, K12; results/sign_learnability/.
- Fully pre-summed CTQMC kernel route: not established sign-free; remains historically interesting but is not the current main route.

Current focus is the fixed-node amplitude solve <-> sign reconstruction loop and its scaling.
> CORRECTION 2026-10-27 (index): stale. The refit loop (FN amplitude solve <-> sign reconstruction) was CLOSED 2026-10-26 (L8, docs/reports/2026-10-26_l8_loop_round_8x8.md). Current focus since the PI's refocus 2026-10-26: network + p Lanczos hops + lattice fixed-node GFMC as an add-on without refit (PLAN.md Goal).
