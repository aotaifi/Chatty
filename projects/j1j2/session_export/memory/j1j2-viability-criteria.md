---
name: j1j2-viability-criteria
description: "What makes the J1-J2 FN/Krylov loop worth publishing (PI's criteria, 2026-10-07)"
metadata:
  node_type: memory
  type: project
  originSessionId: 538ff1fa-50fd-457b-aa22-c58f295c8d22
  modified: 2026-10-07T08:52:10.188Z
---

The PI judges the J1-J2 FN/Krylov project by viability, not by solving angles:
1. **Scale:** merit only if it scales to sizes where it gives new benchmark results that existing methods (NQS, DMRG, PEPS, Lanczos-step VMC, GFMC) do not reach.
2. **Cost:** if it scales but costs much more than brute-force NQS training or other numerics for the same accuracy, it is still bad — always report cost against those.
3. **Separate pitch:** FN/Krylov correcting bad or imperfect guides is a merit on its own ("enhance your NQS" add-on for NQS people), even without beating SOTA.

**Why:** the PI wants the project's viability protected while angles get solved.
**How to apply:** frame every result with an equal-cost baseline and a size-scaling outlook; keep the literature/baselines (incl. Lanczos step + variance extrapolation) in view; raise viability concerns early. Related: [[j1j2-paper-on-overleaf]].
