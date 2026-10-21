---
name: check-method-before-negative
description: "Never declare a research route failed until the failure is understood and shown to be a conceptual, unfixable limit; check optimizer/convergence first"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 538ff1fa-50fd-457b-aa22-c58f295c8d22
  modified: 2026-10-08T09:02:28.864Z
---

When something does not work, do not give up or relay a "failed" verdict. First ask WHY it failed, using our own reasoning and the literature, and look for a better solution. Declare failure only when the cause is pinpointed to something conceptual that blocks success and that cannot be fixed. Minimum check before any negative verdict: optimizer (e.g. Adam vs SR/minSR for NQS), whether the loss was still decreasing when stopped, learning rates, sample sizes, whether the state-of-the-art method was used.

**Why:** PI's standing rule (2026-10-08). The J1-J2 write-back route was declared failed after a test that used Adam (not SR/minSR) and had not converged; the PI caught it, not me or the independent reviewer.
**How to apply:** for every worker verdict, read the method/run config and convergence before relaying; brief workers with the state-of-the-art optimizer and ask for convergence evidence; label results "inconclusive (cause: X)" rather than "failed" until a conceptual limit is shown. Related: [[j1j2-viability-criteria]].
