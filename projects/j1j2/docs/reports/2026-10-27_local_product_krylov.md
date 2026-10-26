# Local products of Krylov factors cannot replace the global step; the global step is linear-cost only on local guides (2026-10-27)

- **Finding.** A product of per-bond Krylov factors psi * prod_b [1 + alpha (eps_b - e_b)] keeps essentially none of one global
  Krylov step's fixed-node gain at 6x6: E_FN - E0 = 3.5247e-5/site (LP) vs 5.5738e-6 (global step G) vs 3.5249e-5 (no step);
  LP's share of G's gain falls from 0.15-0.25 at 4x4 to 1e-4 at 6x6. Pre-registered verdict: KILL (best LP / G = 6.32 > 1.5).
  Exact, /scratch/a/A.Otaifi/local_product/stageA/6x6/lp6x6.json (A40 job 17517448), 4x4 in stageA/4x4_*/.
- **Cause.** Near nodes the product divides by psi once per active bond: |psi_LP/psi| reaches 9.7e6 at a configuration with
  |psi|/max 8.9e-13 (eps_b up to ~1e13). Any alpha above ~1e-5 raises the variational energy, so the optimal alpha (~1e-6)
  flips no signs: G flips 4.1e-2 of the near-node configurations, LP 1e-13 (overlap 0). The global step has no such pole:
  psi G = psi + alpha (H - E) psi. Predicted beforehand by the outside review (Astra, cj-b7c4313e2b88).
- **All local variants fail alike** at 6x6: per-site product, equal-alpha product, plaquette product (3.5249e-5 each).
- **Controls.** alpha = 0 reproduces psi bitwise (E_FN diff 1e-16; broken 3.2e-10); densities from an unrelated configuration
  keep 0.00 of the gain; G reproduces the earlier 6x6 value 5.5738e-6 to all digits. The first-order LP = G check holds at 4x4
  (halving ratio 4.000, broken 2.000) but cannot be reached at 6x6 because alpha*eps_b is not small even at alpha = 1e-6.
- **Side result (Stage A').** On a CNN with a finite receptive field, the GLOBAL step can be evaluated with cached bond energies
  exactly (vs full pass <= 3.9e-14; broken update set 5.6e-2) and its cost per sweep grows ~N^1.0 once the lattice exceeds the
  affected range (L >= 16 for a 7x7 receptive field; L = 6..20 fit 1.79, failing the pre-registered <= 1.2). The price is a
  constant overhead per sample ~ the number of affected bonds: ~1000x plain sampling for 7x7, ~240x for 3x3 (RBM+PP direct at
  8x8: 137x, growing ~N). /scratch/a/A.Otaifi/local_product/stageAprime/full/*.json, code experiments/local_product/gcache.py.
- **Cost.** ~3 A40-slice hours, ~0.2 CPU-h; no Paderborn.
- **Consequence (lead's view).** The Krylov step's sign repair cannot be localised: it needs the whole sum (H - E) psi, and any
  product of local factors blows up at the near-node configurations where the repair matters. The step itself scales linearly
  only on strictly local guides, at an overhead of hundreds per sample.
