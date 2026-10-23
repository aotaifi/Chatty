# Proposal: the next major calculation (lead draft v1, 2026-10-23, for a two-round review)

## Claim it rests on
The loop's edge is that fixed node computes the amplitude given a sign and the hop computes the near-node cancellation
sums; a stored network should carry only the bulk and the global sign. Evidence (exact 6x6, J2/J1 = 0.5, one seed;
details in experiments/compress_6x6/README.md):
- O2b: fitted loop with one hop, oracle sign s_2: 2.37e-5 -> 8.4e-6 per site above E0 in 5 rounds; refit tax ~6e-6/round.
- F1/F2/T1s: better fitting (time, capacity, the exact FN-relevant objective) does not lower the tax; the error sits on
  ~0.1% of violating bonds (55% of it) and does not generalise.
- H1: a second hop cuts the tax to 0.32x (1.90e-6); derived two-hop loop floor ~2.5e-6 (unvalidated).
- S1: the Pfaffian's own sign beats the oracle s_2 on its own amplitude (one hop 1.87e-5 vs 2.37e-5); oracle-free
  ceilings 2.5e-6 (s_1, ViT one-hop sign) and <= 7e-6 (PP own sign); s_2 9.15e-7.
- H1 side result: the signed two-step polynomial has lower <H> than the unsigned one in every case.
- N0 (running): are the top-0.1% bonds near-node cancellation points?

## Proposed calculation L1: the oracle-free loop with RBM+PP and two signed hops (exact 6x6)
- Network a_k: RBM+PP exactly as Nomura-Imada (experiments/rbmpp_6x6/rbmpp_lib.py, M = 16, complex RBM, real f,
  full space group x spin flip), warm start = our converged VMC RBM+PP (G1 reference: one hop with s_2 gives 1.29e-5).
- Guide g_k = psi_net x P_2(H - E), P_2 the joint two-step Krylov polynomial, coefficients by the variational energy
  (H1 rule). Sign of the guide = sign(Re psi_net) x sign(P_2) (signed hop); no oracle sign.
- Amplitude step: exact lattice FN solution phi_k of g_k (stands for GFMC until O4).
- Refit: RBM+PP to phi_k, |phi_k|^2-weighted (bulk) log-amplitude plus sign agreement in the bulk; the tail is not
  fitted (the hops rebuild it). Constant budget per round.
- Measured per round: E_FN(g_k), tax_2, contraction c_k, the guide sign's |psi0|^2-weighted wrong weight vs s0, the
  expelled-set split.
- Controls: round 0 with sign s_2 and one hop reproduces 1.29e-5; with two hops reproduces the H1-type number for
  the RBM+PP table.
- Decision: PASS if E_FN <= 3e-6 within 4 rounds with no oracle except exact solves; viability: tax_2 < 0.5 x the
  exact chain's gain in rounds 1-3; RISE (c_k > 1) stops it.
- Cost: H100 (Paderborn), ~1-1.5 H100-h per round from F1/F2 rates; ~6-8 H100-h.

## Then (if L1 passes): 8x8 with walkers (O4)
- First measurement at 8x8: contraction per hop vs L (domain walls may need more hops at larger L).
- The bar: 8x8 RBM+PP -0.498886 (rel 3.5e-4), no published GPU cost; 10x10 ViT 1.6e-4 for ~1920 A100-h.

## Open risks I see
- The RBM part is complex: the guide sign must be real; is sign(Re psi_net) after projection well defined?
- A signed hop can flip signs inconsistently with the network between rounds (consistency mattered in S1).
- One seed throughout.
