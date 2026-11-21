# 2026-11-21 Bounded network breaks the cubic copy floor

**What was built.** The copy that turns each exact round A_{k+1} = M[chi_k] back into a stored network (frozen base B times
exp(head)) used a cubic polynomial of 11 fixed inputs: 8 L2 moments of B, the round-1 head h_1, l_1, and l' (rebuilt each round from
a one-hop projection of the latest copy, cost z^2 per amplitude, flat in the round). The cubic landed at N = 1.93-1.95e-6 whatever its
target. A tanh MLP 11-64-64-1 on the same inputs and the same weights (n p_T^2) reaches N 1.731e-6 on the round-5 target (copy error
0.17e-6 vs the cubic's 0.37e-6), 1.911e-6 on round 4 (0.215 vs 0.253). Scrambling l' costs it 0.24e-6, so the two-hop input carries the
gain. Runs: /scratch/a/A.Otaifi/ansatz_runs/consol_6x6/ref2hop/NN/{NN4r,NN4sr,NN5r}; code NN/nn_fit.py (numpy backprop).
Record: STORAGE_PROBLEM.md 'BOUNDED NETWORK ON THE COPY INPUTS' (paper check + result).

**Why.** A cubic grows as |z|^3 outside its bulk, so rows with little weight got huge, wrong amplitudes (spikes), and reweighting
made them worse. A tanh network is bounded (|f| <= |b2| + sum|w2|), so it cannot spike, and it fits the bulk better.

**Rejected.** Reweighting the cubic toward the copy measure (blew up: high-leverage rows). An asinh head and the range guard
(they recover part of the spike cost, but the floor stays). Growing the inputs (joint two-exchange tokens): kept as the next
step if the network had also floored at >= 0.35e-6; it did not.

**Three things to understand.**
1. "Floor whatever the target" can mean the fit family or the inputs. The test was to keep the inputs and change only the family.
   The bounded network moved the floor, so the family was the limit. This revises point 2 of handoffs/2026-11-20-two-hop-copy-inputs.md.
2. A chain with constant copy error eps per round converges to x* = N* + eps/(1-c) (c 0.56, N* 1.05 from arm R): eps 0.17 gives
   about 1.44e-6. If the errors compound, eps grows with the round instead.
3. Still open: NN5's input was the cubic's round-4 copy. The network-only chain for rounds 5-6 (ref2hop/NNc, STORAGE 'NETWORK-ONLY
   CHAIN') tests whether it degrades. Predicted chi_5 1.70, chi_6 1.58 (+-0.06, 50%).
