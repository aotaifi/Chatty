# Stored-sign Krylov loop: progress log
- 2026-10-05: started (fresh attempt). Reading CLAUDE_TAKE.md, learned_loop_6x6 verdict/code.
- 2026-10-05: read exact4x4 + anderson loop code and full_4x4_plain.json reference (it5 eps=4.45e-3 w_s=1.12e-3; it15 eps=4.9e-4 w_s=3.2e-5; it30 eps=5.2e-5 w_s=8.9e-7). Checking ws1 env next.
- 2026-10-05: env = ws1 `module load python/3.12-2024.10` (numpy 1.26, scipy 1.13, torch 2.4), partition small.

## Design (2026-10-05)
Key observation: the Krylov change c_k(x) = sgn[T_k - r_k(x)] is built from r_k(x) = (H psi_k)(x)/psi_k(x), a ratio, so
lattice-symmetry characters cancel: if (a_k, s_k) are symmetric, c_k is exactly INVARIANT under the full group G
(translations x D4 x spin flip; |G| = 256 on 4x4, 576 on 6x6). Given an O(1) stored sign s_hat_k, the exact one-step
target s_rec_{k+1}(x) = s_hat_k(x) c_k(x) costs one hop (|a| and s_hat_k on ~N_bonds neighbours). We then REPLACE
s_rec_{k+1} by an O(1) object s_hat_{k+1} fitted on N samples x ~ |a_{k+1}|^2 (i.i.d. exact on 4x4; MCMC on 6x6).
Chosen methods:
 (c) SYMTAB  orbit table: s_hat_{k+1}(x) = s_hat_k(x) * (-1)^{[orbit(x) in F_{k+1}]}, F = orbits of sampled x with c=-1.
     Stored as one cumulative parity table over orbit representatives, so evaluation = canonicalise x under G
     (|G| bit permutations) + hash lookup, independent of k. Exact on every seen orbit; bias = |a|^2 weight of
     flipped-but-unseen orbits (= the measured stored-vs-recursive disagreement). Ablation TAB: same without symmetry.
     Chosen because it is unbiased where it has data, needs no training, and symmetry multiplies coverage by ~|G|.
 (a) CNN     translation-invariant periodic CNN for the Marshall-gauge sign sigma = s*M (sum-pooled logit), trained
     on the N samples with BCE (sampling from |a|^2 = |a|^2-weighted loss), warm-started from the previous iteration.
     Chosen because it can generalise to unseen configs (where SYMTAB is forced to "unchanged").
 (b) learning only c on top of s_hat_k was not chosen as separate method: stacking k classifiers grows cost linearly
     in k, and SYMTAB already is "(b) with a table". (A CNN-for-c fallback on unseen orbits is the natural hybrid.)
Idealisations in the 4x4 test: FN amplitude exact (a <- phi_FN[a, s_hat]); T_k chosen exactly (full-vector
energy-optimal threshold, as in the reference loop) so that only the sign-STORAGE error is tested.
Metrics per iteration: w_s of s_hat vs ED, eps = rel. energy error of guide (a, s_hat), E_FN, and
D_k = sum_x a^2 [s_hat_{k} != s_rec_{k}] (stored vs exact one-hop recursive sign), plus flip weight of c.
- 2026-10-05: wrote experiments/stored_signs/stored_sign_loop_4x4.py + stored_sign_4x4_lmu.sbatch (array 0..19: exact, bench, {symtab,tab,cnn}x N{1e3,1e4,1e5} x seed{0,1}); cluster dir ws1:~/ChattyRun/j1j2/stored_signs. Test job 16824143 (tasks 0,1,2,14, maxiter 3).
- 2026-10-05: test 16824143 OK: exact mode reproduces full_4x4_plain (it2 eps 1.8375e-2, w_s 2.0275e-3); c exactly orbit-invariant (c_nonsym_weight=0 for symmetric s). Bench: symtab 7e-6 s/config (4096 bitops), CNN 5e-5 s/config (6e5 flops, 18.8k params). CNN training slow (110 s/iter) -> profiling.
- 2026-10-05: CNN replaced by translation-averaged MLP (mean_t MLP(T_t x), 16-64-64-1, 5.3k params; = full-kernel periodic CNN): 1e-5 s/config, 8-60 s/iter. Test 16824167 OK. Insight already: at it2 Krylov flips 3.8e-3 of |a|^2 weight, but on LOW-amplitude configs: even N=1e5 (sample cover 0.98) leaves D=2.0e-3 unstored. Main job submitted (array 0-19, maxiter 40, tag main):
  job 16824183 (theorie, partition small).
- 2026-10-05: main 16824183 CANCELLED (mine): FN eigsh time grew 0.1->53 s/iter in stored runs. Cause: unsampled wrong-sign states get a(x)->0 geometrically under FN (diag += |K| a_y/a_x feedback), hit the 1e-15 floor -> spectrum width 1e10 -> Lanczos stall. splu shift-invert too slow (fill-in). Fix: amplitude floor 1e-8 in FN (reference min a ~2e-5, so exact loop unchanged); added min_amp / n_amp<1e-8 diagnostics. Test 16824523 (exact, symtab N1e3, tab N1e3; 20 it).
- 2026-10-05: test3 16824523 (cancelled, mine) confirmed the collapse: with symtab/tab N=1e3, states with a<1e-8: 0 (it5) -> 1600 (it20); still slow at floor 1e-8. Now floor 1e-6 (|a|^2=1e-12) + ARPACK ncv=64; test4 16824581.
- 2026-10-05: test4 16824581 OK (fast, 0.2-2 s/iter). FIRST RESULT: symtab N=1e3 w_s stuck at 9.7e-3 (Marshall 1.27e-2), eps plateau ~4e-3; tab N=1e3 w_s 1.26e-2. Mechanism: Krylov flips live on LOW-|a| configs that |a|^2 samples miss; FN then suppresses a there (nodes), so they are never sampled again -> self-sealing sign error.
  Design extension (table needs no unbiased weights): (i) tempered sampling from a^(2 beta), beta=0.5; (ii) +nbr: also tabulate c on one-hop neighbours of samples (2-hop shell per sample at build time only, not recursive).
  Main run main2 = job 16824183's replacement: job 16824587; array 0-43 = exact, bench, {symtab, symtab_b0.5, symtabnbr, symtabnbr_b0.5, tab, cnn, cnn_b0.5} x N{1e3,1e4,1e5} x seed{0,1}; 40 iterations.
- 2026-10-05: main2 partial (all but CNN done). it30 w_s / eps (exact loop: 8.85e-7 / 5.19e-5):
  symtab b1: N1e3 9e-3/3.8e-3, N1e4 2e-3/1.2e-3, N1e5 2.3e-4/2.7e-4 (fails; self-sealing)
  tab b1 (no sym): 1.26e-2..7.9e-3 (fails at all N)
  symtab b0.5: N1e3 2e-5 & 6.8e-4 (seed-dependent), N1e4 7e-6/5.9e-5, N1e5 1.3e-5/6.3e-5
  symtab+nbr (b1 and b0.5): IDENTICAL to exact loop at N=1e3..1e5 (D=0 or <1e-8)
  cnn b0.5 N1e3: w_s ~1e-3 floor, eps ~1e-3.
  CAVEAT: 4x4 has only ~1e2 symmetry orbits, so symtab+nbr may simply cover all of them; added coverage counters
  (n_seen_keys/n_keys, unseen flip weight) and harder runs: main3 job 16827468 (array 44-71: tab+nbr, tab+nbr b0.5,
  tab b0.5 at N 1e2/1e3/1e4; symtab variants and cnn_b0.5 at N=100).
- 2026-10-05: main3 16827468 done. Coverage diagnostics show the 4x4 successes are COVERAGE successes: symtab+nbr sees 105-107 of 107 orbits; tab+nbr N1e3 sees 7.6e3-1.0e4 of 12870 configs per step. Decisive structural fact: the Krylov flip set lies almost entirely OUTSIDE the |a|^2-typical region: tab_b0.5 N1e3 covers 99.5% of |a|^2 weight yet unseen flip weight == total flip weight (4e-3 at it2); tab+nbr N=100 covers 19% of configs / 99.8% weight and still misses 75% of the flip weight. Also: D_stored_vs_rec is small (1e-5..1e-8) even when w_s is stuck at 1e-2, because FN collapses a(x) on the wrong-sign states (self-sealing hides the error) -> D measured under the current a is a misleading diagnostic.
  it30 (w_s, eps): tab+nbr b1 N1e2 1.5e-3/1.3e-3, N1e3 8.7e-5/1.7e-4, N1e4 1.3e-6/5.2e-5; tab+nbr b0.5 N1e2 2.3e-4/4.8e-4, N1e3 9e-7/5.3e-5; symtab b1 N1e2 1.27e-2 (=Marshall).

## 4x4 result table (2026-10-05; w_s / eps / D at it 5 | 15 | 30; seed mean or range; D = a^2-weighted stored-vs-one-hop-recursive)
exact recursive loop (reproduced)  1.12e-3 4.45e-3 | 3.21e-5 4.91e-4 | 8.85e-7 5.19e-5
symtab  b1  N1e3   9e-3  7.3e-3 1.4e-4 | 9e-3  4.0e-3 | 9e-3   3.8e-3 4e-6
symtab  b1  N1e4   3e-3  5.2e-3 4e-5   | 2e-3  1.5e-3 | 1.9e-3 1.2e-3 5e-8
cnn     b1  N1e3   1.27e-2 9e-3 5e-4   | 1.27e-2 4.7e-3 | 1.27e-2 4.5e-3 1e-5   (= Marshall, never learns flips)
cnn     b1  N1e4   7.9e-3 7.3e-3 3e-4  | 8-10e-3 4.2e-3 | 0.9-1.3e-2 4.0e-3 9e-6
cnn     b0.5 N1e3  2e-3 5.3e-3 8e-5    | 1e-3 1.4e-3    | 1.3e-3 9.5e-4 2e-5
cnn     b0.5 N1e4  1.15e-3 4.5e-3 3e-6 | 7e-5 5.0e-4    | 2.5e-5 7.4e-5 1e-6
symtab+nbr b1 N1e3 = exact loop (D<1e-8) at all checkpoints; tab+nbr b1 N1e3: 1.8e-3 5.1e-3 | 4.6e-4 8.2e-4 | 8.7e-5 1.7e-4
tab+nbr b0.5 N1e3: 1.2e-3 4.6e-3 | 8.7e-5 5.3e-4 | 9e-7 5.3e-5 (but table touches 1.0e4 of 12870 configs/step)
Cost per stored-sign evaluation (CPU, batched): symtab 9e-6 s (|G|*2N=4096 bit ops; 6x6: 576*2*36), net 1e-5 s (1.7e5 flops, 5.3k params); both O(1) in k.
Verdict: storing is cheap, but the Krylov flip set sits on configurations that |a|^2 samples essentially never visit, and FN then collapses a(x) there (self-sealing; D stays small while w_s is stuck). Pure |a|^2-sample storage (task spec) fails at N=1e3-1e4 on 4x4. Only near-complete coverage works (symmetry orbits or one-hop neighbours, tempered sampling), which will not hold on 6x6 (1.6e7 orbits). The generalising net with tempered sampling (b=0.5, N=1e4) is the only route that does not rely on coverage: it30 w_s 2.5e-5, eps 7.4e-5 (exact 8.9e-7 / 5.2e-5). It is the candidate for 6x6, together with neighbour-augmented labels.
Pending at write time: cnn_b0.5 N1e5 (tasks 42/43) and cnn N1e4/1e5 still running in 16824587 (slow FN: amplitude collapse); log parse used for it30.
