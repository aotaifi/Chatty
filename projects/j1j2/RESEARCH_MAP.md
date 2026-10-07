# J1-J2 FN/Krylov project: research map (living document, updated 2026-10-07)

One page that holds the thread of our discussions. Details live in CLAUDE_TAKE.md (log) and results/*/README.md.

## 1. What would make this worth publishing (PI's criteria)
1. **Scale:** new results where other methods have none. Survey (results/literature/J1J2_BENCHMARKS_COST.md):
   - 8x8: an energy below -0.49889 (RBM+PP -0.498886) would be new.
   - 12x12, 14x14, 18x18: no deep-NQS results at all (the clearest gaps).
   - 10x10: saturated (record -0.4976921); only a cost win or a controlled extrapolation counts.
2. **Cost:** cheaper than brute-force NQS at equal accuracy. Bar: the 10x10 ViT took ~1.9e3-3.8e3 A100-h.
3. **Separate pitch:** FN/Krylov repairs the signs of bad or imperfect NQS guides ("enhance your NQS").

## 2. Established (with the file that holds the evidence)
| Result | Where |
|---|---|
| Krylov sign step (label-free, energy-optimal T): exact signs in 3-6 steps given the exact amplitude; depth grows slowly with N (16 -> 36 sites) | paper, sign_design_tests |
| Exact FN/Krylov loop reaches the GS; iteration count flat 16 -> 36 sites; Anderson ~3x faster | paper |
| Bad starts: any in-sector GS content -> GS; exact eigenstates are fixed points; other symmetry sectors trap; Anderson unsafe from bad starts | bad_start/ |
| Stored signs: composite sign net + ONE exact hop = recursion at fixed cost; train on tempered samples + one-hop neighbours (self-sealing); Krylov step factors are unlearnable, composite signs are learnable | stored_signs*/, sign_learnability/ |
| 6x6 signs: composite net + hop w_s 2e-5 vs ViT 1.3e-4 | amp_design/T3 |
| FN-surrogate amplitude objective = one majorise-minimise step on fixed-sign VMC; second-order fixed-sign VMC beats it 4-10x | amp_design/DESIGN_MEMO.md |
| 4x4 from a trained net: Krylov sign + second-order fixed-sign VMC beats the best tuned VMC (signs+amplitude) 3.3-7.2x at equal CPU-h | amp_design/vit_t1t2* |
| 6x6 from the ViT: energy tie; near the ViT the remaining FN error is the amplitude, not the signs | learned_loop_6x6_it2, amp_design/T3 |
| Corrections: our ViT is NOT state of the art at 6x6/8x8 (RBM+PP is better) | CLAUDE_TAKE 2026-10-07 |

## 3. Open questions, in the order we attack them
1. **Why does the loop stall at the ViT level on 6x6?** Compression vs information vs optimizer noise. Running: stall_6x6/.
2. ~~Do large-sample amplitude steps beat their own control?~~ **Answered 2026-10-07: no.** Large N fixes generalisation (steps verify at z 2-8), but at the ViT amplitude the fixed-sign optimum is within ~1e-5/site; loop arm ties, control D gains -0.35(0.31)e-5 (amp_design/AMP6_table.md).
3. **Cost race from scratch:** Marshall + amplitude-only net + Krylov signs vs training a full NQS, total GPU-h to a target energy (6x6 validation -> 8x8 target < -0.49889). Waits for 1 (architecture choice).
4. **Add-on pitch at scale:** one Krylov sign step on the public 10x10 ViT checkpoint (nqs-models); cheap check.
5. **Baseline to beat for any energy claim:** Lanczos step + variance extrapolation (Hu et al. 2013).

## 4. Decisions taken (and why)
- FN = signs + referee/bound; amplitude objective = fixed-sign VMC energy with a second-order optimizer (test A, 2026-10-06).
- Signs are stored as composite nets, always used with one exact hop (angle 2).
- Claims at the 1e-5/site level need FN at M >= 512, medians, and the reference guide on the same seeds (M=128 has a heavy low tail).
- "Beat the ViT" is not a result by itself; 6x6 is a lab with exact references.

## 5. Parked ideas (not dropped)
- Walker-histogram / MLE amplitude fitting (no zero-variance property; ledger).
- 8x8 sign machinery alone (cheap; useful for the add-on pitch).
- Separate paper repo for Overleaf; paper rewrite after the angles settle.
- Possible NQS collaborators (ViT authors, CQSL).

## 6. Running now
| Work | Owner | Budget |
|---|---|---|
| Stall diagnosis (6x6) | Opus | <= 15 GPU-h + 100 CPU-h |
