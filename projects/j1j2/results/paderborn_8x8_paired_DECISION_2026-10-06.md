# Decision on the 8x8 paired A/B jobs on Paderborn (2026-10-06)

Jobs (submitted 2026-10-04 by the ChatGPT-loop agent, all pending: Paderborn GPU maintenance):
3579917-3579920 (4 paired replicas, one process), array 3579925 (arms split per job), array 3579934[0-3]
(4 replicates, arms sequential). All three are repairs of OOM job 3578114.

What they test: on 8x8 with parent signs s1 fixed, whether a retrained amplitude (physical-H SR or
frozen-H_FN SR checkpoints) gives a larger next-K1 energy decrease than the baseline ViT amplitude.
128 train + 128 validation states per replica.

Decision: not needed; cancel all three sets when Paderborn is reachable.
- Redundant: three versions of the same repair.
- Resolution: the earlier single runs gave +-0.07 total energy (~1e-3/site); 512 matched states cannot
  resolve the ~1e-5..1e-4/site differences that matter (6x6 iteration-2 verdict).
- Superseded: the amplitude design study (results/amp_design/DESIGN_MEMO.md, test A) shows the frozen-H_FN
  objective is a single majorise-minimise step on the fixed-sign VMC energy and is beaten 4-10x by a
  second-order fixed-sign VMC optimizer; first-order SR checkpoints are not the amplitudes we will use.
- Cost: ~3 H100-h and ~80 GB per arm.
