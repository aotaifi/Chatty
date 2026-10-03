# 6x6 Krylov-FN matched-energy verdict — 2026-09-28

Matched Marshall-FN control, 8 independent M=32 replicas:
- mean of run means: -17.97267437
- between-replica SE: 0.02954654
- mean tail4: -18.00366842
- tail4 between-replica SE: 0.02520821.

Direct-ViT Krylov-FN pilot:
- Emean: -18.12798165
- naive within-run SE: 0.00595.

Central difference is about -0.15531 total energy in favor of the Krylov sign guide, consistent with the earlier fixed-amplitude improvement. This passes the 6x6 physics gate: the label-free Krylov signs materially improve the fixed-node projector over Marshall.

Initial four-bin g(r) fit from 416 mixed walkers:
g = [-0.00949, -0.19851, +0.16378, +0.04238],
but chronological chunk SD is ~0.21-0.27 per bin, so the current mixed sample is too small to freeze the correction.

A larger two-replica direct-ViT Krylov-FN collection is running under Chatty job 20260928-090211-49031. It keeps the expensive ViT/r_M cache alive, runs M=32 to beta=2.0 for seeds 9001 and 9002, saves every post-burn population, and fits per-replica plus combined four-bin g(r). If the two g vectors agree, freeze the combined g with eta=0.5 and launch the next 6x6 FN iteration.
