# Direct callable FN-amplitude integration — 2026-09-30

## State change
The static four-bin \(g(r)\) handoff is no longer on the production path. The K1/FN code now
has a flat amplitude interface: a backend returns \(\log a_{\rm FN}(x)\) directly for batches
of spin bitstrings, and the main loop recomputes the local-energy field, label-free K1
threshold, signs, and fixed-node projector from that callable amplitude.

Files:
- \`experiments/callable_fn_loop.py\`
- \`experiments/vit8_callable_amplitude_adapter.py\`
- \`experiments/regress_callable_fn_loop_vit8.py\`
- \`AMPLITUDE_HANDOFF_API.md\`


## Regression
Using the original 8x8 ViT as a backend through the new interface, 64 stored K1 training
states were recomputed from scratch.

Result:
- max \(|r_{\rm callable}-r_{\rm stored}|\) = \(7.1765\times10^{-13}\)
- mean absolute difference = \(7.2053\times10^{-14}\)
- direct amplitude evaluations = 8,094.

A separate constant-amplitude 4x4 engine smoke test also completed with finite FN energies.

## Remaining dependency
No finished learned-FN amplitude backend/checkpoint from the separate amplitude-learning branch
is present on this Mac or in the searched Library yet. Therefore no learned-amplitude physics
rerun has been claimed.

Next executable step: drop in that branch's backend/checkpoint plus threshold/pool samples,
then run the direct callable loop. The first comparison is learned-amplitude K1 threshold/node
change and FN energy versus the established K1-FN baseline; do not reintroduce the static
four-bin \(g(r)\) layer.
