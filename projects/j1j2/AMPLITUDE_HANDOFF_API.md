# Direct callable FN-amplitude handoff

## Purpose
The main K1/FN loop now accepts one flat positive-amplitude model rather than a recursive
static \(g(r)\) correction. This is the interface for the separate amplitude-learning branch.

## Required backend contract
Provide a Python module with either:

\`\`\`python
def load(checkpoint_path):
    return model  # model.log_amplitude_bits(states_uint64) -> 1D float array
\`\`\`

or a module-level \`log_amplitude_bits(states_uint64)\` function.

The returned numbers are \(\log a_{\rm FN}(x)\) for a positive amplitude. An arbitrary
global additive constant is fine. Inputs are a 1D NumPy \`uint64\` array of spin bitstrings;
outputs must preserve order and be finite.


## Important constraint
The backend may internally use the old ViT plus a learned residual, but one call must directly
return the final learned log-amplitude. It must **not** call a local-energy coordinate, K1
threshold, or another recursive \(r\to g(r)\to a\) layer.

## Threshold / sampling handoff
The integration driver also needs:
- a threshold sample NPZ containing states and weights representing the learned-amplitude
  target distribution used for the label-free robust two-means K1 threshold;
- a pool NPZ containing states for FN initialization (optional explicit weights).

The current driver is:
\`projects/j1j2/experiments/callable_fn_loop.py\`.

## First acceptance checks
1. Same-state repeated calls are deterministic.
2. Neighbor log-ratios are finite and batch-order independent.
3. On a known backend (the original 8x8 ViT), fresh local-energy values reproduce stored K1
   values to numerical precision.
4. With a learned FN backend, recomputing K1 and running FN must not reference the old
   four-bin \(g(r)\) layer or base-\(r_0\) cache.
