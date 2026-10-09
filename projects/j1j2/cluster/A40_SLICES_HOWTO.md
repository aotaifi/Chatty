# Running our JAX GPU code on the partial A40 GPUs ("slices") of the LMU theorie cluster

Date tested: 2026-10-09, ws1 -> Slurm partition `cip`. Total GPU time used for this study: well under 1 GPU-h.

## Status: works (jax 0.8.2 venv; jax 0.10.2 does NOT work on slices)

| gres | GPU seen by CUDA | XLA `bytes_limit` (default mem fraction 0.75) | node RAM / cores | works? |
|---|---|---|---|---|
| `gpu:a40-8gb:1`  | `NVIDIA A40-8Q`  | 6.30 GB  | 26 GB / 12 | matmul + ViT fwd yes; wt exact-sector run **OOM** (needs ~7.3 GB of sector on device) |
| `gpu:a40-16gb:1` | `NVIDIA A40-16Q` | 12.72 GB | 26 GB / 12 | yes, incl. `wt_run.py` smoke arm |
| `gpu:a40-24gb:1` | `NVIDIA A40-24Q` | 19.15 GB | 41 GB / 12 | yes, incl. `wt_run.py` smoke arm |

Slices exist only in partition `cip`. Use `--mem` <= 24G on 8gb/16gb nodes (26 GB RAM) and <= 38G on 24gb nodes, otherwise
`sbatch` says "Requested node configuration is not available".

## Cause

The slices are NVIDIA **vGPU** (time-sliced "RTX Virtual Workstation", `nvidia-smi -q`: `Virtualization Mode : VGPU`,
`MIG Mode: N/A`, driver 580.159.03). They are normal CUDA devices 0 (no MIG UUID / `CUDA_VISIBLE_DEVICES` handling needed), but
vGPU does not support **CUDA Virtual Memory Management (VMM)**. Recent XLA (jax 0.10.2 in `~/chatty_ll6/venv_gpu`, also used
by `wt.sbatch`) requires VMM for all device allocations and refuses to create the device:

    E ... platform_util.cc:271] Failed to create stream executor for device CUDA:0: Device 0 does not support CUDA Virtual
    Memory Management (VMM). VMM is required for device memory allocation in XLA.
    RuntimeError: Unable to initialize backend 'cuda': INTERNAL: no supported devices found for platform CUDA

Not the cause (tried, no effect): `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `--xla_gpu_autotune_level=0`, memory fraction, driver/CUDA
mismatch (driver 580 / CUDA 13 runs the cu12 wheels fine). The V100 cuDNN failures are a separate issue.

## Fix

Use an older jax whose XLA still has a non-VMM allocator. Tested: **jax 0.6.2, 0.8.2, 0.9.2 all initialise and run on the slice**
(matmul test; 0.8.2 and 0.6.2 also used for the full tests below), jax 0.10.2 fails. 0.10.0/0.10.1 were not tried.

New venv (the existing venvs were not touched). It lives under /project because `~` was over its disk quota and `/var/tmp`
(the default `TMPDIR` on ws1) was full when this was done; use `TMPDIR=/tmp/...` and `--no-cache-dir` for pip:

    PY=/software/opt/el_9/x86_64/python/3.11-2023.09/bin/python3
    V=/project/theorie/a/A.Otaifi/a40slice/venv_jax_0.8.2
    mkdir -p /tmp/${USER}_pip; $PY -m venv $V
    TMPDIR=/tmp/${USER}_pip $V/bin/pip install --no-cache-dir "jax[cuda12]==0.8.2" flax einops optax msgpack numpy scipy

(Installed here: jax/jaxlib 0.8.2, jax-cuda12-plugin 0.8.2, nvidia-cudnn-cu12 9.27.0.42, nvidia-cublas-cu12 12.9.2.10, flax 0.12.8.
`netket`/`nqsmagic` were NOT installed - not needed by `wt_run.py` or `ll6_core.py`; install them separately if a script needs them.)
This venv also works on the full A40 and the 2080 Ti (same code path, checked on the 2080 Ti), so it can be used everywhere.

Harmless noise in the logs on slices: `Nvml call failed with 3(Not Supported). Assuming PCIe gen 3 x16 bandwidth.`
(and, for jax 0.9.2, `xtile_compiler.cc` Triton autotune messages).

## Exact header + env lines

See `a40_slice_wt.sbatch` (tested file, next to this one). Core of it:

    #SBATCH --partition=cip
    #SBATCH --gres=gpu:a40-24gb:1          # or gpu:a40-16gb:1 (then --mem=24G); a40-8gb only for small jobs
    #SBATCH --cpus-per-task=4
    #SBATCH --mem=38G                      # 24G on 8gb/16gb nodes
    VENV=/project/theorie/a/A.Otaifi/a40slice/venv_jax_0.8.2
    export XLA_FLAGS="--xla_gpu_autotune_level=0" XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
    "$VENV/bin/python" script.py ...

Usage: `sbatch --job-name=NAME [--gres=gpu:a40-16gb:1 --mem=24G] a40_slice_wt.sbatch wt_run.py SPEC.json`.
Keep `XLA_PYTHON_CLIENT_PREALLOCATE=false` (otherwise XLA grabs a fraction of a small card up front). Slices are shared
time-sliced vGPUs of one physical A40; there is no per-job isolation guarantee on throughput.

## Verification (2026-10-09)

1. **Matmul** `jnp.ones((1000,1000))@jnp.ones((1000,1000))` summed = `1e+09` on 8gb/16gb/24gb (jax 0.6.2, 0.8.2, 0.9.2).
2. **ViT forward pass**, fp32 with `jax_default_matmul_precision=highest` (`ll6_core.Net`, 6x6 checkpoint, 1024 random Sz=0 states,
   script `a40_slice_test.py`), output `log|psi|`:
   * 8gb vs 16gb vs 24gb slice: bit-identical (max diff 0).
   * float64 version of the same network: bit-identical between slice (jax 0.8.2) and 2080 Ti (jax 0.10.2), max diff 0 -> the
     model code/data path is exactly the same.
   * fp32: slice (0.8.2) vs 2080 Ti (0.8.2) max |d| 1.3e-3, rms 8.1e-5; slice vs the production 2080 Ti jax 0.10.2: max 2.2e-3,
     rms 1.1e-4. **This is not 1e-7**: this ViT amplifies fp32 rounding (LayerNorm fast variance etc.); the *existing* production
     pair (2080 Ti 0.8.2 vs 0.10.2) differs by the same amount (max 1.9e-3, rms 1.0e-4), and the fp32 error against the float64
     result is rms 1.7e-4 (2080 Ti, jax 0.10.2) vs 1.9e-4 (slice, jax 0.8.2). So the slice is exactly as accurate as the
     production build; it is not bit-reproducible against it. No full-A40 reference was run (all full A40s were busy/QoS-limited),
     the 2080 Ti was used as reference instead.
3. **`wt_run.py` smoke** (first arm of `specs/smoke.json`, `s-old`, 300 steps, check mode: 8 estimator CHECKs, exact-sector FN solve):
   ran to completion on 24gb (10.4 min) and 16gb (10.5 min). `G0 = 3.0108013e-05`, `E_FN = 1.0183073e-04` identical to the
   earlier full-A40 smoke (job 16901902: `3.0108013045075193e-05`, `1.0183073736769188e-04`; the 16gb slice reproduces the
   G0 to all printed digits, 24gb to 1e-11 relative). Validation traces `val_LE` at steps 0/100/200/300: slice 16gb
   3.0038615e-5 / 2.99995e-5 / 2.99287e-5 / 2.98682e-5 vs earlier A40 run 3.0038615e-5 / 2.99568e-5 / 2.99267e-5 / 2.98576e-5
   (differences from stochastic training and a later version of wt_run.py, not from the hardware). 16gb and 24gb results agree
   to ~1e-7 relative. On the **8gb** slice it fails with `RESOURCE_EXHAUSTED: Out of memory while trying to allocate 241.16MiB`
   in `st6_sector.py` `Hm` (sector matrix, 7.3 GB, does not fit into 6.3 GB).

Logs: `/project/theorie/a/A.Otaifi/a40slice/logs`, runs: `/project/theorie/a/A.Otaifi/a40slice/runs` and
`/project/theorie/a/A.Otaifi/chatty_writeback_tail/runs/wtslice_*`.

## Practical notes

* Jobs for anything needing > ~6 GB on device need >= 16gb; the 6x6 exact-sector `wt_*` jobs need >= 16gb.
* `ws1` side effects found on the way: `~` home was over its disk quota (pip failed with "No space left" / "Disk quota exceeded")
  and `/var` was 98 % full; both unrelated to this change, I did not delete anything of other sessions.
* Observed: full-A40 jobs sat `PENDING (QOSMaxGRESPerUser)` while slice jobs were started immediately (slice jobs were not
  blocked by that limit in this test), which is the main practical reason to use slices.
