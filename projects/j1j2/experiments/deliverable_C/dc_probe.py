"""Throughput probe of the projected 8x8 ViT (fp32, highest matmul precision) on the current GPU, for cost planning.
  python dc_probe.py OUT"""
import os, sys, time, json
import numpy as np
import l8_core as C
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); res = dict(device=str(C.jax.devices()))
rg = np.random.default_rng(0); S = C.spins_to_bits_np(C.random_sz0(rg, 16384))
for sym, batches in ((True, (1024, 2048, 4096)), (False, (8192, 16384))):
    for b in batches:
        g = C.Guide('vit_J2=0.50_N=8x8_k=0.mpack', sym=sym, batch=b, phi=0.056006)
        n = 4 * b if sym else 4 * b
        X = S[:min(n, len(S))]; g.eval(X[:b])
        t = time.time(); g.eval(X); dt = time.time() - t
        res[f'sym{int(sym)}_b{b}'] = len(X) / dt; C.log(sym, b, len(X) / dt)
json.dump(res, open(os.path.join(OUT, 'probe.json'), 'w'), indent=1); print(res)
