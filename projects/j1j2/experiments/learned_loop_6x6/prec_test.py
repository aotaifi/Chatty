"""Precision audit of the float32 variants of the 6x6 ViT vs the original float64 network.
Usage: python prec_test.py VARIANT   (writes prec_VARIANT.npy; 'float64' is the reference)"""
import sys, time, numpy as np
import ll6_core as C
v = sys.argv[1]
net = C.Net('vit_J2=0.50_N=6x6_k=0.mpack', dtype=v)
ab = np.load('krylov_scaling_6x6_a1.20_tr4096_va2048.npz')
S = np.concatenate([ab['train_states'], ab['val_states']]).astype(np.uint64)
la = net.logabs(net.flat0, S); np.save(f'prec_{v}.npy', la)
big = np.tile(S, 40); net.logabs(net.flat0, big[:16384]); t = time.time(); net.logabs(net.flat0, big); dt = time.time() - t
out = dict(variant=v, evals_per_s=len(big) / dt)
try:
    ref = np.load('prec_float64.npy'); d = la - ref
    out.update(max_abs=float(np.max(np.abs(d))), rms=float(np.std(d)), q999=float(np.quantile(np.abs(d), .999)))
except FileNotFoundError: pass
print('PREC', out, flush=True)
