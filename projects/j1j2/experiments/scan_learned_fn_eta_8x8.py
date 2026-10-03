import os, sys
import numpy as np
import jax.numpy as jnp

ROOT = "/Users/aliotaifi/Chatty/projects/j1j2"
WB = "/Users/aliotaifi/j1j2_vit_bench"
EXP = os.path.join(ROOT, "experiments")
sys.path.insert(0, EXP)
os.environ["BASE_VIT_CHECKPOINT"] = os.path.join(WB, "vit_J2=0.50_N=8x8_k=0.mpack")

from learned_fn8_callable_amplitude_adapter import Backend, bits2spins
from callable_fn_loop import SquareJ1J2, robust_two_means_threshold

BASE_NPZ = os.path.join(ROOT, "results/8x8_krylov_3471544/krylov_scaling_8x8_a1.20_tr4096_va2048.npz")
HANDOFF = os.path.join(ROOT, "results/learned_fn_handoff_8x8.npz")
RESID = os.path.join(ROOT, "results/fn_residual_cnn_8x8.mpack")
OUT = os.path.join(ROOT, "results/learned_fn_eta_scan_8x8.npz")
T0 = -28.37107876288694
ETAS = np.array([0.0, .0025, .005, .01, .02, .04, .08, .12, .18, .24, .36])

b = np.load(BASE_NPZ)
h = np.load(HANDOFF)
states = np.concatenate([b["train_states"], b["val_states"]]).astype(np.uint64)
r0 = np.concatenate([b["rtrain"], b["rval"]]).astype(float)
old = r0 <= T0
frac = len(b["train_states"]) / len(states)
lat = SquareJ1J2(8, .5)
# Build the exact threshold-state neighborhood once.
diag = np.array([lat.diag_energy(int(x)) for x in states], float)
rows, xs, ys, coeff = [], [], [], []
all_states = [states]
for i, x in enumerate(states):
    n = lat.neigh(int(x))
    if n:
        yy = np.array([q[0] for q in n], np.uint64)
        all_states.append(yy)
        rows.extend([i] * len(n))
        xs.extend([int(x)] * len(n))
        ys.extend([q[0] for q in n])
        coeff.extend([0.5 * q[1] * q[2] for q in n])
all_states = np.unique(np.concatenate(all_states))
rows = np.asarray(rows, np.int32)
coeff = np.asarray(coeff, float)
xidx = np.searchsorted(all_states, np.asarray(xs, np.uint64))
yidx = np.searchsorted(all_states, np.asarray(ys, np.uint64))
print("UNIQUE_STATES", len(all_states), "EDGE_TERMS", len(rows), flush=True)

# Evaluate base ViT and residual CNN once on all required states.
backend = Backend(RESID)
base = np.empty(len(all_states), float)
g = np.empty(len(all_states), float)
for i in range(0, len(all_states), 4096):
    u = all_states[i:i+4096]
    X = bits2spins(u)
    flat = X.reshape(len(X), 64)
    base[i:i+len(u)] = np.asarray(backend.v0.log_value(jnp.asarray(flat))).real
    g[i:i+len(u)] = np.asarray(backend.res_apply(jnp.asarray(X)))
    if i == 0 or (i // 4096) % 25 == 0:
        print("AMP_CACHE", min(i+len(u), len(all_states)), "/", len(all_states), flush=True)

rows_out, rall = [], []
for eta in ETAS:
    loga = base + eta * g
    contrib = coeff * np.exp(loga[yidx] - loga[xidx])
    rr = diag + np.bincount(rows, weights=contrib, minlength=len(states))

    wt = b["iwtrain"].astype(float) * np.exp(2.0 * eta * h["train_g"].astype(float))
    wv = b["iwval"].astype(float) * np.exp(2.0 * eta * h["val_g"].astype(float))
    wt /= wt.sum(); wv /= wv.sum()
    w = np.concatenate([frac * wt, (1.0 - frac) * wv])

    T, info = robust_two_means_threshold(rr, w)
    new = rr <= T
    change = float(np.sum(w * (new != old)) / w.sum())
    d10 = float(np.sum(w * (old & ~new)) / w.sum())
    d01 = float(np.sum(w * (~old & new)) / w.sum())
    corr = float(np.corrcoef(r0, rr)[0, 1])
    ess = float(1.0 / np.sum((w / w.sum()) ** 2))
    row = [eta, T, change, d10, d01, corr, ess]
    rows_out.append(row); rall.append(rr)
    print("ETA_SCAN", row, flush=True)
rows_out = np.asarray(rows_out, float)
rall = np.asarray(rall, float)
np.savez_compressed(OUT, etas=ETAS, rows=rows_out, threshold_r=rall, T0=T0,
                    unique_states=len(all_states), edge_terms=len(rows))
print("ETA0_MAX_R_DIFF", float(np.max(np.abs(rall[0] - r0))), flush=True)
print("SAVED", OUT, flush=True)
