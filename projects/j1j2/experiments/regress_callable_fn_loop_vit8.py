import sys
import numpy as np

EXP = "/Users/aliotaifi/Chatty/projects/j1j2/experiments"
RES = "/Users/aliotaifi/Chatty/projects/j1j2/results/8x8_krylov_3471544"
BENCH = "/Users/aliotaifi/j1j2_vit_bench"
sys.path.insert(0, EXP)

from callable_fn_loop import CachedAmplitude, SquareJ1J2, K1FNEngine
from vit8_callable_amplitude_adapter import load

n = int(sys.argv[1]) if len(sys.argv) > 1 else 64
backend = load(f"{BENCH}/vit_J2=0.50_N=8x8_k=0.mpack")
z = np.load(f"{RES}/krylov_scaling_8x8_a1.20_tr4096_va2048.npz")
states = z["train_states"][:n].astype(np.uint64)
reference = z["rtrain"][:n].astype(float)

eng = K1FNEngine(
    SquareJ1J2(8, 0.5),
    CachedAmplitude(backend.log_amplitude_bits),
)
eng.ensure_r(states)
fresh = np.asarray([eng.r_cache[int(x)] for x in states])
err = np.abs(fresh - reference)
print("CALLABLE_REGRESSION", "n", n, "max_abs", float(err.max()),
      "mean_abs", float(err.mean()), "amp_eval", eng.amp.neval)
assert float(err.max()) < 1e-9
