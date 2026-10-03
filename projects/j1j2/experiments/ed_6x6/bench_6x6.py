"""Timing benchmark: 6x6 A1,+ basis build, one ls matrix-free matvec, CSR-chunk build."""
import os, sys, time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ed_common as ec

L = 6
t = time.time()
basis, G = ec.make_basis(L, k=(0, 0), point="A1", flip=1)
basis.build()
dim = basis.number_states
print(f"group {G} (x2 flip) dim={dim} build {time.time()-t:.1f}s", flush=True)
op = ec.make_operator(basis, L)
print("max_buffer_size", op.max_buffer_size, flush=True)
st = basis.states
x = np.random.default_rng(0).standard_normal(dim)
x /= np.linalg.norm(x)
t = time.time(); y = op(x); print(f"ls matvec {time.time()-t:.1f}s  <x|H|x>={x@y:.4f}", flush=True)
m = 100000
t = time.time()
sp_, co, cnt = op.batched_apply(st[:m])
t1 = time.time() - t
idx = basis.batched_index(sp_[:, 0])
print(f"batched_apply {m} states: {t1:.2f}s  index {time.time()-t-t1:.2f}s  nnz/row {cnt.mean():.1f}  imag max {np.abs(co.imag).max()}", flush=True)
nthr = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
chunks = [st[s:s + 20000] for s in range(m, m + nthr * 20000 * 2, 20000)]
def work(c):
    a, b, cc = op.batched_apply(c)
    return basis.batched_index(a[:, 0]).size
t = time.time()
with ThreadPoolExecutor(nthr) as ex:
    tot = sum(ex.map(work, chunks))
dt = time.time() - t
print(f"threaded {len(chunks)*20000} states: {dt:.2f}s -> est full CSR {dt*dim/(len(chunks)*20000)/60:.1f} min, nnz est {cnt.mean()*dim:.3e}", flush=True)
