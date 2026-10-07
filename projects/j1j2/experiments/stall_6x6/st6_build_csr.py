#!/usr/bin/env python3
"""Build the symmetric-sector (k=0, A1, flip+) Hamiltonian of the periodic 6x6 J1-J2 model (J2=0.5)
once on CPU and store it compactly for the GPU code (st6_sector.py).

Stored (out dir):
  indptr.npy  int64 (D+1)       CSR row pointers
  indices.npy int32 (nnz)       column indices (sorted reps order = psi0 table order)
  codes.npy   int16 (nnz)       8 * c_rr',  H_rr' = sqrt(n_r / n_r') * c_rr'  (c = sum_{y in orbit r'} H_{rep r, y})
  norb.npy    int16 (D)         orbit sizes
  meta.json                     dims, checks (E0 from the stored ED vector, max code rounding error)
Usage (ls_env python, ED_COMMON_DIR pointing at the dir with ed_common.py + closed_fn_krylov_sym6x6.py):
  python st6_build_csr.py OUTDIR PSI0_TABLE ORB_CACHE
"""
import json, os, sys, time
import numpy as np
import numba

sys.path.insert(0, os.environ["ED_COMMON_DIR"])
import closed_fn_krylov_sym6x6 as cf  # noqa: E402

out, table, orbc = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)
t0 = time.time()
S = cf.Sym(6, 0.5, cache=orbc)
z = np.load(table)
assert np.array_equal(z["reps"], S.states), "psi0 table reps != sector basis order"
assert np.array_equal(z["orbit"].astype(np.float64), S.n_orb)
v0 = z["amp"] * np.sqrt(S.n_orb)
print("norm v0", float(v0 @ v0), flush=True)
E0 = float(v0 @ S.H(v0) / (v0 @ v0))
print("E0", E0, E0 / 36, "table", float(z["E0"]), flush=True)


@numba.njit(parallel=True)
def mkcodes(indptr, indices, data, n, codes, err):
    D = indptr.shape[0] - 1
    for i in numba.prange(D):
        e = 0.0
        for p in range(indptr[i], indptr[i + 1]):
            c = data[p] * np.sqrt(n[indices[p]] / n[i]) * 8.0
            k = np.rint(c)
            codes[p] = np.int16(k)
            d = abs(c - k)
            if d > e:
                e = d
        err[i] = e


codes = np.empty(S.nnz, np.int16)
err = np.empty(S.D)
mkcodes(S.indptr, S.indices, S.data, S.n_orb, codes, err)
print("max code rounding error", float(err.max()), "code range", int(codes.min()), int(codes.max()), flush=True)
assert err.max() < 1e-6
# symmetry check of the matrix on a random subset of rows
np.save(os.path.join(out, "indptr.npy"), S.indptr.astype(np.int64))
np.save(os.path.join(out, "indices.npy"), S.indices.astype(np.int32))
np.save(os.path.join(out, "codes.npy"), codes)
np.save(os.path.join(out, "norb.npy"), S.n_orb.astype(np.int16))
meta = dict(D=int(S.D), nnz=int(S.nnz), E0=E0, E0_site=E0 / 36, max_code_err=float(err.max()),
            code_min=int(codes.min()), code_max=int(codes.max()), sec=time.time() - t0)
json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
print("DONE", meta, flush=True)
