#!/usr/bin/env python3
"""Convert the stored sector CSR (st6_build_csr.py) into the compact upper-triangular coded format
used on the GPU (fits an 11 GB RTX 2080 Ti):
  H = diag(sqrt n) C diag(1/sqrt n),  C_rr' = codes/8  (H symmetric => only entries with col >= row kept)
  up_indices.npy int32 (nnz_u)  column of each kept entry
  up_codes.npy   int16 (nnz_u)  8 * c_rr'
  up_inc.npy     uint8 (nnz_u)  row(p) - row(p-1)  (row(-1) = 0), so row = cumsum(inc)
Usage: python st6_build_upper.py CSRDIR
"""
import json, os, sys, time
import numpy as np
import numba

d = sys.argv[1]
t0 = time.time()
indptr = np.load(os.path.join(d, 'indptr.npy'))
indices = np.load(os.path.join(d, 'indices.npy'))
codes = np.load(os.path.join(d, 'codes.npy'))
D = indptr.shape[0] - 1
print('loaded', time.time() - t0, flush=True)


@numba.njit(parallel=True)
def count(indptr, indices, cnt):
    for i in numba.prange(indptr.shape[0] - 1):
        c = 0
        for p in range(indptr[i], indptr[i + 1]):
            if indices[p] >= i: c += 1
        cnt[i] = c


@numba.njit(parallel=True)
def fill(indptr, indices, codes, uptr, ui, uc, rowof):
    for i in numba.prange(indptr.shape[0] - 1):
        q = uptr[i]
        for p in range(indptr[i], indptr[i + 1]):
            j = indices[p]
            if j >= i:
                ui[q] = j; uc[q] = codes[p]; rowof[q] = i; q += 1


cnt = np.zeros(D, np.int64)
count(indptr, indices, cnt)
uptr = np.zeros(D + 1, np.int64); np.cumsum(cnt, out=uptr[1:])
nnz_u = int(uptr[-1])
ui = np.empty(nnz_u, np.int32); uc = np.empty(nnz_u, np.int16); rowof = np.empty(nnz_u, np.int32)
fill(indptr, indices, codes, uptr, ui, uc, rowof)
inc = np.diff(np.concatenate([[0], rowof.astype(np.int64)]))
assert inc.min() >= 0 and inc.max() <= 255, (inc.min(), inc.max())
# symmetry check on the codes: C_rr' n_r = C_r'r n_r'  (sample of rows)
n = np.load(os.path.join(d, 'norb.npy')).astype(np.float64)
rng = np.random.default_rng(0)
bad = 0
for i in rng.integers(0, D, 2000):
    for p in range(indptr[i], indptr[i + 1]):
        j = indices[p]
        if j == i: continue
        sij = codes[indptr[i]:indptr[i + 1]][indices[indptr[i]:indptr[i + 1]] == j].sum()
        sji = codes[indptr[j]:indptr[j + 1]][indices[indptr[j]:indptr[j + 1]] == i].sum()
        if abs(sij * n[i] - sji * n[j]) > 1e-9: bad += 1
print('symmetry violations in sample', bad, flush=True)
assert bad == 0
np.save(os.path.join(d, 'up_indices.npy'), ui)
np.save(os.path.join(d, 'up_codes.npy'), uc)
np.save(os.path.join(d, 'up_inc.npy'), inc.astype(np.uint8))
meta = json.load(open(os.path.join(d, 'meta.json')))
meta.update(nnz_upper=nnz_u, inc_max=int(inc.max()), n_rows_without_upper=int(np.sum(cnt == 0)),
            n_diag_entries=int(np.sum(ui == rowof)), sec_upper=time.time() - t0)
json.dump(meta, open(os.path.join(d, 'meta.json'), 'w'), indent=1)
print('DONE', meta, flush=True)
