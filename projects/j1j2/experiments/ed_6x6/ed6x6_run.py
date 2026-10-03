"""Symmetric exact diagonalisation of the 6x6 periodic J1-J2 model (J2/J1=0.5),
S^z=0, using lattice_symmetries 0.8.4 for the symmetry-adapted basis and the
Hamiltonian action.  The Hamiltonian of each sector is stored once as a CSR
matrix (int64 indptr, int32 indices, float64 data; all characters are real),
built with threaded ls batched_apply/batched_index calls, and diagonalised
with ARPACK (scipy eigsh) using a numba-parallel CSR matvec.

Usage:
  python ed6x6_run.py OUTDIR "kx,ky,POINT,FLIP[,NEV[,save]]" ["..."] ...
  e.g. python ed6x6_run.py out "0,0,A1,1,3,save" "0,0,B1,1,1"
kx,ky in units of 2pi/6 (0 or 3 here), POINT in A1,A2,B1,B2,E,none, FLIP=+-1.
"""
import gc, json, os, sys, time
import numpy as np
import numba
import scipy.sparse.linalg as sla
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ed_common as ec

L = 6
J1, J2 = 1.0, 0.5
NTHR = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))


@numba.njit(parallel=True, cache=False)
def csr_matvec(indptr, indices, data, x, y):
    n = indptr.shape[0] - 1
    for i in numba.prange(n):
        acc = 0.0
        for p in range(indptr[i], indptr[i + 1]):
            acc += data[p] * x[indices[p]]
        y[i] = acc


def build_csr(basis, op, chunk=20000):
    st = basis.states
    dim = st.shape[0]
    starts = list(range(0, dim, chunk))

    def work(s):
        sp_, co, cnt = op.batched_apply(st[s:s + chunk])
        idx = basis.batched_index(sp_[:, 0]).astype(np.int32)
        imax = float(np.abs(co.imag).max()) if co.size else 0.0
        return cnt, idx, np.ascontiguousarray(co.real), imax

    parts = []
    imag_max = 0.0
    with ThreadPoolExecutor(NTHR) as ex:
        for cnt, idx, dat, im in ex.map(work, starts):
            parts.append((cnt, idx, dat))
            imag_max = max(imag_max, im)
    assert imag_max < 1e-12, imag_max
    counts = np.concatenate([p[0] for p in parts])
    indptr = np.zeros(dim + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    nnz = int(indptr[-1])
    indices = np.empty(nnz, dtype=np.int32)
    data = np.empty(nnz, dtype=np.float64)
    pos = 0
    while parts:
        cnt, idx, dat = parts.pop(0)
        indices[pos:pos + idx.size] = idx
        data[pos:pos + dat.size] = dat
        pos += idx.size
        del cnt, idx, dat
    assert pos == nnz
    return indptr, indices, data


def run_sector(outdir, kx, ky, point, flip, nev, save):
    tag = f"k{kx}{ky}_{point}_{'p' if flip > 0 else 'm'}"
    rec = {"tag": tag, "k": [kx, ky], "point": point, "flip": flip, "nev": nev}
    t0 = time.time()
    basis, G = ec.make_basis(L, k=(kx, ky), point=point, flip=flip)
    basis.build()
    dim = basis.number_states
    rec.update(group_size_spatial=G, group_size_total=G * 2, dim=dim, t_basis=time.time() - t0)
    print(f"[{tag}] dim={dim} group={G}x2 basis {rec['t_basis']:.1f}s", flush=True)
    op = ec.make_operator(basis, L, J1, J2)
    t0 = time.time()
    indptr, indices, data = build_csr(basis, op)
    rec.update(nnz=int(indptr[-1]), t_csr=time.time() - t0)
    print(f"[{tag}] nnz={rec['nnz']:.3e} csr {rec['t_csr']:.1f}s", flush=True)

    nmv = [0]
    y = np.empty(dim)

    def mv(x):
        nmv[0] += 1
        x = np.ascontiguousarray(x, dtype=np.float64).reshape(-1)
        csr_matvec(indptr, indices, data, x, y)
        return y.copy()

    # symmetry check of the stored matrix on random vectors
    rng = np.random.default_rng(1234)
    a, b = rng.standard_normal(dim), rng.standard_normal(dim)
    asym = abs(a @ mv(b) - b @ mv(a)) / (np.linalg.norm(a) * np.linalg.norm(b))
    rec["asym_check"] = float(asym)
    Hop = sla.LinearOperator((dim, dim), matvec=mv, dtype=np.float64)
    t0 = time.time()
    v0 = rng.standard_normal(dim)
    w, V = sla.eigsh(Hop, k=nev, which="SA", tol=1e-13 if save else 1e-11,
                     ncv=max(20, 4 * nev + 8), v0=v0, maxiter=5000)
    o = np.argsort(w)
    w, V = w[o], V[:, o]
    res = [float(np.linalg.norm(mv(V[:, i]) - w[i] * V[:, i])) for i in range(nev)]
    rec.update(energies=w.tolist(), residuals=res, n_matvec=nmv[0], t_lanczos=time.time() - t0)
    print(f"[{tag}] E={np.array2string(w, precision=10)} E/N={w[0]/36:.8f} "
          f"res={res} matvecs={nmv[0]} {rec['t_lanczos']:.1f}s", flush=True)
    if save:
        np.save(os.path.join(outdir, f"states_{tag}.npy"), np.asarray(basis.states).copy())
        np.save(os.path.join(outdir, f"vectors_{tag}.npy"), V)
        rec["saved"] = [f"states_{tag}.npy", f"vectors_{tag}.npy"]
    del indptr, indices, data, V, basis, op
    gc.collect()
    return rec


def main():
    outdir = sys.argv[1]
    os.makedirs(outdir, exist_ok=True)
    summ_path = os.path.join(outdir, "sectors.json")
    summary = json.load(open(summ_path)) if os.path.exists(summ_path) else {}
    for spec in sys.argv[2:]:
        f = spec.split(",")
        kx, ky, point, flip = int(f[0]), int(f[1]), f[2], int(f[3])
        nev = int(f[4]) if len(f) > 4 else 1
        save = len(f) > 5 and f[5] == "save"
        rec = run_sector(outdir, kx, ky, point, flip, nev, save)
        summary[rec["tag"]] = rec
        with open(summ_path, "w") as fh:
            json.dump(summary, fh, indent=1)


if __name__ == "__main__":
    main()
