#!/usr/bin/env python3
"""Exact closed FN / Krylov loop in the fully symmetric (k=0, A1, spin-flip +) sector
of the periodic L x L J1-J2 Heisenberg model (L=4 for validation, L=6 for the real run).

Same loop as closed_fn_krylov_anderson.py (plain / Anderson on log a), but every object
lives on symmetry orbit representatives (lattice_symmetries basis, 288x2 elements for L=6).

Conventions (derivation in the notes at the bottom of this docstring)
  v_r  : normalised vector component in the symmetric basis, psi(x) = s_r v_r / sqrt(n_r) for x in orbit(r)
         (n_r = orbit size under space group x spin flip).  Per-configuration amplitude a_x = v_r/sqrt(n_r).
  H_rr': symmetric-basis Hamiltonian (all entries >= 0 off the diagonal, real characters).
  r(x)  = (H_sym (s v))_r / (s_r v_r)                   (constant on orbits, = full-basis local energy)
  H_FN  : K_ij = s_i H_ij s_j in the full basis.  Signs / amplitudes are orbit-constant and H_ij>0, so
          for orbit pair (r,r'): s_r != s_r' -> off-diagonal element -H_rr' (kept);
          s_r == s_r'  -> moved to the diagonal with weight a_x'/a_x, which after summing over the orbit
          of x' is  H_rr' * w_r'/w_r,  w_r = max(a_r,1e-15)*sqrt(n_r)  (= v_r unless floored).
          The r'=r term (self-orbit hops + physical diagonal, both in H_rr) gets weight 1.
  Krylov: s' = s*sgn[T-r], exact energy-optimal threshold over all cuts of the sorted r (equal r grouped);
          E(s') = (s' v)^T H_sym (s' v) exactly.
  Scoring: w_s = sum_r v0_r^2 [s_r != sgn v0_r] (global sign fixed), eps=(E-E0)/|E0|, F_amp=(sum v|v0|)^2.
  Anderson (type II, memory m) on x=log v with sqrt(n_r) weights in the least squares, which makes it
  identical to the full-basis uniform-weight version of closed_fn_krylov_anderson.py.
The target ED ground state is used ONLY for scoring.
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import numba
import scipy.sparse.linalg as sla

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("ED_COMMON_DIR", HERE))
import ed_common as ec  # noqa: E402

NTHR = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
numba.set_num_threads(NTHR)


# ----------------------------------------------------------------------------- kernels
@numba.njit(parallel=True, cache=False)
def k_matvec(indptr, indices, data, x, y):
    n = indptr.shape[0] - 1
    for i in numba.prange(n):
        acc = 0.0
        for p in range(indptr[i], indptr[i + 1]):
            acc += data[p] * x[indices[p]]
        y[i] = acc


@numba.njit(parallel=True, cache=False)
def k_fn_diag(indptr, indices, data, s, w, D):
    n = indptr.shape[0] - 1
    for i in numba.prange(n):
        si = s[i]
        acc = 0.0
        for p in range(indptr[i], indptr[i + 1]):
            j = indices[p]
            if s[j] == si:
                acc += data[p] * w[j]
        D[i] = acc / w[i]


@numba.njit(parallel=True, cache=False)
def k_fn_matvec(indptr, indices, data, s, D, x, y):
    n = indptr.shape[0] - 1
    for i in numba.prange(n):
        si = s[i]
        acc = D[i] * x[i]
        for p in range(indptr[i], indptr[i + 1]):
            j = indices[p]
            if s[j] != si:
                acc -= data[p] * x[j]
        y[i] = acc


@numba.njit(parallel=True, cache=False)
def k_kry(indptr, indices, data, gid, u, delta_blocks, e0_blocks):
    n = indptr.shape[0] - 1
    nb = delta_blocks.shape[0]
    chunk = (n + nb - 1) // nb
    for b in numba.prange(nb):
        lo_i = b * chunk
        hi_i = min(n, lo_i + chunk)
        e = 0.0
        for i in range(lo_i, hi_i):
            gi = gid[i]
            for p in range(indptr[i], indptr[i + 1]):
                j = indices[p]
                if j > i:
                    val = 2.0 * data[p] * u[i] * u[j]
                    e += val
                    gj = gid[j]
                    if gi < gj:
                        delta_blocks[b, gi] -= 2.0 * val
                        delta_blocks[b, gj] += 2.0 * val
                    elif gj < gi:
                        delta_blocks[b, gj] -= 2.0 * val
                        delta_blocks[b, gi] += 2.0 * val
                elif j == i:
                    e += data[p] * u[i] * u[i]
        e0_blocks[b] = e


# ----------------------------------------------------------------------------- basis / H
def make_op(basis, L, J2, cluster=None):
    import lattice_symmetries as ls
    nn, nnn = ec.bonds(L) if cluster is None else (cluster.nn, cluster.nnn)
    terms = [ls.Interaction(ec.HEIS, [list(b) for b in nn])]
    if J2 != 0.0:
        terms.append(ls.Interaction(J2 * ec.HEIS, [list(b) for b in nnn]))
    return ls.Operator(basis, terms)


def build_csr(basis, op, chunk=20000):
    from concurrent.futures import ThreadPoolExecutor
    st = basis.states
    dim = st.shape[0]
    starts = list(range(0, dim, chunk))

    def work(s0):
        sp_, co, cnt = op.batched_apply(st[s0:s0 + chunk])
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


def csr_op(indptr, indices, data):
    dim = indptr.shape[0] - 1
    cnt = [0]

    def mv(x):
        cnt[0] += 1
        x = np.ascontiguousarray(x, dtype=np.float64).reshape(-1)
        y = np.empty(dim)
        k_matvec(indptr, indices, data, x, y)
        return y

    return sla.LinearOperator((dim, dim), matvec=mv, dtype=np.float64), cnt


def ground(op, v0=None, tol=1e-12, ncv=None, maxiter=300000):
    ew, ev = sla.eigsh(op, k=1, which="SA", v0=v0, tol=tol, ncv=ncv, maxiter=maxiter)
    v = np.asarray(ev[:, 0], float)
    return float(ew[0]), v / np.linalg.norm(v)


class Sym:
    """Symmetric-sector data: basis states, orbit sizes, Marshall signs and H (CSR)."""

    def __init__(self, L, J2=0.5, log=print, cache=None, cluster=None):
        self.L = L
        self.cluster = cluster
        self.N = L * L if cluster is None else cluster.N
        self.log = log
        t0 = time.time()
        if cluster is None:
            basis, G = ec.make_basis(L, k=(0, 0), point="A1", flip=1)
        else:
            basis, G = cluster.make_basis(k=(0, 0), flip=1)
        basis.build()
        self.basis = basis
        self.states = np.asarray(basis.states).copy()
        self.D = self.states.shape[0]
        assert np.all(np.diff(self.states.astype(np.int64)) > 0) or self.N < 40
        log(f"[sym] L={L} dim={self.D} group={G}x2 basis {time.time()-t0:.1f}s")
        t0 = time.time()
        self.T = ec.byte_tables(ec.space_group(L) if cluster is None else cluster.group, self.N)
        if cache is not None and os.path.exists(cache):
            z = np.load(cache)
            assert z["n"].shape[0] == self.D
            self.n_orb = z["n"].astype(np.float64)
            log(f"[sym] orbit sizes from cache {cache}")
        else:
            reps, orb = ec.rep_and_orbit(self.states, self.T, self.N)
            assert np.array_equal(reps, self.states), "ls representatives != min representatives"
            self.n_orb = orb.astype(np.float64)
            if cache is not None:
                np.savez(cache, n=orb.astype(np.int16))
            log(f"[sym] orbits {time.time()-t0:.1f}s sum={int(orb.sum())}")
        mask = ec.sublattice_mask(L) if cluster is None else cluster.mask
        self.mask = mask
        pc = np.zeros(self.D, dtype=np.uint8)
        x = self.states & np.uint64(mask)
        for b in range(self.N):
            pc ^= ((x >> np.uint64(b)) & np.uint64(1)).astype(np.uint8)
        self.marshall = np.where(pc == 0, 1, -1).astype(np.int8)
        self.sqrt_n = np.sqrt(self.n_orb)
        self.J2 = J2
        t0 = time.time()
        op = make_op(basis, L, J2, cluster)
        self.indptr, self.indices, self.data = build_csr(basis, op)
        del op
        self.nnz = int(self.indptr[-1])
        log(f"[sym] H built nnz={self.nnz:.4e} {time.time()-t0:.1f}s")
        self.nmv = 0

    def H(self, x):
        y = np.empty(self.D)
        k_matvec(self.indptr, self.indices, self.data, np.ascontiguousarray(x, dtype=np.float64), y)
        self.nmv += 1
        return y

    # ---- amplitude helpers
    def lognorm_v(self, x):
        m = np.max(x)
        return x - (m + 0.5 * np.log(np.sum(np.exp(2.0 * (x - m)))))

    # ---- FN
    def fn_solve(self, v, s, tol, ncv=None, solver="arpack", maxmv=1500):
        """Perron ground state of H_FN[a=v,s] (warm-started from v). Returns E_FN, v_FN (>=0, normalised), nmv.
        solver 'arpack': eigsh(SA), tol = relative residual (ARPACK convention).
        solver 'lobpcg': block-size-1 LOBPCG with the diagonal preconditioner 1/max(D_FN - theta, 1)
          (D_FN has entries up to ~1e3..1e14 for tiny-amplitude configurations, which wrecks plain Lanczos);
          stops at residual norm <= tol*|theta|; falls back to ARPACK (warm-started) if not converged or if the
          returned vector is not a positive (Perron) vector."""
        a = np.maximum(v / self.sqrt_n, 1e-15)
        w = a * self.sqrt_n
        Dg = np.empty(self.D)
        k_fn_diag(self.indptr, self.indices, self.data, s, w, Dg)
        cnt = [0]

        def mv(x):
            cnt[0] += 1
            x = np.ascontiguousarray(x, dtype=np.float64).reshape(-1)
            y = np.empty(self.D)
            k_fn_matvec(self.indptr, self.indices, self.data, s, Dg, x, y)
            return y

        def mvm(X):
            X = np.asarray(X)
            return mv(X) if X.ndim == 1 else np.stack([mv(X[:, i]) for i in range(X.shape[1])], axis=1)

        op = sla.LinearOperator((self.D, self.D), matvec=mv, matmat=mvm, dtype=np.float64)
        self.last_solver = solver
        if solver == "lobpcg":
            w0 = w / np.linalg.norm(w)
            th = float(w0 @ mv(w0))
            den = np.maximum(Dg - th, 1.0)
            Minv = sla.LinearOperator((self.D, self.D), matvec=lambda x: np.asarray(x).reshape(-1) / den,
                                      matmat=lambda X: np.asarray(X) / den[:, None], dtype=np.float64)
            import warnings
            maxit = max(10, maxmv)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lam, V = sla.lobpcg(op, w0[:, None].copy(), M=Minv, largest=False, tol=tol * abs(th), maxiter=maxit)
            u = np.asarray(V[:, 0], float)
            u /= np.linalg.norm(u)
            e = float(lam[0])
            resid = float(np.linalg.norm(mv(u) - e * u))
            negw = float(min(np.sum(u[u < 0] ** 2), np.sum(u[u > 0] ** 2)))
            self.last_resid = resid
            if resid > 5.0 * tol * abs(th) or negw > 1e-8 or not np.isfinite(e):
                self.last_solver = "lobpcg->arpack"
                e, u = ground(op, v0=np.abs(u) + 1e-300 if np.all(np.isfinite(u)) else w, tol=tol, ncv=ncv)
        else:
            e, u = ground(op, v0=w, tol=tol, ncv=ncv)
        u = np.abs(u)
        u /= np.linalg.norm(u)
        return e, u, cnt[0]

    # ---- Krylov
    def krylov(self, v, s):
        """Current-sign projected Krylov update with exact energy-optimal grouped threshold."""
        psi = v * s
        Hpsi = self.H(psi)
        r = Hpsi / np.where(np.abs(psi) > 1e-300, psi, 1.0)
        gid, G = group_ids(r)
        c0 = -s
        u = c0.astype(np.float64) * v
        nb = NTHR
        delta_blocks = np.zeros((nb, G))
        e0_blocks = np.zeros(nb)
        k_kry(self.indptr, self.indices, self.data, gid, u, delta_blocks, e0_blocks)
        e0 = float(e0_blocks.sum())
        delta = delta_blocks.sum(axis=0)
        del delta_blocks
        cand = np.r_[e0, e0 + np.cumsum(delta)]
        kbest = int(np.argmin(cand)) - 1
        sn = c0.copy()
        if kbest >= 0:
            sn[gid <= kbest] *= -1
        return canonical(sn), float(np.min(cand)), G, r


def canonical(s):
    s = np.asarray(s, dtype=np.int8).copy()
    if s[0] < 0:
        s = -s
    return s


def group_ids(r, atol=1e-11, rtol=1e-11):
    D = len(r)
    order = np.argsort(r, kind="mergesort")
    rs = r[order]
    scale = np.maximum(np.maximum(np.abs(rs[1:]), np.abs(rs[:-1])), 1.0)
    br = np.empty(D, dtype=bool)
    br[0] = True
    br[1:] = np.abs(rs[1:] - rs[:-1]) > (atol + rtol * scale)
    gs = np.cumsum(br, dtype=np.int32) - 1
    gid = np.empty(D, dtype=np.int32)
    gid[order] = gs
    return gid, int(gs[-1]) + 1


class Anderson:
    def __init__(self, m, beta):
        self.m = m
        self.beta = beta
        self.X = []
        self.F = []

    def reset(self):
        self.X = []
        self.F = []

    def step(self, x, f, sqrtw=None):
        self.X.append(x.copy())
        self.F.append(f.copy())
        if len(self.X) > self.m + 1:
            self.X.pop(0)
            self.F.pop(0)
        if len(self.X) == 1:
            return x + self.beta * f, 0, None
        dX = np.stack([self.X[i + 1] - self.X[i] for i in range(len(self.X) - 1)], axis=1)
        dF = np.stack([self.F[i + 1] - self.F[i] for i in range(len(self.F) - 1)], axis=1)
        if sqrtw is not None:
            A = dF * sqrtw[:, None]
            b = f * sqrtw
        else:
            A = dF
            b = f
        gam, *_ = np.linalg.lstsq(A, b, rcond=1e-12)
        xn = x + self.beta * f - (dX + self.beta * dF) @ gam
        return xn, dX.shape[1], gam


# ----------------------------------------------------------------------------- init / oracle
def init_amplitude(S, init_j2, tol, log=print, cache=None):
    """|ground state| of the J2=init_j2 Hamiltonian in this sector (oracle-free initial amplitude)."""
    if cache is not None and os.path.exists(cache):
        z = np.load(cache)
        log(f"[init] loaded {cache} E={float(z['E'])}")
        return np.abs(z["v"]), float(z["E"])
    t0 = time.time()
    op = make_op(S.basis, S.L, init_j2, S.cluster)
    ip, ix, da = build_csr(S.basis, op)
    del op
    lop, cnt = csr_op(ip, ix, da)
    rng = np.random.default_rng(7)
    E, v = ground(lop, v0=rng.standard_normal(S.D), tol=tol)
    log(f"[init] J2={init_j2} E={E:.12f} matvecs={cnt[0]} {time.time()-t0:.1f}s")
    del ip, ix, da, lop
    if cache is not None:
        np.savez(cache, v=v, E=E)
    return np.abs(v), E


def target_gs(S, args, log=print):
    if args.gs_vec:
        v0 = np.load(args.gs_vec)
        if v0.ndim != 1:
            v0 = v0[:, 0]
        assert v0.shape[0] == S.D
        if args.gs_states:
            st = np.load(args.gs_states)
            assert np.array_equal(st, S.states), "saved GS basis ordering differs"
        v0 = np.asarray(v0, float)
        v0 /= np.linalg.norm(v0)
        E0 = float(v0 @ S.H(v0))
        log(f"[oracle] loaded GS vector, Rayleigh E0={E0:.12f}")
        return E0, v0
    lop, cnt = csr_op(S.indptr, S.indices, S.data)
    E0, v0 = ground(lop, tol=args.ed_tol)
    log(f"[oracle] ED E0={E0:.12f} matvecs={cnt[0]}")
    return E0, v0


# ----------------------------------------------------------------------------- loop
def save_ckpt(path, it, v, s, x, efn_prev, nreset, acc, hist):
    tmp = path + ".tmp.npz"
    arrs = dict(it=it, v=v, s=s, x=x, efn_prev=efn_prev, nreset=nreset,
                nX=0 if acc is None else len(acc.X), hist=json.dumps(hist))
    if acc is not None:
        for i, (a, b) in enumerate(zip(acc.X, acc.F)):
            arrs[f"X{i}"] = a
            arrs[f"F{i}"] = b
    np.savez(tmp, **arrs)
    os.replace(tmp, path)


def run(args):
    t00 = time.time()
    log = lambda *a: print(*a, flush=True)
    log("threads", NTHR, "args", vars(args))
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = args.tag
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cluster = None
    ctag = f"L{args.L}"
    if args.torus:
        import torus_cluster
        tv = [int(z) for z in args.torus.split(",")]
        cluster = torus_cluster.Cluster((tv[0], tv[1]), (tv[2], tv[3]))
        ctag = f"T{'_'.join(str(z) for z in tv)}".replace("-", "m")
        log(f"[cluster] torus T1={cluster.t1} T2={cluster.t2} N={cluster.N} point gens={[k for k,_ in cluster.point_gens]} group={len(cluster.group)}")
    S = Sym(args.L, args.target_j2, log=log, cache=str(cache_dir / f"orb_{ctag}.npz"), cluster=cluster)
    E0, v0 = target_gs(S, args, log)
    a0abs = np.abs(v0)
    sgn0 = np.where(v0 >= 0, 1, -1).astype(np.int8)
    p0 = v0 * v0
    vinit, Einit = init_amplitude(S, args.init_source_j2, args.ed_tol, log,
                                  cache=str(cache_dir / f"init_{ctag}_J2{args.init_source_j2}.npz"))
    vinit = vinit / np.linalg.norm(vinit)
    # sanity: Marshall signs are symmetric (consistent gauge) -- spot check on random full configs
    sM = S.marshall.copy()
    chk = np.random.default_rng(3).integers(0, S.D, size=min(2000, S.D))
    img = ec.all_images(S.states[chk], S.T, S.N)
    mask = S.mask
    par = np.zeros(img.shape, dtype=np.uint8)
    xx = img & np.uint64(mask)
    for b in range(S.N):
        par ^= ((xx >> np.uint64(b)) & np.uint64(1)).astype(np.uint8)
    assert np.all(par == (S.marshall[chk] < 0)[:, None].astype(np.uint8)), "Marshall sign not orbit-constant"
    log("[check] Marshall sign constant on orbits (spot check)")

    def score(v, s):
        psi = v * s
        E = float(psi @ S.H(psi) / (psi @ psi))
        O = abs(float(np.sum(p0 * s * sgn0)))
        return dict(E_trial=E, eps=(E - E0) / abs(E0), w_s=max(0.0, (1 - O) / 2),
                    F_amp=float(abs(np.dot(v, a0abs)) ** 2))

    acc = Anderson(args.m, args.beta) if args.method == "anderson" else None
    ckpt = str(outdir / f"ckpt_{tag}.npz")
    hist_path = outdir / f"{tag}_history.jsonl"
    if args.resume and os.path.exists(ckpt):
        z = np.load(ckpt, allow_pickle=False)
        it0 = int(z["it"]); v = z["v"]; s = z["s"]; x = z["x"]
        efn_prev = float(z["efn_prev"]); nreset = int(z["nreset"])
        hist = json.loads(str(z["hist"]))
        if acc is not None:
            for i in range(int(z["nX"])):
                acc.X.append(z[f"X{i}"]); acc.F.append(z[f"F{i}"])
        log(f"[resume] it={it0}")
    else:
        v = vinit.copy(); s = sM.copy()
        s = canonical(s)
        x = S.lognorm_v(np.log(np.maximum(v, 1e-300)))
        hist = [dict(it=0, **score(v, s))]
        log("ITER 0", json.dumps(hist[-1]))
        efn_prev = np.inf; nreset = 0; it0 = 0
        hist_path.write_text(json.dumps(hist[-1]) + "\n")
    terminal = "maxiter"
    nfull = S.n_orb
    for it in range(it0 + 1, args.maxiter + 1):
        if time.time() - t00 > args.time_budget:
            terminal = "time_budget"; break
        tt = time.time()
        sold = s.copy()
        nm0 = S.nmv
        tfn = time.time()
        efn, afn, nmv_fn = S.fn_solve(v, s, args.fn_tol, args.ncv, solver=args.solver)
        tfn = time.time() - tfn
        g = S.lognorm_v(np.log(np.maximum(afn, 1e-300)))
        f = g - x
        info = dict(E_FN=efn, eps_FN=(efn - E0) / abs(E0),
                    res_log_l2=float(np.sqrt(np.sum(nfull * f * f))),
                    res_log_wl2=float(np.sqrt(np.sum(afn ** 2 * f ** 2))),
                    res_log_inf=float(np.max(np.abs(f))))
        reset = False; reason = None
        if args.method != "plain" and efn > efn_prev + args.e_tol * abs(efn):
            reset = True; reason = "E_FN_up"
        if args.method == "plain" or reset:
            xn = g; nhist = 0
            if acc is not None: acc.reset()
        else:
            xn, nhist, gam = acc.step(x, f, S.sqrt_n)
        if args.method != "plain" and not reset:
            jump = float(np.max(np.abs(xn - g))) if np.all(np.isfinite(xn)) else np.inf
            if not np.isfinite(jump) or jump > args.max_jump:
                reset = True; reason = f"jump>{args.max_jump}" if np.isfinite(jump) else "nonfinite"
                xn = g; nhist = 0
                if acc is not None: acc.reset()
        if reset: nreset += 1
        xn = S.lognorm_v(xn)
        an = np.exp(xn); an /= np.linalg.norm(an)
        kamp = an if args.krylov_amp == "accel" else afn
        tk = time.time()
        snew, eth, G, _ = S.krylov(kamp, sold)
        tk = time.time() - tk
        efn_prev = efn
        amp_delta = float(np.linalg.norm(an - v))
        x = xn; v = an; s = snew
        sc = score(v, s)
        rec = dict(it=it, **sc, **info, n_hist=int(nhist), reset=reason, amp_delta_l2=amp_delta,
                   n_sign_flips=int(np.sum(nfull[s != sold])), r_groups=G, min_amp=float(np.min(v / S.sqrt_n)),
                   fn_matvecs=nmv_fn, fn_solver=S.last_solver, sec_fn=tfn, sec_kry=tk, sec=time.time() - tt)
        hist.append(rec)
        with open(hist_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        log("ITER", it, json.dumps(rec))
        if it % args.ckpt_every == 0:
            save_ckpt(ckpt, it, v, s, x, efn_prev, nreset, acc, hist)
        if rec["w_s"] < 1e-14 and rec["eps"] < args.stop_eps:
            terminal = f"converged_eps<{args.stop_eps}"; break
        if np.array_equal(s, sold) and amp_delta < 1e-12:
            terminal = "fixed_pair"; break
    save_ckpt(ckpt, hist[-1]["it"], v, s, x, efn_prev, nreset, acc, hist)

    def first(pred):
        for r in hist:
            if pred(r): return r["it"]
        return None

    def settled(pred):
        k = None
        for r in reversed(hist):
            if pred(r): k = r["it"]
            else: break
        return k

    summary = dict(
        it_exact_signs_first=first(lambda r: r["w_s"] < 1e-14),
        it_exact_signs_settled=settled(lambda r: r["w_s"] < 1e-14),
        it_eps_1e5=first(lambda r: r["eps"] < 1e-5),
        it_eps_1e6=first(lambda r: r["eps"] < 1e-6),
        it_eps_1e7=first(lambda r: r["eps"] < 1e-7),
        n_resets=nreset, n_iter=hist[-1]["it"], final_eps=hist[-1]["eps"], final_w_s=hist[-1]["w_s"],
        terminal=terminal)
    out = dict(method=dict(lattice=(f"{args.L}x{args.L} periodic" if cluster is None else f"torus T1={cluster.t1} T2={cluster.t2}") + ", symmetric sector k=0,A1,flip+",
                           D=S.D, target_J2=args.target_j2, init_amplitude_source_J2=args.init_source_j2,
                           initial_signs="Marshall", accel=args.method, m=args.m, beta=args.beta,
                           krylov_amp=args.krylov_amp, e_tol=args.e_tol, max_jump=args.max_jump,
                           fn_tol=args.fn_tol, ncv=args.ncv, solver=args.solver, nnz=S.nnz,
                           oracle_policy="target ED only for scoring (w_s, eps, F_amp)"),
               E0=E0, E_init_source=Einit, summary=summary, elapsed_sec=time.time() - t00, history=hist)
    (outdir / f"{tag}.json").write_text(json.dumps(out, indent=1))
    log("SUMMARY", json.dumps(summary))
    log("WROTE", str(outdir / f"{tag}.json"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--torus", default=None, help="t1x,t1y,t2x,t2y: tilted torus instead of LxL (needs torus_cluster.py)")
    ap.add_argument("--method", choices=["plain", "anderson"], default="plain")
    ap.add_argument("--m", type=int, default=3)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--krylov-amp", choices=["accel", "fn"], default="accel")
    ap.add_argument("--e-tol", type=float, default=1e-10)
    ap.add_argument("--max-jump", type=float, default=5.0)
    ap.add_argument("--fn-tol", type=float, default=2e-11)
    ap.add_argument("--ncv", type=int, default=None)
    ap.add_argument("--solver", choices=["arpack", "lobpcg"], default="arpack")
    ap.add_argument("--ed-tol", type=float, default=1e-12)
    ap.add_argument("--target-j2", type=float, default=0.5)
    ap.add_argument("--init-source-j2", type=float, default=0.0)
    ap.add_argument("--maxiter", type=int, default=150)
    ap.add_argument("--stop-eps", type=float, default=1e-8)
    ap.add_argument("--gs-vec", default=None, help="npy with target GS comps in this basis ordering (scoring only)")
    ap.add_argument("--gs-states", default=None, help="npy with the states array of that GS (ordering check)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--ckpt-every", type=int, default=10)
    ap.add_argument("--time-budget", type=float, default=1e18, help="seconds; stop cleanly (checkpointed) afterwards")
    run(ap.parse_args())
