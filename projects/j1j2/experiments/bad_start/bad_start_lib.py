#!/usr/bin/env python3
"""Bad-start tests for the exact FN/Krylov loop (J1-J2, Sz=0 full basis).

Re-uses the building blocks of
  krylov_sign_structure/experiments/closed_fn_krylov_anderson.py
(Model, fixed_node_solve, projected_krylov_update, Anderson, lognorm); the only change w.r.t.
its run() is the initial guide (a0, s0), which is passed in, and the per-guide metrics.
Target ED ground state is used ONLY for scoring.

The full Sz=0 basis (4x4: 12870, 20 sites: 184756) is used, with NO symmetry restriction in the
loop; symmetry sectors are used only to construct the starting guides (penalty-method
diagonalisation in a sector) and to measure symmetry expectation values.
"""
import sys, json, time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "krylov_sign_structure" / "experiments"))
import closed_fn_krylov_anderson as A  # noqa: E402


# ----------------------------------------------------------------------------- lattice symmetries
def red_fn(lattice):
    if lattice == "4x4":
        return 4, 4, (lambda x, y: (x % 4) + 4 * (y % 4))
    LX, LY = 4, 5
    def red(x, y):
        q = y // LY; y = y - q * LY; x = x - q
        return (x % LX) + LX * y
    return LX, LY, red


def site_perm(lattice, op):
    LX, LY, red = red_fn(lattice)
    p = np.zeros(LX * LY, dtype=np.int64)
    for y in range(LY):
        for x in range(LX):
            i = red(x, y)
            if op == "Tx": j = red(x + 1, y)
            elif op == "Ty": j = red(x, y + 1)
            elif op == "rot": j = red(-y, x)       # C4 about a site (4x4 only)
            elif op == "mir": j = red(y, x)        # diagonal mirror (4x4 only)
            elif op == "inv": j = red(-x, -y)      # inversion
            else: raise ValueError(op)
            p[i] = j
    return p


class Sym:
    """Permutation / spin-flip operators on the Sz=0 basis."""
    def __init__(self, model, lattice):
        self.m = model; self.lattice = lattice
        self.cache = {}
        self.S2 = None

    def pidx(self, op):
        if op in self.cache: return self.cache[op]
        m = self.m; codes = m.basis.astype(np.uint64)
        if op == "flip":
            new = (~codes) & np.uint64((1 << m.N) - 1)
        else:
            sp_ = site_perm(self.lattice, op)
            new = np.zeros_like(codes)
            for i in range(m.N):
                new |= ((codes >> np.uint64(i)) & np.uint64(1)) << np.uint64(sp_[i])
        idx = m.state_to_idx[new.astype(np.int64)].astype(np.int64)
        assert (idx >= 0).all()
        self.cache[op] = idx
        return idx

    def U(self, op, v):
        out = np.empty_like(v); out[self.pidx(op)] = v; return out

    def Ut(self, op, v):
        return v[self.pidx(op)]

    def expect(self, op, v):
        # <v|U|v>/<v|v>  (real part of the character for real v)
        return float(v @ self.U(op, v) / (v @ v))

    def build_S2(self):
        if self.S2 is not None: return self.S2
        m = self.m; N = m.N; D = m.D; basis = m.basis; rows = np.arange(D, dtype=np.int32)
        diag = np.full(D, 0.75 * N)
        rr = []; cc = []; vv = []
        for u in range(N):
            for v in range(u + 1, N):
                bu = (basis >> u) & 1; bv = (basis >> v) & 1; same = bu == bv
                diag += np.where(same, 0.5, -0.5)
                sel = ~same
                t = basis[sel] ^ (np.uint32(1 << u) | np.uint32(1 << v))
                rr.append(rows[sel]); cc.append(m.state_to_idx[t]); vv.append(np.ones(sel.sum()))
        rr = np.concatenate(rr); cc = np.concatenate(cc); vv = np.concatenate(vv)
        S2 = sp.coo_matrix((np.r_[vv, diag], (np.r_[rr, rows], np.r_[cc, rows])), shape=(D, D)).tocsr()
        self.S2 = S2
        return S2


SECTORS = {
    "4x4": {
        "A1":    dict(gens=dict(Tx=1, Ty=1, rot=1, mir=1, flip=1), s2=True),
        "S1":    dict(gens=dict(Tx=1, Ty=1, rot=1, mir=1, flip=-1), s2=False),
        "kPiPi": dict(gens=dict(Tx=-1, Ty=-1, rot=1, mir=1, flip=1), s2=True),
        "Brot":  dict(gens=dict(Tx=1, Ty=1, rot=-1, mir=1, flip=1), s2=True),
    },
    "20": {
        "A1":    dict(gens=dict(Tx=1, Ty=1, inv=1, flip=1), s2=True),
        "S1":    dict(gens=dict(Tx=1, Ty=1, inv=1, flip=-1), s2=False),
        "kPiPi": dict(gens=dict(Tx=-1, Ty=-1, inv=1, flip=1), s2=True),
        "Iodd":  dict(gens=dict(Tx=1, Ty=1, inv=-1, flip=1), s2=True),
    },
}


def sector_states(model, sym, H, spec, k, cP=20.0, cS=20.0, tol=0.0):
    """Lowest k eigenpairs of H restricted to the sector, via penalty method."""
    D = model.D
    gens = [(sym.pidx(o), chi) for o, chi in spec["gens"].items()]
    S2 = sym.build_S2() if spec["s2"] else None
    def mv(v):
        v = np.asarray(v, float).ravel()
        out = H @ v
        if S2 is not None: out = out + cS * (S2 @ v)
        for pidx, chi in gens:
            Uv = np.empty_like(v); Uv[pidx] = v
            out = out + cP * (v - chi * 0.5 * (Uv + v[pidx]))
        return out
    op = sla.LinearOperator((D, D), matvec=mv, dtype=float)
    rng = np.random.default_rng(7)
    ew, ev = sla.eigsh(op, k=k, which="SA", tol=tol, ncv=max(40, 4 * k), maxiter=200000,
                       v0=rng.standard_normal(D))
    o = np.argsort(ew); ew = ew[o]; ev = ev[:, o]
    out = []
    for i in range(k):
        v = ev[:, i] / np.linalg.norm(ev[:, i])
        E = float(v @ (H @ v)); res = float(np.linalg.norm(H @ v - E * v))
        info = dict(E=E, resid=res, S2=float(v @ (sym.build_S2() @ v)),
                    chars={o_: sym.expect(o_, v) for o_ in spec["gens"]})
        out.append((v, info))
    return out


def project_sector(sym, spec, v, sweeps=80):
    """Project v onto the joint chi-eigenspace of the sector generators (alternating projections
    over the cyclic groups generated by each generator; converges to the intersection)."""
    v = np.asarray(v, float).copy()
    cyc = []
    for op, chi in spec["gens"].items():
        p = sym.pidx(op); ident = np.arange(len(p)); order = 1; q = p.copy()
        while not np.array_equal(q, ident):
            q = p[q]; order += 1
        cyc.append((op, chi, order))
    for _ in range(sweeps):
        for op, chi, order in cyc:
            acc = np.zeros_like(v); w = v.copy()
            for k in range(order):
                acc += (chi ** k) * w
                w = sym.U(op, w)
            v = acc / order
    return v / np.linalg.norm(v)


def sgn(v, tiny=0.0):
    return np.where(v >= -tiny, 1, -1).astype(np.int8)


def guide_from_psi(psi):
    psi = np.asarray(psi, float); psi = psi / np.linalg.norm(psi)
    return np.abs(psi), A.canonical(sgn(psi))


# ----------------------------------------------------------------------------- the loop
def run_loop(model, H, diag, ei, ej, hij, a0, s0, E0, refs, method="plain", maxiter=300,
             stop_eps=1e-9, m=5, beta=1.0, e_tol=1e-10, max_jump=5.0, fn_tol=None,
             sym=None, sym_ops=(), verbose=False, stall_window=300, mask_tol=1e-12):
    """Plain / Anderson FN-Krylov loop from guide (a0, s0).  refs: dict name -> unit vector
    (first entry must be the ED ground state 'gs').  Records per guide k: E_FN[a_k,s_k], <H>_guide,
    wrong-sign probability vs ED GS, overlaps with refs, optional symmetry expectation values."""
    D = model.D
    if fn_tol is None: fn_tol = 2e-11 if model.N == 16 else 2e-10
    gs = refs["gs"]; ptrue = gs ** 2; strue = A.canonical(np.where(gs >= 0, 1, -1).astype(np.int8))
    a = np.asarray(a0, float) / np.linalg.norm(a0); s = A.canonical(s0)
    x = A.lognorm(np.log(np.maximum(a, 1e-300)))
    acc = A.Anderson(m, beta) if method == "anderson" else None

    def score(a, s):
        psi = a * s; n2 = float(psi @ psi)
        Eg = float(psi @ (H @ psi) / n2)
        O = abs(float(np.sum(ptrue * s * strue)))
        r = dict(E_guide=Eg, eps_guide=(Eg - E0) / abs(E0), w_s=max(0.0, (1 - O) / 2))
        for nm, v in refs.items():
            r["ov_" + nm] = float((psi @ v) ** 2 / n2)
        if sym is not None:
            for o in sym_ops:
                r["sym_" + o] = sym.expect(o, psi)
            if sym.S2 is not None:
                r["S2"] = float(psi @ (sym.S2 @ psi) / n2)
        return r

    # Guides with symmetry-enforced exact zeros (other-sector eigenstates): the lattice-FN diagonal
    # correction k_ij a_j/a_i with a_i -> 1e-15 reaches 1e15 and ARPACK does not converge.  Those
    # configurations are inert (amplitude ~1e-15 in the original code), so the FN eigenproblem is solved on
    # the active subspace (a > mask_tol) and the inactive amplitudes are set to 1e-15.  Only used if the
    # STARTING guide has such zeros (use_mask); all other runs use A.fixed_node_solve unchanged.
    use_mask = bool(a.min() < mask_tol)

    def fn_solve(a_, s_):
        act = a_ > mask_tol
        if (not use_mask) or act.all():
            return A.fixed_node_solve(H, diag, ei, ej, hij, a_, s_, fn_tol)
        remap = -np.ones(D, dtype=np.int64); remap[act] = np.arange(int(act.sum()))
        keep = act[ei] & act[ej]
        e_, v_ = A.fixed_node_solve(None, diag[act], remap[ei[keep]].astype(np.int32), remap[ej[keep]].astype(np.int32),
                                    hij[keep], a_[act], s_[act], fn_tol)
        full = np.full(D, 1e-15); full[act] = v_
        return e_, full / np.linalg.norm(full)

    hist = []; efn_prev = np.inf; nreset = 0; terminal = "maxiter"
    t0 = time.time()
    for it in range(0, maxiter + 1):
        sold = s.copy()
        efn, afn = fn_solve(a, s)
        rec = dict(it=it, E_FN=efn, eps_FN=(efn - E0) / abs(E0), **score(a, s))
        # converged?
        if rec["eps_FN"] < stop_eps and rec["w_s"] < 1e-13 and rec["eps_guide"] < stop_eps:
            hist.append(rec); terminal = f"converged_eps<{stop_eps}"; break
        if it == maxiter:
            hist.append(rec); break
        g = A.lognorm(np.log(np.maximum(afn, 1e-300))); f = g - x
        reset = False
        if method != "plain" and efn > efn_prev + e_tol * abs(efn): reset = True
        if method == "plain" or reset:
            xn = g
            if acc is not None: acc.reset()
        else:
            xn, nh, gam = acc.step(x, f, None)
        if method != "plain" and not reset:
            jump = float(np.max(np.abs(xn - g))) if np.all(np.isfinite(xn)) else np.inf
            if not np.isfinite(jump) or jump > max_jump:
                reset = True; xn = g
                if acc is not None: acc.reset()
        if reset: nreset += 1
        xn = A.lognorm(xn); an = np.exp(xn); an /= np.linalg.norm(an)
        snew, eth, G = A.projected_krylov_update(H, diag, ei, ej, hij, an, sold)
        rec["amp_delta_l2"] = float(np.linalg.norm(an - a))
        rec["n_sign_flips"] = int(np.sum(snew != sold)); rec["reset"] = bool(reset)
        efn_prev = efn
        hist.append(rec)
        if verbose and (it % 10 == 0 or it < 5):
            print(f"  it {it:4d} E_FN eps {rec['eps_FN']:.3e} guide eps {rec['eps_guide']:.3e} "
                  f"w_s {rec['w_s']:.3e} ov_gs {rec['ov_gs']:.4f} flips {rec['n_sign_flips']} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        x = xn; a = an; s = snew
        # stall: E_FN and guide frozen for a long window
        if it >= stall_window:
            e_old = hist[it - stall_window]["E_FN"]
            if abs(efn - e_old) < 1e-14 * abs(E0) and rec["amp_delta_l2"] < 1e-11:
                terminal = f"stalled(frozen for {stall_window} it)"
                hist.append(dict(it=it + 1, E_FN=efn, eps_FN=(efn - E0) / abs(E0), **score(a, s)))
                break
    run_loop.last = (a.copy(), s.copy())
    return hist, terminal, nreset


def first_it(hist, key, thr):
    for r in hist:
        if r[key] < thr: return r["it"]
    return None
