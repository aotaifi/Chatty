#!/usr/bin/env python3
"""Run the exact FN/Krylov loop from bad starting guides (see bad_start_lib.py)."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse, json, time, sys
from pathlib import Path
import multiprocessing as mp
import numpy as np
import bad_start_lib as B
A = B.A

ROOT = B.HERE.parents[1]
RESDIR = ROOT / "results" / "bad_start"


def rnd(o):
    if isinstance(o, float): return float(f"{o:.6g}")
    if isinstance(o, dict): return {k: rnd(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [rnd(v) for v in o]
    return o


def setup(lattice, cache_dir):
    t0 = time.time()
    M = A.Model(lattice)
    H, diag, ei, ej, hij = M.build_H(0.5)
    sym = B.Sym(M, lattice)
    cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    cf = cache_dir / f"sectors_{lattice}.npz"; ci = cache_dir / f"sectors_{lattice}.json"
    # sanity: symmetries commute with H
    rng = np.random.default_rng(1); v = rng.standard_normal(M.D)
    for o in B.SECTORS[lattice]["A1"]["gens"]:
        c = np.linalg.norm(sym.U(o, H @ v) - H @ sym.U(o, v)) / np.linalg.norm(H @ v)
        assert c < 1e-12, (o, c)
    if cf.exists():
        z = np.load(cf); info = json.loads(ci.read_text())
        vecs = {k: z[k] for k in z.files}
    else:
        vecs = {}; info = {}
        for name, spec in B.SECTORS[lattice].items():
            k = 5 if name == "A1" else 2
            res = B.sector_states(M, sym, H, spec, k)
            for i, (v_, inf) in enumerate(res):
                vecs[f"{name}_{i}"] = v_; info[f"{name}_{i}"] = inf
            print(name, [round(inf["E"], 8) for _, inf in res], flush=True)
        np.savez(cf, **vecs); ci.write_text(json.dumps(info, indent=1))
    # gauge: positive overlap with Marshall
    sM = A.canonical(M.marshall_signs()).astype(float)
    for k in list(vecs):
        if vecs[k] @ sM < 0: vecs[k] = -vecs[k]
    return M, H, diag, ei, ej, hij, sym, vecs, info


def make_starts(M, H, vecs, lattice):
    sec_other = [k for k in B.SECTORS[lattice] if k != "A1"]
    p0, p1, p2 = vecs["A1_0"], vecs["A1_1"], vecs["A1_2"]
    S = {}
    sM = A.canonical(M.marshall_signs())
    # (a) baseline: J2=0 amplitude + Marshall
    H0, *_ = M.build_H(0.0)
    _, psi_init = A.ground(H0, tol=1e-12)
    if psi_init @ sM < 0: psi_init = -psi_init
    S["a_base_J2zero_Marshall"] = (np.abs(psi_init), sM, "J2=0 amplitude + Marshall signs")
    # (b) pure excited states in the GS sector
    S["b_ex1"] = (*B.guide_from_psi(p1), "1st excited state, same sector")
    S["b_ex2"] = (*B.guide_from_psi(p2), "2nd excited state, same sector")
    # (c) mixtures with ground-state weight w
    for w in (0.5, 0.1, 0.01, 1e-4, 1e-8):
        S[f"c_mix1_w{w:g}"] = (*B.guide_from_psi(np.sqrt(w) * p0 + np.sqrt(1 - w) * p1),
                               f"sqrt(w) phi0 + sqrt(1-w) phi1, w={w:g}")
    for w in (0.5, 0.01):
        S[f"c_mix1m_w{w:g}"] = (*B.guide_from_psi(np.sqrt(w) * p0 - np.sqrt(1 - w) * p1),
                                f"sqrt(w) phi0 - sqrt(1-w) phi1, w={w:g}")
        S[f"c_mix2_w{w:g}"] = (*B.guide_from_psi(np.sqrt(w) * p0 + np.sqrt(1 - w) * p2),
                               f"sqrt(w) phi0 + sqrt(1-w) phi2, w={w:g}")
    # (d) lowest state of other symmetry sectors (pure) and mixed with the ground state
    for sec in sec_other:
        q = vecs[f"{sec}_0"]
        S[f"d_{sec}"] = (*B.guide_from_psi(q), f"lowest state of sector {sec}")
        for w in (0.5, 0.01):
            S[f"d_mix_{sec}_w{w:g}"] = (*B.guide_from_psi(np.sqrt(w) * p0 + np.sqrt(1 - w) * q),
                                        f"sqrt(w) phi0 + sqrt(1-w) [{sec} state], w={w:g}")
    # (f) extra: weak cross-sector mixtures and random-noise perturbations of eigenstates (escape rate)
    for sec in sec_other:
        q = vecs[f"{sec}_0"]
        for w in (1e-4, 1e-8):
            S[f"d_mix_{sec}_w{w:g}"] = (*B.guide_from_psi(np.sqrt(w) * p0 + np.sqrt(1 - w) * q),
                                        f"sqrt(w) phi0 + sqrt(1-w) [{sec} state], w={w:g}")
    for nm, q in [("ex1", p1)] + [(sec, vecs[f"{sec}_0"]) for sec in sec_other]:
        for eta in (1e-2, 1e-4):
            r = np.random.default_rng(55).standard_normal(M.D); r /= np.linalg.norm(r)
            S[f"f_noise_{nm}_eta{eta:g}"] = (*B.guide_from_psi(q + eta * r),
                                             f"[{nm} eigenstate] + {eta:g} x random unit vector")
    # (g) symmetric perturbations: eigenstate + eta x random vector projected onto ITS OWN sector
    symm = B.Sym(M, lattice)
    for nm, q, sec in [("ex1", p1, "A1")] + [(sec_, vecs[f"{sec_}_0"], sec_) for sec_ in sec_other]:
        for eta in (0.1, 0.5):
            r = np.random.default_rng(77).standard_normal(M.D)
            r = B.project_sector(symm, B.SECTORS[lattice][sec], r)
            S[f"g_symnoise_{nm}_eta{eta:g}"] = (*B.guide_from_psi(q + eta * r),
                                                f"[{nm} eigenstate] + {eta:g} x random vector projected on its own sector")
    # (h) coordinator request: w=1e-6 admixture; random multiplicative perturbation of the amplitude of
    #     phi_1 (log|a| -> log|a| + sigma*g, signs untouched, no explicit phi_0 component)
    S["c_mix1_w1e-06"] = (*B.guide_from_psi(np.sqrt(1e-6) * p0 + np.sqrt(1 - 1e-6) * p1),
                          "sqrt(w) phi0 + sqrt(1-w) phi1, w=1e-06")
    a1, s1 = B.guide_from_psi(p1)
    for sig in (1e-3, 1e-2):
        g = np.random.default_rng(31).standard_normal(M.D)
        an = a1 * np.exp(sig * g); an /= np.linalg.norm(an)
        S[f"h_logamp_ex1_sigma{sig:g}"] = (an, s1, f"phi1 signs, log|a| perturbed by N(0,{sig:g}^2) per config")
    # (e) random signs
    a0 = np.abs(p0)
    for seed in range(3):
        rng = np.random.default_rng(100 + seed)
        sr = A.canonical(np.where(rng.random(M.D) < 0.5, 1, -1).astype(np.int8))
        S[f"e_randsign_exactamp_s{seed}"] = (a0 / np.linalg.norm(a0), sr, "exact |phi0| amplitude, random signs")
        S[f"e_randsign_uniform_s{seed}"] = (np.full(M.D, 1 / np.sqrt(M.D)), sr, "uniform amplitude, random signs")
    S["e_uniform_Marshall"] = (np.full(M.D, 1 / np.sqrt(M.D)), sM, "uniform amplitude + Marshall (reference)")
    return S


_G = {}


def worker(args):
    try:
        return _worker(args)
    except Exception as e:
        print('FAILED', args[2], type(e).__name__, str(e)[:200], flush=True)
        return None


def _worker(args):
    lattice, method, name, maxiter, cache_dir = args
    if "setup" not in _G:
        _G["setup"] = setup(lattice, cache_dir)
    M, H, diag, ei, ej, hij, sym, vecs, info = _G["setup"]
    starts = make_starts(M, H, vecs, lattice)
    a0, s0, desc = starts[name]
    E0 = info["A1_0"]["E"]
    refs = dict(gs=vecs["A1_0"], ex1=vecs["A1_1"], ex2=vecs["A1_2"])
    for sec in B.SECTORS[lattice]:
        if sec != "A1": refs["sec_" + sec] = vecs[f"{sec}_0"]
    ops = list(B.SECTORS[lattice]["A1"]["gens"])
    t0 = time.time()
    hist, term, nreset = B.run_loop(M, H, diag, ei, ej, hij, a0, s0, E0, refs, method=method,
                                    maxiter=maxiter, sym=sym, sym_ops=ops,
                                    stall_window=int(os.environ.get('BAD_START_STALL', 300)))
    summ = dict(
        terminal=term, n_iter=hist[-1]["it"], final_eps_FN=hist[-1]["eps_FN"],
        final_eps_guide=hist[-1]["eps_guide"], final_w_s=hist[-1]["w_s"],
        final_ov_gs=hist[-1]["ov_gs"], final_E_FN=hist[-1]["E_FN"],
        it_epsFN_1e2=B.first_it(hist, "eps_FN", 1e-2), it_epsFN_1e4=B.first_it(hist, "eps_FN", 1e-4),
        it_epsFN_1e6=B.first_it(hist, "eps_FN", 1e-6), it_epsFN_1e9=B.first_it(hist, "eps_FN", 1e-9),
        it_epsguide_1e6=B.first_it(hist, "eps_guide", 1e-6),
        it_ws_1e8=B.first_it(hist, "w_s", 1e-8), n_resets=nreset, sec=time.time() - t0)
    out = dict(name=name, desc=desc, lattice=lattice, method=method, E0=E0,
               start=hist[0], summary=summ, history=hist)
    print(f"[{lattice} {method}] {name}: {term} n_iter={summ['n_iter']} eps_FN={summ['final_eps_FN']:.2e} "
          f"w_s={summ['final_w_s']:.2e} ov_gs={summ['final_ov_gs']:.4f} ({summ['sec']:.0f}s)", flush=True)
    (Path(cache_dir) / "parts").mkdir(exist_ok=True)
    (Path(cache_dir) / "parts" / f"{lattice}_{method}_{name}{os.environ.get('BAD_START_TAG', '')}.json").write_text(json.dumps(rnd(out)))
    return name


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lattice", default="4x4")
    ap.add_argument("--method", default="plain")
    ap.add_argument("--starts", default="all")
    ap.add_argument("--maxiter", type=int, default=600)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    # build sector states once in the parent (cached on disk), then fan out
    if a.starts == "all":
        Mx, Hx, *_r = setup(a.lattice, a.cache)
        names = list(make_starts(Mx, Hx, _r[-2], a.lattice).keys())
        del _r, Mx, Hx
    else:
        names = a.starts.split(",")
        Path(a.cache).mkdir(parents=True, exist_ok=True)
    jobs = [(a.lattice, a.method, n, a.maxiter, a.cache) for n in names]
    with mp.get_context("spawn").Pool(a.workers) as p:
        done = p.map(worker, jobs, chunksize=1)
    parts = []
    for n in names:
        if not (Path(a.cache) / "parts" / f"{a.lattice}_{a.method}_{n}.json").exists(): continue
        parts.append(json.loads((Path(a.cache) / "parts" / f"{a.lattice}_{a.method}_{n}.json").read_text()))
    out = a.out or str(RESDIR / "data" / f"bad_start_{a.lattice}_{a.method}.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(dict(lattice=a.lattice, method=a.method, runs=parts)))
    print("WROTE", out, flush=True)
