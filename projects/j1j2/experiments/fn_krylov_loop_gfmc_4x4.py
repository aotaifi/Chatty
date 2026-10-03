#!/usr/bin/env python3
"""Sampled 4x4 FN/Krylov loop, level (B): real fixed-node GFMC walkers instead of i.i.d. samples.

Same loop as fn_krylov_loop_sampled_4x4.py, but the mixed distribution f = a phi_FN is sampled by
continuous-time importance-sampled lattice FN GFMC for H_FN[a, s]:
  walker at x hops to allowed neighbour y with rate |K_xy| a(y)/a(x)   (K_xy = s_x H_xy s_y < 0 kept),
  log-weight accumulates -int E_L(x) dtau,  E_L = d_FN(x) - sum_y rate(x->y)   (exact in time, no dt error),
  fixed population M, systematic resampling every block of length dtau.
Histogram = weighted walker positions (weights taken just before each resampling) after a burn-in
projection time tau_burn, accumulated for n_blocks blocks. phi_hat = hist/(sum(hist) a) with the same
unvisited-state rule as the i.i.d. script ("cap"). The walker population is carried over between loop
iterations (warm start) and re-burned under the new H_FN.

Modes:
  --mode ess  : first loop iteration only (guide = exact J2=0 amplitude, Marshall). For several (M, n_blocks)
                compares the GFMC histogram to the exact f: chi2 = sum (fhat-f)^2/f and the equivalent
                i.i.d. sample size N_eff = (D-1)/chi2 (for multinomial E[chi2]=(D-1)/N), plus an
                i.i.d. multinomial control at N=N_eff.
  --mode loop : full sampled loop with GFMC histograms; same observables as the i.i.d. script.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, "/Users/aliotaifi/Chatty-organize/projects/j1j2/experiments")
sys.path.insert(0, "/Users/aliotaifi/Chatty-organize/projects/j1j2/krylov_sign_structure/experiments")
import closed_fn_krylov_exact4x4 as ref  # noqa: E402
import fn_krylov_loop_sampled_4x4 as iid  # noqa: E402

D = ref.D


class FNWalk:
    def __init__(self, diag, rows, cols, h, a, s):
        af = np.maximum(np.asarray(a, float), 1e-15)
        k = s[rows].astype(float) * h * s[cols].astype(float)
        keep = k < 0
        d = diag.astype(float) + np.bincount(rows[~keep], weights=k[~keep] * af[cols[~keep]] / af[rows[~keep]],
                                             minlength=D)
        r, c, kk = rows[keep], cols[keep], k[keep]
        rate = -kk * af[c] / af[r]
        order = np.lexsort((c, r))
        r, c, rate = r[order], c[order], rate[order]
        deg = np.bincount(r, minlength=D)
        maxd = max(int(deg.max()), 1)
        start = np.r_[0, np.cumsum(deg)[:-1]]
        pos = np.arange(r.size) - start[r]
        self.nei = np.zeros((D, maxd), np.int64)
        rt = np.zeros((D, maxd))
        self.nei[r, pos] = c
        rt[r, pos] = rate
        self.R = rt.sum(1)
        self.cum = np.cumsum(rt, 1)
        self.cum[np.arange(maxd)[None, :] >= deg[:, None]] = np.inf
        self.deg = deg
        self.EL = d - self.R

    def block(self, x, dtau, rng):
        M = x.size
        rem = np.full(M, dtau)
        logw = np.zeros(M)
        act = np.arange(M)
        hops = 0
        while act.size:
            xa = x[act]
            R = self.R[xa]
            with np.errstate(divide="ignore"):
                t = rng.exponential(1.0, act.size) / R
            stay = t >= rem[act]
            fin = act[stay]
            logw[fin] -= rem[fin] * self.EL[x[fin]]
            mv = act[~stay]
            tm = t[~stay]
            xm = x[mv]
            logw[mv] -= tm * self.EL[xm]
            rem[mv] -= tm
            u = rng.random(mv.size) * self.R[xm]
            kidx = (u[:, None] >= self.cum[xm]).sum(1)
            kidx = np.minimum(kidx, self.deg[xm] - 1)
            x[mv] = self.nei[xm, kidx]
            hops += mv.size
            act = mv
        return x, logw, hops


def sysres(w, rng, M):
    c = np.cumsum(w); c /= c[-1]
    u = (rng.random() + np.arange(M)) / M
    return np.minimum(np.searchsorted(c, u, side="right"), w.size - 1)


def gfmc_hist(walk, x, n_blocks, n_burn, dtau, rng):
    M = x.size
    hist = np.zeros(D)
    emix_num = 0.0; emix_den = 0.0
    hops = 0
    for b in range(n_burn + n_blocks):
        x, logw, hp = walk.block(x, dtau, rng)
        hops += hp
        w = np.exp(logw - logw.max())
        if b >= n_burn:
            wn = w * (M / w.sum())
            hist += np.bincount(x, weights=wn, minlength=D)
            emix_num += float(np.sum(wn * walk.EL[x])); emix_den += float(np.sum(wn))
        x = x[sysres(w, rng, M)]
    return hist, x, emix_num / max(emix_den, 1e-300), hops


def amp_from_hist(a, hist, Nnom, rule="cap"):
    vis = hist > 0
    fh = hist / hist.sum()
    ph = np.zeros(D)
    ph[vis] = fh[vis] / a[vis]
    C = float(np.sum(ph[vis] * a[vis]) / np.sum(a[vis] ** 2))
    if rule == "cap":
        ph[~vis] = np.minimum(C * a[~vis], 0.5 / (Nnom * a[~vis]))
    else:
        ph[~vis] = C * a[~vis]
    return ph / np.linalg.norm(ph), int(np.sum(~vis))


def setup(J2=0.5, J2init=0.0):
    H, diag, rows, cols, h = iid.prep(J2)
    E0, psi0 = ref.ground(H)
    sM = ref.canonical(ref.marshall_signs())
    if np.dot(psi0, sM) < 0:
        psi0 = -psi0
    a0 = np.abs(psi0); a0 /= np.linalg.norm(a0)
    strue = ref.canonical(np.where(psi0 >= 0, 1, -1).astype(np.int8))
    Hi, _ = ref.build_H(J2init)
    _, pin = ref.ground(Hi)
    a = np.abs(pin); a /= np.linalg.norm(a)
    return H, diag, rows, cols, h, E0, a0, strue, a, sM.copy()


def mode_ess(args):
    H, diag, rows, cols, h, E0, a0, strue, a, s = setup()
    efn, phi = iid.fn_solve(diag, rows, cols, h, a, s, 1e-11)
    f = a * phi; f /= f.sum()
    walk = FNWalk(diag, rows, cols, h, a, s)
    print("E_FN", efn, "mixed E_L exact", float(np.sum(f * walk.EL)), flush=True)
    rng = np.random.default_rng(args.seed)
    out = []
    for M in args.M:
        for nb in args.blocks:
            x = rng.choice(D, M, p=a * a / np.sum(a * a))
            t0 = time.time()
            hist, x, emix, hops = gfmc_hist(walk, x, nb, args.burn, args.dtau, rng)
            el = time.time() - t0
            fh = hist / hist.sum()
            chi2 = float(np.sum((fh - f) ** 2 / f))
            neff = (D - 1) / chi2
            nraw = M * nb
            # i.i.d. control at N = N_eff
            ci = rng.multinomial(int(max(neff, 1)), f) / int(max(neff, 1))
            chi2_iid = float(np.sum((ci - f) ** 2 / f))
            hi = f > 1e-5
            chi2_hi = float(np.sum((fh[hi] - f[hi]) ** 2 / f[hi]))
            neff_hi = (hi.sum() - 1) / chi2_hi
            rec = dict(M=M, n_blocks=nb, burn_blocks=args.burn, dtau=args.dtau, raw_snapshots=nraw,
                       N_eff=neff, N_eff_highf=neff_hi, eff_per_snapshot=neff / nraw, chi2=chi2,
                       chi2_iid_control=chi2_iid, mixed_E=emix, E_FN=float(efn), bias_E=emix - float(efn),
                       hops=int(hops), seconds=el, TV=float(0.5 * np.abs(fh - f).sum()))
            print(json.dumps(rec), flush=True)
            out.append(rec)
    return dict(mode="ess", records=out)


def mode_loop(args):
    H, diag, rows, cols, h, E0, a0, strue, a, s = setup()
    p0 = a0 * a0

    def ws(sg):
        w = float(np.sum(p0[sg != strue])); return min(w, 1.0 - w)

    def eps(aa, sg):
        return (ref.physical_energy(H, aa, sg) - E0) / abs(E0)

    rng = np.random.default_rng(args.seed)
    M = args.M[0]; nb = args.blocks[0]
    x = rng.choice(D, M, p=a * a / np.sum(a * a))
    hist_rec = [dict(it=0, w_s=ws(s), eps=eps(a, s))]
    t0 = time.time()
    for it in range(1, args.maxiter + 1):
        walk = FNWalk(diag, rows, cols, h, a, s)
        hist, x, emix, hops = gfmc_hist(walk, x, nb, args.burn, args.dtau, rng)
        efn, phi = iid.fn_solve(diag, rows, cols, h, a, s, 1e-10)  # diagnostics only
        f = a * phi; f /= f.sum()
        fh = hist / hist.sum()
        chi2 = float(np.sum((fh - f) ** 2 / f))
        amp, nunv = amp_from_hist(a, hist, M * nb)
        snew, T, *_ = ref.projected_krylov_update(H, diag, amp, s)
        rec = dict(it=it, w_s=ws(snew), eps=eps(amp, snew), E_FN_of_guide=float(efn), mixed_E_gfmc=emix,
                   N_eff=(D - 1) / chi2, n_unvisited=nunv, T=float(T), amp_fidelity=float(np.dot(amp, phi) ** 2),
                   sign_hash=ref.sign_hash(snew))
        hist_rec.append(rec)
        a, s = amp, snew
        print(f"GFMC M={M} nb={nb} it={it} w_s={rec['w_s']:.3e} eps={rec['eps']:.3e} Neff={rec['N_eff']:.3g} "
              f"Emix={emix:.5f} Efn={efn:.5f} t={time.time()-t0:.0f}s", flush=True)
    return dict(mode="loop", M=M, n_blocks=nb, burn_blocks=args.burn, dtau=args.dtau, seed=args.seed,
                E0=float(E0), history=hist_rec)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["ess", "loop"], required=True)
    ap.add_argument("--M", type=int, nargs="+", default=[1000])
    ap.add_argument("--blocks", type=int, nargs="+", default=[1000])
    ap.add_argument("--burn", type=int, default=50)
    ap.add_argument("--dtau", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--maxiter", type=int, default=30)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    res = mode_ess(args) if args.mode == "ess" else mode_loop(args)
    res["method"] = __doc__
    res["args"] = vars(args)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print("WROTE", args.out, flush=True)
