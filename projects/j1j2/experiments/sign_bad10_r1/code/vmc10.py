"""One-step Lanczos moments of the guide v = Marshall x |ViT| (10x10 symm ViT, f32h), the 8x8 recipe (headline_s306 vmc_moments.py) with E_L-type quantities evaluated exactly:
  vt = v/|v| = M(x);  h1 = (Hv)/|psi|(x) = H_xx M(x) + sum_b (J_b/2) M(y_b) |psi(y_b)/psi(x)|;  h2 = (H^2 v)/|psi|(x) = H_xx h1(x) + sum_b (J_b/2) h1(y_b) |psi(y_b)/psi(x)|
  (h1 at every one-swap neighbour y_b: the TWO-HOP set of x, deduplicated).  x ~ |psi|^2 (Metropolis-exchange chains, K steps between measurements).
  H' = H - E_ref:  h1' = h1 - E_ref vt,  h2' = h2 - 2 E_ref h1 + E_ref^2 vt;  s0 = <vt^2> = 1, s1 = <vt h1'>, s2 = <h1'^2>, s3 = <h1' h2'>, s4 = <h2'^2>  (check: <vt h2'> = s2).
  Moments -> lanczos_from_moments (copied verbatim from vmc_moments.py).  Totals, spin-1/2 units, N = 100."""
import time, json, numpy as np
import vitlib as VL
from vitlib import BI, BJ, HW, JB, N

def lanczos_from_moments(m, E_ref):
    m1, m2, m3, m4 = m
    mu2 = m2 - m1 ** 2; mu3 = m3 - 3 * m1 * m2 + 2 * m1 ** 3; mu4 = m4 - 4 * m1 * m3 + 6 * m1 ** 2 * m2 - 3 * m1 ** 4
    alpha = (mu3 - np.sqrt(mu3 ** 2 + 4 * mu2 ** 3)) / (2 * mu2 ** 2)
    n = 1 + alpha ** 2 * mu2; dE1 = (2 * alpha * mu2 + alpha ** 2 * mu3) / n
    var1 = (mu2 + 2 * alpha * mu3 + alpha ** 2 * mu4) / n - dE1 ** 2
    return dict(E=E_ref + m1, var=mu2, mu3=mu3, mu4=mu4, alpha=alpha, E1=E_ref + m1 + dE1, var1=var1, E1_ritz=E_ref + m1 + alpha * mu2)

def from_sums(s, E_ref):
    return lanczos_from_moments((s[1] / s[0], s[2] / s[0], s[3] / s[0], s[4] / s[0]), E_ref)

def h1_one_hop(ev, X, chunk=2048):
    """h1(x) for walkers X (n,100) (one-hop only: x and N1(x)); returns h1, vt, lpx."""
    n = len(X); wi, bi, Y = VL.neighbours(X); l = ev(np.concatenate([X, Y]), chunk).real; lx = l[:n]; ly = l[n:]
    M = VL.marshall(np.concatenate([X, Y])).astype(float); Mx = M[:n]; My = M[n:]
    off = np.bincount(wi, weights=HW[bi] * My * np.exp(ly - lx[wi]), minlength=n)
    return VL.diag0(X) * Mx + off, Mx, lx, len(Y) + n

def two_hop(ev, x, chunk=2048):
    """vt, h1, h2 at ONE walker x (100,) with the exact two-hop evaluation; returns dict incl. counts and timings."""
    t0 = time.perf_counter(); x = np.asarray(x, np.float32)[None]
    wi1, bi1, Y1 = VL.neighbours(x); L1 = np.concatenate([x, Y1]); n1 = len(L1)
    wi2, bi2, Z = VL.neighbours(L1)
    allS = np.concatenate([L1, Z]); k = VL.keys_of(allS); _, first, inv = np.unique(k, return_index=True, return_inverse=True)
    U = allS[first]; t1 = time.perf_counter()
    lpU = ev(U, chunk).real; t2 = time.perf_counter()
    lp = lpU[inv]; M = VL.marshall(U).astype(float)[inv]
    lpL1 = lp[:n1]; lpZ = lp[n1:]; MZ = M[n1:]
    off = np.bincount(wi2, weights=HW[bi2] * MZ * np.exp(lpZ - lpL1[wi2]), minlength=n1)
    h1 = VL.diag0(L1) * M[:n1] + off                                    # h1 at x (index 0) and every neighbour y_b (1..)
    h2 = VL.diag0(x)[0] * h1[0] + np.sum(HW[bi1] * h1[1:] * np.exp(lpL1[1:] - lpL1[0]))
    t3 = time.perf_counter()
    return dict(vt=float(M[0]), h1=float(h1[0]), h2=float(h2), n_nbr=int(n1 - 1), n_unique=int(len(U)), n_raw=int(len(allS)), sec_prep=t1 - t0, sec_vit=t2 - t1, sec_post=t3 - t2)

def metropolis(ev, X, rng, steps, chunk=1024):
    """exchange Metropolis on |psi|^2 (swap a uniform antiparallel bond; Hastings n_a(x)/n_a(y)), as vit10_vmc.py.  X (W,100) float32 modified in place; returns acceptance."""
    W = len(X); lp = ev(X, chunk).real; acc = 0.0
    for _ in range(steps):
        anti = X[:, BI] != X[:, BJ]; na = anti.sum(1); b = np.argmax(rng.random(anti.shape) * anti, 1); r = np.arange(W)
        Y = X.copy(); Y[r, BI[b]], Y[r, BJ[b]] = X[r, BJ[b]], X[r, BI[b]]
        lq = ev(Y, chunk).real; nay = (Y[:, BI] != Y[:, BJ]).sum(1)
        ok = np.log(rng.random(W)) < 2 * (lq - lp) + np.log(na / nay)
        X[ok] = Y[ok]; lp = np.where(ok, lq, lp); acc += ok.mean()
    return acc / max(steps, 1)


def sub_sample(ev, X, rng, Kb, chunk=1024):
    """(vt, h1, h2) at each walker of X (C,100) with the second hop of h2 estimated by Kb bonds drawn WITH replacement with probability p_b ~ w_b = (J_b/2)|psi(y_b)/psi(x)|:
       h2 = H_xx h1(x) + sum_b w_b h1(y_b)  ~  H_xx h1(x) + (W/Kb) sum_k h1(y_{b_k}),  W = sum_b w_b  (unbiased; h1(y) exact: all N1(y)).  Returns vt, h1, h2, n_evals."""
    C = len(X); wi, bi, Y = VL.neighbours(X); S = np.concatenate([X, Y]); l = ev(S, chunk).real; lx = l[:C]; ly = l[C:]
    M = VL.marshall(S).astype(float); Mx = M[:C]; My = M[C:]; r = np.exp(ly - lx[wi]); w = HW[bi] * r
    h1x = VL.diag0(X) * Mx + np.bincount(wi, weights=HW[bi] * My * r, minlength=C)
    sel = []; W = np.bincount(wi, weights=w, minlength=C); own = []
    for c in range(C):
        idx = np.nonzero(wi == c)[0]; p = w[idx] / w[idx].sum(); pick = idx[rng.choice(len(idx), Kb, p=p)]; sel.append(pick); own += [c] * Kb
    sel = np.concatenate(sel); own = np.array(own)
    h1y, _, _, ne = h1_one_hop(ev, Y[sel], chunk)
    Ssum = np.bincount(own, weights=h1y, minlength=C) * W / Kb
    h2 = VL.diag0(X) * h1x + Ssum
    return Mx, h1x, h2, len(S) + ne
