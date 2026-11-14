"""vardec_lib.py (T-VARDEC, PLAN_sign.md): variational decomposition of the gated sign flip, s306 8x8 J1-J2 (J2 = 0.5).

Arm A: psi_A = psi_s306 (real sign s(x) = sgn Re[w psi]).  Arm B: psi_B = c(x) psi_A, c = GatedGuide's gated exact one-step sign, F = {c = -1}.  |psi_B| = |psi_A|.
Per-sample exact algebra (c = +-1, H_xy = HW_b for an antiparallel bond b, r_y = psi_A(y)/psi_A(x) = sx sy |r_b|):
   D(x) := E_L^B(x) - E_L^A(x) = sum_y H_xy (c(x) c(y) - 1) r_y
   x in F      : D = -2 sum_{y not in F} H_xy r_y = 2 (H_xx - E_L^A) + 2 P(x)              P(x) := sum_{y in F} H_xy r_y   [E_L^A - H_xx = sum_y H_xy r_y]
   x not in F  : D = -2 Q(x),  Q(x) := sum_{y in F} H_xy r_y  (= P(x) formula, y in F)
The two pieces of D are equal IN EXPECTATION under |psi|^2 (H symmetric, psi real up to the sign carried by r):  E[1_{x notin F} (-2 Q)] = E[1_F (-2 sum_{y notin F} H_xy r_y)] = E[1_F (2 (H_xx - E_L) + 2 P)],
so   dE_var = E[D] = E[S],   S(x) := 1_F(x) [ single(x) + pair(x) ],  single = 4 (H_xx - E_L^A),  pair = 4 P(x).
The lead's split is the symmetrised one: correct for the MEAN, NOT per sample (per sample D = 2 (single + pair)/4 * ... see test_vardec_identity.py: exact per-sample identity is the D formula above;
the mean identity E[D] = E[S] is exact and tested algebraically to 1e-12 on a random small system).  Production measures S only on x with decade(x) <= gate (E_L, c needed there only) and, for x in F, c on its neighbours y with decade(y) <= gate."""
import math, os, sys
import numpy as np

LN10 = math.log(10.0)


def hxx(g, X):
    X = np.asarray(X, np.float64); return 0.25 * (g.JB[None] * X[:, g.BI] * X[:, g.BJ]).sum(1)


def sgn(v):
    return np.where(v >= 0, 1.0, -1.0)


def pair_terms(g, X, lB, r, sx, F_idx, rule='c'):
    """P(x) = sum_{y in F, y nbr of x} H_xy psi_A(y)/psi_A(x) for the flipped configurations X[F_idx]; neighbour c(y) from y's OWN call (decade(y) <= gate and c(y) < 0).
    rule 'c': F = {c = -1}; rule 'win': F' = {decade <= gate, E_L^A > H_xx} (no Lanczos condition).  Returns (P (nF,), n_nbr_calls, n_nbr_F)."""
    import gated_sign as GS
    if len(F_idx) == 0: return np.zeros(0), 0, 0
    Xf = X[F_idx]; lBf = lB[F_idx]; rf = r[F_idx]; sxf = sx[F_idx]
    anti = Xf[:, g.BI] != Xf[:, g.BJ]; wi, bi = np.nonzero(anti)
    ph = g.w * np.exp(1j * np.imag(lBf)); sy = sgn((ph[wi] * rf[wi, bi]).real)
    est = 2.0 * np.real(lBf[wi]) + 2.0 * np.log(np.abs(rf[wi, bi]))
    ci = np.nonzero(np.floor((est - GS.MREF + 1e-6) / GS.LN10) <= g.gate)[0]
    P = np.zeros(len(F_idx)); nF = 0
    if len(ci):
        Y = Xf[wi[ci]].copy(); q = np.arange(len(ci)); i_, j_ = g.BI[bi[ci]], g.BJ[bi[ci]]; yi = Y[q, i_].copy(); Y[q, i_] = Y[q, j_]; Y[q, j_] = yi
        lBy, ry, _, _ = g.ratios(Y); _, ELy, cy, dy = g.el_c(Y, lBy, ry); inF = (dy <= g.gate) & ((cy < 0) if rule == 'c' else (ELy > hxx(g, Y))); nF = int(inF.sum())
        k = ci[inF]
        term = sxf[wi[k]] * g.HW[bi[k]] * sy[k] * np.abs(rf[wi[k], bi[k]])
        P = np.bincount(wi[k], weights=term, minlength=len(F_idx))
    return P, int(len(ci)), nF


def eval_gate(g, X, cand):
    """X (K, 64) +-1 (float64); cand: bool (K,) preselected (f32 decade <= gate + margin).  Evaluates only the candidates.
    Returns dict of per-gated-sample records (gated = authoritative decade <= gate): idx (into X), dec, lw (2 Re ln psi, f64 authoritative), Hxx, EL, F (c = -1), pair, Fp (E_L > H_xx, the F' set), pairp, and counters.
    Note S(x) = single + pair = -4 sum_{y not in F} H_xy r_y = the boundary contribution of x (reviewer form dE = -4 sum_{bonds with one end in F} t_xy a_x a_y; internal bonds enter as 8 x sum, since pair = 4 sum_ordered)."""
    ic = np.nonzero(cand)[0]; out = dict(idx=np.zeros(0, np.int64), dec=np.zeros(0), lw=np.zeros(0), Hxx=np.zeros(0), EL=np.zeros(0), F=np.zeros(0, bool), pair=np.zeros(0), n_cand=len(ic), n_nbr=0, n_nbrF=0)
    if len(ic) == 0: return out
    Xc = np.asarray(X[ic], np.float64); lB, r, _, _ = g.ratios(Xc); sx, EL, c, dec = g.el_c(Xc, lB, r)
    gated = dec <= g.gate; ig = np.nonzero(gated)[0]
    if len(ig) == 0: return out
    Xg = Xc[ig]; lBg = lB[ig]; rg = r[ig]; sxg = sx[ig]; ELg = EL[ig]; cg = c[ig]; decg = dec[ig]
    F = cg < 0; F_idx = np.nonzero(F)[0]
    P, nn, nF = pair_terms(g, Xg, lBg, rg, sxg, F_idx)
    pair = np.zeros(len(ig)); pair[F_idx] = P
    Hg = hxx(g, Xg); Fp = ELg > Hg; Fp_idx = np.nonzero(Fp)[0]
    Pp, _, _ = pair_terms(g, Xg, lBg, rg, sxg, Fp_idx, 'win'); pairp = np.zeros(len(ig)); pairp[Fp_idx] = Pp
    out.update(idx=ic[ig], dec=decg, lw=2.0 * np.real(lBg), Hxx=Hg, EL=ELg, F=F, pair=pair, Fp=Fp, pairp=pairp, n_nbr=nn, n_nbrF=nF)
    return out


def direct_D(g, X):
    """INDEPENDENT direct E_L^B - E_L^A for configurations X, from signs_of_states on x and on every antiparallel neighbour y (own calls); returns (D, E_L^A, E_L^B, sA, sB at x, ok flags).
    E_L^arm(x) = H_xx + sum_{b anti} HW_b s_arm(x) s_arm(y_b) |r_b|, |r_b| from x's call, s_A = s306, s_B = s306 * c, each from the state's OWN call."""
    X = np.asarray(X, np.float64); lB, r, _, _ = g.ratios(X); anti = X[:, g.BI] != X[:, g.BJ]
    sA, sB, _, _ = g.signs_of_states(X)
    wi, bi = np.nonzero(anti); Y = X[wi].copy(); q = np.arange(len(wi)); i_, j_ = g.BI[bi], g.BJ[bi]; yi = Y[q, i_].copy(); Y[q, i_] = Y[q, j_]; Y[q, j_] = yi
    sAy, sBy, _, _ = g.signs_of_states(Y)
    ar = np.abs(r[wi, bi]); h = hxx(g, X)
    ELA = h + np.bincount(wi, weights=g.HW[bi] * sA[wi] * sAy * ar, minlength=len(X)); ELB = h + np.bincount(wi, weights=g.HW[bi] * sB[wi] * sBy * ar, minlength=len(X))
    return ELB - ELA, ELA, ELB, sA, sB


def D_formula(g, X):
    """D(x) from the F / P / Q formula: x in F: 2 (H_xx - E_L^A) + 2 P ; x not in F: -2 Q  (P = Q = sum over neighbours y in F of H_xy r_y).  Uses the pair machinery of production (pair_terms) for BOTH cases:
    for x not in F the neighbour sum is evaluated by the same routine on x itself."""
    X = np.asarray(X, np.float64); lB, r, _, _ = g.ratios(X); sx, EL, c, dec = g.el_c(X, lB, r)
    cx = np.where(dec <= g.gate, c, 1.0); F = cx < 0
    P, _, _ = pair_terms(g, X, lB, r, sx, np.arange(len(X)))             # P(x) = sum_{y in F} H_xy r_y for every x
    h = hxx(g, X)
    return np.where(F, 2.0 * (h - EL) + 2.0 * P, -2.0 * P), F, EL, h, P


def mean_identity_selftest(seed=0, n=10, trials=5):
    """Exact algebraic check of E_{|psi|^2}[D] = E[S] on a random small real system: random symmetric H (arbitrary off-diagonal signs), random real psi, random flip set F; max abs error."""
    rg = np.random.default_rng(seed); worst = 0.0
    for _ in range(trials):
        H = rg.normal(size=(n, n)); H = H + H.T
        H = H * (rg.random((n, n)) < 0.6); H = np.triu(H) + np.triu(H, 1).T                 # sparse symmetric
        psi = rg.normal(size=n) * np.exp(rg.normal(size=n)); c = np.where(rg.random(n) < 0.35, -1.0, 1.0); a2 = psi ** 2 / (psi ** 2).sum()
        off = H - np.diag(np.diag(H)); Hxx = np.diag(H); r = psi[None, :] / psi[:, None]; EL = Hxx + (off * r).sum(1)
        EB = Hxx + (off * (c[None, :] * psi[None, :]) / (c[:, None] * psi[:, None])).sum(1)
        D = EB - EL; F = c < 0; P = (off * r * F[None, :]).sum(1); S = np.where(F, 4 * (Hxx - EL) + 4 * P, 0.0)
        Dform = np.where(F, 2 * (Hxx - EL) + 2 * P, -2 * P)
        worst = max(worst, float(np.max(np.abs(D - Dform))), abs(float((a2 * D).sum() - (a2 * S).sum())))
    return worst


# ---------------------------------------------------------------------------------------------------------------- tempered sampler
def make_tempered(sm, ptemp):
    """Metropolis exchange sampler for |psi|^(2 ptemp) (Hastings factor as in vmc_pp2s.PPSampler), float32 acceptance.  Returns (sweep(S, key, nstep) -> S, lnpsi32(S) -> 2 Re ln psi (f32 array))."""
    import jax, jax.numpy as jnp
    from functools import partial
    import pp2_lemma as PL
    BIj, BJj = jnp.asarray(sm.BI), jnp.asarray(sm.BJ); p32 = sm.p32; pt = float(ptemp)
    logpsi = lambda S: PL._logB_j(p32, S.astype(jnp.float32))

    def nanti(S): return jnp.sum(S[:, BIj] != S[:, BJj], 1)

    def step(S, lS, key):
        k1, k2 = jax.random.split(key); anti = S[:, BIj] != S[:, BJj]
        sc = jnp.where(anti, jax.random.uniform(k1, anti.shape), -1.0); bsel = jnp.argmax(sc, 1)
        i = BIj[bsel]; j = BJj[bsel]; r = jnp.arange(S.shape[0]); si = S[r, i]; sj = S[r, j]
        Y = S.at[r, i].set(sj).at[r, j].set(si); lY = logpsi(Y)
        lacc = pt * 2.0 * jnp.real(lY - lS) + jnp.log(nanti(S) / nanti(Y))
        acc = jnp.log(jax.random.uniform(k2, (S.shape[0],))) < lacc
        return jnp.where(acc[:, None], Y, S), jnp.where(acc, lY, lS), acc

    @partial(jax.jit, static_argnums=(2,))
    def sweep(S, key, nstep):
        lS = logpsi(S)
        def body(cr, k):
            S, lS, na = cr; S, lS, acc = step(S, lS, k); return (S, lS, na + jnp.mean(acc)), None
        (S, lS, na), _ = jax.lax.scan(body, (S, lS, 0.0), jax.random.split(key, nstep)); return S, 2.0 * jnp.real(lS), na / nstep

    return sweep, jax.jit(lambda S: 2.0 * jnp.real(logpsi(S)))
