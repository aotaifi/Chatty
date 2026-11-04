"""pp2_lemma_sign_test.py: savings item (4), the sign angle's 1-PP complex-phase stored sign through pp2_lemma's determinant-ratio path (pp2_lemma.PhaseSign).
usage (JAX_PLATFORMS=cpu):
  python pp2_lemma_sign_test.py OUT ref                      reference validation: the copy of the sign angle's margin vs its stored logits.npy / S_a1.npy / marshall.npy
  python pp2_lemma_sign_test.py OUT a SYS M [VARIANTS...]    check (a): lemma sign (+ margin) at x and every valid one- and two-swap endpoint vs the direct full evaluation
  python pp2_lemma_sign_test.py OUT b SYS M [VARIANTS...]    check (b): fJ_fixed / aug_ratios with the lemma-path sign vs with the direct sign_fn (eval_path old / pack / dedup)
  python pp2_lemma_sign_test.py OUT time SYS M VARIANT       check (c): CPU seconds per walker step (one aug_ratios call) for VARIANT in
                                                             hash (no real sign), lemma_old / lemma_pack / lemma_dedup, direct_dedup, direct_old
  python pp2_lemma_sign_test.py OUT default                  check (d2): default path outputs bit-equal to the frozen 7c4dddf pp2_lemma.py
SYS a6: 6x6, sign = S_a1 (sign_tcap runs_a40s/a1_pp2_m4j/theta.npy), walkers = 192 random group images of teacher states + 128 uniform random S^z = 0 configurations,
        amplitude (checks b, time) = pfaffian_r5 pp2_s1 theta_5;
    a8: 8x8, sign f = rb8_prod f * (1 + 0.4 i) + 0.15 std(f) complex noise, theta0 0.37 (a complex 1-PP: no trained 8x8 phase PP exists yet), walkers = first M of the
        |P_1|^2 Metropolis walkers (savings/walkers_8x8_prod_M256_seed0.npy), amplitude = rb8_prod + P_2 (savings a8two construction).
Reference ("the sign angle's own code"): s2b_train_m.py (pp arm) log_one / P_one / margin_one, copied below with L generalised (N, NH from L; Lattice(L, 0.5, 'full') from the
sign_tcap rbmpp_lib.py); S = M sgn(l), sgn(0) = +1 (s2b_eval.py: np.where(lg >= 0, 1, -1)), M = (-1)^{#up, sublattice x + y even}.
"""
import os, sys, json, time, hashlib, importlib.util
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import jax, jax.numpy as jnp
import pp2_lemma as PL

OUT = sys.argv[1]; STAGE = sys.argv[2]
os.makedirs(OUT, exist_ok=True); T0 = time.time()
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
R = '/scratch/a/A.Otaifi/ansatz_runs'
SIGN_CODE = f'{R}/sign_tcap/code'
THETA_A1 = f'{R}/sign_tcap/runs_a40s/a1_pp2_m4j/theta.npy'; LOGITS_A1 = f'{R}/sign_tcap/runs_a40s/a1_pp2_m4j/logits.npy'; S_A1 = f'{R}/sign_tcap/S_a1.npy'
SDATA = f'{R}/sign_s2/data/N36'
PROD = '/scratch/a/A.Otaifi/rbmpp8/rb8_prod/params_final.npz'
WALK8 = f'{R}/pfaffian_8x8build/savings/walkers_8x8_prod_M256_seed0.npy'
THETA5 = f'{R}/pfaffian_r5/pp2_s1/tab/theta_5.npy'
EP6 = None
TOL = 1e-12


def log(m):
    s = f'[{time.time() - T0:7.0f}s] {m}'; print(s, flush=True); open(f'{OUT}/test.log', 'a').write(s + '\n')


# ------------------------------------------------------------------------------------------ the sign angle's margin (copy of s2b_train_m.py, L generalised)
def _sign_lib():
    sp = importlib.util.spec_from_file_location('rbmpp_lib_sign', f'{SIGN_CODE}/rbmpp_lib.py'); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m


def make_direct_margin(theta, L, with_kappa=False):
    """jitted batch margin l(x) = Re(e^{-i theta0} P)/|P| for X (B, N) of +-1 (float64): s2b_train_m.py 'pp' arm, c128."""
    RL = _sign_lib(); N = L * L; NH = N // 2; lat = RL.Lattice(L, 0.5, 'full'); IMG = lat.IMG; RDT = jnp.float64; CDT = jnp.complex128
    FLP = lat.FL.astype(RDT); th = jnp.asarray(np.asarray(theta, np.float64))

    def log_one(f, zz):          # cx_r1fit.py C1 MODEL, COMPLEX branch, BF = False (copied)
        order = jnp.argsort(-zz, stable=True); Uo = order[:NH]; Do = order[NH:]
        Fm = f[Uo][:, Do]
        sg, ld = jnp.linalg.slogdet(Fm)
        up = (zz > 0).astype(jnp.int32); dn = 1 - up
        inv = jnp.sum(up * (jnp.cumsum(dn) - dn))
        return ld, (1.0 - 2.0 * (inv % 2).astype(RDT)) * sg

    def P_one(th, sigma):
        f = (th[:N * N] + 1j * th[N * N:2 * N * N]).astype(CDT).reshape(N, N)
        Z = FLP[:, None] * sigma[IMG]
        ld, sgn = jax.vmap(lambda zz: log_one(f, zz))(Z)
        m = jax.lax.stop_gradient(jnp.max(ld))
        return jnp.sum(sgn * jnp.exp(ld - m)), jnp.sum(jnp.exp(ld - m))             # (P / e^m, sum of |image terms| / e^m)

    def margin_one(th, sigma):
        S_, A_ = P_one(th, sigma); ph = jnp.exp(-1j * th[-1])
        return jnp.real(ph * S_) / jnp.maximum(jnp.abs(S_), 1e-300), A_ / jnp.maximum(jnp.abs(S_), 1e-300)   # margin, cancellation factor kappa = sum |a_g| / |P|
    mj = jax.jit(jax.vmap(margin_one, in_axes=(None, 0)))
    ch = 256 if L == 6 else 4

    def margin(X):
        X = np.asarray(X, np.float64); out = np.empty(len(X)); kap = np.empty(len(X))
        for i in range(0, len(X), ch):
            b = X[i:i + ch]; m = len(b)
            if m < ch: b = np.concatenate([b, np.repeat(b[:1], ch - m, 0)])
            o1, o2 = mj(th, jnp.asarray(b)); out[i:i + m] = np.asarray(o1)[:m]; kap[i:i + m] = np.asarray(o2)[:m]
        return (out, kap) if with_kappa else out
    return margin


def margin_ld(theta, L, cfgs):
    """extended-precision (numpy longdouble, ~1e-19) margin of the sign angle's definition: batched complex LU with partial pivoting of every image.  For the worst entries only."""
    RL = _sign_lib(); N = L * L; NH = N // 2; lat = RL.Lattice(L, 0.5, 'full'); IMG = np.asarray(lat.IMG); FL = np.asarray(lat.FL, np.float64); th = np.asarray(theta, np.float64)
    f = (th[:N * N] + 1j * th[N * N:2 * N * N]).reshape(N, N).astype(np.clongdouble); out = []
    for c in np.asarray(cfgs, np.float64):
        Z = FL[:, None] * c[IMG]; order = np.argsort(-Z, axis=1, kind='stable'); U = order[:, :NH]; D = order[:, NH:]
        A = f[U[:, :, None], D[:, None, :]].copy(); nI = len(A); det = np.ones(nI, np.clongdouble)
        for k in range(NH):
            piv = k + np.argmax(np.abs(A[:, k:, k]), axis=1); ar = np.arange(nI)
            sw = piv != k; det = np.where(sw, -det, det)
            rk = A[ar, k].copy(); rp = A[ar, piv].copy(); A[ar, k] = rp; A[ar, piv] = rk
            d = A[:, k, k]; det = det * d
            fac = A[:, k + 1:, k] / d[:, None]; A[:, k + 1:, k:] -= fac[:, :, None] * A[:, k, None, k:]
        up = (Z > 0).astype(np.int64); dn = 1 - up
        inv = np.sum(up * (np.cumsum(dn, 1) - dn), 1); par = 1.0 - 2.0 * (inv % 2)
        P = np.sum(par * det); ph = np.cos(np.longdouble(th[-1])) - 1j * np.sin(np.longdouble(th[-1]))
        out.append(float(np.real(ph * P) / np.abs(P)))
    return np.array(out)


def marshall_of(X, L):
    """(-1)^{#up on the sublattice x + y even}, site = x + L y."""
    X = np.asarray(X); A = np.array([(i % L + i // L) % 2 == 0 for i in range(L * L)])
    return np.where(((X > 0)[:, A].sum(1)) % 2 == 0, 1, -1).astype(np.int8)


def make_direct_sign_fn(theta, L, count=None):
    """the direct-phase sign_fn (numpy in, +-1 int8 out): S = M sgn l of the sign angle's own margin; a drop-in for fJ_fixed's sign_fn."""
    mg = make_direct_margin(theta, L)

    def sign_fn(cfg):
        cfg = np.asarray(cfg)
        if count is not None: count.append(len(cfg))
        m = mg(cfg); return (marshall_of(cfg, L) * np.where(m >= 0, 1, -1)).astype(np.int8)
    return sign_fn


# ------------------------------------------------------------------------------------------ systems
def cfg_keys(X):
    return PL._cfg_keys(np.asarray(X))


def rand_cfg(rng, B, N):
    S = np.ones((B, N)); S[:, N // 2:] = -1
    for r in S: rng.shuffle(r)
    return S


def setup(system, M):
    """-> dict(theta (sign theta), L, X (M, N) walkers, amp (PP2 for b / time), EP, f0)"""
    if system == 'a6':
        th = np.load(THETA_A1); L = 6; rng = np.random.default_rng(2026)
        states = np.load(f'{SDATA}/states.npy', mmap_mode='r'); tr = np.load(f'{SDATA}/train_b0.5_s0.npz')['nbr_30000'].astype(np.int64)
        lat = PL.lattice(6, 'full'); n_t = (M * 3) // 5; n_r = M - n_t
        pick = np.sort(rng.choice(tr, n_t, replace=False)); sh = np.arange(36, dtype=np.uint64)
        base = (((np.asarray(states[pick])[:, None] >> sh[None]) & np.uint64(1)).astype(np.float64) * 2 - 1)
        g = rng.integers(0, lat.nI, n_t); X1 = np.stack([lat.FL[gi] * base[i][lat.IMG[gi]] for i, gi in enumerate(g)])
        X = np.concatenate([X1, rand_cfg(rng, n_r, 36)])
        EP = json.load(open(f'{R}/pfaffian_r5/readout/readout.json'))['tables']['pp2_s1']['E_FN']['E']
        amp = PL.PP2.from_cx_theta(np.load(THETA5))
        return dict(theta=th, L=L, X=X, amp=amp, EP=EP, f0=2.0, info=dict(theta=THETA_A1, theta_sha256=sha(THETA_A1), theta5=THETA5))
    assert system == 'a8'
    z = np.load(PROD); f0 = np.asarray(z['f']); rng = np.random.default_rng(88); N = 64
    f = f0 * (1 + 0.4j) + 0.15 * np.std(f0) * (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N)))
    th = np.concatenate([np.real(f).ravel(), np.imag(f).ravel(), [0.37]])
    X = np.load(WALK8).astype(np.float64)[:M]
    p1 = PL.PP2.from_rbmpp_npz(PROD)
    r = np.random.default_rng(1); f2 = np.asarray(p1.f1) + 0.3 * np.std(np.real(np.asarray(p1.f1))) * (r.standard_normal((64, 64)) + 1j * r.standard_normal((64, 64)))
    x0 = np.ones((1, 64)); x0[0, 32:] = -1; np.random.default_rng(5).shuffle(x0[0])
    pP2 = PL.PP2(f2, L=8, proj=p1.proj, rbm='inner', b=p1.b, W=p1.W)
    lam = 0.5 * np.exp(np.real(complex(PL.logB(p1, x0)[0]) - complex(PL.logB(pP2, x0)[0])))
    amp = p1.with_p2(f2, lam)
    return dict(theta=th, L=8, X=X, amp=amp, EP=-0.4987565 * 64, f0=3.0, info=dict(prod=PROD, prod_sha256=sha(PROD), walkers=WALK8, walkers_sha256=sha(WALK8)))


def valid_endpoint_configs(X, d):
    """all valid y_a, z_ab configurations (int8) with their (walker, a) / (walker, a, b) indices, from pp2_lemma._configs."""
    Xj = jnp.asarray(X, jnp.float64)
    Y, Z = PL._configs(Xj, {k: d[k] for k in ('upa', 'dna', 'am', 'ru', 'rd', 'k')}, with_z=True)
    return np.asarray(Y), np.asarray(Z)


# ------------------------------------------------------------------------------------------ reference validation
def stage_ref():
    th = np.load(THETA_A1); mg = make_direct_margin(th, 6); res = {}
    states = np.load(f'{SDATA}/states.npy', mmap_mode='r'); Mt = np.load(f'{SDATA}/marshall.npy'); lg = np.load(LOGITS_A1); Sa = np.load(S_A1)
    rng = np.random.default_rng(0); idx = np.sort(rng.choice(len(states), 6000, replace=False)); sh = np.arange(36, dtype=np.uint64)
    X = (((np.asarray(states[idx])[:, None] >> sh[None]) & np.uint64(1)).astype(np.float64) * 2 - 1)
    m = mg(X); S = marshall_of(X, 6) * np.where(m >= 0, 1, -1)
    res['n'] = len(idx); res['max_abs_margin_vs_stored_logits_float32'] = float(np.abs(m - lg[idx]).max())
    res['S_equals_S_a1_fraction'] = float(np.mean(S == Sa[idx])); res['marshall_formula_equals_marshall_npy_fraction'] = float(np.mean(marshall_of(X, 6) == Mt[idx]))
    allX = (((np.asarray(states[:300000])[:, None] >> sh[None]) & np.uint64(1)).astype(np.float64) * 2 - 1)
    res['marshall_formula_first_3e5_fraction'] = float(np.mean(marshall_of(allX, 6) == Mt[:300000]))
    res['theta_sha256'] = sha(THETA_A1); res['logits_sha256'] = sha(LOGITS_A1); res['S_a1_sha256'] = sha(S_A1)
    res['PASS'] = res['max_abs_margin_vs_stored_logits_float32'] < 1e-6 and res['S_equals_S_a1_fraction'] == 1.0 and res['marshall_formula_equals_marshall_npy_fraction'] == 1.0 and res['marshall_formula_first_3e5_fraction'] == 1.0
    json.dump(res, open(f'{OUT}/check_ref.json', 'w'), indent=1); log(f'REF {res}')


# ------------------------------------------------------------------------------------------ (a)
def stage_a(system, M, variants):
    S = setup(system, M); L = S['L']; N = L * L; X = S['X']; th = S['theta']; log(f'a {system} M {len(X)}: direct reference ...')
    ps0 = PL.PhaseSign.from_theta(th, L)
    d = PL.two_swap_ratios(ps0.params, X, 'f64', None, walker_chunk=32 if L == 8 else 128, eval_path='dedup'); dA = d['am'].shape[1]
    Y, Z = valid_endpoint_configs(X, d); am = np.asarray(d['am']); bm = np.asarray(d['bm']); Mw, _, dB = bm.shape
    allc = np.concatenate([X[:, None, :].astype(np.int8), Y, Z.reshape(Mw, -1, N)], 1)                    # (M, 1 + dA + dA dB, N)
    valid = np.concatenate([np.ones((Mw, 1), bool), am, bm.reshape(Mw, -1)], 1)
    keys = np.stack([cfg_keys(allc[i]) for i in range(Mw)])
    uk, first = np.unique(keys[valid], return_index=True)
    cfg_valid = allc[valid]; ucfg = cfg_valid[first].astype(np.float64)
    log(f'a {system}: walkers {Mw}, valid endpoints {int(valid.sum())}, distinct configurations {len(uk)}')
    mg = make_direct_margin(th, L, True); t = time.time(); mu = np.empty(len(ucfg)); ku = np.empty(len(ucfg)); step = 64 if L == 8 else 4096
    for i in range(0, len(ucfg), step):
        mu[i:i + step], ku[i:i + step] = mg(ucfg[i:i + step]); log(f'  direct {min(i + step, len(ucfg))}/{len(ucfg)}')
    sec_direct = time.time() - t
    pos = np.searchsorted(uk, keys); pos = np.clip(pos, 0, len(uk) - 1)
    kap_dir = np.where(valid, ku[pos], 1.0)
    m_dir = np.where(valid, mu[pos], np.nan)                                                              # (M, E+1), nan where invalid
    Mar = marshall_of(allc.reshape(-1, N), L).reshape(Mw, -1)
    S_dir = Mar * np.where(m_dir >= 0, 1, -1); S_dir = np.where(valid, S_dir, 0)
    sxd = S_dir[:, 0]; syd = S_dir[:, 1:1 + dA]; szd = S_dir[:, 1 + dA:].reshape(Mw, dA, dB); mxd = m_dir[:, 0]; myd = m_dir[:, 1:1 + dA]; mzd = m_dir[:, 1 + dA:].reshape(Mw, dA, dB)
    res = dict(system=system, M=Mw, L=L, valid_endpoints=int(valid.sum()), distinct=int(len(uk)), sec_direct=sec_direct, dA=dA, dB=dB, theta_sha256=hashlib.sha256(th.tobytes()).hexdigest(),
               info=S['info'], code_sha256=sha(os.path.join(HERE, 'pp2_lemma.py')), test_sha256=sha(__file__), host=os.uname().nodename, variants={})
    mabs = np.abs(np.where(valid, m_dir, 1.0)); res['margin_stats'] = dict(min_abs=float(np.nanmin(np.where(valid, np.abs(m_dir), np.nan))), frac_abs_lt_1em3=float(np.mean(np.abs(m_dir[valid]) < 1e-3)),
                                                                          frac_negative=float(np.mean(m_dir[valid] < 0)), mean_abs=float(np.mean(np.abs(m_dir[valid]))))
    for v in variants:
        brk = None if v in ('old', 'pack', 'dedup') else v.split(':')[1]; path = v if brk is None else v.split(':')[0]
        ps = PL.PhaseSign.from_theta(th, L, broken=brk) if brk else ps0
        t = time.time(); PL.STATS.clear()
        (sx, sy, sz), (mx, my, mz) = ps.endpoints(X, d, True, 'f64', path, walker_chunk=32 if L == 8 else 128, margins=True)
        sec = time.time() - t
        sx = np.asarray(sx); sy = np.asarray(sy); sz = np.asarray(sz); mx = np.asarray(mx); my = np.asarray(my); mz = np.asarray(mz)
        # invalid entries: z = x (the old padded configs), so their sign must be sign(x): compare everything against the padded-layout expectation
        syx = np.where(am, syd, sxd[:, None]); szx = np.where(bm, szd, sxd[:, None, None])
        mism = dict(x=int((sx != sxd).sum()), y_valid=int(((sy != syd) & am).sum()), z_valid=int(((sz != szd) & bm).sum()),
                    y_all=int((sy != syx).sum()), z_all=int((sz != szx).sum()))
        dm = np.concatenate([np.abs(mx - mxd)[:, None], np.where(am, np.abs(my - myd), 0), np.where(bm, np.abs(mz - mzd), 0).reshape(Mw, -1)], 1)
        mref = np.concatenate([np.abs(mxd)[:, None], np.where(am, np.abs(myd), 1), np.where(bm, np.abs(mzd), 1).reshape(Mw, -1)], 1)
        big = mref >= 0.1
        dmk = np.where(valid, dm / (2.2e-16 * np.maximum(kap_dir, 1.0)), 0)
        rec = dict(path=path, broken=brk, sec=sec, mismatches=mism, mismatch_total=int(sum(mism.values())), max_abs_dmargin=float(dm.max()), mean_abs_dmargin=float(dm[valid].mean()),
                   max_rel_dmargin_for_abs_margin_ge_0p1=float((dm[big] / mref[big]).max()), max_rel_dmargin_all=float((dm[valid] / np.maximum(mref[valid], 1e-300)).max()),
                   max_dmargin_over_eps_kappa=float(dmk.max()), max_kappa=float(kap_dir[valid].max()), median_kappa=float(np.median(kap_dir[valid])),
                   max_abs_dmargin_kappa_le_1e3=float(dm[valid & (kap_dir <= 1e3)].max()), frac_kappa_gt_1e3=float(np.mean(kap_dir[valid] > 1e3)),
                   max_rel_dmargin_big_kappa_le_1e3=float((dm / np.maximum(mref, 1e-300))[big & valid & (kap_dir <= 1e3)].max()),
                   n_sign_flips_valid=int(mism['y_valid'] + mism['z_valid'] + mism['x']), stats=dict(PL.STATS))
        if brk is None and v == variants[0]:                                                              # which one is closer to the truth on the worst entries? (longdouble LU)
            worst = np.argsort(-np.where(valid, dm, 0).ravel())[:6]; ww, ee = np.divmod(worst, dm.shape[1])
            mlem = np.concatenate([mx[:, None], my, mz.reshape(Mw, -1)], 1)[ww, ee]; mdir = m_dir[ww, ee]; mld = margin_ld(th, L, allc[ww, ee])
            rec['worst_entries'] = dict(kappa=kap_dir[ww, ee].tolist(), abs_margin=np.abs(mdir).tolist(), lemma_minus_direct=(mlem - mdir).tolist(),
                                        lemma_minus_longdouble=(mlem - mld).tolist(), direct_minus_longdouble=(mdir - mld).tolist())
            log(f'  worst-entry check vs longdouble: lemma {np.abs(mlem - mld).max():.2e}, direct {np.abs(mdir - mld).max():.2e}')
        res['variants'][v] = rec
        log(f'  {v}: max |dm| {rec["max_abs_dmargin"]:.3e} (rel for |m|>=0.1: {rec["max_rel_dmargin_for_abs_margin_ge_0p1"]:.3e}), sign mismatches {mism}, {sec:.1f}s')
    good = [v for v in variants if ':' not in v]; bad = [v for v in variants if ':' in v]
    res['PASS'] = bool(good) and all(res['variants'][v]['mismatch_total'] == 0 and res['variants'][v]['max_abs_dmargin'] <= TOL for v in good)
    res['PASS_strict_relative_margin_ge_0p1'] = bool(good) and all(res['variants'][v]['max_rel_dmargin_for_abs_margin_ge_0p1'] <= TOL for v in good)
    res['BROKEN_FAILS'] = bool(bad) and all((res['variants'][v]['mismatch_total'] > 0 or res['variants'][v]['max_abs_dmargin'] > 1e-6) for v in bad)   # (a Marshall error leaves the margin unchanged: sign mismatches are the signal)
    res['tol'] = TOL
    json.dump(res, open(f'{OUT}/check_a_{system}.json', 'w'), indent=1, default=float)
    log(f'CHECK a {system}: PASS {res["PASS"]} BROKEN_FAILS {res["BROKEN_FAILS"]}')


# ------------------------------------------------------------------------------------------ (b)
KEYS = ('ratio', 'add_x', 'add_y', 'logG_x', 'sx', 'sy', 'sz', 'r1', 'r2', 'fx', 'fy')


def cmp_out(ref, o, am, bm):
    out = {}
    for k in KEYS:
        a = np.asarray(ref[k]); b = np.asarray(o[k])
        a = a.astype(complex) if a.dtype.kind == 'c' else a.astype(float); b = b.astype(complex) if b.dtype.kind == 'c' else b.astype(float)
        fa = np.isfinite(a); fb = np.isfinite(b)
        if (fa != fb).any(): out[k] = float('inf'); out[k + '_rel'] = float('inf'); continue
        dd = np.abs(a[fa] - b[fb]); out[k] = float(dd.max()) if dd.size else 0.0
        out[k + '_rel'] = float((dd / np.maximum(np.abs(a[fa]), 1.0)).max()) if dd.size else 0.0
    out['sign_mismatch_valid'] = int((np.asarray(ref['sx']) != np.asarray(o['sx'])).sum() + ((np.asarray(ref['sy']) != np.asarray(o['sy'])) & am).sum() + ((np.asarray(ref['sz']) != np.asarray(o['sz'])) & bm).sum())
    return out


def stage_b(system, M, variants):
    S = setup(system, M); L = S['L']; X = S['X']; th = S['theta']; amp = S['amp']; wc = 32 if L == 8 else 128
    dmax = PL._dmax_of(amp.lat, X)
    kw = dict(f0=S['f0'], mode='fix', walker_chunk=wc, sign_chunk=4096 if L == 6 else 64, dmax=dmax)
    cnt = []; dsf = make_direct_sign_fn(th, L, cnt)
    log(f'b {system} M {len(X)}: direct sign_fn (dedup) aug_ratios ...'); t = time.time()
    ref = PL.aug_ratios(amp, X, dsf, S['EP'], 4.0, eval_path='dedup', **kw); jax.block_until_ready(ref['ratio']); sec_ref = time.time() - t
    log(f'  direct done {sec_ref:.0f}s, sign_fn configs {sum(cnt)}')
    am = np.asarray(ref['am']); bm = np.asarray(ref['bm']); res = dict(system=system, M=len(X), sec_direct=sec_ref, sign_fn_configs=int(sum(cnt)), info=S['info'], variants={},
                                                                  code_sha256=sha(os.path.join(HERE, 'pp2_lemma.py')), test_sha256=sha(__file__), host=os.uname().nodename)
    refd = {k: np.asarray(ref[k]) for k in KEYS}
    for v in variants:
        brk = None if v in ('old', 'pack', 'dedup') else v.split(':')[1]; path = v if brk is None else v.split(':')[0]
        ps = PL.PhaseSign.from_theta(th, L, broken=brk); t = time.time(); PL.STATS.clear()
        o = PL.aug_ratios(amp, X, ps, S['EP'], 4.0, eval_path=path, **kw); jax.block_until_ready(o['ratio']); sec = time.time() - t
        md = cmp_out(refd, {k: np.asarray(o[k]) for k in KEYS}, am, bm)
        worst_rel = max(md[k + '_rel'] for k in KEYS); worst_abs = max(md[k] for k in KEYS)
        res['variants'][v] = dict(path=path, broken=brk, sec=sec, maxdiff=md, max_rel=worst_rel, max_abs=worst_abs, sign_mismatch_valid=md['sign_mismatch_valid'], stats=dict(PL.STATS))
        log(f'  {v}: max rel {worst_rel:.3e} max abs {worst_abs:.3e} sign mismatches {md["sign_mismatch_valid"]} ({sec:.0f}s)')
    good = [v for v in variants if ':' not in v]; bad = [v for v in variants if ':' in v]
    res['PASS'] = bool(good) and all(res['variants'][v]['max_rel'] <= TOL and res['variants'][v]['sign_mismatch_valid'] == 0 for v in good)
    res['BROKEN_FAILS'] = bool(bad) and all(res['variants'][v]['max_rel'] > 1e-6 or res['variants'][v]['sign_mismatch_valid'] > 0 for v in bad)
    res['tol'] = TOL
    json.dump(res, open(f'{OUT}/check_b_{system}.json', 'w'), indent=1, default=float)
    log(f'CHECK b {system}: PASS {res["PASS"]} BROKEN_FAILS {res["BROKEN_FAILS"]}')


# ------------------------------------------------------------------------------------------ (c)
def hash_sign(cfg):
    """deterministic, batch-independent +-1 function of a configuration (the savings test's stand-in sign: no real sign cost)."""
    c = (np.asarray(cfg) > 0).astype(np.uint64); N = c.shape[1]
    w = (np.arange(N, dtype=np.uint64) * np.uint64(0x9E3779B97F4A7C15) + np.uint64(12345)) | np.uint64(1)
    h = (c * w[None]).sum(1, dtype=np.uint64); h ^= h >> np.uint64(29); h *= np.uint64(0xBF58476D1CE4E5B9); h ^= h >> np.uint64(32)
    return np.where(h & np.uint64(1), 1, -1).astype(np.int8)


def stage_time(system, M, variant):
    S = setup(system, M); L = S['L']; X = S['X']; th = S['theta']; amp = S['amp']; wc = 32 if L == 8 else 128
    dmax = PL._dmax_of(amp.lat, X)
    kw = dict(f0=S['f0'], mode='fix', walker_chunk=wc, sign_chunk=4096 if L == 6 else 64, dmax=dmax)
    if variant.startswith('hash'):                                                                        # hash (= dedup path), hash_old, hash_pack: no real sign cost, the amplitude lemma alone
        sf = hash_sign; path = variant[5:] or 'dedup'
    elif variant.startswith('lemma_'):
        sf = PL.PhaseSign.from_theta(th, L); path = variant[6:]
    else:
        sf = make_direct_sign_fn(th, L); path = variant[7:]
    reps = []
    for rep in range(1 if variant.startswith('direct') else 2):                                                                                  # rep 0 includes compile of new shapes
        PL.STATS.clear(); t = time.time()
        o = PL.aug_ratios(amp, X, sf, S['EP'], 4.0, eval_path=path, **kw); jax.block_until_ready(o['ratio']); reps.append(dict(sec=time.time() - t, stats=dict(PL.STATS)))
        log(f'time {system} {variant} M={len(X)} rep {rep}: {reps[-1]["sec"]:.1f}s ({reps[-1]["sec"] / len(X):.3f} s/walker)')
    res = dict(system=system, variant=variant, M=len(X), path=path, reps=reps, sec_per_walker=reps[-1]['sec'] / len(X), host=os.uname().nodename, threads=os.environ.get('OMP_NUM_THREADS', '-'),
               code_sha256=sha(os.path.join(HERE, 'pp2_lemma.py')), test_sha256=sha(__file__))
    json.dump(res, open(f'{OUT}/time_{system}_{variant}_M{M}.json', 'w'), indent=1, default=float)


# ------------------------------------------------------------------------------------------ (d2)
def stage_default():
    """default path of the working-tree pp2_lemma.py == the frozen 7c4dddf module, bit for bit (one-swap, logB, two_swap_ratios / fJ_fixed / aug_ratios on old, pack, dedup)."""
    os.environ.pop('PP2_DEFAULT_PATH', None)
    FZ = f'{R}/pfaffian_8x8build/signlemma/frozen_7c4dddf/pp2_lemma.py'
    sp = importlib.util.spec_from_file_location('pp2_frozen', FZ); old = importlib.util.module_from_spec(sp); sys.modules['pp2_frozen'] = old; sp.loader.exec_module(old)
    new = PL; assert new.DEFAULT_PATH == 'old'
    hs = lambda c: np.where(((np.asarray(c) > 0).astype(np.int64) @ (np.arange(np.asarray(c).shape[1]) * 7 + 3)) % 5 < 2, 1, -1)
    out = dict(frozen_sha256=sha(FZ), new_sha256=sha(os.path.join(HERE, 'pp2_lemma.py')), systems={})
    ok = True
    for name, mk, X in (('6x6_theta5', lambda m: m.PP2.from_cx_theta(np.load(THETA5)), None),
                        ('8x8_prod', lambda m: m.PP2.from_rbmpp_npz(PROD), np.load(WALK8)[:4].astype(float))):
        if X is None:
            rng = np.random.default_rng(3); X = np.ones((6, 36)); X[:, 18:] = -1
            for r in X: rng.shuffle(r)
        po, pn = mk(old), mk(new); N = X.shape[1]; EP = -0.4987565 * N; rec = {}
        for path in ('default', 'old', 'pack', 'dedup'):
            kw = {} if path == 'default' else dict(eval_path=path)
            a = old.aug_ratios(po, X, hs, EP, 4.0, 3.0, 'fix', **kw); b = new.aug_ratios(pn, X, hs, EP, 4.0, 3.0, 'fix', **kw)
            md = {}
            for k in a:
                if k == 'Y': continue
                A = np.asarray(a[k]); B = np.asarray(b[k])
                md[k] = bool(A.dtype == B.dtype and A.shape == B.shape and np.array_equal(A, B, equal_nan=True))
            rec[path] = dict(all_bit_equal=all(md.values()), keys_equal=sorted(a) == sorted(b), per_key=md)
            ok &= rec[path]['all_bit_equal'] and rec[path]['keys_equal']
        o1 = np.asarray(old.one_swap_ratios(po, X)); n1 = np.asarray(new.one_swap_ratios(pn, X)); ol = np.asarray(old.logB(po, X)); nl = np.asarray(new.logB(pn, X))
        rec['one_swap_bit_equal'] = bool(np.array_equal(o1, n1)); rec['logB_bit_equal'] = bool(np.array_equal(ol, nl)); ok &= rec['one_swap_bit_equal'] and rec['logB_bit_equal']
        out['systems'][name] = rec; log(f'default {name}: {all(rec[p]["all_bit_equal"] for p in ("default", "old", "pack", "dedup"))}')
    out['PASS'] = bool(ok); json.dump(out, open(f'{OUT}/check_default.json', 'w'), indent=1); log(f'CHECK default PASS {ok}')


if __name__ == '__main__':
    jax.config.update('jax_enable_x64', True)
    if STAGE == 'ref': stage_ref()
    elif STAGE == 'a': stage_a(sys.argv[3], int(sys.argv[4]), sys.argv[5:] or ['old', 'pack', 'dedup'])
    elif STAGE == 'b': stage_b(sys.argv[3], int(sys.argv[4]), sys.argv[5:] or ['old', 'pack', 'dedup'])
    elif STAGE == 'time': stage_time(sys.argv[3], int(sys.argv[4]), sys.argv[5])
    elif STAGE == 'default': stage_default()
    else: raise SystemExit(__doc__)
