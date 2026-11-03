"""pp2_lemma_test.py: checks (a)-(d) of pp2_lemma.py (B design item (i)), each with a deliberately broken version that must fail.
usage: JAX_PLATFORMS=cpu python pp2_lemma_test.py OUTDIR {a|b|c|d|all}    -> OUTDIR/check_<x>.json, PASS / BROKEN_FAILS per check
(a) L=6, 20 random configs, two models: round-5 pp2_s1 theta_5 ('outer' RBM, full projection, lam != 0) and a random two-PP + 'inner'
    complex RBM, translations x flip (the 8x8 production structure at L=6).  Every one-swap ratio and every two-swap ratio vs direct
    logB (slogdet of all images) of the endpoint, max rel < 1e-10.  Broken: the slot term P+Q of the lemma sign dropped.
(b) L=6, 1000 random orbit reps: Re logB(theta_5) vs la_table (pp2_s1 tab_t5) up to a constant, max |dev| < 1e-10.  Broken: lam = 0.
(c) L=8, production params (lam = 0, no P_2), 1000 random S^z=0 configs: logB vs rbmpp_lib.make_fns(...).logpsi (the production
    evaluator, experiments/rbmpp_6x6/rbmpp_lib.py), |d Re| and |d Im mod 2 pi| < 1e-10.  Broken: RBM bias conjugated.
    Extra: one- and two-swap ratios at L=8 (production + random P_2) on 2 configs vs direct, max rel < 1e-10.
(d) L=6, 1000 random orbit reps: fJ_fixed(mode 'old', c=4, E' = E_FN of pp2_s1) at x vs r5_bare.fJ_of arithmetic on the table rows
    (la_table, s_G, nb tables) with |d f_J| < 1e-9; f_J(y) for every neighbour of 20 of them vs fJ_of at the neighbour's row.
    Broken: kept and blocked bonds exchanged.
"""
import os, sys, json, time, hashlib
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import jax.numpy as jnp
import pp2_lemma as PL

OUT = sys.argv[1]; WHICH = sys.argv[2] if len(sys.argv) > 2 else 'all'
os.makedirs(OUT, exist_ok=True); T0 = time.time()
R5 = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r5/pp2_s1/tab'
THETA5 = f'{R5}/theta_5.npy'; LA = f'{R5}/tab_t5/round_1/la_table.npy'
SG = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_lswo2/s_G.npy'
REPS = '/project/theorie/a/A.Otaifi/Chatty/runs/ll6/data/psi0_6x6_table.npz'
TAB = '/scratch/a/A.Otaifi/ansatz_runs/c2_t0b_6x6/tables'
PROD = '/scratch/a/A.Otaifi/rbmpp8/rb8_prod/params_final.npz'
PRODLIB = os.path.join(HERE, '..', '..', 'experiments', 'rbmpp_6x6')
EP_R5 = json.load(open('/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r5/readout/readout.json'))['tables']['pp2_s1']['E_FN']['E']
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()


def log(m):
    s = f'[{time.time() - T0:7.0f}s] {m}'; print(s, flush=True)
    open(f'{OUT}/test.log', 'a').write(s + '\n')


def rand_cfg(rng, B, N):
    S = np.ones((B, N)); S[:, N // 2:] = -1
    for r in S: rng.shuffle(r)
    return S


def swap(x, b, lat):
    y = x.copy(); i, j = lat.BI[b], lat.BJ[b]; y[i], y[j] = x[j], x[i]; return y


def direct_ratios(p, x, lat, d=None):
    """direct logB of every one-swap neighbour and (if d) every valid two-swap endpoint of x."""
    lx = complex(PL.logB(p, x[None])[0])
    v = np.nonzero(x[lat.BI] != x[lat.BJ])[0]
    Y = np.array([swap(x, b, lat) for b in v]); r1 = np.exp(np.asarray(PL.logB(p, Y)) - lx)
    if d is None: return v, r1
    a = np.asarray(d['a']); am = np.asarray(d['am']); bb = np.asarray(d['b']); bm = np.asarray(d['bm'])
    Z, ix = [], []
    for i in np.nonzero(am)[0]:
        y = swap(x, a[i], lat)
        for j in np.nonzero(bm[i])[0]:
            assert y[lat.BI[bb[i, j]]] != y[lat.BJ[bb[i, j]]]; Z.append(swap(y, bb[i, j], lat)); ix.append((i, j))
    r2 = np.exp(np.asarray(PL.logB(p, np.array(Z))) - lx)
    return v, r1, np.array(ix), r2


def relmax(a, b): return float(np.max(np.abs(a - b) / np.abs(b)))


def model_inner6(seed=7):
    rng = np.random.default_rng(seed); N = 36
    f1 = (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))) / np.sqrt(2)
    f2 = (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))) / np.sqrt(2)
    b = 0.05 * (rng.standard_normal(16) + 1j * rng.standard_normal(16)); W = 0.1 * (rng.standard_normal((N, 16)) + 1j * rng.standard_normal((N, 16)))
    p = PL.PP2(f1, L=6, proj='translations', rbm='inner', b=b, W=W)
    x0 = rand_cfg(np.random.default_rng(5), 1, N)
    l1 = complex(PL.logB(p, x0)[0]); l2 = complex(PL.logB(PL.PP2(f2, L=6, proj='translations', rbm='inner', b=b, W=W), x0)[0])
    lam = 0.3 * np.exp(0.7j) * np.exp(l1 - l2)                              # comparable weights of the two terms (walker_cost.py)
    return p.with_p2(f2, lam)


# ------------------------------------------------------------------ (a)
def check_a():
    rng = np.random.default_rng(11); res = {}
    models = dict(r5_outer_full=PL.PP2.from_cx_theta(np.load(THETA5)), inner_transl=model_inner6())
    for name, p in models.items():
        lat = p.lat; rows = []
        for c in range(20):
            x = rand_cfg(rng, 1, lat.N)
            r1 = np.asarray(PL.one_swap_ratios(p, x))[0]; r1b = np.asarray(PL.one_swap_ratios(p, x, _broken=True))[0]
            d = PL.two_swap_ratios(p, x); db = PL.two_swap_ratios(p, x, _broken=True)
            v, rd1, ix, rd2 = direct_ratios(p, x[0], lat, jax_first(d))
            R2 = np.asarray(d['r2'])[0][ix[:, 0], ix[:, 1]]; R2b = np.asarray(db['r2'])[0][ix[:, 0], ix[:, 1]]
            K = np.asarray(d['k'])[0][ix[:, 0], ix[:, 1]]
            a = np.asarray(d['a'])[0]; am = np.asarray(d['am'])[0]
            r1c = dict(zip(a[am].tolist(), np.asarray(d['r1'])[0][am].tolist()))
            row = dict(d=int(len(v)), n2=int(len(ix)), nk={int(k): int((K == k).sum()) for k in (0, 1, 2)},
                       one=relmax(r1[v], rd1), one_from_two=relmax(np.array([r1c[b] for b in v]), rd1),
                       two=relmax(R2, rd2), two_k={int(k): relmax(R2[K == k], rd2[K == k]) for k in (1, 2) if (K == k).any()},
                       one_broken=relmax(r1b[v], rd1), two_broken=relmax(R2b, rd2),
                       aligned_zero=bool(np.all(r1[np.setdiff1d(np.arange(lat.NB), v)] == 0)))
            rows.append(row); log(f'(a) {name} conf {c}: {row}')
        res[name] = dict(max_one=max(r['one'] for r in rows), max_one_from_two=max(r['one_from_two'] for r in rows),
                         max_two=max(r['two'] for r in rows), n_two=sum(r['n2'] for r in rows),
                         min_one_broken=min(r['one_broken'] for r in rows), min_two_broken=min(r['two_broken'] for r in rows),
                         aligned_zero=all(r['aligned_zero'] for r in rows), rows=rows)
    ok = all(max(v['max_one'], v['max_one_from_two'], v['max_two']) < 1e-10 and v['aligned_zero'] for v in res.values())
    brk = all(min(v['min_one_broken'], v['min_two_broken']) > 1e-3 for v in res.values())
    return dict(PASS=ok, BROKEN_FAILS=brk, tol=1e-10, models=res)


def jax_first(d):
    return {k: np.asarray(v)[0] for k, v in d.items() if v is not None}


# ------------------------------------------------------------------ orbit helpers (6x6 sector)
_REPS = None


def reps():
    global _REPS
    if _REPS is None: _REPS = np.load(REPS)['reps']
    return _REPS


SH = np.arange(36, dtype=np.uint64)


def sig_of(codes):
    return (((np.asarray(codes, np.uint64)[:, None] >> SH[None]) & np.uint64(1)).astype(np.float64) * 2 - 1)


def canon_idx(cfg, chunk=2048):
    lat = PL.lattice(6, 'full'); R = reps(); out = np.empty(len(cfg), np.int64); POW = np.uint64(1) << SH
    for i in range(0, len(cfg), chunk):
        c = np.asarray(cfg[i:i + chunk]); im = lat.FL[None, :, None] * c[:, lat.IMG]
        code = ((im > 0).astype(np.uint64) * POW).sum(-1).min(1)
        ix = np.searchsorted(R, code); assert np.all(R[np.minimum(ix, len(R) - 1)] == code), 'config not in the sector reps'
        out[i:i + chunk] = ix
    return out


# ------------------------------------------------------------------ (b)
def check_b():
    rng = np.random.default_rng(22); R = reps(); la = np.load(LA, mmap_mode='r')
    rows = np.sort(rng.choice(len(R), 1000, replace=False)); X = sig_of(R[rows])
    assert np.all(canon_idx(X) == rows), 'rep convention'
    p = PL.PP2.from_cx_theta(np.load(THETA5))
    lB = np.real(np.asarray(PL.logB(p, X))); dev = lB - la[rows]; dev -= np.median(dev)
    p0 = PL.PP2(p.f1, p.f2, 0.0, p.b, p.W, L=6, proj='full', rbm='outer')
    lb0 = np.real(np.asarray(PL.logB(p0, X))); dv0 = lb0 - la[rows]; dv0 -= np.median(dv0)
    out = dict(n=1000, la_table=LA, la_sha256=sha(LA), theta5_sha256=sha(THETA5), lam=[float(np.real(p.lam)), float(np.imag(p.lam))],
               const=float(np.median(lB - la[rows])), max_abs_dev=float(np.abs(dev).max()), broken_lam0_max_abs_dev=float(np.abs(dv0).max()))
    out['PASS'] = out['max_abs_dev'] < 1e-10; out['BROKEN_FAILS'] = out['broken_lam0_max_abs_dev'] > 1e-3
    log(f'(b) {out}'); return out


# ------------------------------------------------------------------ (c)
def check_c():
    sys.path.insert(0, PRODLIB)
    import rbmpp_lib as RL
    z = np.load(PROD); proj = str(z['proj']); lat = RL.Lattice(8, 0.5, proj)
    pr = {k: jnp.asarray(z[k]) for k in ('f', 'bre', 'bim', 'Wre', 'Wim')}
    fns = RL.make_fns(lat, use_rbm=bool(z['rbm']))
    import jax
    lp = jax.jit(jax.vmap(fns['logpsi'], in_axes=(None, 0)))
    rng = np.random.default_rng(33); X = rand_cfg(rng, 1000, 64)
    ref = np.concatenate([np.asarray(lp(pr, jnp.asarray(X[i:i + 100]))) for i in range(0, 1000, 100)])
    p = PL.PP2.from_rbmpp_npz(PROD); mine = np.asarray(PL.logB(p, X))
    dre = np.abs(np.real(mine - ref)); dim = np.abs(np.angle(np.exp(1j * np.imag(mine - ref))))
    pb = PL.PP2(p.f1, None, 0.0, np.conj(p.b), p.W, L=8, proj=p.proj, rbm='inner'); mb = np.asarray(PL.logB(pb, X))
    brk = float(np.max(np.abs(np.real(mb - ref)) + np.abs(np.angle(np.exp(1j * np.imag(mb - ref))))))
    out = dict(params=PROD, params_sha256=sha(PROD), proj=proj, n_images=int(p.lat.nI), prodlib=os.path.realpath(os.path.join(PRODLIB, 'rbmpp_lib.py')),
               prodlib_sha256=sha(os.path.join(PRODLIB, 'rbmpp_lib.py')), n=1000, max_dRe=float(dre.max()), max_dIm_mod2pi=float(dim.max()),
               broken_conj_b_max=brk)
    out['PASS'] = max(out['max_dRe'], out['max_dIm_mod2pi']) < 1e-10; out['BROKEN_FAILS'] = brk > 1e-3
    # ratios at L=8: production + random P_2 (lam scaled to comparable weight)
    r = np.random.default_rng(44); f2 = (r.standard_normal((64, 64)) + 1j * r.standard_normal((64, 64))) * 0.05
    x0 = rand_cfg(np.random.default_rng(5), 1, 64)
    l1 = complex(PL.logB(p, x0)[0]); l2 = complex(PL.logB(PL.PP2(f2, L=8, proj=p.proj, rbm='inner', b=p.b, W=p.W), x0)[0])
    p2 = p.with_p2(f2, 0.3 * np.exp(0.7j) * np.exp(l1 - l2)); rr = []
    for c in range(2):
        x = rand_cfg(r, 1, 64); d = PL.two_swap_ratios(p2, x); r1 = np.asarray(PL.one_swap_ratios(p2, x))[0]
        v, rd1, ix, rd2 = direct_ratios(p2, x[0], p2.lat, jax_first(d))
        R2 = np.asarray(d['r2'])[0][ix[:, 0], ix[:, 1]]
        rr.append(dict(one=relmax(r1[v], rd1), two=relmax(R2, rd2), n2=int(len(ix)))); log(f'(c) L=8 ratios conf {c}: {rr[-1]}')
    out['L8_ratios'] = rr; out['PASS_L8_ratios'] = max(max(q['one'], q['two']) for q in rr) < 1e-10
    log(f'(c) {out}'); return out


# ------------------------------------------------------------------ (d)
def fJ_of_rows(rows, lF, s1, EP, t1, t2, HW, JW):
    """r5_bare.fJ_of arithmetic restricted to rows (verbatim per-row formula)."""
    t = np.concatenate([t1[rows], t2[rows]], 1); valid = t >= 0; tt = np.where(valid, t, 0)
    ss = s1[rows, None] * s1[tt]; blk = valid & (ss > 0); kpt = valid & (ss < 0)
    lmax = lF.max()
    ratio = np.exp(np.clip((lF[tt] - lmax) - (lF[rows, None] - lmax), -700, 700))
    VA = (blk * HW[None, :] * ratio).sum(1); SK = (kpt * HW[None, :] * ratio).sum(1)
    den = 27.0 - 0.5 * (valid * JW[None, :]).sum(1) + VA - EP
    with np.errstate(divide='ignore', invalid='ignore'): return np.log(SK) - np.log(den), valid.sum(1)


def check_d():
    rng = np.random.default_rng(55); R = reps(); lF = np.load(LA); s1 = np.load(SG).astype(np.float64)
    t1 = np.load(f'{TAB}/nb_J1.npy', mmap_mode='r'); t2 = np.load(f'{TAB}/nb_J2.npy', mmap_mode='r')
    HW = np.concatenate([np.full(72, 0.5), np.full(72, 0.25)]); JW = np.concatenate([np.ones(72), np.full(72, 0.5)])
    rows = np.sort(rng.choice(len(R), 1000, replace=False)); X = sig_of(R[rows])
    p = PL.PP2.from_cx_theta(np.load(THETA5)); C = 4.0
    sign_fn = lambda cfg: s1[canon_idx(cfg)]
    ref, nvalid = fJ_of_rows(rows, lF, s1, EP_R5, t1, t2, HW, JW)
    nanti = np.sum(X[:, PL.lattice(6, 'full').BI] != X[:, PL.lattice(6, 'full').BJ], 1)
    mine = []; brk = []
    for i in range(0, 1000, 100):
        o = PL.fJ_fixed(p, X[i:i + 100], sign_fn, EP_R5, C, mode='old', at_y=False); mine.append(np.asarray(o['fx']))
        assert np.allclose(np.asarray(o['add_x']), C * np.where(np.isfinite(mine[-1]), mine[-1], 0))
        ob = PL.fJ_fixed(p, X[i:i + 100], sign_fn, EP_R5, C, mode='old', at_y=False, _broken=True, _two=o); brk.append(np.asarray(ob['fx']))
    mine = np.concatenate(mine); brk = np.concatenate(brk)
    fin = np.isfinite(ref)
    out = dict(n=1000, EP=EP_R5, c=C, mode='old', s_G=SG, s_G_sha256=sha(SG), table_valid_equals_anti=bool(np.all(nvalid == nanti)),
               n_nonfinite_ref=int((~fin).sum()), nonfinite_agree=bool(np.array_equal(np.isfinite(mine), fin)),
               max_abs_fx=float(np.abs(mine[fin] - ref[fin]).max()), broken_min_abs_fx=None,
               broken_max_abs_fx=float(np.nanmax(np.abs(brk[fin] - ref[fin]))))
    # f_J(y) on 20 orbits: every neighbour vs fJ_of at the neighbour's row
    ey = []; ny = 0
    for i in range(20):
        o = PL.fJ_fixed(p, X[i:i + 1], sign_fn, EP_R5, C, mode='old', at_y=True)
        am = np.asarray(o['am'])[0]; Y = np.asarray(o['Y'])[0][am].astype(np.float64)
        ry, _ = fJ_of_rows(canon_idx(Y), lF, s1, EP_R5, t1, t2, HW, JW); fy = np.asarray(o['fy'])[0][am]
        f_ = np.isfinite(ry); assert np.array_equal(f_, np.isfinite(fy)); ey.append(float(np.abs(fy[f_] - ry[f_]).max())); ny += int(am.sum())
        # the fixed modes run (no reference here: unit sanity only)
        for md in ('a1', 'fix'):
            q = PL.aug_ratios(p, X[i:i + 1], sign_fn, EP_R5, C, f0=2.0, mode=md); assert np.all(np.isfinite(np.asarray(q['ratio'])))
    out['n_y'] = ny; out['max_abs_fy'] = max(ey)
    out['PASS'] = out['max_abs_fx'] < 1e-9 and out['max_abs_fy'] < 1e-9 and out['nonfinite_agree']
    out['BROKEN_FAILS'] = out['broken_max_abs_fx'] > 1e-3
    log(f'(d) {out}'); return out


if __name__ == '__main__':
    todo = 'abcd' if WHICH == 'all' else WHICH
    for w in todo:
        r = dict(check=w, code_sha256=sha(os.path.join(HERE, 'pp2_lemma.py')), test_sha256=sha(__file__), host=os.uname().nodename)
        r.update({'a': check_a, 'b': check_b, 'c': check_c, 'd': check_d}[w]())
        json.dump(r, open(f'{OUT}/check_{w}.json', 'w'), indent=1, default=float)
        log(f'CHECK ({w}) PASS {r["PASS"]} BROKEN_FAILS {r["BROKEN_FAILS"]}' + (f' PASS_L8_ratios {r["PASS_L8_ratios"]}' if w == 'c' else ''))
