"""pool_checks.py: CPU checks of pool_fit.py (PLAN_pfaffian.md "B design" item (iii), "V (narrowed)"), each with a deliberately broken version.
  a  : python pool_checks.py a OUT REF_DIR       vs pool_ref_r1.py (cx_r1fit.py's own code):
       (a1) the table fit's batches as a pool: loss, gradient (Jacobian path) and GN step (minSR and parameter-space branches) to 1e-10;
            broken: partner rows shifted by one entry.
       (a2) pool = every orbit of U with ALL its blocked partners (nb tables, my sign test, my labels), weighted as the table fit (p_x = n_x / n(U_g), 1/9 per
            decade, b_xy = lam_b |H_xy| T_y/T_x / Wbar_g): population loss and gradient vs cx_r1fit's to 1e-10;
            broken: (i) partners blocked under the opposite sign test, (ii) orbit weights without n_x.
  b  : python pool_checks.py b OUT POOL_DIR      pool vs the exact 6x6 tables: realised counts per (beta, decade) vs the exact multinomial expectation; reweighted
       within-decade means of ln|B|^2, c f_J = lnT - lnB, and W_x (from rho) vs exact configuration-uniform means (z-scores); the band-uniform pool's mass per
       |T|^2 decade (the table fit's decade labels) next to the table fit's 1/9;  broken: the same means without the importance weights.
  c  : python pool_checks.py c OUT POOL_SPEC      a 5-step CPU dry fit from a small pool: held-out and fixed-train loss must decrease; broken: break_negate_step."""
import os, sys, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.append('/project/theorie/a/A.Otaifi/Chatty/runs/compress6/code')
import pool_fit as PF
mode, OUT = sys.argv[1:3]; os.makedirs(OUT, exist_ok=True); T0 = time.time()
log = lambda m: print(f'[{time.time() - T0:6.0f}s] {m}', flush=True)
R5 = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r5'; T5 = f'{R5}/target_T5_sG.npy'; SG = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_lswo2/s_G.npy'
LB4 = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r4/pdb_tabs/tab_t4/round_1/la_table.npy'
rel = lambda a, b: float(np.max(np.abs(np.asarray(a) - np.asarray(b))) / max(np.max(np.abs(np.asarray(b))), 1e-300))
res = dict(mode=mode, pool_fit_sha256=PF.sha(os.path.join(HERE, 'pool_fit.py')))

if mode == 'a':
    import jax; jax.config.update('jax_enable_x64', True); import jax.numpy as jnp
    REF = sys.argv[3]; rj = json.load(open(f'{REF}/ref_r1.json')); rz = np.load(f'{REF}/ref_r1.npz'); th = jnp.asarray(rz['theta']); c = rj['c']
    ex = PF.Exact6(np.load(T5), np.load(SG)); lab = PF.TableLabels(ex, T5)
    mdl = PF.make_model('pp2'); net = PF.Net(mdl); per_sample, grad_from_G, solve_dir = PF._jfuns()
    for Bm in rj['spec']['B_list']:
        p = lambda k: rz[f'a1_{Bm}_{k}']; t0 = time.time()
        x = p('x'); y = p('y'); M, K = y.shape
        assert np.array_equal(ex.reps[p('idx')], x), 'batch x are the reps of idx'
        lx = lab(x); ly = lab(y.reshape(-1)).reshape(M, K)                    # labels through labels_fn(configs) (canonicalisation), not cx_r1fit's ls2 lookup
        wb = np.repeat(p('wb')[:, None], K, 1)

        def mine(yy, lyy):
            Xall = jnp.concatenate([jnp.asarray(x), jnp.asarray(yy).reshape(-1)]); S = net.fwd(th, Xall); G = net.jac(th, Xall)
            L = float(jnp.mean(per_sample(S, jnp.asarray(lx), jnp.asarray(lyy), jnp.asarray(wb), c)[0]))
            g, _ = grad_from_G(G, S, jnp.asarray(lx), jnp.asarray(lyy), jnp.asarray(wb), c, jnp.full(M, 1.0 / M))
            dP, dc, prms, sc, rr = solve_dir(G, S, jnp.asarray(lx), jnp.asarray(lyy), jnp.asarray(wb), c, 1e-2, 0.05)
            return L, np.asarray(g), np.asarray(dP), float(dc), np.asarray(S)
        L, g, dP, dc, S = mine(y, ly); r = rj[f'a1_{Bm}']
        d = dict(branch=r['branch'], M=M, labels_vs_ref_max_abs=float(max(np.max(np.abs(lx - p('lx'))), np.max(np.abs(ly - p('ly'))))), S_rel=rel(S, p('S')),
                 L_mine=L, L_ref=r['L'], L_rel=abs(L - r['L']) / abs(r['L']), grad_rel=rel(g, p('grad')), dP_rel=rel(dP, p('dP')), dc_abs=abs(dc - r['dc']))
        Lb, gb, dPb, _, _ = mine(np.roll(y, 1, 0), np.roll(ly, 1, 0))
        d['broken_shift'] = dict(L_rel=abs(Lb - r['L']) / abs(r['L']), grad_rel=rel(gb, p('grad')), dP_rel=rel(dPb, p('dP')))
        d['pass'] = bool(max(d['L_rel'], d['grad_rel'], d['dP_rel']) < 1e-10 and d['labels_vs_ref_max_abs'] < 1e-10 and min(d['broken_shift'].values()) > 1e-6)
        d['sec'] = time.time() - t0; res[f'a1_{Bm}'] = d; log(f'a1 B={Bm}: {d}')
    # (a2) population on U
    U = rz['a2_U']; gU = rz['a2_g']; ND = int(gU.max()) + 1; x = ex.reps[U]; LAMB = float(rj['spec'].get('lam_b', 1.0))

    def pop(flip_sign=False, no_n=False):
        ycfg, hw, valid, blocked, _, yo = ex.neighbours_fn(x, U)
        if flip_sign: blocked = valid & ~blocked
        lx = lab(x); lyall = lab.la[yo]
        w = np.where(blocked, hw[None, :] * np.exp(np.clip(lyall - lx[:, None], -700, 700)), 0.0); W = w.sum(1)
        a = np.zeros(U.size); Wb = np.zeros(ND)
        for k in range(ND):
            m = gU == k; q = (np.ones(m.sum()) if no_n else ex.n[U][m]); a[m] = q / q.sum() / ND; Wb[k] = np.sum(a[m] * ND * W[m])
        Kmax = int(blocked.sum(1).max()); yy = np.repeat(x[:, None], Kmax, 1); ly = np.repeat(lx[:, None], Kmax, 1); wb = np.zeros((U.size, Kmax))
        for i in range(U.size):
            j = np.nonzero(blocked[i])[0]; yy[i, :j.size] = ycfg[i, j]; ly[i, :j.size] = lab.la[yo[i, j]]; wb[i, :j.size] = LAMB * w[i, j] / Wb[gU[i]]
        Xall = jnp.concatenate([jnp.asarray(x), jnp.asarray(yy).reshape(-1)]); S = net.fwd(th, Xall); G = net.jac(th, Xall)
        Lx = per_sample(S, jnp.asarray(lx), jnp.asarray(ly), jnp.asarray(wb), c)[0]
        g, _ = grad_from_G(G, S, jnp.asarray(lx), jnp.asarray(ly), jnp.asarray(wb), c, jnp.asarray(a))
        return float(jnp.sum(jnp.asarray(a) * Lx)), np.asarray(g), dict(W=W, Wbar=Wb, nblk=blocked.sum(1), Kmax=Kmax)
    t0 = time.time(); L2, g2, info = pop(); r = rj['a2']
    d = dict(n_U=int(U.size), Kmax=info['Kmax'], nblk_equal_ref=bool(np.array_equal(info['nblk'], rz['a2_nbad'])), W_rel=rel(info['W'], rz['a2_W']),
             Wbar_rel=rel(info['Wbar'], r['Wbar']), L_mine=L2, L_ref=r['L'], L_rel=abs(L2 - r['L']) / abs(r['L']), grad_rel=rel(g2, rz['a2_grad']))
    Lf, gf, _ = pop(flip_sign=True); Ln, gn, _ = pop(no_n=True)
    d['broken_opposite_sign'] = dict(L_rel=abs(Lf - r['L']) / abs(r['L']), grad_rel=rel(gf, rz['a2_grad']))
    d['broken_no_n'] = dict(L_rel=abs(Ln - r['L']) / abs(r['L']), grad_rel=rel(gn, rz['a2_grad']))
    d['pass'] = bool(max(d['L_rel'], d['grad_rel'], d['W_rel'], d['Wbar_rel']) < 1e-10 and d['nblk_equal_ref']
                     and min(d['broken_opposite_sign']['L_rel'], d['broken_no_n']['L_rel']) > 1e-6)
    d['sec'] = time.time() - t0; res['a2'] = d; log(f'a2: {d}')
    res['pass'] = bool(all(res[k]['pass'] for k in res if k.startswith('a1_') or k == 'a2'))

elif mode == 'b':
    PD = sys.argv[3]; P = PF.load_pool(f'{PD}/pool.npz'); SP = json.load(open(f'{PD}/pool_summary.json'))['spec']
    n = np.load(f'{PF.CSRD6}/norb.npy').astype(np.float64); lB = np.load(SP['la_B']).astype(np.float64); s = np.load(SP['sign']).astype(np.float64)
    lT = PF.norm_la_np(np.load(SP['target']).astype(np.float64), n); D = n.size; med = float(P['med2lnB']); betas = P['betas'].tolist(); decs = P['decs_rel'].tolist()
    drel = np.floor((2 * lB - med) / PF.LN10).astype(np.int64); dabsT = np.floor(2 * lT / PF.LN10).astype(np.int64)
    # exact W_x = sum_blocked |H_xy| T_y/T_x over every orbit (configuration-uniform means need it on all orbits of a decade)
    t1 = np.load(f'{PF.TAB6}/nb_J1.npy', mmap_mode='r'); t2 = np.load(f'{PF.TAB6}/nb_J2.npy', mmap_mode='r'); Wx = np.empty(D)
    WXC = os.path.join(OUT, 'W_exact.npy'); WXS = PF.sha(SP['target']) + PF.sha(SP['sign'])
    if os.path.exists(WXC) and open(WXC + '.key').read() == WXS: Wx = np.load(WXC); D_todo = 0
    else: D_todo = D
    for i0 in range(0, D_todo, 262144):
        i1 = min(D, i0 + 262144); t = np.concatenate([t1[i0:i1], t2[i0:i1]], 1); v = t >= 0; tt = np.where(v, t, 0)
        blk = v & (s[i0:i1, None] * s[tt] > 0); Wx[i0:i1] = np.sum(np.where(blk, PF.HW6[None, :] * np.exp(np.clip(lT[tt] - lT[i0:i1, None], -700, 700)), 0.0), 1)
    if D_todo: np.save(WXC, Wx); open(WXC + '.key', 'w').write(WXS)
    log('exact W_x done')
    HO = PF.ho_hash(np.arange(D))                                        # the pool's training entries are the non-HO orbits: compare with the exact means over those
    # (b1) counts per (beta, decade) vs the exact multinomial expectation (all entries, train + held-out)
    Sb = [int((P['chain'] == q).sum()) for q in range(len(betas))]; b1 = {}; zmax = 0.0
    for q, b in enumerate(betas):
        lw = np.log(n) + 2 * b * (lB - lB.max()); pr = np.exp(lw - lw.max()); pr /= pr.sum()
        for d in range(int(P['dec'].max()), int(P['dec'].min()) - 1, -1):
            pe = float(pr[drel == d].sum()); e = Sb[q] * pe; o = int(((P['dec'] == d) & (P['chain'] == q)).sum())
            z = (o - e) / np.sqrt(max(e * (1 - pe), 1e-12)) if e > 0 else (0.0 if o == 0 else np.inf)
            b1[f'{b}:{d}'] = dict(obs=o, exp=e, z=float(z));
            if e >= 5: zmax = max(zmax, abs(z))
    nz = [v['z'] for v in b1.values() if v['exp'] >= 5]
    res['b1'] = dict(cells=b1, max_abs_z_exp_ge5=zmax, n_cells=len(nz), chi2=float(np.sum(np.square(nz))))
    log(f'b1: max |z| {zmax:.2f} over {len(nz)} cells, chi2 {res["b1"]["chi2"]:.1f}')
    # (b2) reweighted within-decade means vs exact configuration-uniform means; broken: unweighted
    tr = ~P['ho']; gq = dict(lnB2=2 * P['lnB_x'], cfJ=P['lnT_x'] - P['lnB_x'], W=P['rho'].mean(1))
    ex_ = dict(lnB2=2 * lB, cfJ=lT - lB, W=Wx); b2 = {}; zmx = 0.0; zbr = 0.0
    lTx_off = float(np.mean(P['lnT_x'] - lT[P['x_orb']]))               # pool labels are norm_la'd the same way: offset 0
    for d in decs:
        m = (P['dec'] == d) & tr; w = P['w_in'][m]; me = (drel == d) & ~HO; rec = {}
        for k in gq:
            g = gq[k][m]; mu = float(np.sum(w * g)); se = float(np.sqrt(np.sum(w ** 2 * (g - mu) ** 2))); muE = float(np.sum(n[me] * ex_[k][me]) / n[me].sum())
            raw = float(np.mean(g)); seR = float(np.std(g) / np.sqrt(g.size))
            rec[k] = dict(pool=mu, se=se, exact=muE, z=(mu - muE) / se, broken_unweighted=raw, broken_z=(raw - muE) / seR)
            zmx = max(zmx, abs(rec[k]['z'])); zbr = max(zbr, abs(rec[k]['broken_z']))
        rec['Wbar_pool'] = float(P['Wbar'][decs.index(d)]); rec['n_train'] = int(m.sum()); b2[str(d)] = rec
    res['b2'] = dict(by_decade=b2, max_abs_z=zmx, broken_max_abs_z=zbr, label_offset_vs_table=lTx_off)
    log(f'b2: max |z| {zmx:.2f} (broken unweighted {zbr:.1f})')
    # (b3) the band-uniform pool mass per |T|^2 decade (table fit's labels) and the exact mass of the same weighting; the table fit puts 1/9 on -5..-13
    m = tr & np.isin(P['dec'], decs); dT = dabsT[P['x_orb'][m]]; wm = P['w_in'][m] / len(decs)
    exact = {}
    for d in decs:
        me = (drel == d) & ~HO
        for q in np.unique(dabsT[me]): exact[int(q)] = exact.get(int(q), 0.0) + float(n[me & (dabsT == q)].sum() / n[me].sum()) / len(decs)
    res['b3'] = dict(pool_mass_by_T_decade={int(q): float(wm[dT == q].sum()) for q in np.unique(dT)}, exact_mass_by_T_decade=exact,
                     table_fit=({str(q): 1 / 9 for q in range(-5, -14, -1)}), mass_outside_T_m5_m13=float(wm[(dT > -5) | (dT < -13)].sum()))
    log(f'b3: {res["b3"]}')
    res['pass'] = bool(zmx < 4.0 and zbr > 10.0 and res['b1']['max_abs_z_exp_ge5'] < 5.0)

elif mode == 'c':
    SPF = sys.argv[3]; SP = json.load(open(SPF)); PD = os.path.join(OUT, 'pool_small')
    if not os.path.exists(f'{PD}/pool.npz'): PF.build6(PD, SP['pool_spec'])
    runs = {}
    for tag, extra in (('ok', {}), ('broken_negate_step', dict(break_negate_step=True))):
        fs = dict(SP['fit_spec'], pool=f'{PD}/pool.npz', **extra); r = PF.fit(os.path.join(OUT, f'fit_{tag}'), fs)
        cv = r['fit']['curve']; runs[tag] = dict(L_ho=[q['L_ho'] for q in cv], L_tr=[q['L_tr'] for q in cv], accepted=[h['ok'] for h in r['fit']['hist']],
                                                verdicts=[h['verdict'] for h in r['fit']['hist']], sec_per_step=r['fit']['median_sec_per_step'])
        log(f'c {tag}: {runs[tag]}')
    ok = runs['ok']; br = runs['broken_negate_step']
    res['c'] = runs; res['pass'] = bool(ok['L_ho'][-1] < ok['L_ho'][0] and ok['L_tr'][-1] < ok['L_tr'][0] and not (br['L_ho'][-1] < br['L_ho'][0] * (1 - 1e-9)))
res['sec'] = time.time() - T0; PF.jdump(os.path.join(OUT, f'check_{mode}.json'), res); log(f'CHECK {mode} pass {res.get("pass")}')
