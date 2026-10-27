"""q_ownnode.py: pre-registered test q (PLAN_pfaffian.md "Test q"): does A's own sign flip across the costly bonds of the O2b copy?

usage: JAX_PLATFORMS=cpu python q_ownnode.py OUTDIR [NSAMP] [NSEC]
Stage 1 (numpy, mmap nb tables): one pass over all orbit reps exactly as bond_split.cost_pass (same lp normalisation, violating
        pairs of s2, small side xs / large side ys, c, class q = ln r - ln r*); weighted reservoir (with replacement, probability
        proportional to c, per class '<1/3' and '1/3-3') of NSAMP pairs.  Also whole-pass: total cost, class cost, cost-weighted
        e_x histogram (bin 0.01, as bond_guide.py) with e = norm(la_table) - norm(target_la).
Stage 2 (JAX CPU, float64): signed projected sum A of the O2b network (log_one / logA_one copied from cx_t1s.py l.140-180 with
        the sign of the sum returned in addition) at the sampled reps; t = sgn A * s2; checks; q, q_ref, q_any, warm-start PP.
"""
import os, sys, json, time, hashlib
import numpy as np
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
CODE = '/project/theorie/a/A.Otaifi/Chatty/runs/compress6/runs/o2b_17431886/code'
sys.path.insert(0, CODE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
assert hashlib.sha256(open(f'{CODE}/rbmpp_lib.py', 'rb').read()).hexdigest() == '6d09da0665a4d13a2420c0c0ad9ae14862ea111e07b0122ccef23a1f04782a4a'

OUT = sys.argv[1]; NS_ = int(sys.argv[2]) if len(sys.argv) > 2 else 5000; NSEC = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
os.makedirs(OUT, exist_ok=True)
T0 = time.time()
LOGF = open(os.path.join(OUT, 'q_ownnode.log'), 'a')


def log(m):
    s = f'[{time.time() - T0:7.0f}s] {m}'; print(s, flush=True); LOGF.write(s + '\n'); LOGF.flush()


TAB = '/scratch/a/A.Otaifi/ansatz_runs/c2_t0b_6x6/tables'
CSRD = '/project/theorie/a/A.Otaifi/Chatty/runs/stall6/csr'
TABLE = '/project/theorie/a/A.Otaifi/Chatty/runs/stall6/data/psi0_6x6_table.npz'
S2 = '/project/theorie/a/A.Otaifi/Chatty/runs/compress6/runs/refresh_17409419/s2.npy'
RD = '/project/theorie/a/A.Otaifi/Chatty/runs/compress6/runs/o2b_17431886/round_1'
PPZ = '/project/theorie/a/A.Otaifi/Chatty/runs/rbmpp6/runs/rpT1_pp_17163493/params_final.npz'
NS = 36
HW = np.concatenate([np.full(72, 0.5), np.full(72, 0.25)])
LN3 = np.log(3.0)
NEB = 4001; EB0 = -20.0; EBW = 0.01        # e_x histogram: bins of 0.01 on [-20, 20]


def norm_la(la, n):
    la = la - la.max(); return la - 0.5 * np.log((n * np.exp(2 * la)).sum())


# ===================================================================================== stage 1
def stage1():
    rng = np.random.default_rng(12345)
    n = np.load(f'{CSRD}/norb.npy').astype(np.float64); s = np.load(S2).astype(np.float64)
    la_phi = np.load(f'{RD}/target_la.npy'); la_cp = np.load(f'{RD}/la_table.npy')
    D = n.shape[0]; assert n.sum() == 9075135300
    lp = la_phi - la_phi.max(); lp = lp - 0.5 * np.log((n * np.exp(2 * lp)).sum())     # as cost_pass
    lpn = norm_la(la_phi, n); lcn = norm_la(la_cp, n)
    sa = s
    t1 = np.load(f'{TAB}/nb_J1.npy', mmap_mode='r'); t2 = np.load(f'{TAB}/nb_J2.npy', mmap_mode='r')
    R = 65536
    cls_tot = np.zeros(4); tot = 0.0; npair = 0
    hexw = np.zeros(NEB); hexn = np.zeros(NEB)               # e_x histogram, class '<1/3' (cost weights / counts)
    W = [0.0, 0.0]
    slot = [dict(xs=np.full(NS_, -1, np.int64), ys=np.full(NS_, -1, np.int64), kk=np.full(NS_, -1, np.int64), c=np.zeros(NS_), q=np.zeros(NS_), ex=np.zeros(NS_)) for _ in range(2)]
    for i0 in range(0, D, R):
        i1 = min(D, i0 + R)
        t = np.concatenate([t1[i0:i1], t2[i0:i1]], 1); valid = t >= 0; tt = np.where(valid, t, 0)
        viol = valid & (sa[i0:i1, None] * sa[tt] > 0)
        rr, kk = np.nonzero(viol); x = rr + i0; y = tt[rr, kk]; H = HW[kk]; del t, valid, tt, viol, rr
        lpx, lpy = lp[x], lp[y]; lcx, lcy = la_cp[x], la_cp[y]
        sw = lpx > lpy
        ls_, ll_ = np.where(sw, lpy, lpx), np.where(sw, lpx, lpy)
        cs_, cl_ = np.where(sw, lcy, lcx), np.where(sw, lcx, lcy)
        xs = np.where(sw, y, x); ys = np.where(sw, x, y)
        lrs = np.clip(ls_ - ll_, -300, 0); lr = np.clip(cs_ - cl_, -300, 300)
        rs = np.exp(lrs); r = np.exp(lr)
        c = H * np.exp(2 * ll_) * (r - rs) ** 2 / r * n[x] * 0.5
        tot += c.sum(); npair += c.size
        q = lr - lrs; qb = np.where(q < -LN3, 0, np.where(q < LN3, 1, np.where(q < np.log(30), 2, 3)))
        cls_tot += np.bincount(qb, weights=c, minlength=4)
        ex = lcn[xs] - lpn[xs]
        m0 = qb == 0
        ib = np.clip(np.floor((ex[m0] - EB0) / EBW).astype(int), 0, NEB - 1)
        hexw += np.bincount(ib, weights=c[m0], minlength=NEB); hexn += np.bincount(ib, minlength=NEB)
        for j, m in ((0, qb == 0), (1, qb == 1)):
            cj = c[m]; w = cj.sum()
            if w <= 0: continue
            W[j] += w
            rep = rng.random(NS_) < w / W[j]; k = int(rep.sum())
            if k == 0: continue
            cdf = np.cumsum(cj); pick = np.minimum(np.searchsorted(cdf, rng.random(k) * cdf[-1]), cj.size - 1)
            sl = np.nonzero(rep)[0]; S = slot[j]
            S['xs'][sl] = xs[m][pick]; S['ys'][sl] = ys[m][pick]; S['kk'][sl] = kk[m][pick]; S['c'][sl] = cj[pick]; S['q'][sl] = q[m][pick]; S['ex'][sl] = ex[m][pick]
        if (i0 // R) % 20 == 0: log(f'pass {i1}/{D}  cost/site so far {tot / NS:.4e}')
    res = dict(total=tot, total_site=tot / NS, npairs=npair, cls_cost=cls_tot.tolist(), cls_share=(cls_tot / tot).tolist())
    # cost-weighted median of e_x (class <1/3), same convention as bond_guide.wq (bin centre)
    cw = np.cumsum(hexw) / hexw.sum(); ib = int(np.searchsorted(cw, 0.5)); res['ex_median_costw_lt13'] = EB0 + (ib + 0.5) * EBW
    res['ex_mean_costw_lt13'] = float((hexw * (EB0 + (np.arange(NEB) + 0.5) * EBW)).sum() / hexw.sum())
    np.savez(os.path.join(OUT, 'sample.npz'), **{f'{k}{j}': slot[j][k] for j in range(2) for k in slot[j]})
    json.dump(res, open(os.path.join(OUT, 'stage1.json'), 'w'), indent=1)
    log(f'stage1 done: {res}')
    return res


# ===================================================================================== stage 2
def stage2(res1):
    import jax, jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    from rbmpp_lib import Lattice
    N = 36
    AR = jnp.arange(N, dtype=jnp.uint64)

    def bits(Sx):                                           # cx_lib.bits
        return (((Sx[:, None] >> AR[None, :]) & jnp.uint64(1)).astype(jnp.float32) * 2 - 1)
    NBR = jnp.asarray(np.array([[((x + dx) % 6) + 6 * ((y + dy) % 6) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
                                for y in range(6) for x in range(6)], np.int32))            # cx_lib.NBR
    lat = Lattice(6, 0.5, 'full')
    IMG = lat.IMG; FLP = lat.FL.astype(jnp.float64); NH = N // 2
    xy = np.array([(i % 6, i // 6) for i in range(N)])
    DISP = jnp.asarray(np.array([[((xy[j, 0] - xy[i, 0]) % 6) + 6 * ((xy[j, 1] - xy[i, 1]) % 6) for j in range(N)]
                                 for i in range(N)], np.int32))

    def make(BF):
        def log_one(f, u1, u2, zz):                          # cx_t1s.py l.140-153 verbatim
            order = jnp.argsort(-zz, stable=True); Uo = order[:NH]; Do = order[NH:]
            Fm = f[Uo][:, Do]
            if BF:
                zn = zz[NBR]
                M1 = zn[Uo] @ u1
                M2 = zn[Do] @ u2
                Fm = Fm + jnp.take_along_axis(M1, DISP[Uo][:, Do], 1) + jnp.take_along_axis(M2, DISP[Do][:, Uo], 1).T
            sg, ld = jnp.linalg.slogdet(Fm)
            up = (zz > 0).astype(jnp.int32); dn = 1 - up
            inv = jnp.sum(up * (jnp.cumsum(dn) - dn))
            par = (jnp.where(sg < 0, 1, 0) + inv) % 2
            return ld, 1.0 - 2.0 * par

        def logA_one(th, sigma):                             # cx_t1s.py l.155-168; also returns sgn of the projected sum
            f = th[:N * N].reshape(N, N)
            if BF:
                u = th[N * N:].reshape(2, 9, N); u1, u2 = u[0], u[1]
            else:
                u1 = u2 = None
            Z = FLP[:, None] * sigma.astype(jnp.float64)[IMG]
            ld, sgn = jax.vmap(lambda zz: log_one(f, u1, u2, zz))(Z)
            m = jax.lax.stop_gradient(jnp.max(ld))
            S = jnp.sum(sgn * jnp.exp(ld - m))
            return m + jnp.log(jnp.maximum(jnp.abs(S), 1e-300)), jnp.sign(S)
        return jax.jit(lambda th, X: jax.vmap(lambda s_: logA_one(th, s_))(X.astype(jnp.float64)))
    fwd_bf = make(True); fwd_pp = make(False)

    th_bf = jnp.asarray(np.load(f'{RD}/theta.npy'), jnp.float64); assert th_bf.size == N * N + 2 * 9 * N
    z = np.load(PPZ); assert not bool(z['rbm']); f0 = jnp.asarray(z['f'], jnp.float64); del z
    th_pp = f0.reshape(-1)
    reps = np.load(TABLE)['reps']; s2 = np.load(S2)
    la_cp = np.load(f'{RD}/la_table.npy', mmap_mode='r')
    CH = 256

    def ev(fn, th, cfg):                                     # cfg uint64 (M,)  -> logA, sgn
        outl, outs = [], []
        for i in range(0, len(cfg), CH):
            c_ = cfg[i:i + CH]; m = len(c_)
            if m < CH: c_ = np.concatenate([c_, np.repeat(c_[:1], CH - m)])
            l_, s_ = fn(th, bits(jnp.asarray(c_))); outl.append(np.asarray(l_)[:m]); outs.append(np.asarray(s_)[:m])
            if (i // CH) % 20 == 0: log(f'  eval {i + m}/{len(cfg)}')
        return np.concatenate(outl), np.concatenate(outs)

    d = np.load(os.path.join(OUT, 'sample.npz'))
    out = dict(stage1=res1, nsamp=NS_, nsec=NSEC)
    # ---- orbit-invariance check on 20 random reps (images under the 288 space-group elements x spin flip)
    rng = np.random.default_rng(7)
    G = lat.G; inv_ok = True; inv_rows = []
    for r_ in rng.choice(len(reps), 20, replace=False):
        sig = np.where((int(reps[r_]) >> np.arange(N)) & 1, 1.0, -1.0)
        gi = rng.choice(len(G), 6, replace=False)
        imgs = np.array([sig[G[g]] for g in gi] + [-sig[G[g]] for g in gi[:3]] + [sig, -sig])
        cfg = np.array([sum(1 << i for i in range(N) if v[i] > 0) for v in imgs], np.uint64)
        l_, s_ = ev(fwd_bf, th_bf, cfg)
        inv_rows.append(dict(rep=int(r_), sgn=s_.tolist(), dlogA=float(l_.max() - l_.min())))
        inv_ok &= bool(np.all(s_ == s_[0])) and float(l_.max() - l_.min()) < 1e-6
    out['orbit_invariance'] = dict(ok=inv_ok, rows=inv_rows)
    log(f'orbit invariance (sgn A and |A| over images, 20 reps): {inv_ok}')
    json.dump(out, open(os.path.join(OUT, 'q_ownnode.json'), 'w'), indent=1)
    if not inv_ok:
        log('ORBIT INVARIANCE FAILED: stop'); return out
    # ---- sampled pairs
    for j, nm in ((0, '<1/3'), (1, '1/3-3')):
        xs, ys = d[f'xs{j}'], d[f'ys{j}']; assert (xs >= 0).all()
        u = np.unique(np.concatenate([xs, ys]))
        l_, s_ = ev(fwd_bf, th_bf, reps[u].astype(np.uint64))
        pos = {int(a): i for i, a in enumerate(u)}
        ix = np.array([pos[int(a)] for a in xs]); iy = np.array([pos[int(a)] for a in ys])
        sgx, sgy = s_[ix], s_[iy]
        tx, ty = sgx * s2[xs], sgy * s2[ys]
        offs = np.concatenate([l_[ix] - la_cp[xs], l_[iy] - la_cp[ys]])
        r = dict(n=len(xs), n_unique=len(u), q=float((tx * ty == -1).mean()), t_xs_minus1=float((tx == -1).mean()), t_ys_minus1=float((ty == -1).mean()),
                 s2s2_plus=float((s2[xs] * s2[ys] > 0).mean()), offset_mean=float(offs.mean()), offset_min=float(offs.min()), offset_max=float(offs.max()),
                 offset_spread=float(offs.max() - offs.min()), offset_std=float(offs.std()), offset_c_file=float(np.load(f'{RD}/offset_c.npy')),
                 ex_median_sample=float(np.median(d[f'ex{j}'])), ex_mean_sample=float(d[f'ex{j}'].mean()), cost_mean=float(d[f'c{j}'].mean()))
        r['q_err'] = float(np.sqrt(r['q'] * (1 - r['q']) / len(xs)))
        out[nm] = r
        log(f'class {nm}: {r}')
        if j == 0:   # warm-start plain PP
            u_ = u
            lp_, sp_ = ev(fwd_pp, th_pp, reps[u_].astype(np.uint64))
            spx, spy = sp_[ix], sp_[iy]
            r['pp_sgnx_equal_o2b'] = float((spx == sgx).mean()); r['pp_sgny_equal_o2b'] = float((spy == sgy).mean())
            r['pp_both_equal_o2b'] = float(((spx == sgx) & (spy == sgy)).mean())
            r['pp_q'] = float((spx * spy == -1).mean())
            offp = np.concatenate([lp_[ix] - la_cp[xs], lp_[iy] - la_cp[ys]]); r['broken_check_pp_offset_spread'] = float(offp.max() - offp.min())   # plain PP vs la_table must NOT be constant
            r['pp_pair_product_equal_o2b'] = float((spx * spy == sgx * sgy).mean())
            log(f'  warm-start PP: {r}')
        json.dump(out, open(os.path.join(OUT, 'q_ownnode.json'), 'w'), indent=1)
        # ---- secondary: any neighbour z of xs with t(z) != t(xs), on the first NSEC samples
        t1 = np.load(f'{TAB}/nb_J1.npy', mmap_mode='r'); t2 = np.load(f'{TAB}/nb_J2.npy', mmap_mode='r')
        sub = np.arange(min(NSEC, len(xs)))
        nbl = []
        for a in xs[sub]:
            row = np.concatenate([t1[a], t2[a]]); nbl.append(np.unique(row[row >= 0]))
        zu = np.unique(np.concatenate(nbl))
        log(f'  secondary {nm}: {len(sub)} samples, {len(zu)} unique neighbour configs')
        lz, sz = ev(fwd_bf, th_bf, reps[zu].astype(np.uint64))
        tz = dict(zip(zu.tolist(), (sz * s2[zu]).tolist()))
        anyf, anyv = [], []
        for k, a in enumerate(xs[sub]):
            txa = sgx[k] * s2[a]
            ts = np.array([tz[int(b)] for b in nbl[k]])
            anyf.append(bool((ts != txa).any()))
        r['q_any'] = float(np.mean(anyf)); r['q_any_n'] = len(sub); r['q_any_err'] = float(np.sqrt(r['q_any'] * (1 - r['q_any']) / len(sub)))
        log(f'  q_any {nm} = {r["q_any"]:.4f} +- {r["q_any_err"]:.4f}')
        json.dump(out, open(os.path.join(OUT, 'q_ownnode.json'), 'w'), indent=1)
    # ---- verdict (PLAN_pfaffian.md)
    q = out['<1/3']['q']; qr = out['1/3-3']['q']
    if q >= 0.5 and q >= 3 * qr: v = 'SUPPORTS'
    elif q <= 0.1: v = 'REFUTES'
    else: v = 'INCONCLUSIVE'
    out['verdict'] = v
    line = f'VERDICT q test: {v} (q={q:.4f}+-{out["<1/3"]["q_err"]:.4f}, q_ref={qr:.4f}+-{out["1/3-3"]["q_err"]:.4f}; rule: q>=0.5 and q>=3 q_ref supports, q<=0.1 refutes)'
    out['verdict_line'] = line; log(line)
    json.dump(out, open(os.path.join(OUT, 'q_ownnode.json'), 'w'), indent=1)
    return out


if __name__ == '__main__':
    if os.path.exists(os.path.join(OUT, 'stage1.json')) and os.path.exists(os.path.join(OUT, 'sample.npz')):
        res1 = json.load(open(os.path.join(OUT, 'stage1.json'))); log('stage1 cached')
    else:
        res1 = stage1()
    stage2(res1)
    log('DONE')
