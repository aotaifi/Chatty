"""bond_split.py: which pairs carry the refit tax of a copy a of a target phi (exact 6x6 symmetric sector, numpy, one CPU process).

For every exchange pair {x,y} (a bond flip) that is VIOLATING for the copy (sa_x sa_y > 0, i.e. psi_T(x) H_xy psi_T(y) > 0 with
H_xy = J/2 > 0 in the S^z basis), let y be the larger |phi| side, r* = |phi_x/phi_y|, r = |a_x/a_y|:
    cost = |H_xy| phi_y^2 (r - r*)^2 / r          (phi normalised: sum_orbits n phi^2 = 1, per-config amplitude)
Orbit bookkeeping: rows are orbit reps with multiplicity n_r; the config sum over ordered (x, bond) is sum_r n_r sum_bonds and each
unordered pair is seen from both ends, so total = 0.5 * sum_r n_r sum_bonds(...).  Reported per site (N = 36).
Compared with the measured zero-hop E_FN(a) - E_FN(phi) (FN ground energies of H_FN[a, s2] and H_FN[phi, s2], own numpy LOBPCG).

usage: python3 bond_split.py OUTDIR ROUND_DIR [ROUND_DIR ...]
  ROUND_DIR has target_la.npy (ln|phi|) and la_table.npy (ln|a|); signs: s2.npy of refresh_17409419 for BOTH (O2b fits only
  log|A| and the target phi_k is the FN solution of H_FN[g_k, s2], sign s2; cx_t1s.py lines 6, 107).
Tables (read only, mmap): nb_J1.npy / nb_J2.npy of /scratch/a/A.Otaifi/ansatz_runs/c2_t0b_6x6/tables; n from stall6/csr/norb.npy.
"""
import os, sys, json, time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, lobpcg

TAB = '/scratch/a/A.Otaifi/ansatz_runs/c2_t0b_6x6/tables'
CSRD = '/project/theorie/a/A.Otaifi/Chatty/runs/stall6/csr'
S2 = '/project/theorie/a/A.Otaifi/Chatty/runs/compress6/runs/refresh_17409419/s2.npy'
E0 = -18.137147540071613          # csr/meta.json
NS = 36
HW = np.concatenate([np.full(72, 0.5), np.full(72, 0.25)])      # J/2 per bond column (J1 then J2 = 0.5)
JW = np.concatenate([np.ones(72), np.full(72, 0.5)])
T0 = time.time()
DEC = np.arange(-12, 1)                                            # r* decade labels floor(log10 r*), clipped to [-12, 0]
QB = ['<1/3', '1/3-3', '3-30', '>30']


def log(m):
    print(f'[{time.time() - T0:7.0f}s] {m}', flush=True)


def build(n, s):
    """CSR of kept entries K (data J/2 sqrt(n_r/n_r')) and violating entries Vm (data J/2), rep basis; and diagonal D0."""
    D = n.shape[0]; sn = np.sqrt(n); isn = 1.0 / sn
    t1 = np.load(f'{TAB}/nb_J1.npy', mmap_mode='r'); t2 = np.load(f'{TAB}/nb_J2.npy', mmap_mode='r')
    assert t1.shape == (D, 72) and t2.shape == (D, 72)
    R = 131072
    ki, kd, kc, vi, vd, vc, D0 = [], [], [], [], [], [], np.empty(D)
    for i0 in range(0, D, R):
        i1 = min(D, i0 + R)
        t = np.concatenate([t1[i0:i1], t2[i0:i1]], 1); valid = t >= 0; tt = np.where(valid, t, 0)
        D0[i0:i1] = 27.0 - 0.5 * (valid * JW[None, :]).sum(1)
        kept = valid & (s[i0:i1, None] * s[tt] < 0); viol = valid & ~kept
        ki.append(tt[kept].astype(np.int32)); kc.append(kept.sum(1)); kd.append((HW[None, :] * sn[i0:i1, None] * isn[tt])[kept])
        vi.append(tt[viol].astype(np.int32)); vc.append(viol.sum(1)); vd.append(np.broadcast_to(HW[None, :], viol.shape)[viol])
        if (i0 // R) % 30 == 0: log(f'build {i1}/{D}')

    def mk(idx, dat, cnt):
        ip = np.zeros(D + 1, np.int64); ip[1:] = np.cumsum(np.concatenate(cnt))
        i_ = np.concatenate(idx); idx.clear(); d_ = np.concatenate(dat); dat.clear()
        return sp.csr_matrix((d_, i_, ip), shape=(D, D))
    K = mk(ki, kd, kc); Vm = mk(vi, vd, vc)
    return K, Vm, D0


def guide(la, n):
    la = la - la.max(); a = np.exp(la); a = a / np.sqrt((n * a * a).sum()); return np.maximum(a, 1e-15)   # as sector.fn_solve


def efn(K, Vm, D0, n, la, tol=1e-6, maxit=60):
    """FN ground energy of H_FN[a, s] (rotated basis, nonpositive kept hops), a = exp(la) floored 1e-15; sector.fn_solve convention."""
    sn = np.sqrt(n); a = guide(la, n); w = a * sn
    Dg = D0 + (Vm @ a) / a
    N_ = len(a)
    A = LinearOperator((N_, N_), matvec=lambda x: Dg * np.ravel(x) - K @ np.ravel(x), matmat=lambda X: Dg[:, None] * X - K @ X, dtype=np.float64)
    x = (w / np.linalg.norm(w))[:, None]
    th = float(x[:, 0] @ (Dg * x[:, 0] - K @ x[:, 0]))
    den = 1.0 / np.maximum(Dg - th, 1.0)
    M = LinearOperator((N_, N_), matvec=lambda r: den * np.ravel(r), matmat=lambda Rr: den[:, None] * Rr, dtype=np.float64)
    val, vec, hist = lobpcg(A, x, M=M, largest=False, tol=tol, maxiter=maxit, retResidualNormsHistory=True)
    e = float(val[0])
    return dict(E=e, dE_site=(e - E0) / NS, E_guide_var=th, dE_guide_site=(th - E0) / NS, iters=len(hist), res=float(np.ravel(hist[-1])[0]))


def cost_pass(n, s, la_phi, la_cp, rows=None, sa=None, sphi=None, board_eps=None):
    """accumulate the split over rep rows (default all).  sa / sphi: signs of copy / target (default both s).
    board_eps: per-rep eps with copy = phi e^eps: also returns the second-order formula sum 0.5 n_x H phi_x phi_y (eps_x - eps_y)^2."""
    D = n.shape[0]; i0g, i1g = rows or (0, D)
    sa = s if sa is None else sa; sphi = s if sphi is None else sphi
    lp = la_phi - la_phi.max(); lp = lp - 0.5 * np.log((n * np.exp(2 * lp)).sum())          # sum n phi^2 = 1
    ND = len(DEC); R = 65536
    hist = np.zeros((ND, 4)); by_cls = np.zeros(2); by_sign = np.zeros(4); by_p = np.zeros(40)
    npair = 0; tot = 0.0; board = 0.0; nflag = 0
    t1 = np.load(f'{TAB}/nb_J1.npy', mmap_mode='r'); t2 = np.load(f'{TAB}/nb_J2.npy', mmap_mode='r')
    for i0 in range(i0g, i1g, R):
        i1 = min(i1g, i0 + R)
        t = np.concatenate([t1[i0:i1], t2[i0:i1]], 1); valid = t >= 0; tt = np.where(valid, t, 0)
        viol = valid & (sa[i0:i1, None] * sa[tt] > 0)
        rr, kk = np.nonzero(viol); x = rr + i0; y = tt[rr, kk]; H = HW[kk]; del t, valid, tt, viol, rr
        lpx, lpy = lp[x], lp[y]; lcx, lcy = la_cp[x], la_cp[y]
        sw = lpx > lpy                                       # y := larger |phi| side
        ls_, ll_ = np.where(sw, lpy, lpx), np.where(sw, lpx, lpy)
        cs_, cl_ = np.where(sw, lcy, lcx), np.where(sw, lcx, lcy)
        xs = np.where(sw, y, x); ys = np.where(sw, x, y)     # small / large (y) side rep
        lrs = np.clip(ls_ - ll_, -300, 0); lr = np.clip(cs_ - cl_, -300, 300)
        rs = np.exp(lrs); r = np.exp(lr)
        c = H * np.exp(2 * ll_) * (r - rs) ** 2 / r * n[x] * 0.5
        tot += c.sum(); npair += c.size
        d = np.clip(np.floor(lrs / np.log(10.0)).astype(int), -12, 0) - DEC[0]
        q = lr - lrs; qb = np.where(q < -np.log(3), 0, np.where(q < np.log(3), 1, np.where(q < np.log(30), 2, 3)))
        hist += np.bincount(d * 4 + qb, weights=c, minlength=ND * 4).reshape(ND, 4)
        by_cls += [c[kk < 72].sum(), c[kk >= 72].sum()]
        mx = sa[xs] != sphi[xs]; my = sa[ys] != sphi[ys]
        by_sign += [c[~mx & ~my].sum(), c[mx & ~my].sum(), c[~mx & my].sum(), c[mx & my].sum()]
        by_p += np.bincount(np.clip(np.floor(2 * ll_ / np.log(10.0)).astype(int), -39, 0) + 39, weights=c, minlength=40)
        if board_eps is not None:
            board += float((H * np.exp(lpx + lpy) * (board_eps[x] - board_eps[y]) ** 2 * n[x] * 0.5).sum())
    return dict(total=tot, total_site=tot / NS, npairs=npair, hist=hist, by_class=by_cls, by_sign=by_sign, by_phi2_decade=by_p, board=board)


def table_str(h, tot):
    out = ['r* decade   | ' + ' | '.join(f'r/r*{q:>6}' for q in QB) + ' |   row   (% of total cost)']
    for i, d in enumerate(DEC):
        if h[i].sum() > 0:
            out.append(f'1e{d:>3}..1e{d + 1:>3} | ' + ' | '.join(f'{100 * v / tot:11.2f}' for v in h[i]) + f' | {100 * h[i].sum() / tot:7.2f}')
    out.append('column      | ' + ' | '.join(f'{100 * h[:, j].sum() / tot:11.2f}' for j in range(4)))
    return '\n'.join(out)


def checks(n, s, la_phi, rows, out):
    rng = np.random.default_rng(1)
    D = n.shape[0]
    c1 = cost_pass(n, s, la_phi, la_phi.copy(), rows=rows)
    eps = 0.01 * rng.standard_normal(D)
    c2 = cost_pass(n, s, la_phi, la_phi + eps, rows=rows, board_eps=eps)
    sa = s.copy(); flip = rng.random(D) < 0.01; sa[flip] *= -1
    c3 = cost_pass(n, s, la_phi, la_phi + eps, rows=rows, sa=sa)       # sign path: some copy signs flipped
    c4 = cost_pass(n, s, la_phi, la_phi + 0.3 * rng.standard_normal(D), rows=rows)   # broken (large) copy: must be >> c2
    rec = dict(rows=rows, control_copy_eq_phi=c1['total'], npairs=c1['npairs'], noisy_total=c2['total'], noisy_board=c2['board'],
               ratio_noisy_over_board=c2['total'] / c2['board'], large_noise_total=c4['total'],
               sign_flip_total=c3['total'], sign_flip_by_sign=c3['by_sign'].tolist(), sign_flip_npairs=c3['npairs'],
               plain_by_sign=c2['by_sign'].tolist())
    json.dump(rec, open(os.path.join(out, 'checks.json'), 'w'), indent=1)
    log(f'CHECKS {rec}')
    return rec


if __name__ == '__main__':
    OUT = sys.argv[1]; RDS = sys.argv[2:]; os.makedirs(OUT, exist_ok=True)
    n = np.load(f'{CSRD}/norb.npy').astype(np.float64); s = np.load(S2).astype(np.float64)
    assert n.sum() == 9075135300 and set(np.unique(s)) <= {-1.0, 1.0}
    log('sum n = C(36,18) ok; building K, Vm')
    K, Vm, D0 = build(n, s)
    log(f'built: K nnz {K.nnz}, Vm nnz {Vm.nnz}, valid total {K.nnz + Vm.nnz} (tables n_anti 1170496152)')
    for rd in RDS:
        tag = os.path.basename(rd.rstrip('/'))
        lp = np.load(f'{rd}/target_la.npy'); la = np.load(f'{rd}/la_table.npy')
        pn = lp - lp.max(); pn = pn - 0.5 * np.log((n * np.exp(2 * pn)).sum())
        wsum = float((n * np.exp(2 * pn)).sum())
        if tag == os.path.basename(RDS[0].rstrip('/')) and not os.path.exists(os.path.join(OUT, 'checks.json')):
            checks(n, s, lp, (0, 3000000), OUT)
        res = cost_pass(n, s, lp, la)
        json.dump(dict(total=res['total'], hist=res['hist'].tolist(), by_class=res['by_class'].tolist(), by_sign=res['by_sign'].tolist(), npairs=res['npairs'], by_phi2=res['by_phi2_decade'].tolist()), open(os.path.join(OUT, f'cost_raw_{tag}.json'), 'w'))
        ea = efn(K, Vm, D0, n, la); log(f'{tag}: E_FN(copy) dE_site {ea["dE_site"]:.6e} (res {ea["res"]:.1e}, it {ea["iters"]})')
        ep = efn(K, Vm, D0, n, lp); log(f'{tag}: E_FN(target) dE_site {ep["dE_site"]:.6e} (res {ep["res"]:.1e}, it {ep["iters"]})')
        dE = ea['dE_site'] - ep['dE_site']
        out = dict(round=tag, weight_sum_phi=wsum, cost_total=res['total'], cost_site=res['total_site'], npairs=res['npairs'],
                   E_FN_copy=ea, E_FN_target=ep, dE_FN_site=dE, ratio_cost_over_dE=res['total_site'] / dE,
                   by_class=dict(J1=res['by_class'][0] / res['total'], J2=res['by_class'][1] / res['total']),
                   by_sign=dict(none=res['by_sign'][0] / res['total'], x_small_side=res['by_sign'][1] / res['total'],
                                y_large_side=res['by_sign'][2] / res['total'], both=res['by_sign'][3] / res['total']),
                   hist_share=(res['hist'] / res['total']).tolist(), dec_labels=DEC.tolist(), ratio_bins=QB,
                   by_phi2_decade_share={int(d - 39): float(v / res['total']) for d, v in enumerate(res['by_phi2_decade']) if v > 0})
        json.dump(out, open(os.path.join(OUT, f'split_{tag}.json'), 'w'), indent=1)
        open(os.path.join(OUT, f'split_{tag}.txt'), 'w').write(table_str(res['hist'], res['total']) + '\n')
        log(f'{tag}: sum n phi^2 = {wsum:.12f}; cost/site {res["total_site"]:.4e}; dE_FN {dE:.4e}; ratio {res["total_site"] / dE:.3f}')
        log('\n' + table_str(res['hist'], res['total']))
        log(f'J1 {out["by_class"]["J1"]:.3f} J2 {out["by_class"]["J2"]:.3f}; sign {out["by_sign"]}')
    log('DONE')
