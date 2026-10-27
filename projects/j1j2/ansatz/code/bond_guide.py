"""bond_guide.py: follow-up of bond_split.py. For violating pairs (same selection and cost weights as bond_split.cost_pass) report
 (1) which side is wrong: cost-weighted mean / median of e_x, e_y (x = small, y = big |phi| side; e = ln|copy| - ln|phi|),
 (2) guide vs FN vs copy: rho_g, rho_phi, rho_c = ln(.(x)/.(y)); share of cost with rho_g < rho_phi; median of
     f = (rho_c - rho_g)/(rho_phi - rho_g) for |rho_phi - rho_g| > 0.05; per-config by phi^2 decade mean(ln phi - ln g), mean(ln copy - ln g).
The GUIDE g_k of round k is the O2a hop of the previous copy (cx_o2b.py hop()): g = |a_k| |1 + alpha_V (E_L - E)|, a_k = round_{k-1}/la_table.npy,
alpha_V = bounded argmin of the exact <H> of the hop state, E_L = D0 + V - W of (|a_k|, s2); phi_k = FN(g_k, s2) = round_k/target_la.npy.
usage: python3 bond_guide.py OUT K [K ...]   (K = round number, copy = round_K/la_table.npy; needs K >= 1)
"""
import os, sys, json, time
import numpy as np
from scipy.optimize import minimize_scalar
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bond_split as B
from bond_split import HW, NS, TAB, log

RUN = '/project/theorie/a/A.Otaifi/Chatty/runs/compress6/runs/o2b_17431886'
EB = np.arange(-30, 30.0001, 0.02); NEB = len(EB) - 1          # e histogram
FB = np.arange(-5, 6.0001, 0.01); NFB = len(FB) - 1            # f histogram
CL = ['<1/3', '1/3-3', '3-30', '>30']


def norm_la(la, n):
    la = la - la.max(); return la - 0.5 * np.log((n * np.exp(2 * la)).sum())


def hop(la0, n, s, K, Vm, D0, rec):
    sn = np.sqrt(n); lb = norm_la(la0, n); a = np.exp(lb)
    def EL_of(a):
        return D0 + (Vm @ a) / a - (K @ (a * sn)) / (sn * a)
    EL = EL_of(a); p = n * a * a; Eb = float((p * EL).sum()); sdE = float(np.sqrt((p * (EL - Eb) ** 2).sum()))
    def lac(al): return lb + np.log(np.maximum(np.abs(1.0 + al * (EL - Eb)), 1e-30))
    def Hs(al):
        ac = np.exp(lac(al) - lac(al).max()); e = (n * ac * ac * EL_of(ac)).sum() / (n * ac * ac).sum()
        return (e - B.E0) / NS
    lo, hi = -3.0 / sdE, 3.0 / sdE
    r = minimize_scalar(Hs, bounds=(lo, hi), method='bounded', options=dict(maxiter=100, xatol=1e-4 / sdE))
    log(f'hop: E_base {Eb:.12f} (rec {rec["E_base"]:.12f}) sdE {sdE:.6f} (rec {rec["sdE"]:.6f}) alpha_V {r.x:.6f} (rec {rec["alpha_V"]:.6f}) H_V {r.fun:.6e} (rec {rec["H_V"]:.6e}) nfev {r.nfev}')
    return norm_la(lac(r.x), n), dict(E_base=Eb, sdE=sdE, alpha=float(r.x), H_V=float(r.fun))


def wq(h, edges, q):
    c = np.cumsum(h); 
    if c[-1] <= 0: return float('nan')
    i = int(np.searchsorted(c, q * c[-1])); return float(0.5 * (edges[i] + edges[min(i + 1, len(edges) - 1)]))


def pair_pass(n, s, lg, lp, lc, rows=None):
    D = n.shape[0]; i0g, i1g = rows or (0, D); R = 65536
    t1 = np.load(f'{TAB}/nb_J1.npy', mmap_mode='r'); t2 = np.load(f'{TAB}/nb_J2.npy', mmap_mode='r')
    hex_ = np.zeros((4, 2, NEB)); hex_n = np.zeros((4, 2, NEB)); sm = np.zeros((4, 2)); tot = np.zeros(4)
    sh_lt = np.zeros(4); sh_valid = np.zeros(4); hf = np.zeros((4, NFB)); hfn = np.zeros((4, NFB)); nval = np.zeros(4)
    for i0 in range(i0g, i1g, R):
        i1 = min(i1g, i0 + R)
        t = np.concatenate([t1[i0:i1], t2[i0:i1]], 1); valid = t >= 0; tt = np.where(valid, t, 0)
        viol = valid & (s[i0:i1, None] * s[tt] > 0)
        rr, kk = np.nonzero(viol); x = rr + i0; y = tt[rr, kk]; H = HW[kk]; del t, valid, tt, viol, rr
        sw = lp[x] > lp[y]; xs = np.where(sw, y, x); ys = np.where(sw, x, y)       # small / big |phi| side
        lrs = np.clip(lp[xs] - lp[ys], -300, 0); lr = np.clip(lc[xs] - lc[ys], -300, 300)
        rs = np.exp(lrs); r = np.exp(lr)
        c = H * np.exp(2 * lp[ys]) * (r - rs) ** 2 / r * n[x] * 0.5
        q = lr - lrs; cls = np.where(q < -np.log(3), 0, np.where(q < np.log(3), 1, np.where(q < np.log(30), 2, 3)))
        ex = lc[xs] - lp[xs]; ey = lc[ys] - lp[ys]
        rg = lg[xs] - lg[ys]; rp = lp[xs] - lp[ys]; rc = lc[xs] - lc[ys]
        dd = rp - rg; ok = np.abs(dd) > 0.05
        f = np.where(ok, (rc - rg) / np.where(ok, dd, 1.0), 0.0)
        for j, e in enumerate((ex, ey)):
            ib = np.clip(np.searchsorted(EB, e) - 1, 0, NEB - 1)
            hex_[:, j] += np.bincount(cls * NEB + ib, weights=c, minlength=4 * NEB).reshape(4, NEB)
            hex_n[:, j] += np.bincount(cls * NEB + ib, minlength=4 * NEB).reshape(4, NEB)
            sm[:, j] += np.bincount(cls, weights=c * e, minlength=4)
        tot += np.bincount(cls, weights=c, minlength=4); nval += np.bincount(cls, minlength=4)
        sh_lt += np.bincount(cls, weights=c * (rg < rp), minlength=4); sh_valid += np.bincount(cls, weights=c * ok, minlength=4)
        fb = np.clip(np.searchsorted(FB, f) - 1, 0, NFB - 1)
        hf += np.bincount((cls * NFB + fb)[ok], weights=c[ok], minlength=4 * NFB).reshape(4, NFB)
        hfn += np.bincount((cls * NFB + fb)[ok], minlength=4 * NFB).reshape(4, NFB)
    return dict(hex=hex_, hexn=hex_n, sm=sm, tot=tot, sh_lt=sh_lt, sh_valid=sh_valid, hf=hf, hfn=hfn, nval=nval)


def summarize(P):
    out = {}
    for j, c in enumerate(CL):
        t = P['tot'][j]
        out[c] = dict(cost_share=None, n_pairs=int(P['nval'][j]),
                      ex_mean=P['sm'][j, 0] / t if t > 0 else None, ex_median=wq(P['hex'][j, 0], EB, .5),
                      ey_mean=P['sm'][j, 1] / t if t > 0 else None, ey_median=wq(P['hex'][j, 1], EB, .5),
                      ex_median_count=wq(P['hexn'][j, 0], EB, .5), ey_median_count=wq(P['hexn'][j, 1], EB, .5),
                      share_rho_g_lt_rho_phi=P['sh_lt'][j] / t if t > 0 else None,
                      share_cost_with_dphi_gt_0p05=P['sh_valid'][j] / t if t > 0 else None,
                      f_median_costw=wq(P['hf'][j], FB, .5), f_median_count=wq(P['hfn'][j], FB, .5),
                      f_q25_costw=wq(P['hf'][j], FB, .25), f_q75_costw=wq(P['hf'][j], FB, .75))
    T = P['tot'].sum()
    for j, c in enumerate(CL): out[c]['cost_share'] = P['tot'][j] / T
    out['total_cost_site'] = T / NS
    return out


def decade_means(n, lg, lp, lc):
    dec = np.clip(np.floor(2 * lp / np.log(10.0)).astype(int), -40, 0) + 40
    w = n; res = {}
    cnt = np.bincount(dec, weights=w, minlength=41)
    a = np.bincount(dec, weights=w * (lp - lg), minlength=41) / np.maximum(cnt, 1)
    b = np.bincount(dec, weights=w * (lc - lg), minlength=41) / np.maximum(cnt, 1)
    wt = np.bincount(dec, weights=w * np.exp(2 * lp), minlength=41)
    return {int(d - 40): dict(mean_phi_minus_g=float(a[d]), mean_copy_minus_g=float(b[d]), configs=float(cnt[d]), weight=float(wt[d])) for d in range(41) if cnt[d] > 0}


if __name__ == '__main__':
    OUT = sys.argv[1]; KS = [int(k) for k in sys.argv[2:]]; os.makedirs(OUT, exist_ok=True)
    n = np.load(f'{B.CSRD}/norb.npy').astype(np.float64); s = np.load(B.S2).astype(np.float64)
    cx = json.load(open(f'{RUN}/cx_o2b.json'))
    log('building K, Vm'); K, Vm, D0 = B.build(n, s)
    for k in KS:
        la0 = np.load(f'{RUN}/round_{k - 1}/la_table.npy')
        lg, hinfo = hop(la0, n, s, K, Vm, D0, cx['rounds'][k]['hop_a'])
        lp = norm_la(np.load(f'{RUN}/round_{k}/target_la.npy'), n); lc = norm_la(np.load(f'{RUN}/round_{k}/la_table.npy'), n)
        # checks on a row subset: copy = phi -> f = 1 ; copy = g -> f = 0 (count-based medians; cost weight is 0 for copy = phi)
        rows = (0, 2000000)
        ck = {}
        for nm, l_ in (('copy=phi', lp), ('copy=g', lg)):
            P = pair_pass(n, s, lg, lp, l_, rows=rows); S = summarize(P)
            ck[nm] = {c: dict(f_median_count=S[c]['f_median_count'], n=S[c]['n_pairs']) for c in CL}
        log(f'round {k} CHECKS {ck}')
        P = pair_pass(n, s, lg, lp, lc); S = summarize(P); S['hop'] = hinfo; S['checks'] = ck
        S['decade'] = decade_means(n, lg, lp, lc); S['round'] = k
        json.dump(S, open(f'{OUT}/guide_split_round_{k}.json', 'w'), indent=1, default=float)
        log(f'round {k}: ' + json.dumps({c: S[c] for c in CL}, default=float))
        log(f'round {k} decades: ' + json.dumps(S['decade'], default=float))
    log('DONE')
