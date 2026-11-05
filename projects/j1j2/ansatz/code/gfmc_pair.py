"""gfmc_pair.py (c1 paired pilot, PLAN_pfaffian.md "c1 paired design"; worker 2026-11-05).  NEW file; gfmc_aug.py is imported, not changed.

Two fixed-node GFMC arms on ONE amplitude table |B| (6x6 orbit table la[rep]) with two sign tables (arm S, arm G):
  guide_a(x) = s_a(x) |B(x)|,  c = 0.  Walker dynamics = the launched M1 code (gfmc_fn_lib.GFMC.block: continuous time, exponential waiting
  time with rate R(x) = sum of kept rates, bond chosen with prob rate_b / R, log-weight -= E_L dt, reconfiguration every dtau; ten Haaf FN rows
  from GuideBase.finish), re-implemented here with COUNTER-BASED common random numbers so that both arms can share them:
  * every walker carries a 64-bit lineage label; its k-th event in block b uses u = H(label, b, k, stream) (splitmix64), stream 0 = waiting
    time (t = -ln u / R), stream 1 = bond choice;  a walker with the same label and configuration in both arms therefore makes the same moves;
  * reconfiguration ('comb', default): walkers ordered by label, ONE systematic comb with the shared uniform u0 = H(seed, b) (the c3 comb up to
    the walker order); 'local': each walker gets floor(M w_i/W + u_i) copies, u_i = H(label, b, 0, 2) (population size fluctuates, coupling local);
  * a child's label = H(parent label, b, copy number), so a parent with identical copies in both arms gives identical children.
  Arms that agree stay in lockstep (identical labels, configurations and weights); a walker that meets a bond whose kept/blocked status
  differs between the signs (S != s_G at x or at a neighbour) diverges, and each arm then keeps its own walkers on the same random stream.
  --indep: arm G gets its own seed (labels, comb uniform) and its own initial walkers = the naive independent estimator (broken control).
Rows: one canonicalisation (st6_sector byte tables, as gfmc_aug.guide_table6) of the union of both arms' configurations and neighbours,
  ratio r = s(y) s(x) exp(la(y) - la(x)) (= LogpsiGuide.ratio on the table6 logpsi la + i pi [s < 0]); test_rows() checks the identity.
Outputs OUT/: progress.txt (PROGRESS lines, last 'GFMC_PAIR_DONE ...'), ckpt.npz (every --ckpt_every blocks; the RNG is counter-based, so a
  resume is exact), blocks.npz (per block: emix/egrow per arm, coincidence counters), E_S.npy, E_G.npy, E_D.npy (= E_S - E_G; per site, after
  burn, time order: quotable.py input), result.json.
  python gfmc_pair.py OUT --sgnS S.npy --sgnG s_G.npy --M 1024 --dtau 0.1 --burn 200 --blocks B --seed s [--indep --seedG s2] [--E0 e]
"""
import argparse, json, os, sys, time, hashlib
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gfmc_aug as GA                                         # (imports the launched M1 walker library)
from gfmc_fn_lib import GuideBase, blocked_error, iat

U64 = np.uint64
C1, C2, C3, C4 = U64(0x9E3779B97F4A7C15), U64(0xD1B54A32D192ED03), U64(0xA0761D6478BD642F), U64(0xE7037ED1A0B428DB)


def mix(z):
    """splitmix64 finaliser, vectorised (uint64 wrap-around)."""
    with np.errstate(over='ignore'):
        z = np.asarray(z, U64) + C1
        z = (z ^ (z >> U64(30))) * U64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> U64(27))) * U64(0x94D049BB133111EB)
        return z ^ (z >> U64(31))


def H(label, b, k, stream):
    """uniform in (0, 1) from (label, block, event, stream)."""
    with np.errstate(over='ignore'):
        h = mix(np.asarray(label, U64) ^ mix(U64(b) * C2 + U64(stream) * C3))
        h = mix(h + np.asarray(k, U64) * C4)
    return ((h >> U64(11)).astype(np.float64) + 0.5) * 2.0 ** -53


def child_labels(parent_labels, b, copy_no):
    with np.errstate(over='ignore'):
        return mix(np.asarray(parent_labels, U64) ^ mix(U64(b) * C3 + np.asarray(copy_no, U64) * C2 + U64(7)))


class TablePair(GuideBase):
    """6x6 orbit-table amplitude la[rep] with K sign tables; rows for several arms from one canonicalisation of the union."""

    def __init__(self, la, signs, reps_path=GA.REPS6):
        import st6_sector as SS
        self.L = 6; self.N = 36; self.BI, self.BJ, self.JB = GA.bonds(6); self.NB = len(self.BI)
        self.reps = np.asarray(np.load(reps_path)['reps']).astype(U64)
        self.la = np.asarray(la, np.float64); self.s = [np.asarray(s).astype(np.float64) for s in signs]
        assert all(s.shape == self.reps.shape for s in self.s) and self.la.shape == self.reps.shape
        self.T = SS.byte_tables(SS.space_group()); self.FULL = U64((1 << 36) - 1)
        self.code = (U64(1) << np.arange(36, dtype=U64))
        self.flip = (U64(1) << self.BI.astype(U64)) | (U64(1) << self.BJ.astype(U64))
        self.n_canon = 0; self.n_req = 0; self.t_eval = 0.0

    def canon_idx(self, x):
        img = np.zeros((x.shape[0], self.T.shape[0]), U64)
        for b in range(5):
            img |= self.T[:, b, :][:, ((x >> U64(8 * b)) & U64(255)).astype(np.int64)].T
        m = np.minimum(img.min(1), (img ^ self.FULL).min(1))
        i = np.clip(np.searchsorted(self.reps, m), 0, len(self.reps) - 1)
        if not np.all(self.reps[i] == m):
            raise KeyError(f'TablePair: {int(np.sum(self.reps[i] != m))} configurations without an orbit rep')
        return i

    def rows_multi(self, Xs, arms):
        """Xs: list of (B_k, 36) int8 walker sets, arms: their sign-table index -> list of (rate, dFN)."""
        t0 = time.time(); parts = []; allc = []
        for X in Xs:
            X = np.asarray(X); cx = ((X > 0).astype(U64) * self.code).sum(1, dtype=U64)
            anti = X[:, self.BI] != X[:, self.BJ]; wi, bi = np.nonzero(anti)
            cy = cx[wi] ^ self.flip[bi]
            parts.append((X, anti, wi, bi, len(cx))); allc += [cx, cy]
        allc = np.concatenate(allc) if allc else np.zeros(0, U64)
        uc, inv = np.unique(allc, return_inverse=True)
        idx = self.canon_idx(uc)[inv]; self.n_canon += len(uc); self.n_req += len(allc)
        out = []; o = 0
        for (X, anti, wi, bi, B), a in zip(parts, arms):
            ix = idx[o:o + B]; iy = idx[o + B:o + B + len(wi)]; o += B + len(wi)
            s = self.s[a]
            rr = s[iy] * s[ix][wi] * np.exp(self.la[iy] - self.la[ix][wi])
            out.append(self.finish(X, anti, wi, bi, rr))
        self.t_eval += time.time() - t0
        return out


class Arm:
    def __init__(self, X0, labels):
        self.X = np.ascontiguousarray(X0, np.int8); self.lab = np.asarray(labels, U64).copy(); self.M = len(self.lab)
        self.rate = None; self.R = None; self.EL = None

    def set_rows(self, idx, r, d):
        if self.rate is None or self.rate.shape[0] != self.M:
            self.rate = np.zeros((self.M, r.shape[1] if r.size else 144)); self.R = np.zeros(self.M); self.EL = np.zeros(self.M)
        self.rate[idx] = r; self.R[idx] = r.sum(1); self.EL[idx] = d - self.R[idx]


class PairGFMC:
    def __init__(self, g, X0S, X0G, labS, labG, dtau, seedS, seedG, resample='comb', Mt=None, move='cdf'):
        self.move = move; self.g = g; self.dtau = dtau; self.seeds = (int(seedS), int(seedG)); self.resample = resample
        self.arms = [Arm(X0S, labS), Arm(X0G, labG)]; self.Mt = Mt or len(labS)
        self.refresh([np.arange(a.M) for a in self.arms])

    def refresh(self, idxs):
        sel = [k for k in range(2) if idxs[k].size]
        if not sel:
            return
        res = self.g.rows_multi([self.arms[k].X[idxs[k]] for k in sel], sel)
        for k, (r, d) in zip(sel, res):
            self.arms[k].set_rows(idxs[k], r, d)

    def resample_arm(self, A, w, b, seed):
        M = self.Mt
        order = np.argsort(A.lab, kind='stable')                       # canonical walker order = label order (shared across arms)
        wo = w[order]
        if self.resample == 'comb':
            c = np.cumsum(wo); c /= c[-1]
            u0 = float(H(U64(seed), b, 0, 3))
            par = np.minimum(np.searchsorted(c, (u0 + np.arange(M)) / M, side='right'), wo.size - 1)
        else:
            nc = np.floor(M * wo / wo.sum() + H(A.lab[order], b, 0, 2)).astype(np.int64)
            par = np.repeat(np.arange(wo.size), nc)
        par = np.sort(par)
        first = np.searchsorted(par, par, side='left'); copy_no = np.arange(par.size) - first
        src = order[par]
        A.X = A.X[src]; A.rate = A.rate[src]; A.R = A.R[src]; A.EL = A.EL[src]
        A.lab = child_labels(A.lab[src], b, copy_no); A.M = A.lab.size

    def xsame_pairs(self):
        S, G = self.arms
        com, iS, iG = np.intersect1d(S.lab, G.lab, return_indices=True)
        xs = np.all(S.X[iS] == G.X[iG], 1)
        return iS[xs], iG[xs]

    def block(self, b):
        pS0, pG0 = self.xsame_pairs()
        st = []
        for A in self.arms:
            st.append(dict(rem=np.full(A.M, self.dtau), logw=np.zeros(A.M), ev=np.zeros(A.M, np.int64), act=np.arange(A.M), hops=0, iters=0,
                           nmoves=np.zeros(A.M, np.int64)))
        while any(s['act'].size for s in st):
            mvs = []
            for A, s in zip(self.arms, st):
                act = s['act']
                if not act.size:
                    mvs.append(act); continue
                if self.move == 'clock':                                # competing per-bond exponential clocks (same law)
                    nb_ = A.rate.shape[1]
                    Ub = H(A.lab[act][:, None], b, s['ev'][act][:, None], 10 + np.arange(nb_, dtype=U64)[None, :])
                    with np.errstate(divide='ignore'):
                        tb_ = -np.log(Ub) / A.rate[act]
                    kc = np.argmin(tb_, 1); t = tb_[np.arange(act.size), kc]
                else:
                    with np.errstate(divide='ignore'):
                        t = -np.log(H(A.lab[act], b, s['ev'][act], 0)) / A.R[act]
                stay = t >= s['rem'][act]
                fin = act[stay]; s['logw'][fin] -= s['rem'][fin] * A.EL[fin]
                mv = act[~stay]; tm = t[~stay]
                s['logw'][mv] -= tm * A.EL[mv]; s['rem'][mv] -= tm
                if mv.size and self.move == 'clock':
                    k = kc[~stay]
                if mv.size and self.move != 'clock':
                    cum = np.cumsum(A.rate[mv], 1)
                    u = H(A.lab[mv], b, s['ev'][mv], 1) * cum[:, -1]
                    k = np.minimum((cum <= u[:, None]).sum(1), self.g.NB - 1)
                    bad = A.rate[mv, k] <= 0
                    if bad.any():
                        k[bad] = np.argmax(A.rate[mv[bad]], 1)
                if mv.size:
                    i, j = self.g.BI[k], self.g.BJ[k]
                    xi = A.X[mv, i].copy(); A.X[mv, i] = A.X[mv, j]; A.X[mv, j] = xi
                    s['ev'][mv] += 1; s['hops'] += mv.size; s['nmoves'][mv] += 1
                s['act'] = mv; s['iters'] += 1; mvs.append(mv)
            self.refresh(mvs)
        out = {}
        # coincidence at the block end, before reconfiguration: same label, same configuration, same log-weight
        S, G = self.arms
        com, iS, iG = np.intersect1d(S.lab, G.lab, assume_unique=False, return_indices=True)
        xs = np.all(S.X[iS] == G.X[iG], 1); dlw = np.abs(st[0]['logw'][iS] - st[1]['logw'][iG])
        same = xs & (dlw == 0)
        out['n_common'] = int(com.size); out['n_ident'] = int(same.sum()); out['n_xsame'] = int(xs.sum())
        out['dlogw_max_xsame'] = float(dlw[xs].max()) if xs.any() else 0.0
        out['MS'] = S.M; out['MG'] = G.M
        n_div_move = int(np.sum(np.any(S.X[pS0] != G.X[pG0], 1)))      # pairs identical at the block start whose configurations split
        out['block_ident'] = bool(S.M == G.M and same.sum() == S.M)
        for tag, A, s, seed in (('S', S, st[0], self.seeds[0]), ('G', G, st[1], self.seeds[1])):
            lw = s['logw']; mx = lw.max(); w = np.exp(lw - mx); sw = w.sum()
            em = float(np.sum(w * A.EL) / sw)
            out[f'emix_{tag}'] = em; out[f'egrow_{tag}'] = float(-(mx + np.log(sw / A.M)) / self.dtau)
            out[f'ess_{tag}'] = float(sw * sw / np.sum(w * w)); out[f'hops_{tag}'] = s['hops']; out[f'iters_{tag}'] = s['iters']
            self.resample_arm(A, w, b, seed)
        out['lab_overlap_after'] = int(np.intersect1d(S.lab, G.lab).size)
        out['xsame_start'] = int(pS0.size); out['n_div_move'] = n_div_move; out['xsame_after'] = int(self.xsame_pairs()[0].size)
        return out


def test_rows(g, la_path, sgn_path, which, n=64, seed=5):
    """identity of TablePair rows with gfmc_aug.LogpsiGuide(guide_table6) rows on random Sz = 0 walkers."""
    from types import SimpleNamespace
    fn, _, idt, _ = GA.guide_table6(SimpleNamespace(table=GA.REPS6, la=la_path, sgn=sgn_path, syma=None))
    ref = GA.LogpsiGuide(fn, 6)
    rng = np.random.default_rng(seed); X = np.ones((n, 36), np.int8); X[:, 18:] = -1
    for r in range(n): rng.shuffle(X[r])
    r1, d1 = ref.rows(X); (r2, d2), = g.rows_multi([X], [which])
    return float(max(np.max(np.abs(r1 - r2)), np.max(np.abs(d1 - d2))))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('--la', default=f'{GA.R6}/pfaffian_r4/pdb_tabs/tab_t4/round_1/la_table.npy')
    ap.add_argument('--sgnS', required=True); ap.add_argument('--sgnG', default=f'{GA.R6}/pfaffian_lswo2/s_G.npy')
    ap.add_argument('--M', type=int, default=1024); ap.add_argument('--dtau', type=float, default=0.1)
    ap.add_argument('--blocks', type=int, default=200); ap.add_argument('--burn', type=int, default=200)
    ap.add_argument('--seed', type=int, default=1); ap.add_argument('--indep', action='store_true'); ap.add_argument('--seedG', type=int, default=None)
    ap.add_argument('--resample', default='comb', choices=['comb', 'local'])
    ap.add_argument('--move', default='cdf', choices=['cdf', 'clock'], help="cdf: t ~ -ln u/R, bond by inverse CDF (M1 code); clock: per-bond clocks")
    ap.add_argument('--X0', help='(M, 36) int8 initial walkers (both arms; arm S only with --indep)'); ap.add_argument('--X0G', help='--indep: arm G walkers')
    ap.add_argument('--nsweep', type=int, default=2000)
    ap.add_argument('--ckpt_every', type=int, default=200); ap.add_argument('--resume', action='store_true'); ap.add_argument('--maxsec', type=float, default=1e18)
    ap.add_argument('--E0', type=float, default=-0.5038096538908782)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); prog = os.path.join(a.out, 'progress.txt'); N = 36

    def P(msg):
        line = f'{time.strftime("%Y-%m-%dT%H:%M:%S")} PROGRESS {msg}'
        print(line, flush=True); open(prog, 'a').write(line + '\n')
    sha = lambda f: hashlib.sha256(open(f, 'rb').read()).hexdigest()
    info = dict(la=a.la, la_sha256=sha(a.la), sgnS=a.sgnS, sgnS_sha256=sha(a.sgnS), sgnG=a.sgnG, sgnG_sha256=sha(a.sgnG),
                code_sha256=sha(os.path.abspath(__file__)), gfmc_aug_sha256=sha(GA.__file__))
    g = TablePair(np.load(a.la), [np.load(a.sgnS), np.load(a.sgnG)])
    dmax = max(test_rows(g, a.la, a.sgnS, 0), test_rows(g, a.la, a.sgnG, 1))
    info['rows_identity_maxabs'] = dmax
    P(f'pair M {a.M} dtau {a.dtau} seed {a.seed} indep {a.indep} seedG {a.seedG} resample {a.resample} move {a.move} rows identity vs LogpsiGuide max|d| {dmax:.2e} '
      f'sgnS {info["sgnS_sha256"][:8]} sgnG {info["sgnG_sha256"][:8]} la {info["la_sha256"][:8]} code {info["code_sha256"][:8]}')
    if dmax > 1e-12:
        P('ABORT rows identity failed'); sys.exit(2)
    seedG = a.seedG if a.indep else a.seed
    assert not a.indep or (seedG is not None and seedG != a.seed)
    ck = os.path.join(a.out, 'ckpt.npz'); rec = []; b0 = 0
    if a.resume and os.path.exists(ck):
        z = np.load(ck, allow_pickle=True); b0 = int(z['nb']); rec = list(z['rec'].tolist())
        pg = PairGFMC(g, z['XS'], z['XG'], z['labS'], z['labG'], a.dtau, a.seed, seedG, a.resample, a.M, a.move)
        P(f'RESUMED at block {b0} from {ck}')
    else:
        la6 = np.load(a.la, mmap_mode='r'); fn = lambda S: np.asarray(la6[g.canon_idx(((S > 0).astype(U64) * g.code).sum(1, dtype=U64))], np.float64).astype(np.complex128)
        t0 = time.time()
        if a.X0:
            XS = np.load(a.X0).astype(np.int8); acc = float('nan')
        else:
            XS, acc = GA.metropolis_init(fn, 6, a.M, a.seed, a.nsweep)
        if a.indep:
            XG = np.load(a.X0G).astype(np.int8) if a.X0G else GA.metropolis_init(fn, 6, a.M, seedG, a.nsweep)[0]
        else:
            XG = XS.copy()
        assert XS.shape == (a.M, N) and XG.shape == (a.M, N)
        np.save(os.path.join(a.out, 'X0S.npy'), XS); np.save(os.path.join(a.out, 'X0G.npy'), XG)
        with np.errstate(over="ignore"):
            labS = mix(U64(a.seed) * C3 + np.arange(a.M, dtype=U64)); labG = mix(U64(seedG) * C3 + np.arange(a.M, dtype=U64))
        P(f'init walkers acc {acc:.3f} {time.time() - t0:.0f}s Sz {np.abs(XS.sum(1)).max()} {np.abs(XG.sum(1)).max()} X0 identical {bool(np.array_equal(XS, XG))}')
        pg = PairGFMC(g, XS, XG, labS, labG, a.dtau, a.seed, seedG, a.resample, a.M, a.move)
    t_start = time.time(); tot = a.burn + a.blocks

    def save(nb):
        tmp = ck + '.tmp.npz'
        np.savez(tmp, XS=pg.arms[0].X, XG=pg.arms[1].X, labS=pg.arms[0].lab, labG=pg.arms[1].lab, rec=np.array(rec, dtype=object), nb=nb)
        os.replace(tmp, ck)
    status = 'OK'; stopped = False
    for b in range(b0, tot):
        tb = time.time(); nc0 = g.n_canon; nr0 = g.n_req
        r = pg.block(b)
        r['sec'] = time.time() - tb; r['block'] = b; r['canon'] = g.n_canon - nc0; r['req'] = g.n_req - nr0
        r['xhS'] = int(hashlib.sha256(np.ascontiguousarray(pg.arms[0].X).tobytes()).hexdigest()[:15], 16)
        r['xhG'] = int(hashlib.sha256(np.ascontiguousarray(pg.arms[1].X).tobytes()).hexdigest()[:15], 16)
        rec.append(r)
        if (b + 1) % 10 == 0 or b < 3:
            P(f'block {b + 1}/{tot} dS {r["emix_S"] / N - a.E0:+.4e} dG {r["emix_G"] / N - a.E0:+.4e} diff {(r["emix_S"] - r["emix_G"]) / N:+.3e} '
              f'ident {r["n_ident"]} xsame {r["n_xsame"]}/{r["MS"]},{r["MG"]} dlw {r["dlogw_max_xsame"]:.1e} block_ident {r["block_ident"]} hops {r["hops_S"]},{r["hops_G"]} canon {r["canon"]}/{r["req"]} sec {r["sec"]:.2f}')
        if not (np.isfinite(r['emix_S']) and np.isfinite(r['emix_G'])):
            status = 'NAN energy'; P(f'ABORT block {b + 1}: non-finite energy'); break
        if (b + 1) % a.ckpt_every == 0: save(b + 1)
        if time.time() - t_start > a.maxsec:
            save(b + 1); stopped = True; P(f'MAXSEC reached at block {b + 1}'); break
    nb = len(rec); save(nb)
    keys = [k for k in rec[0] if k not in ('block',)] if rec else []
    np.savez(os.path.join(a.out, 'blocks.npz'), burn=a.burn, dtau=a.dtau, M=a.M, N=N, **{k: np.array([q[k] for q in rec]) for k in keys})
    eS = np.array([q['emix_S'] for q in rec[a.burn:]]) / N; eG = np.array([q['emix_G'] for q in rec[a.burn:]]) / N
    np.save(os.path.join(a.out, 'E_S.npy'), eS); np.save(os.path.join(a.out, 'E_G.npy'), eG); np.save(os.path.join(a.out, 'E_D.npy'), eS - eG)
    out = dict(args=vars(a), info=info, nblocks=nb, completed=(not stopped and status == 'OK' and nb == tot), status=status)
    if len(eS) >= 32:
        for name, x in (('S', eS), ('G', eG), ('D', eS - eG)):
            if np.all(x == x[0]):
                m, errs, tau = float(x[0]), [0.0] * 4, 0.0
            else:
                m, errs, tau = blocked_error(x)
            out[name] = dict(mean=m, err=max(errs[-4:]), err_level0=errs[0], var_block=float(np.var(x, ddof=1)), tau_blocking=tau)
        out['S']['d_E0'] = out['S']['mean'] - a.E0; out['G']['d_E0'] = out['G']['mean'] - a.E0
        q = rec[a.burn:]
        out['coinc'] = dict(frac_walkers_ident=float(np.mean([x['n_ident'] / max(x['MS'], x['MG']) for x in q])),
                            frac_walkers_xsame=float(np.mean([x['n_xsame'] / max(x['MS'], x['MG']) for x in q])),
                            frac_blocks_ident=float(np.mean([x['block_ident'] for x in q])),
                            frac_blocks_diff_zero=float(np.mean(eS == eG)))
        out['walker_tau_per_arm'] = float(np.sum([0.5 * (x['MS'] + x['MG']) for x in q]) * a.dtau)
    out['sec_per_block_median'] = float(np.median([q['sec'] for q in rec])) if rec else None
    out['canon_frac'] = g.n_canon / max(g.n_req, 1)
    json.dump(out, open(os.path.join(a.out, 'result.json'), 'w'), indent=1, default=float)
    P(f'GFMC_PAIR_DONE status {status} blocks {nb}/{tot} completed {out["completed"]}' +
      (f' tax {out["D"]["mean"]:.4e} +- {out["D"]["err"]:.1e} identW {out["coinc"]["frac_walkers_ident"]:.3f} xsameW {out["coinc"]["frac_walkers_xsame"]:.3f}' if 'D' in out else ''))
