"""Deliverable C, E2: lattice fixed-node DMC (FN-GFMC) with a callable guide, for L x L periodic J1-J2 (J2/J1 = 0.5).

Algorithm = the calibrated referee of experiments/fn_calibration_6x6/fncal_dmc.py (results/fn_calibration_6x6/README.md):
  propagator 1 - tau (H_FN - Eref_pop), Eref_pop = mean guide local energy of the population at the start of the step,
  tau = min(tau_max, 0.8 / max(0, max_walkers(d_FN - Eref_pop))), one move per walker per step (stay or one kept hop),
  walker weight tot = stay + sum of kept hop weights, systematic resampling every step (per population).
  H_FN: kept edges (s_x s_y < 0, i.e. H_xy psi_x psi_y < 0 since H_xy = +J/2) keep H_xy; violating edges are removed and
  their J/2 a(y)/a(x) is added to the diagonal: d_FN = H_xx + V.  Guide local energy E_L = d_FN - W (W = kept sum).
  Estimator: beta-TIME-AVERAGED population energy per population over the window [burn, beta_target] (fncal_analyze.pop_means).
Difference to fncal_dmc.py: the guide is a callable (projected ViT, or a table for the 4x4 gate); every walker carries its
one-hop data (d_FN, E_L, kept rates) and only walkers that hopped are re-evaluated (stay / resampled copies reuse it).

  8x8:  python dc_fn.py OUT vit ...   (guide 'vit' = psi_P, 'psi1' = one Lanczos step on psi_P, Paderborn)
        python dc_fn.py OUT vit --npop_max 64 --batch_pops 4 --se_target 1e-5 --min_pop 16 --hours 8.5 --pool a.npz,b.npz
  4x4 gate: dc_fn_gate4x4.py (imports this module)
"""
import os, sys, time, json, argparse
import numpy as np

ONE = np.uint64(1)


def log(*a):
    print(time.strftime('[%H:%M:%S]'), *a, flush=True)


class Lattice:
    def __init__(self, L, J2=0.5):
        self.L = L; self.N = L * L
        q = lambda x, y: (x % L) + L * (y % L)
        nn_, nnn = [], []
        for y in range(L):
            for x in range(L):
                i = q(x, y); nn_ += [(i, q(x + 1, y)), (i, q(x, y + 1))]
                nnn += [(i, q(x + 1, y + 1)), (i, q(x + 1, y - 1))]
        B = [(i, j, 1.0) for i, j in nn_] + [(i, j, J2) for i, j in nnn]
        self.BI = np.array([b[0] for b in B], np.uint64); self.BJ = np.array([b[1] for b in B], np.uint64)
        self.JB = np.array([b[2] for b in B]); self.NB = len(B)
        self.MASKS = (ONE << self.BI) | (ONE << self.BJ)

    def neighbors(self, S):
        S = np.asarray(S, np.uint64)[:, None]
        valid = (((S >> self.BI) ^ (S >> self.BJ)) & ONE).astype(bool)
        return S ^ self.MASKS, valid

    def diag(self, valid):
        return 0.25 * self.JB.sum() - 0.5 * (valid.astype(float) @ self.JB)


def local_data(lat, guide, U):
    """U: unique states (n,).  guide(states) -> (log a, sign +-1).  Returns d_FN, E_L, kept rates wa (n, NB)."""
    U = np.asarray(U, np.uint64); n = len(U)
    NBs, V = lat.neighbors(U)
    st = np.concatenate([U, NBs[V]])
    u, inv = np.unique(st, return_inverse=True)
    la_u, s_u = guide(u)
    la = la_u[inv]; s = s_u[inv]
    lax = la[:n]; sx = s[:n]
    lan = np.zeros((n, lat.NB)); sn = np.zeros((n, lat.NB)); lan[V] = la[n:]; sn[V] = s[n:]
    r = np.where(V, 0.5 * lat.JB[None, :] * np.exp(np.clip(lan - lax[:, None], -60, 60)), 0.0)
    kept = V & (sx[:, None] * sn < 0)
    wa = np.where(kept, r, 0.0)
    dfn = lat.diag(V) + np.where(V & ~kept, r, 0.0).sum(1)
    return dfn, dfn - wa.sum(1), wa, len(u)


def _data_for(lat, guide, S):
    U, inv = np.unique(S, return_inverse=True)
    dfn, el, wa, nev = local_data(lat, guide, U)
    return dfn[inv], el[inv], wa[inv], nev


def run_batch(lat, guide, pool, P, M, rg, beta_target=2.4, tau_max=0.025, max_steps=5000, verbose=False):
    """P independent populations of M walkers.  Returns Eref (P, S), tau (P, S), ndist (P, S), evaluations."""
    x = np.concatenate([pool[rg.choice(len(pool), M, replace=len(pool) < M)] for _ in range(P)]).astype(np.uint64)
    dfn, el, wa, nev = _data_for(lat, guide, x); nevals = nev
    beta = np.zeros(P); Er, Ta, Nd = [], [], []
    pid = np.repeat(np.arange(P), M); t0 = time.time()
    for it in range(max_steps):
        if not np.any(beta < beta_target): break
        Eref = el.reshape(P, M).mean(1)
        mxd = np.maximum(0.0, (dfn - Eref[pid]).reshape(P, M).max(1))
        tau = np.minimum(tau_max, np.where(mxd > 0, 0.8 / np.maximum(mxd, 1e-300), tau_max))
        tau = np.where(beta < beta_target, tau, 0.0)
        tw = tau[pid]
        stay = 1.0 - tw * (dfn - Eref[pid])
        ws = tw[:, None] * wa
        tot = stay + ws.sum(1)
        u = rg.random(P * M) * tot
        hop = u >= stay
        y = x.copy()
        if hop.any():
            h = np.nonzero(hop)[0]
            cs = np.cumsum(ws[h], 1)
            j = np.sum(cs <= (u[h] - stay[h])[:, None], 1)
            last = np.max(np.where(wa[h] > 0, np.arange(lat.NB)[None, :], 0), 1)
            j = np.minimum(j, last)
            y[h] = x[h] ^ lat.MASKS[j]
            d2, e2, w2, nev = _data_for(lat, guide, y[h]); nevals += nev
            dfn = dfn.copy(); el = el.copy(); wa = wa.copy()
            dfn[h] = d2; el[h] = e2; wa[h] = w2
        # systematic resampling inside each population
        T = tot.reshape(P, M); bw = T / T.sum(1, keepdims=True)
        c = np.cumsum(bw, 1); c[:, -1] = 1.0
        pos = (rg.random((P, 1)) + np.arange(M)[None, :]) / M
        idx = np.minimum(np.array([np.searchsorted(c[p], pos[p], side='right') for p in range(P)]), M - 1)
        g = (idx + (np.arange(P) * M)[:, None]).reshape(-1)
        x = y[g]; dfn = dfn[g]; el = el[g]; wa = wa[g]
        Er.append(Eref); Ta.append(tau); Nd.append(1 + np.sum(np.diff(idx, axis=1) != 0, 1))
        beta = beta + tau
        if verbose and it % 10 == 0:
            log(f'step {it} beta {beta.min():.3f} tau {tau.mean():.4f} hop {hop.mean():.3f} E/N {Eref.mean() / lat.N:.7f} '
                f'evals {nevals} {time.time() - t0:.0f}s')
    return np.stack(Er, 1), np.stack(Ta, 1), np.stack(Nd, 1), nevals


def pop_means(Eref, tau, burn, beta, N):
    """beta-time average (fncal_analyze.pop_means): population at the start of step t lives for tau_t."""
    b_after = np.cumsum(tau, 1); b_before = b_after - tau
    w = np.clip(np.minimum(b_after, beta) - np.maximum(b_before, burn), 0, None) * (tau > 0)
    return (Eref * w).sum(1) / np.maximum(w.sum(1), 1e-30) / N


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('guide')
    ap.add_argument('--M', type=int, default=512); ap.add_argument('--batch_pops', type=int, default=4)
    ap.add_argument('--npop_max', type=int, default=64); ap.add_argument('--min_pop', type=int, default=16)
    ap.add_argument('--se_target', type=float, default=1e-5); ap.add_argument('--hours', type=float, default=8.5)
    ap.add_argument('--beta', type=float, default=2.4); ap.add_argument('--burn', type=float, default=0.8)
    ap.add_argument('--tau_max', type=float, default=0.025); ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--pool', required=True); ap.add_argument('--phi', type=float, default=0.056006)
    ap.add_argument('--ckpt', default='vit_J2=0.50_N=8x8_k=0.mpack'); ap.add_argument('--evbatch', type=int, default=1024)
    ap.add_argument('--alpha', type=float, default=-0.160951); ap.add_argument('--Eref', type=float, default=-31.92677568)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); T0 = time.time()
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'loop_8x8'))
    import l8_core as C
    g = C.Guide(a.ckpt, sym=True, batch=a.evbatch, phi=a.phi)
    if a.guide == 'vit':
        guide = lambda S: (lambda la, ph: (la, g.sign(ph)))(*g.eval(S))
    elif a.guide == 'psi1':                     # one Lanczos step psi_1 = psi_P (1 + alpha (E_L - E_ref)): one hop per amplitude
        def guide(S):
            d = C.onehop(g, S); f = 1 + a.alpha * (d['ELa'] - a.Eref)
            return d['la'] + np.log(np.maximum(np.abs(f), 1e-300)), d['s'] * np.where(f >= 0, 1.0, -1.0)
    else:
        raise ValueError(a.guide)
    lat = Lattice(8)
    assert np.allclose(lat.JB, C.JB) and np.all(lat.MASKS == C.MASKS)
    pool = np.unique(np.concatenate([np.load(p)['X'] for p in a.pool.split(',')]).astype(np.uint64))
    log('pool', len(pool), 'guide params', g.npar)
    f = os.path.join(a.out, 'fn.npz'); Es, Ts, Ns = [], [], []; bi = 0; nev_tot = 0
    if os.path.exists(f):
        z = np.load(f); Es = [z['Eref']]; Ts = [z['tau']]; Ns = [z['ndist']]; bi = int(z['nbatch']); nev_tot = int(z['nevals'])
    rg = np.random.default_rng([a.seed, bi])
    while True:
        npop = sum(e.shape[0] for e in Es)
        if npop >= a.npop_max: break
        if npop > 0:
            S = max(e.shape[1] for e in Es)
            E = np.concatenate([np.pad(e, ((0, 0), (0, S - e.shape[1]))) for e in Es])
            T = np.concatenate([np.pad(t, ((0, 0), (0, S - t.shape[1]))) for t in Ts])
            pm = pop_means(E, T, a.burn, a.beta, 64)
            se = pm.std(ddof=1) / np.sqrt(len(pm)) if len(pm) > 1 else np.inf
            el_h = (time.time() - T0) / 3600
            log(f'npop {npop} E_FN/N {pm.mean():.7f} +- {se:.2e} sd_pop {pm.std(ddof=1) if len(pm) > 1 else 0:.2e} {el_h:.2f} h')
            if npop >= a.min_pop and se <= a.se_target: break
            per_batch = el_h / max(bi, 1)
            if el_h + per_batch > a.hours: log('time cap'); break
        rg = np.random.default_rng([a.seed, bi]); t = time.time()
        E, T, Nd, nev = run_batch(lat, guide, pool, a.batch_pops, a.M, rg, a.beta, a.tau_max, verbose=True)
        Es.append(E); Ts.append(T); Ns.append(Nd); bi += 1; nev_tot += nev
        S = max(e.shape[1] for e in Es)
        pad = lambda arrs, v: np.concatenate([np.pad(x, ((0, 0), (0, S - x.shape[1])), constant_values=v) for x in arrs])
        np.savez(f, Eref=pad(Es, 0.0), tau=pad(Ts, 0.0), ndist=pad(Ns, 0), nbatch=bi, nevals=nev_tot, M=a.M, beta=a.beta,
                 burn=a.burn, tau_max=a.tau_max)
        log(f'batch {bi} done {time.time() - t:.0f}s steps {E.shape[1]} evals {nev}')
    z = np.load(f); pm = pop_means(z['Eref'], z['tau'], a.burn, a.beta, 64)
    pm2 = pop_means(z['Eref'], z['tau'], 0.4, 1.2, 64)
    res = dict(args=vars(a), npop=len(pm), E_FN_site=float(pm.mean()), se=float(pm.std(ddof=1) / np.sqrt(len(pm))),
               sd_pop=float(pm.std(ddof=1)), E_window_0p4_1p2=float(pm2.mean()), se_0p4_1p2=float(pm2.std(ddof=1) / np.sqrt(len(pm2))),
               pop_means=pm.tolist(), evals=int(z['nevals']), hours=(time.time() - T0) / 3600)
    json.dump(res, open(os.path.join(a.out, 'fn.json'), 'w'), indent=1)
    log('DONE', json.dumps({k: v for k, v in res.items() if k != 'pop_means'}))


if __name__ == '__main__':
    main()
