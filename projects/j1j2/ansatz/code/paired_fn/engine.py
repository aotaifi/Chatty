"""engine.py (paired_fn): paired lattice fixed-node GFMC for two guides A, B on one joint run.

Spec: /scratch/a/A.Otaifi/astra/paired_fn/ASTRA_paired.md section (3) "explicit split coupling plus correct paired resampling" (mode
'coupled', default), with the experimental modes 'union' (eq 8 + common resampling eq 9) and 'reweight_A' (eq 5, only on established
support; a B-only kept edge at any visited state raises SupportViolation).  Also mode 'single' (one standalone multinomial arm; used to
measure the standalone variances for kappa_work).  Delta = E_FN[B] - E_FN[A] (total energy; per site in the summaries).

Continuous time, exact in tau: per pair of walkers the joint process has, for each physical bond b,
    both arms swap b at c_b = min(r_A(b|x_A), r_B(b|x_B)),   A only at r_A - c_b,   B only at r_B - c_b,
so each marginal has exactly its own rates r_g (generator check in stage 1).  Each arm integrates its own E_L over the common elapsed time;
the block end (dtau) is handled exactly (no event beyond it).  At the block end, parents are drawn by the maximal categorical coupling of
the two arms' multinomial parent laws aggregated by configuration key (resample_coupled); arm weights reset within each arm (growth
normalisers are recorded).  Block estimators (as gfmc_fn_lib.GFMC.block): mixed E_g = sum_i w_g,i E_L,g(x_g,i) / sum_i w_g,i with the
block-local weights w = exp(-int E_L dt), growth E_g = -ln(mean w_g)/dtau.

Union: one state x, two log masses U_A, U_B (eq 8) w.r.t. the reference rates r_* = max(r_A, r_B); masses persist across blocks;
common multinomial parents p_i = (v_A,i/Z_A + v_B,i/Z_B)/2, offspring mass v/(M p_i) (eq 9); arm masses renormalised each block.
reweight_A: one state x walked with A's rates, masses v_A (block-local A weight) and v_B = v_A * R_{B/A} (eq 5; hop ratio r_B/r_A, dwell
-(d_B - d_A)); parents by p_A; B masses inherit v_B/(M p_A).  memory_tau (blocks) optionally resets R_{B/A} to 1 every memory_tau blocks
(finite-memory, biased: eq 10).  broken='inverse_hop_ratio' uses r_A/r_B (validation only).
"""
import json, time
import numpy as np


class SupportViolation(RuntimeError):
    pass


class EstimatorFailure(RuntimeError):
    pass


# ------------------------------------------------------------------------------------------------ resampling
def multinomial_parents(p, M, rng):
    c = np.cumsum(p); c /= c[-1]
    return np.minimum(np.searchsorted(c, rng.random(M), side='right'), len(p) - 1)


def _within_group(gid, p, zsel, rng, ng):
    """choose, for each requested group z in zsel, an index i with gid[i] == z with probability p_i / sum_{gid == z} p."""
    order = np.argsort(gid, kind='stable'); ps = p[order]; gs = gid[order]
    cum = np.cumsum(ps); start = np.searchsorted(gs, np.arange(ng), side='left'); end = np.searchsorted(gs, np.arange(ng), side='right')
    base = np.where(start > 0, cum[np.maximum(start - 1, 0)], 0.0); base[start == 0] = 0.0
    tot = np.where(end > start, cum[np.maximum(end - 1, 0)] - base, 0.0)
    v = base[zsel] + rng.random(len(zsel)) * tot[zsel]
    k = np.searchsorted(cum, v, side='right')
    k = np.clip(k, start[zsel], end[zsel] - 1)                      # floating guard: stay inside the group
    bad = ps[k] <= 0                                                  # (cannot happen except at fp edges) take the heaviest in the group
    if bad.any():
        for m in np.flatnonzero(bad):
            s, e = start[zsel[m]], end[zsel[m]]; k[m] = s + int(np.argmax(ps[s:e]))
    return order[k]


def resample_coupled(pA, pB, keyA, keyB, M, rng, force_A_law=False):
    """Maximal categorical coupling of the two multinomial parent laws, aggregated by configuration (ASTRA_paired.md section 3).
    Returns (iA, iB, alpha, n_common).  Each marginal is exactly multinomial(p_g); P(equal configuration) = alpha = sum_z min(pi_A, pi_B).
    force_A_law=True is the BROKEN control (B uses A's parents)."""
    pA = pA / pA.sum(); pB = pB / pB.sum()
    if force_A_law:
        i = multinomial_parents(pA, M, rng); return i, i.copy(), float('nan'), M
    allk = np.concatenate([keyA, keyB]); uz, inv = np.unique(allk, return_inverse=True); nz = len(uz)
    gA, gB = inv[:len(keyA)], inv[len(keyA):]
    piA = np.bincount(gA, pA, nz); piB = np.bincount(gB, pB, nz); mn = np.minimum(piA, piB); alpha = float(mn.sum())
    if alpha > 1 - 1e-13: alpha = 1.0
    u = rng.random(M); com = u < alpha; nc = int(com.sum()); nr = M - nc
    iA = np.empty(M, np.int64); iB = np.empty(M, np.int64)
    if nc:
        z = multinomial_parents(mn, nc, rng)
        iA[com] = _within_group(gA, pA, z, rng, nz); iB[com] = _within_group(gB, pB, z, rng, nz)
    if nr:
        rA = np.maximum(piA - mn, 0.0); rB = np.maximum(piB - mn, 0.0)
        zA = multinomial_parents(rA, nr, rng); zB = multinomial_parents(rB, nr, rng)
        iA[~com] = _within_group(gA, pA, zA, rng, nz); iB[~com] = _within_group(gB, pB, zB, rng, nz)
    return iA, iB, alpha, nc


# ------------------------------------------------------------------------------------------------ arm state
class Arm:
    def __init__(self, guide, S):
        self.g = guide; self.S = np.array(S, copy=True); self.M = len(self.S)
        self.rate = np.zeros((self.M, guide.NB)); self.d = np.zeros(self.M); self.R = np.zeros(self.M); self.EL = np.zeros(self.M)
        self.nonfinite = 0
        self.refresh(np.arange(self.M))

    def refresh(self, idx):
        if len(idx) == 0: return
        rb = self.g.rows(self.S[idx])
        r = rb.rate; d = rb.dFN
        bad = ~(np.isfinite(r).all(1) & np.isfinite(d))
        if bad.any():
            self.nonfinite += int(bad.sum()); raise EstimatorFailure(f'non-finite FN row for {int(bad.sum())} states ({self.g.guide_id[:12]})')
        self.rate[idx] = r; self.d[idx] = d; self.R[idx] = r.sum(1); self.EL[idx] = d - self.R[idx]

    def take(self, idx):
        self.S = self.S[idx]; self.rate = self.rate[idx]; self.d = self.d[idx]; self.R = self.R[idx]; self.EL = self.EL[idx]


def _wstats(logw):
    mx = np.max(logw)
    if not np.isfinite(mx): return dict(ess=0.0, mmax=float('nan'), top10=float('nan'), span=float('nan'), zero=1.0, lnZ=-np.inf), np.zeros_like(logw)
    w = np.exp(logw - mx); sw = w.sum(); fin = np.isfinite(logw)
    srt = np.sort(w)[::-1]
    st = dict(ess=float(sw * sw / np.sum(w * w)), mmax=float(srt[0] / sw), top10=float(srt[:10].sum() / sw),
              span=float(mx - logw[fin].min()), zero=float(1.0 - fin.mean() if np.isneginf(logw).any() else (w == 0).mean()), lnZ=float(mx + np.log(sw)))
    return st, w


# ------------------------------------------------------------------------------------------------ the engine
class PairedFN:
    """One joint run.  guide_A, guide_B: providers (providers.py contract); X0_A, X0_B: initial states (M each; matched within a pair)."""

    def __init__(self, guide_A, guide_B, X0_A, X0_B=None, mode='coupled', dtau=0.1, seed=0, reference_energy=0.0,
                 memory_tau=None, broken=None, anc_lags=(10, 100, 1000), diag_every=1, check_support=True):
        assert mode in ('coupled', 'union', 'reweight_A', 'single'), mode
        self.mode = mode; self.dtau = float(dtau); self.rng = np.random.default_rng(seed); self.seed = seed
        self.Eref = float(reference_energy); self.memory_tau = memory_tau; self.broken = broken
        self.gA, self.gB = guide_A, guide_B
        if guide_B is not None: assert guide_A.NB == guide_B.NB, 'both arms must use the same bond IDs'
        X0_B = X0_A if X0_B is None else X0_B
        self.M = len(X0_A); self.block_no = 0; self.anc_lags = tuple(anc_lags); self.n_support_checked = 0
        self.A = Arm(guide_A, X0_A)
        if mode == 'coupled':
            self.B = Arm(guide_B, X0_B)
        elif mode in ('union', 'reweight_A'):
            assert np.array_equal(np.asarray(X0_A), np.asarray(X0_B)), 'one-coordinate modes need one initial population'
            self.B = Arm(guide_B, X0_A)
            self.logvA = np.zeros(self.M); self.logvB = np.zeros(self.M)
            self.check_support = check_support
            if mode == 'reweight_A' and check_support: self._support(np.arange(self.M))
        else:
            self.B = None
        self.lnZ = {'A': 0.0, 'B': 0.0}
        self.ancA = np.arange(self.M); self.ancB = np.arange(self.M)
        self.anc = {L: (np.arange(self.M), np.arange(self.M)) for L in self.anc_lags}

    # -------------------------------------------------- support check (reweight_A)
    def _support(self, idx):
        bonly = (self.B.rate[idx] > 0) & (self.A.rate[idx] <= 0)
        self.n_support_checked += len(idx)
        if bonly.any():
            w = np.flatnonzero(bonly.any(1))[0]
            raise SupportViolation(f'B-only kept edge at state {self.A.S[idx][w]!r} bond {int(np.flatnonzero(bonly[w])[0])}: A-only reweighting has no support')

    # -------------------------------------------------- one block
    def block(self):
        if self.mode == 'coupled': return self._block_coupled()
        if self.mode == 'single': return self._block_single()
        return self._block_onecoord()

    def _finish_arm(self, logw, arm):
        st, w = _wstats(logw)
        sw = w.sum()
        emix = float(np.sum(w * arm.EL) / sw) if sw > 0 else float('nan')
        egrow = float(-(st['lnZ'] - np.log(self.M)) / self.dtau) + self.Eref
        st.update(emix=emix, egrow=egrow, num=float(np.sum(w * arm.EL) / self.M), den=float(sw / self.M), lnscale=float(np.max(logw)))
        return st, w

    def _block_single(self):
        A = self.A; M = self.M; rem = np.full(M, self.dtau); logw = np.zeros(M); act = np.arange(M); hops = 0
        while act.size:
            with np.errstate(divide='ignore'):
                t = self.rng.exponential(1.0, act.size) / A.R[act]
            stay = t >= rem[act]; fin = act[stay]; logw[fin] -= rem[fin] * (A.EL[fin] - self.Eref)
            mv = act[~stay]; tm = t[~stay]; logw[mv] -= tm * (A.EL[mv] - self.Eref); rem[mv] -= tm
            if mv.size:
                b = _choose(A.rate[mv], self.rng)
                A.S[mv] = A.g.move(A.S[mv], b); A.refresh(mv); hops += mv.size
            act = mv
        st, w = self._finish_arm(logw, A)
        p = w / w.sum(); par = multinomial_parents(p, M, self.rng); A.take(par); self._anc_update(par, par)
        self.lnZ['A'] += st['lnZ'] - np.log(M)
        self.block_no += 1
        return dict(block=self.block_no, emix_A=st['emix'], egrow_A=st['egrow'], A=st, hops_A=hops, **self._anc_report())

    def _block_coupled(self):
        A, B = self.A, self.B; M = self.M; NB = A.g.NB
        rem = np.full(M, self.dtau); lA = np.zeros(M); lB = np.zeros(M); act = np.arange(M)
        cnt = dict(shared=0, A_only=0, B_only=0, iters=0)
        sep_flux = 0.0; cut_flux = 0.0; matched_time = 0.0
        while act.size:
            ra = A.rate[act]; rb = B.rate[act]; c = np.minimum(ra, rb)
            J = np.concatenate([c, ra - c, rb - c], 1); Rj = J.sum(1)
            # diagnostics: separation (sum_b |r_A - r_B|) and exclusive-support (h_cut, eq 14) rates while matched, integrated over dwell
            matched = A.g.key(A.S[act]) == B.g.key(B.S[act])
            with np.errstate(divide='ignore'):
                t = self.rng.exponential(1.0, act.size) / Rj
            dt = np.minimum(t, rem[act])
            if matched.any():
                sep = np.abs(ra - rb).sum(1); hc = (ra * ((ra > 0) & (rb <= 0))).sum(1) + (rb * ((rb > 0) & (ra <= 0))).sum(1)
                sep_flux += float((sep * dt)[matched].sum()); cut_flux += float((hc * dt)[matched].sum()); matched_time += float(dt[matched].sum())
            lA[act] -= dt * (A.EL[act] - self.Eref); lB[act] -= dt * (B.EL[act] - self.Eref)
            mvm = t < rem[act]; mv = act[mvm]; rem[mv] -= t[mvm]
            if mv.size:
                k = _choose(J[mvm], self.rng); typ = k // NB; b = k % NB
                ma = typ != 2; mb = typ != 1
                ia = mv[ma]; ib = mv[mb]
                if ia.size: A.S[ia] = A.g.move(A.S[ia], b[ma]); A.refresh(ia)
                if ib.size: B.S[ib] = B.g.move(B.S[ib], b[mb]); B.refresh(ib)
                cnt['shared'] += int((typ == 0).sum()); cnt['A_only'] += int((typ == 1).sum()); cnt['B_only'] += int((typ == 2).sum())
            act = mv; cnt['iters'] += 1
        stA, wA = self._finish_arm(lA, A); stB, wB = self._finish_arm(lB, B)
        kA = A.g.key(A.S); kB = B.g.key(B.S)
        same_pre = float((kA == kB).mean())
        diag = self._flip_diag(wA, wB)
        iA, iB, alpha, nc = resample_coupled(wA, wB, kA, kB, M, self.rng, force_A_law=(self.broken == 'force_A_parents'))
        A.take(iA); B.take(iB); self._anc_update(iA, iB)
        same_post = float((A.g.key(A.S) == B.g.key(B.S)).mean())
        self.lnZ['A'] += stA['lnZ'] - np.log(M); self.lnZ['B'] += stB['lnZ'] - np.log(M)
        self.block_no += 1
        return dict(block=self.block_no, emix_A=stA['emix'], emix_B=stB['emix'], egrow_A=stA['egrow'], egrow_B=stB['egrow'],
                    delta=stB['emix'] - stA['emix'], delta_grow=stB['egrow'] - stA['egrow'], A=stA, B=stB,
                    alpha=alpha, n_common=nc, same_pre=same_pre, same_post=same_post, sep_flux=sep_flux, cut_flux=cut_flux,
                    matched_time=matched_time, **cnt, **diag, **self._anc_report())

    def _block_onecoord(self):
        A, B = self.A, self.B; M = self.M; union = self.mode == 'union'
        rem = np.full(M, self.dtau); act = np.arange(M); cnt = dict(hops=0, A_excl=0, B_excl=0, iters=0)
        vA, vB = self.logvA, self.logvB
        inv = self.broken == 'inverse_hop_ratio'
        while act.size:
            ra = A.rate[act]; rb = B.rate[act]
            rs = np.maximum(ra, rb) if union else ra
            lam = rs.sum(1)
            with np.errstate(divide='ignore'):
                t = self.rng.exponential(1.0, act.size) / lam
            dt = np.minimum(t, rem[act])
            if union:   # eq 8: dwell -(d_g - lambda_*) dt
                vA[act] -= dt * (A.d[act] - lam - self.Eref); vB[act] -= dt * (B.d[act] - lam - self.Eref)
            else:       # A weight: -E_L,A dt;  B mass = A mass * R_{B/A}: extra -(d_B - d_A) dt  (eq 5)
                vA[act] -= dt * (A.EL[act] - self.Eref); vB[act] -= dt * (A.EL[act] - self.Eref + B.d[act] - A.d[act])
            mvm = t < rem[act]; mv = act[mvm]; rem[mv] -= t[mvm]
            if mv.size:
                b = _choose(rs[mvm], self.rng); rr = np.arange(mv.size)
                pa = A.rate[mv, b]; pb = B.rate[mv, b]; ps = rs[mvm][rr, b]
                with np.errstate(divide='ignore'):
                    if union:
                        vA[mv] += np.log(pa) - np.log(ps); vB[mv] += np.log(pb) - np.log(ps)
                    else:
                        vB[mv] += (np.log(pa) - np.log(pb)) if inv else (np.log(pb) - np.log(pa))
                cnt['A_excl'] += int(((pa > 0) & (pb <= 0)).sum()); cnt['B_excl'] += int(((pb > 0) & (pa <= 0)).sum())
                S2 = A.g.move(A.S[mv], b); A.S[mv] = S2; B.S[mv] = S2; A.refresh(mv); B.refresh(mv)
                if self.mode == 'reweight_A' and self.check_support: self._support(mv)
                cnt['hops'] += mv.size
            act = mv; cnt['iters'] += 1
        stA, wA = self._finish_arm(vA, A); stB, wB = self._finish_arm(vB, B)
        if not (np.isfinite(vA).any() and np.isfinite(vB).any()):
            raise EstimatorFailure(f'{self.mode}: an arm has no surviving mass (A {np.isfinite(vA).sum()}, B {np.isfinite(vB).sum()})')
        # overlap of normalised masses on the common paths
        pA = wA / wA.sum(); pB = wB / wB.sum(); overlap = float(np.minimum(pA, pB).sum())
        diag = self._flip_diag(wA, wB)
        p = 0.5 * (pA + pB) if union else pA
        par = multinomial_parents(p, M, self.rng)
        with np.errstate(divide='ignore'):
            nA = np.log(pA[par]) - np.log(M * p[par]); nB = np.log(pB[par]) - np.log(M * p[par])
        A.take(par); B.take(par); self._anc_update(par, par)
        self.lnZ['A'] += stA['lnZ'] - np.log(M); self.lnZ['B'] += stB['lnZ'] - np.log(M)
        self.block_no += 1
        if self.mode == 'reweight_A' and self.memory_tau and self.block_no % int(self.memory_tau) == 0:
            nB = nA.copy()
        self.logvA = nA + np.log(M); self.logvB = nB + np.log(M)        # offspring masses (eq 9), normalised to mean ~1
        return dict(block=self.block_no, emix_A=stA['emix'], emix_B=stB['emix'], egrow_A=float('nan'), egrow_B=float('nan'),
                    delta=stB['emix'] - stA['emix'], A=stA, B=stB, overlap=overlap, **cnt, **diag, **self._anc_report())

    # -------------------------------------------------- diagnostics
    def _flip_diag(self, wA, wB):
        """end-weight occupancy of F = {x : s_A(x) != s_B(x)} for each arm's walkers (needs provider.sign); counts and weights."""
        out = {}
        if self.B is None or getattr(self.gA, 'sign', None) is None: return out
        try:
            for nm, arm, w in (('A', self.A, wA), ('B', self.B, wB)):
                sa = self.gA.sign(arm.S); sb = self.gB.sign(arm.S)
                if sa is None or sb is None: return {}
                inF = np.asarray(sa) != np.asarray(sb); sw = w.sum()
                out[f'F_count_{nm}'] = int(inF.sum()); out[f'F_weight_{nm}'] = float(w[inF].sum() / sw) if sw > 0 else float('nan')
                if inF.any():
                    wf = w[inF]; out[f'F_ess_frac_{nm}'] = float((wf.sum() ** 2 / np.sum(wf * wf)) / inF.sum())
        except Exception as e:                                   # diagnostics must never stop a run
            out['F_diag_error'] = repr(e)[:200]
        return out

    def _anc_update(self, iA, iB):
        for L in self.anc_lags:
            a, b = self.anc[L]; self.anc[L] = (a[iA], b[iB])
        self._anc_last = {}
        for L in self.anc_lags:
            if (self.block_no + 1) % L == 0:
                a, b = self.anc[L]
                ca = np.bincount(a, minlength=self.M); cb = np.bincount(b, minlength=self.M)
                self._anc_last[L] = dict(fam_A=int((ca > 0).sum()), fam_B=int((cb > 0).sum()),
                                         aess_A=float(self.M ** 2 / np.sum(ca.astype(float) ** 2)), aess_B=float(self.M ** 2 / np.sum(cb.astype(float) ** 2)))
                self.anc[L] = (np.arange(self.M), np.arange(self.M))

    def _anc_report(self):
        d = getattr(self, '_anc_last', {}); out = {}
        for L, v in d.items():
            for k, x in v.items(): out[f'anc{L}_{k}'] = x
        return out

    # -------------------------------------------------- checkpoint (both populations, all accumulated masses/normalisers, RNG)
    def state(self):
        st = dict(mode=self.mode, block_no=self.block_no, SA=self.A.S, lnZ=json.dumps(self.lnZ),
                  rng=json.dumps(self.rng.bit_generator.state))
        if self.B is not None: st['SB'] = self.B.S
        if self.mode in ('union', 'reweight_A'): st.update(logvA=self.logvA, logvB=self.logvB)
        return st


def _choose(rates, rng):
    """event index per row with probability rate / row sum (rows with positive sum)."""
    cum = np.cumsum(rates, 1); u = rng.random(len(rates)) * cum[:, -1]
    k = (cum <= u[:, None]).sum(1); k = np.minimum(k, rates.shape[1] - 1)
    bad = rates[np.arange(len(k)), k] <= 0
    if bad.any(): k[bad] = np.argmax(rates[bad], 1)
    return k


# ------------------------------------------------------------------------------------------------ driver
SCALARS = ('emix_A', 'emix_B', 'egrow_A', 'egrow_B', 'delta', 'delta_grow', 'alpha', 'n_common', 'same_pre', 'same_post', 'sep_flux', 'cut_flux',
           'matched_time', 'shared', 'A_only', 'B_only', 'hops', 'hops_A', 'A_excl', 'B_excl', 'overlap',
           'F_count_A', 'F_weight_A', 'F_ess_frac_A', 'F_count_B', 'F_weight_B', 'F_ess_frac_B')
ARMST = ('ess', 'mmax', 'top10', 'span', 'zero', 'num', 'den', 'lnscale')


def run(guide_A, guide_B, X0_A, X0_B=None, *, mode='coupled', M=None, dtau=0.1, burn_blocks=0, measure_blocks=100, pair_seed=0,
        reference_energy=0.0, output=None, memory_tau=None, coupling='split_rates+config_parent_coupling', broken=None, NS=1,
        progress_every=0, log=print, ckpt_every=0, meta=None, maxsec=None):
    """Run one joint pair ensemble; returns dict(series=..., summary=...).  output: directory for blocks.npz / summary.json / ckpt.npz.
    The series include burn blocks (flag 'burn'); estimators use the measured blocks only."""
    assert coupling == 'split_rates+config_parent_coupling'
    if M is not None: assert len(X0_A) == M
    t0 = time.time()
    eng = PairedFN(guide_A, guide_B, X0_A, X0_B, mode=mode, dtau=dtau, seed=pair_seed, reference_energy=reference_energy,
                   memory_tau=memory_tau, broken=broken)
    t_init = time.time() - t0
    nb = burn_blocks + measure_blocks
    ser = {k: np.full(nb, np.nan) for k in SCALARS}
    for g in ('A', 'B'):
        for k in ARMST: ser[f'{g}_{k}'] = np.full(nb, np.nan)
    anc = {}
    for i in range(nb):
        r = eng.block()
        for k in SCALARS:
            if k in r: ser[k][i] = r[k]
        for g in ('A', 'B'):
            if g in r:
                for k in ARMST: ser[f'{g}_{k}'][i] = r[g][k]
        for k, v in r.items():
            if k.startswith('anc'): anc.setdefault(k, []).append((i, v))
        if progress_every and (i + 1) % progress_every == 0:
            msg = f'PROGRESS block {i + 1}/{nb} emix_A {r["emix_A"]:.6f}'
            if 'delta' in r: msg += f' delta/site {r["delta"] / NS:+.3e} same {r.get("same_post", r.get("overlap", float("nan"))):.3f}'
            log(msg + f' {time.time() - t0:.0f}s')
        if output and ckpt_every and (i + 1) % ckpt_every == 0:
            np.savez(f'{output}/ckpt.npz', **eng.state())
        if maxsec and time.time() - t0 > maxsec and i + 1 > burn_blocks:
            nb = i + 1; ser = {k: v[:nb] for k, v in ser.items()}; log(f'PROGRESS maxsec {maxsec} reached after {nb} blocks'); break
    wall = time.time() - t0
    ser['burn'] = np.arange(nb) < burn_blocks
    out = dict(series=ser, anc=anc, eng=eng, wall=wall, t_init=t_init,
               meta=dict(mode=mode, M=len(X0_A), dtau=dtau, burn_blocks=burn_blocks, measure_blocks=measure_blocks, pair_seed=pair_seed,
                         reference_energy=reference_energy, memory_tau=memory_tau, coupling=coupling, broken=broken, NS=NS,
                         guide_A=guide_A.guide_id, guide_B=None if guide_B is None else guide_B.guide_id, wall_sec=wall, init_sec=t_init,
                         n_support_checked=eng.n_support_checked, blocks_done=nb, estimator='mixed: block-end sum w E_L / sum w with block-local weights '
                         '(coupled/single) or carried masses (union/reweight_A); Delta = E_B - E_A', **(meta or {})))
    if output:
        np.savez(f'{output}/blocks.npz', **{k: v for k, v in ser.items()})
        np.savez(f'{output}/ckpt.npz', **eng.state())
        json.dump(dict(meta=out['meta'], anc={k: v for k, v in anc.items()}), open(f'{output}/run_meta.json', 'w'), indent=1, default=float)
    return out
