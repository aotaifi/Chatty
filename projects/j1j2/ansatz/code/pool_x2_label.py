"""pool_x2_label.py: FAST equivalent of b2_prep.label_task (same output format, same partner rng stream, same numbers), for the combined-pool build (PLAN_pfaffian.md "Pool expansion" stage 2/3).
b2_prep.label_task evaluates the lemma at x twice (NeighboursM1 for the partners, Labels8 for f_J(x)); here the x evaluation of NeighboursM1 is reused for f_J(x) (pp2_lemma.fJ_fixed(..., _two = the same
one-swap evaluation)), so a task costs 3 instead of 4 lemma evaluations per entry.  Everything else (M1Sign, partner draws, Labels8 at the partners, pool format) is b2_prep's own code.
 usage: python pool_x2_label.py OUT.npz SPEC.json DRAWS.npz FILE_INDEX [--limit N] [--check]      (--check: also run b2_prep.label_task's path on the same rows and print the max deviations)"""
import os, sys, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import jax, jax.numpy as jnp
import pp2_lemma as PL, pool_fit as PF, b2_prep as B2


class FastNeighbours(B2.NeighboursM1):
    """b2_prep.NeighboursM1 that also stores add_x (mode 'fix' label term at x) and Re ln B(x) of every block it sees, in call order."""
    def __init__(self, guide, sign, labels, walker_chunk=256):
        super().__init__(guide, sign, walker_chunk); self.lab = labels; self.cw = []; self.clt = []

    def __call__(self, cfg, orb=None, sub=4096):
        w = np.asarray(cfg, np.uint64)
        if len(w) > sub:
            parts = [self(w[i:i + sub]) for i in range(0, len(w), sub)]
            return (np.concatenate([q[0] for q in parts]), self.hw, np.concatenate([q[2] for q in parts]), np.concatenate([q[3] for q in parts]),
                    np.concatenate([q[4] for q in parts]), np.zeros((len(w), len(self.BI)), np.int64))
        X = PF.unpack64(w); Xf = X.astype(np.float64)
        lB, r = PL.one_swap_ratios(self.g, Xf, return_logB=True, walker_chunk=self.wc); lB = np.asarray(lB); r = np.asarray(r)
        # f_J(x) from THIS evaluation (the guide is the sign network here: M1Sign.endpoints reads d['logB'], d['r1'])
        lat = self.g.lat; Xj = jnp.asarray(Xf); BIj = jnp.asarray(lat.BI); BJj = jnp.asarray(lat.BJ); am = Xj[:, BIj] != Xj[:, BJj]
        d = dict(logB=jnp.asarray(lB), a=jnp.where(am, jnp.broadcast_to(jnp.arange(lat.NB), am.shape), -1), am=am, r1=jnp.asarray(r),
                 upa=jnp.where(Xj[:, BIj] > 0, BIj, BJj), dna=jnp.where(Xj[:, BIj] > 0, BJj, BIj))
        L = self.lab; o = PL.fJ_fixed(self.g, Xf, L.sign_fn, L.EP, L.c, L.f0, L.mode, at_y=False, dtype=L.dtype, _two=d)
        self.cw.append(w); self.clt.append(np.asarray(np.real(lB), np.float64) + np.asarray(o['add_x'], np.float64))
        lBs, rs = (lB, r) if self.s.sp is self.g else self.s.ratios_s(Xf, None)
        sx, sy = self.s.from_ratios(X, lBs, rs)
        valid = X[:, self.BI] != X[:, self.BJ]; ar = np.abs(r)
        with np.errstate(divide='ignore'): dl = np.where(valid & (ar > 0), np.log(np.where(ar > 0, ar, 1.0)), -700.0)
        y = np.where(valid, w[:, None] ^ self.MASK[None, :], w[:, None])
        blocked = valid & (sx[:, None].astype(np.int64) * sy > 0)
        return y, self.hw, valid, blocked, dl, np.zeros(y.shape, np.int64)


class FastLabels:
    """labels_fn for partners_labels: x (the exact words the neighbours_fn just saw, in order) -> cached ln T(x); anything else -> Labels8 (the y partners)."""
    def __init__(self, lab, nb): self.lab, self.nb = lab, nb

    def __call__(self, cfg, orb=None):
        cfg = np.asarray(cfg)
        if self.nb.cw and cfg.ndim == 1 and len(cfg) == sum(len(c) for c in self.nb.cw):
            cw = np.concatenate(self.nb.cw)
            if np.array_equal(cw.astype(np.uint64), cfg.astype(np.uint64)): return np.concatenate(self.nb.clt)
        return self.lab(cfg, orb)


def label_task_fast(out, SP, draws, fidx, limit=0, select=None, log=print):
    T0 = time.time(); z = np.load(draws); words = z['words']; lnB = z['lnB']; beta = z['beta']; orb = z['orb']
    if 'final' in z.files: assert int(z['final']) == 1, f'{draws} not final'
    assert str(z['params_sha']) == B2.PROD8_SHA, 'draws are not of rb8_prod'
    med = float(SP['med2lnB']); decs = [int(d) for d in SP.get('decs_rel', B2.DECS_R1)]
    dec = np.floor((2.0 * lnB - med) / B2.LN10).astype(np.int64)
    sel = np.asarray(select, np.int64) if select is not None else np.nonzero(np.isin(dec, decs))[0]
    if limit: sel = sel[:limit]
    guide = B2.guide_of(SP); sgn = B2.M1Sign(sign_params=guide, guide=guide)
    lab = B2.labels_m1(guide, sgn, float(SP.get('c', B2.C_AUG)), float(SP.get('f0', B2.F0)), float(SP.get('EP', B2.EP8)))
    nb = FastNeighbours(guide, sgn, lab); fl = FastLabels(lab, nb); K = int(SP.get('K', 2)); seed = int(SP.get('seed', 1)) * 100000 + int(fidx)
    log(f'[b2 label_task] {draws} rows {len(words)} selected {len(sel)} decades {decs} med {med} K {K} seed {seed} (fast labeller)')
    pl = B2.partners_labels(words[sel], lnB[sel], nb, fl, K, np.random.default_rng(seed), log)
    nchk = min(64, len(sel)); lchk = np.real(np.asarray(PL.logB(guide, PF.unpack64(words[sel[:nchk]]).astype(np.float64)))) if nchk else np.zeros(0)
    dev = float(np.max(np.abs(lchk - lnB[sel[:nchk]]))) if nchk else 0.0
    np.savez(out, idx=sel, x=words[sel], lnB_x=lnB[sel], orb=orb[sel], beta=beta[sel], src=np.asarray(draws), fidx=np.int64(fidx), seed=np.int64(seed), **pl,
             lnB_guide_vs_stored_maxdev=np.float64(dev), guard_checked=np.int64(sgn.n_checked), guard_fallbacks=np.int64(sgn.n_fallback), min_margin=np.float64(sgn.min_margin),
             sec=np.float64(time.time() - T0), code_sha256=np.asarray(B2.sha(os.path.abspath(__file__))), final=np.int64(1))
    log(f'[b2 label_task] done {len(sel)} rows {time.time() - T0:.0f}s; lnB guide vs stored {dev:.2e}; guard {sgn.n_fallback}/{sgn.n_checked}; min margin {sgn.min_margin:.3e}')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('spec'); ap.add_argument('draws'); ap.add_argument('fidx', type=int); ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--check', default=''); a = ap.parse_args()
    label_task_fast(a.out, json.load(open(a.spec)), a.draws, a.fidx, a.limit)
