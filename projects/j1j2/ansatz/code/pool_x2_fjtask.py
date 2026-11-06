"""pool_x2_fjtask.py: stage-1 step (2) for one task's draws: f_J(x) (sign sgn(M1) from the lemma ratios), K = 2 blocked partners, f_J at the partners, for the draws of decades -7..-9
(decade = floor((2 lnB - med)/ln10), med = decades8's beta-1 median).  Resumable: checkpoint every --save chunks (npz of the finished rows; a restart skips them).
Output rows (in draw order within the task): idx (row in the draws file), dec, lnB, fx (raw, -inf if no kept bond), sx, WB, nblk, picks (K bond ids, -1 none), fy (K, raw f_J at the partner), dlnB (K, ln|B_y/B_x|).
usage: python pool_x2_fjtask.py --draws out_T.npz --out fj_T.npz --seed S [--med -67.03209 --lo -9 --hi -7 --K 2 --chunk 128 --save 20 --limit N]"""
import sys, os, json, time, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import pool_x2_fj as F, pool8 as P8

ap = argparse.ArgumentParser(); ap.add_argument('--draws', required=True); ap.add_argument('--out', required=True); ap.add_argument('--seed', type=int, required=True)
ap.add_argument('--med', type=float, default=-67.03209121446969); ap.add_argument('--lo', type=int, default=-9); ap.add_argument('--hi', type=int, default=-7); ap.add_argument('--K', type=int, default=2)
ap.add_argument('--chunk', type=int, default=128); ap.add_argument('--save', type=int, default=20); ap.add_argument('--limit', type=int, default=0); a = ap.parse_args()
logf = open(a.out + '.log', 'a')
def log(m): logf.write(m + '\n'); logf.flush(); print(m, flush=True)
t0 = time.time(); z = np.load(a.draws); words = z['words']; lnB = z['lnB']; assert int(z['final']) == 1, 'draws file is not final'
dec = np.floor((2.0 * lnB - a.med) / F.LN10).astype(np.int64); sel = np.nonzero((dec >= a.lo) & (dec <= a.hi))[0]
if a.limit: sel = sel[:a.limit]
log(f'[pool_x2_fj] start {time.ctime()} host {os.uname().nodename} draws {a.draws} n {len(words)} selected {len(sel)} (decades {a.lo}..{a.hi}, med {a.med}) seed {a.seed} K {a.K} chunk {a.chunk}')
K = a.K; fj = F.FJ8(); keys = ('fx', 'sx', 'WB', 'nblk', 'picks', 'fy', 'dlnB'); R = {k: [] for k in keys}; ndone = 0
if os.path.exists(a.out):
    c = np.load(a.out)
    if int(c['seed']) == a.seed and np.array_equal(c['idx'], sel[:len(c['idx'])]):
        R = {k: [c[k]] for k in keys}; ndone = len(c['idx']); log(f'[pool_x2_fj] resume from {ndone} rows')
def save(final):
    cat = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in R.items()}; n = len(cat['fx']); tmp = a.out + '.tmp.npz'
    np.savez(tmp, idx=sel[:n], dec=dec[sel[:n]], lnB=lnB[sel[:n]], seed=a.seed, med=a.med, K=K, final=final, n_sel=len(sel), guard_checked=fj.m1.n_checked, guard_fallbacks=fj.m1.n_fallback, sec=time.time() - t0, **cat)
    os.replace(tmp, a.out)
rng = np.random.default_rng(a.seed + ndone); nch = 0
for s in range(ndone, len(sel), a.chunk):
    ix = sel[s:s + a.chunk]; X = P8.unpack_words(words[ix], 64).astype(np.float64); M = len(ix)
    ev = fj.evaluate(X); picks, Y = fj.partners(X, ev, rng, K); evy = fj.evaluate(Y.reshape(-1, 64)); fy = evy['fx'].reshape(M, K)
    dl = (np.real(evy['lB']).reshape(M, K) - np.real(ev['lB'])[:, None]); fy = np.where(picks >= 0, fy, np.nan); dl = np.where(picks >= 0, dl, np.nan)
    for k, v in (('fx', ev['fx']), ('sx', ev['sx']), ('WB', ev['WB']), ('nblk', ev['nblk']), ('picks', picks), ('fy', fy), ('dlnB', dl)): R[k].append(np.asarray(v))
    nch += 1; n = s + M
    if nch % a.save == 0: save(0)
    if nch % 4 == 0 or n >= len(sel): log(f'[pool_x2_fj] PROGRESS {n}/{len(sel)} elapsed {time.time() - t0:.0f}s guard {fj.m1.n_fallback}/{fj.m1.n_checked}')
save(1)
log(f'[pool_x2_fj] done {len(sel)} rows {time.time() - t0:.0f}s guard fallbacks {fj.m1.n_fallback}/{fj.m1.n_checked}')
