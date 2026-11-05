"""pool_x_task.py: one array task of the 8x8 round-1 x-draw pool (PLAN_pfaffian.md "8x8 R1 pool, x-draws").
One process = one core = one Pool8Sampler (E independent PT ensembles of the ladder 1, .5, .25), called exactly as pool_fit.build_pool calls it:
  for beta in (1, .5, .25): words, lnB, orb = sampler.draw_fn(beta, S_beta, rng)       (pilot + auto thin ON, informed move OFF, mode 'single')
draw_fn is called in chunks of CH rows (CH a multiple of E, so a chunk is whole time blocks; the sampler advances only on demand in whole thin blocks, so the
chain trajectory does not depend on the chunking) to leave PROGRESS lines for the watcher.
Row r of a rung's output belongs to ensemble (chain) r % E at time block r // E; global chain id = task * E + r % E.
Output npz: words uint64, lnB float64 (= ln|B|), orb int64, beta float64, chain int32 (global), tblock int32, seed, + the rest of the buffer ('x_*': draws the sampler had
already produced for the other rungs and that the targets do not use; NOT part of the registered pool).
usage: python pool_x_task.py --out F.npz --task T --seed S --E 4 --S 8252,11252,8000 [--burn 200 --pilot 400 --ch 200]"""
import sys, os, json, time, math, argparse, hashlib
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import pool8 as P8

ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--task', type=int, required=True); ap.add_argument('--seed', type=int, required=True)
ap.add_argument('--E', type=int, default=4); ap.add_argument('--S', default='8252,11252,8000'); ap.add_argument('--burn', type=int, default=200); ap.add_argument('--pilot', type=int, default=400)
ap.add_argument('--ch', type=int, default=200); a = ap.parse_args()
BETAS = [1.0, 0.5, 0.25]; S = [int(s) for s in a.S.split(',')]; E = a.E
assert all(s % E == 0 for s in S) and a.ch % E == 0
logf = open(a.out + '.log', 'a')
def log(m): logf.write(m + '\n'); logf.flush(); print(m, flush=True)
def sha(p): return hashlib.sha256(open(p, 'rb').read()).hexdigest()
psha = sha(P8.PROD8); assert psha.startswith('0520c5be'), psha                      # guide check before the first draw
t0 = time.time()
log(f'[pool_x] start {time.ctime()} host {os.uname().nodename} task {a.task} seed {a.seed} E {E} S {S} burn {a.burn} pilot {a.pilot} params_sha {psha}')
ev = P8.load_eval(8, 'pp2')
smp = P8.Pool8Sampler(ev, BETAS, E=E, seed=a.seed, burn=a.burn, pilot=a.pilot, log=log)          # mode 'single', thin auto, start 'random'
rng = np.random.default_rng(a.seed + 1); out = {}; tdraw0 = None
for q, b in enumerate(BETAS):
    W, L, O = [], [], []; n = 0
    while n < S[q]:
        m = min(a.ch, S[q] - n); ts = time.time()
        w, l, o = smp.draw_fn(b, m, rng); W.append(w); L.append(l); O.append(o); n += m
        if tdraw0 is None: tdraw0 = (time.time(), smp.sweeps_done)
        log(f'[pool_x] PROGRESS beta {b} {n}/{S[q]} sweeps_done {smp.sweeps_done} elapsed {time.time() - t0:.0f}s')
    out[b] = (np.concatenate(W), np.concatenate(L), np.concatenate(O))
t_end = time.time(); sw_draw = smp.sweeps_done - tdraw0[1]
R = len(BETAS)
def popc(w):
    w = np.asarray(w, np.uint64); c = np.zeros(len(w), np.int64)
    for s in range(64): c += ((w >> np.uint64(s)) & np.uint64(1)).astype(np.int64)
    return c
words = np.concatenate([out[b][0] for b in BETAS]); lnB = np.concatenate([out[b][1] for b in BETAS]); orb = np.concatenate([out[b][2] for b in BETAS])
beta = np.concatenate([np.full(len(out[b][0]), b) for b in BETAS]); row = np.concatenate([np.arange(len(out[b][0])) for b in BETAS])
chain = (a.task * E + row % E).astype(np.int32); tblock = (row // E).astype(np.int32)
assert np.all(popc(words) == 32), 'Sz != 0 word'
# surplus left in the sampler's buffers (draws of the other rungs already produced)
xw, xl, xb, xc, xt = [], [], [], [], []
for b in BETAS:
    if smp.buf[b]:
        c = np.concatenate([c for c, _ in smp.buf[b]]); l = np.concatenate([l for _, l in smp.buf[b]]); r0 = len(out[b][0]); r = r0 + np.arange(len(c))
        xw.append(P8.pack_words(c)); xl.append(l); xb.append(np.full(len(c), b)); xc.append((a.task * E + r % E).astype(np.int32)); xt.append((r // E).astype(np.int32))
cat = lambda x, dt: np.concatenate(x) if x else np.zeros(0, dt)
tau = json.dumps(smp.tau, default=float)
wsw = float((t_end - tdraw0[0]) / max(sw_draw, 1) / (R * E))
np.savez(a.out, words=words, lnB=lnB, orb=orb, beta=beta, chain=chain, tblock=tblock, seed=a.seed, task=a.task, E=E, S=np.array(S), betas=np.array(BETAS), thin=smp.thin, burn=a.burn, pilot=a.pilot,
         tau=tau, med2_pilot=smp.med2, sweeps_done=smp.sweeps_done, sec_total=t_end - t0, s_per_sweep_per_walker_draw=wsw, params_sha=psha, host=os.uname().nodename,
         x_words=cat(xw, np.uint64), x_lnB=cat(xl, np.float64), x_beta=cat(xb, np.float64), x_chain=cat(xc, np.int32), x_tblock=cat(xt, np.int32))
log(f'[pool_x] done task {a.task} thin {smp.thin} sweeps_done {smp.sweeps_done} {t_end - t0:.0f}s total, {wsw:.4f} s per sweep per walker in the draw phase; draws {[len(out[b][0]) for b in BETAS]}; surplus {sum(len(x) for x in xl)}')
