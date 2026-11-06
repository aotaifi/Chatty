"""pool_x2_task.py: one array task of the 8x8 pool expansion stage 1 (PLAN_pfaffian.md "Pool expansion"): SINGLE-RUNG beta = 0.25 x-draws on the production guide
(rb8_prod params_final, sha 0520c5be), Pool8Sampler(ev, [0.25], E, med2 = given decade reference) called through draw_fn exactly as pool_fit.build_pool does (pilot + auto thin on,
mode 'single', no swaps; a 1-rung ladder is the PT code with no swaps).  One process = one core = E independent chains.
Row r belongs to chain r % E at time block r // E; global chain id = task * E + r % E.   Draws are saved as a prefix every --save draws (atomic) so a time-out keeps the data.
usage: python pool_x2_task.py --out F.npz --task T --seed S --E 4 --S 50000 [--med2 -67.03 --burn 200 --pilot 400 --ch 400 --save 5000]"""
import sys, os, json, time, argparse, hashlib
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import pool8 as P8

ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--task', type=int, required=True); ap.add_argument('--seed', type=int, required=True)
ap.add_argument('--E', type=int, default=4); ap.add_argument('--S', type=int, default=50000); ap.add_argument('--med2', type=float, default=-67.03)
ap.add_argument('--burn', type=int, default=200); ap.add_argument('--pilot', type=int, default=400); ap.add_argument('--ch', type=int, default=400); ap.add_argument('--save', type=int, default=5000)
ap.add_argument('--half', default='?'); a = ap.parse_args()
BETA = 0.25; E = a.E; assert a.S % E == 0 and a.ch % E == 0 and a.save % a.ch == 0
logf = open(a.out + '.log', 'a')
def log(m): logf.write(m + '\n'); logf.flush(); print(m, flush=True)
psha = hashlib.sha256(open(P8.PROD8, 'rb').read()).hexdigest(); assert psha.startswith('0520c5be'), psha            # guide check before the first draw
t0 = time.time()
log(f'[pool_x2] start {time.ctime()} host {os.uname().nodename} task {a.task} half {a.half} seed {a.seed} E {E} S {a.S} med2 {a.med2} burn {a.burn} pilot {a.pilot} params_sha {psha}')
ev = P8.load_eval(8, 'pp2')
smp = P8.Pool8Sampler(ev, [BETA], E=E, seed=a.seed, burn=a.burn, pilot=a.pilot, log=log, med2=a.med2)
rng = np.random.default_rng(a.seed + 1); W, L, O = [], [], []; n = 0; tdraw0 = None
def save(final):
    words = np.concatenate(W); lnB = np.concatenate(L); orb = np.concatenate(O); row = np.arange(len(words))
    tmp = a.out + '.tmp.npz'
    np.savez(tmp, words=words, lnB=lnB, orb=orb, beta=np.full(len(words), BETA), chain=(a.task * E + row % E).astype(np.int32), tblock=(row // E).astype(np.int32), seed=a.seed, task=a.task, E=E,
             S=a.S, n_done=len(words), final=final, thin=smp.thin, burn=a.burn, pilot=a.pilot, med2_given=a.med2, tau=json.dumps(smp.tau, default=float), sweeps_done=smp.sweeps_done,
             sec_total=time.time() - t0, params_sha=psha, host=os.uname().nodename, half=a.half)
    os.replace(tmp, a.out)
while n < a.S:
    w, l, o = smp.draw_fn(BETA, a.ch, rng); W.append(w); L.append(l); O.append(o); n += a.ch
    if tdraw0 is None: tdraw0 = (time.time(), smp.sweeps_done)
    if n % 2000 == 0 or n == a.S: log(f'[pool_x2] PROGRESS beta {BETA} {n}/{a.S} sweeps_done {smp.sweeps_done} elapsed {time.time() - t0:.0f}s')
    if n % a.save == 0 and n < a.S: save(0)
save(1)
sw = smp.sweeps_done - tdraw0[1]
log(f'[pool_x2] done task {a.task} thin {smp.thin} sweeps_done {smp.sweeps_done} {time.time() - t0:.0f}s total, {(time.time() - tdraw0[0]) / max(sw, 1) / E:.4f} s per sweep per walker in the draw phase; draws {n}')
