"""pool8_test_drawfn1.py: exactness of Pool8Sampler.draw_fn for a SINGLE-RUNG ladder (beta 0.25 only, no swaps; stage 1 of the pool expansion, PLAN_pfaffian.md "Pool expansion")
at 6x6, same bins / Gaussian rule / bar as pool8_test_drawfn.py (it reuses its test()): max|z| <= 3 and chi2/dof <= 1.5.  The decade reference is given, med2 = the exact 6x6 beta-1
median of 2 ln|B| (pool8_test6.exact_tables), because a 1-rung ladder has no beta = 1 rung to measure it from.  --broken beta sets the PT's 'beta' fault (|B|^beta instead of |B|^(2 beta)).
  python pool8_test_drawfn1.py run  OUT.npz --eval table6 --n 4 --seed0 100 --E 8 --burn 200 --pilot 400 --S 2000 [--broken none|beta|hastings]
  python pool8_test_drawfn1.py test OUT.json F1.npz ...
"""
import sys, os, json, math, time, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import pool8 as P8
import pool8_test6 as T6
import pool8_test_drawfn as DF
BETA = 0.25


def run(a):
    ev = P8.load_eval(6, a.eval); tab = P8.TableEval6() if a.eval == 'pp2' else ev
    med, _ = T6.exact_tables([BETA]); LB = np.zeros((a.n, 1, a.S)); thin = []; tau = []; chk = dict(n=0, max_dlnB=0.0, max_Sz=0, max_dorb=0.0, orb_in_reps=True); t0 = time.time()
    for k in range(a.n):
        smp = P8.Pool8Sampler(ev, [BETA], E=a.E, seed=a.seed0 + k, burn=a.burn, pilot=a.pilot, log=lambda m: None, med2=med); smp.pt.broken = a.broken
        rng = np.random.default_rng(a.seed0 + k); w, l, o = smp.draw_fn(BETA, a.S, rng); LB[k, 0] = l
        if k % 4 == 0:
            c = P8.unpack_words(w[:300], 36); chk['max_Sz'] = max(chk['max_Sz'], int(np.abs(c.sum(1)).max()))
            chk['max_dlnB'] = max(chk['max_dlnB'], float(np.max(np.abs(tab.lnB(c.astype(np.float64)) - l[:300]))))
            ou = o[:300].view(np.uint64); ix = np.searchsorted(tab.reps, ou); ok = bool(np.all(tab.reps[np.minimum(ix, len(tab.reps) - 1)] == ou))
            chk['orb_in_reps'] &= ok; chk['max_dorb'] = max(chk['max_dorb'], float(np.max(np.abs(tab.la[np.minimum(ix, len(tab.reps) - 1)] - l[:300])))); chk['n'] += 300
        print(f'[drawfn1] PROGRESS sampler {k + 1}/{a.n} thin {smp.thin} {time.time() - t0:.0f}s', flush=True); thin.append(smp.thin); tau.append([max(t['tau_lnB2'], t['tau_dec']) for t in smp.tau]); assert smp.med2 == med
    np.savez(a.out, lnB=LB, thin=np.array(thin), tau=np.array(tau), betas=np.array([BETA]), seed0=a.seed0, evalname=a.eval, keep_pilot=0, broken=a.broken, E=a.E, burn=a.burn, pilot=a.pilot, S=a.S,
             med2=med, chk=json.dumps(chk), sec=time.time() - t0)
    print(f'[drawfn1] done {time.time() - t0:.0f}s thin {np.bincount(thin).tolist()} broken {a.broken} med2 {med:.6f} checks {chk}', flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'run':
        ap = argparse.ArgumentParser(); ap.add_argument('cmd'); ap.add_argument('out'); ap.add_argument('--eval', default='table6'); ap.add_argument('--n', type=int, default=4)
        ap.add_argument('--seed0', type=int, default=100); ap.add_argument('--E', type=int, default=8); ap.add_argument('--burn', type=int, default=200); ap.add_argument('--pilot', type=int, default=400)
        ap.add_argument('--S', type=int, default=2000); ap.add_argument('--broken', default='none', choices=['none', 'beta', 'hastings']); run(ap.parse_args())
    else:
        DF.test(sys.argv[2], sys.argv[3:])
        for f in sys.argv[3:]: print('  ', os.path.basename(f), 'broken', str(np.load(f)['broken']), 'thin', np.load(f)['thin'].tolist())
