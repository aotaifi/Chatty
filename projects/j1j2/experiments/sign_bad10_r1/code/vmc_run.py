"""T-BAD10 step 0b: VMC one-step Lanczos moments of the (Marshall x |ViT|) guide.  usage: python vmc_run.py OUT --W 16 --K 100 --maxsec S --seed 5
Stage A: E_ref from 96 shipped samples (one hop).  Then W Metropolis chains from distinct shipped spins (burn 300 steps); each round K steps then the exact two-hop (vt, h1, h2) per walker.
Per-sample records saved after every round (OUT/samples.npz: wid, rnd, vt, h1, h2).  Analysis: vmc_an.py."""
import sys, os, time, json, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--W', type=int, default=16); ap.add_argument('--K', type=int, default=100)
ap.add_argument('--maxsec', type=float, default=1000); ap.add_argument('--seed', type=int, default=5); ap.add_argument('--burn', type=int, default=300)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
import vitlib as VL, vmc10 as VM
P = lambda *s: print(time.strftime('%H:%M:%S'), 'PROGRESS', *s, flush=True)
ev = VL.make_vit('f32h'); S = VL.shipped(); rng = np.random.default_rng(a.seed); t0 = time.time()
ia = rng.choice(len(S), 96, replace=False); h1, vt, _, ne = VM.h1_one_hop(ev, S[ia], 256); Eref = float(np.mean(vt * h1)); P('stage A E_ref', Eref, 'sd', float(np.std(vt * h1)), 'evals', ne, 'sec', time.time() - t0)
json.dump(dict(E_ref=Eref, args=vars(a)), open(a.out + '/stage_A.json', 'w'))
rest = np.setdiff1d(np.arange(len(S)), ia); X = S[rng.choice(rest, a.W, replace=False)].copy()
acc = VM.metropolis(ev, X, rng, a.burn, 256); P('burn acc', acc)
R = {k: [] for k in ('wid', 'rnd', 'vt', 'h1', 'h2', 'n_unique')}; rnd = 0
while time.time() - t0 < a.maxsec:
    acc = VM.metropolis(ev, X, rng, a.K, 256)
    for w in range(a.W):
        d = VM.two_hop(ev, X[w], 256); R['wid'].append(w); R['rnd'].append(rnd)
        for k in ('vt', 'h1', 'h2', 'n_unique'): R[k].append(d[k])
        if time.time() - t0 > a.maxsec + 60: break
    rnd += 1; np.savez(a.out + '/samples.npz.tmp.npz', **{k: np.array(v) for k, v in R.items()}, E_ref=Eref); os.replace(a.out + '/samples.npz.tmp.npz', a.out + '/samples.npz')
    P('round', rnd, 'samples', len(R['vt']), 'acc', acc, 'sec', time.time() - t0)
P('VMC10_DONE samples', len(R['vt']))
