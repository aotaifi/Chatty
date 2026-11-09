"""Readout of the production retrain. python3 retrain_gate.py RUNDIR [RUNDIR2]  (RUNDIR/train/train.json, RUNDIR/VERDICT_RUN).
Single training trace per seed: block errors from the scatter of 10-step sub-block means. Training-trace energies are the biased
mean over SR iterates; the quotable energy is the later independent evaluation (quotable.py), not this.
Also: python3 retrain_gate.py --kill TRAIN_JSON  -> prints KILL / PASS / WAIT for the registered kill (200-step blocks up to step 1000; amended from 600 by the lead 2026-11-09)."""
import json, sys, re, os, numpy as np
KILL_E, KILL_STEP = -0.4988225, 1000
def sub(x):
    m = len(x) // 10 * 10; s = x[:m].reshape(-1, 10).mean(1); return x.mean(), s.std(ddof=1) / np.sqrt(len(s))
def kill(h):
    E = np.array([x['E'] for x in h])
    if len(E) < KILL_STEP: return 'WAIT', []
    b = [E[a:a + 200].mean() for a in range(0, KILL_STEP, 200)]
    return ('PASS' if min(b) < KILL_E else 'KILL'), b
if sys.argv[1] == '--kill':
    v, b = kill(json.load(open(sys.argv[2]))['hist']); print(v, ' '.join(f'{x:.7f}' for x in b)); sys.exit(0)
out = []
for R in sys.argv[1:]:
    h = json.load(open(R + '/train/train.json'))['hist']; want = json.load(open(R + '/train/spec.json'))['steps']
    E = np.array([x['E'] for x in h]); n = len(E)
    vr = open(R + '/VERDICT_RUN').read() if os.path.exists(R + '/VERDICT_RUN') else ''
    rc = (re.findall(r'train_rc=(\d+)', vr) or ['missing'])[-1]
    print(f"== {R}: steps {n}/{want}, train_rc {rc}, t_step median {np.median([x['t_step'] for x in h[3:]]):.2f} s")
    for a in range(0, n, 200):
        m, e = sub(E[a:a + 200]); print(f"  steps {a+1}-{min(a+200,n)}: {m:.7f} +- {e:.1e}")
    kv, kb = kill(h); print(f"  kill check (200-step blocks to step {KILL_STEP} vs {KILL_E}): {kv}")
    t = E[-max(1, n // 5):]; p, c = np.polyfit(np.arange(len(t)), t, 1, cov=True); m, e = sub(t)
    plateau = abs(p[0]) < 2 * np.sqrt(c[0, 0]) and abs(p[0] * len(t)) < 1e-5
    print(f"  last 20% ({len(t)} steps): {m:.7f} +- {e:.1e}; slope {p[0]:.2e} +- {np.sqrt(c[0,0]):.1e}/step, drift {p[0]*len(t):.1e}; plateau {plateau}")
    ok = rc == '0' and n >= want and np.all(np.isfinite(E))
    out.append((ok, plateau, m, e))
if not all(o[0] for o in out): print("VERDICT: INCOMPLETE (a seed has train_rc != 0, missing steps or non-finite E)"); sys.exit(0)
if len(out) == 2:
    d = out[0][2] - out[1][2]; s = np.hypot(out[0][3], out[1][3])
    print(f"  seed difference (last 20%): {d:.1e} +- {s:.1e} ({abs(d)/s:.1f} sigma)")
    print("VERDICT:", "CONVERGED (both plateau, seeds agree within 2 sigma) -> run the 'simple' and 'full' evaluations" if all(o[1] for o in out) and abs(d) < 2 * s
          else "NOT CONVERGED (" + ('no plateau' if not all(o[1] for o in out) else 'seeds disagree > 2 sigma') + ")")
else:
    print("VERDICT:", "PLATEAU" if out[0][1] else "NO PLATEAU", "(one seed; the seed comparison needs both run dirs)")
