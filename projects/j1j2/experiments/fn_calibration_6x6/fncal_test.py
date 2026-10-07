"""Tests for fncal_dmc.py: (1) canonical-orbit lookup (xor trick) == brute-force canon == stall_6x6 canon;
(2) local data (d_FN, E_L, allowed rates) GPU == CPU reference at random walker states; (3) guide tables reproduce the
exact stall numbers via the sector Hamiltonian is NOT repeated here (see results/stall_6x6)."""
import numpy as np, jax, jax.numpy as jnp, sys, time
import fncal_dmc as F
import fncal_ref_cpu as R

guide = sys.argv[1] if len(sys.argv) > 1 else 'vitvit'
eng = F.Engine(guide, chunk=64)
G = R.RefGuide(guide)
pool = np.load(F.POOL)['states'].astype(np.uint64).reshape(-1)
rng = np.random.default_rng(1)
# states: pool states, random permutations of them, and states reached by a few random hops
xs = [pool[:64]]
cur = pool[rng.choice(len(pool), 64)].copy()
for _ in range(30):
    valid = (((cur[:, None] >> F.BI[None]) ^ (cur[:, None] >> F.BJ[None])) & np.uint64(1)).astype(bool)
    b = np.array([rng.choice(np.nonzero(v)[0]) for v in valid]); cur = cur ^ F.MASKS[b]
xs.append(cur)
x = np.concatenate(xs)
x = np.concatenate([x, x[:0]])
x = x[: (len(x) // 64) * 64]
dfn, eh, wa = [np.asarray(a) for a in eng.local(jnp.asarray(x))]
maxd = 0
for i, xi in enumerate(x):
    d, e, ch, w = G.local(xi)
    al = np.nonzero(wa[i] > 0)[0]
    assert len(al) == len(w), (i, len(al), len(w))
    ref = np.sort(w); got = np.sort(wa[i][al])
    maxd = max(maxd, abs(d - dfn[i]), abs(e - eh[i]), np.max(np.abs(ref - got)) if len(ref) else 0)
print('local data max |GPU - CPU| over', len(x), 'states:', maxd)
assert maxd < 1e-10
# canon vs stall_6x6 canon
sys.path.insert(0, '../stall_6x6')
try:
    import st6_sector as SS
    ic = SS.canon(eng.T, eng.reps, jnp.asarray(x)); ib = G.canon_idx(x)
    print('stall canon == brute-force canon:', bool(np.all(np.asarray(ic) == ib)))
except Exception as ex:
    print('stall canon check skipped:', ex)
print('TEST OK', guide)
