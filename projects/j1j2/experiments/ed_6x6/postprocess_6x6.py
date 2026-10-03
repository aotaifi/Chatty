"""Post-processing of the 6x6 A1,+ ground state:
  * build the full-basis amplitude table (canonical rep -> psi0) for psi0_6x6.Psi0
  * exact checks: sum of orbit sizes = C(36,18), full-basis norm = 1
  * Monte-Carlo checks: uniform-sample normalisation, local energy == E0
  * Marshall wrong-sign weight w_s (exact, summed over all orbits)
  * NS exact samples x ~ |psi0|^2

Usage: python postprocess_6x6.py RUNDIR [NS]
"""
import json, math, os, sys, time
import numpy as np
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ed_common as ec
from psi0_6x6 import Psi0

L, n = 6, 36
rundir = sys.argv[1]
NS = int(sys.argv[2]) if len(sys.argv) > 2 else 100000
NTHR = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
tag = "k00_A1_p"
summ = json.load(open(os.path.join(rundir, "sectors.json")))[tag]
E0 = summ["energies"][0]
states = np.load(os.path.join(rundir, f"states_{tag}.npy"))
v = np.load(os.path.join(rundir, f"vectors_{tag}.npy"), mmap_mode="r")[:, 0].copy()
v /= np.linalg.norm(v)
out = {"E0": E0, "E0_per_site": E0 / n, "dim_sector": int(states.size)}

# ---- canonical reps and orbit sizes (threaded numpy) ------------------------
t0 = time.time()
T = ec.byte_tables(ec.space_group(L), n)
CH = 20000
def work(s):
    return ec.rep_and_orbit(states[s:s + CH], T, n, chunk=CH)
reps = np.empty_like(states); orb = np.empty(states.size, dtype=np.int64)
with ThreadPoolExecutor(NTHR) as ex:
    for s, (r, o) in zip(range(0, states.size, CH), ex.map(work, range(0, states.size, CH))):
        reps[s:s + CH] = r; orb[s:s + CH] = o
out["t_orbits"] = time.time() - t0
out["ls_rep_equals_min_rep"] = bool(np.all(reps == states))
out["sum_orbit_sizes"] = int(orb.sum())
out["C36_18"] = math.comb(36, 18)
assert out["sum_orbit_sizes"] == out["C36_18"], out
assert np.unique(reps).size == reps.size
print("orbits ok", out["t_orbits"], out["ls_rep_equals_min_rep"], flush=True)

# ---- Marshall sign and global sign convention -------------------------------
maskA = np.uint64(ec.sublattice_mask(L))
a = states & maskA
cnt = np.zeros(states.size, dtype=np.int64)
for i in range(n):
    cnt += ((a >> np.uint64(i)) & np.uint64(1)).astype(np.int64)
M = np.where(cnt % 2 == 0, 1.0, -1.0)
if np.sum(v * M) < 0:
    v = -v
w = v * v                      # weight of each orbit in |psi0|^2 (sum = 1)
wrong = np.sign(v) != M
out["marshall_wrong_sign_weight"] = float(w[wrong].sum())
out["marshall_wrong_sign_orbit_fraction"] = float(wrong.mean())
out["marshall_wrong_sign_config_fraction"] = float(orb[wrong].sum() / orb.sum())
out["n_zero_amplitude_orbits"] = int(np.sum(v == 0))
out["min_abs_v"] = float(np.abs(v).min())
print("Marshall w_s =", out["marshall_wrong_sign_weight"], flush=True)

# ---- amplitude table ------------------------------------------------------
amp = v / np.sqrt(orb)
o = np.argsort(reps)
tab = os.path.join(rundir, "psi0_6x6_table.npz")
np.savez(tab, L=L, J2=0.5, E0=E0, reps=reps[o], amp=amp[o], orbit=orb[o].astype(np.int16))
out["full_norm_exact"] = float(np.sum(amp ** 2 * orb))
del a, cnt
P = Psi0(tab)

# ---- uniform-sample normalisation and local energy -------------------------
rng = np.random.default_rng(20261003)
def rand_sz0(m):
    keys = rng.random((m, n))
    up = np.argsort(keys, axis=1)[:, : n // 2]
    bits = np.zeros(m, dtype=np.uint64)
    for j in range(n // 2):
        bits |= np.uint64(1) << up[:, j].astype(np.uint64)
    return bits
xu = rand_sz0(200000)
pu = P(xu)
est = pu ** 2 * math.comb(36, 18)
out["uniform_norm_estimate"] = [float(est.mean()), float(est.std() / np.sqrt(est.size))]
print("uniform norm", out["uniform_norm_estimate"], flush=True)

# ---- exact samples from |psi0|^2 -------------------------------------------
G = T.shape[0]
ridx = rng.choice(states.size, size=NS, p=w / w.sum())
gidx = rng.integers(0, 2 * G, size=NS)
r0 = states[ridx]
img = np.zeros(NS, dtype=np.uint64)
for b in range(T.shape[1]):
    byte = ((r0 >> np.uint64(8 * b)) & np.uint64(255)).astype(np.intp)
    img |= T[gidx % G, b, byte]
full = np.uint64((1 << n) - 1)
xs = np.where(gidx >= G, img ^ full, img)
ps = P(xs)
assert np.allclose(ps, v[ridx] / np.sqrt(orb[ridx]), rtol=0, atol=1e-15)
np.savez(os.path.join(rundir, "samples_psi0sq_6x6.npz"), x=xs, psi=ps, marshall=P.marshall(xs),
         seed=20261003, note="x ~ |psi0|^2 exactly (orbit by weight, uniform group element)")
out["samples"] = {"n": NS, "marshall_wrong_frac": float(np.mean(np.sign(ps) != P.marshall(xs)))}

def local_energy(x):
    diag, nbr, coeff = ec.local_connections(x, L)
    px = P(x)
    pn = P(nbr.ravel()).reshape(nbr.shape)
    return (diag * px + (coeff * pn).sum(1)) / px, px
le_s, _ = local_energy(xs[:3000])
le_u, pxu = local_energy(xu[:3000])
out["local_energy_sampled"] = {"max_abs_dev": float(np.max(np.abs(le_s - E0))), "n": 3000}
out["local_energy_uniform"] = {"max_abs_dev": float(np.max(np.abs(le_u - E0))),
                               "median_abs_dev": float(np.median(np.abs(le_u - E0))),
                               "min_abs_psi": float(np.abs(pxu).min()), "n": 3000}
print(json.dumps(out, indent=1), flush=True)
json.dump(out, open(os.path.join(rundir, "postprocess.json"), "w"), indent=1)
