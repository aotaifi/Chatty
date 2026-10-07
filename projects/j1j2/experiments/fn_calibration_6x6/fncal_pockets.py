"""Nodal-pocket structure of the lattice FN Hamiltonian of each calibration guide, in the exact symmetric sector.
Allowed edges = off-diagonal H elements (code > 0) between orbits with s_r s_r' = -1 (H_FN keeps them; all others go to the
diagonal).  Connected components (label propagation on the GPU) = disjoint 'pockets'; DMC cannot leave a pocket, and its
population only equilibrates between pockets at the (tiny) rate of the pocket energy differences.
Output pockets.json: per guide the largest components with their size, psi0^2 mass, exact-FN-vector mass and pool membership."""
import json, os, time
import numpy as np
import jax, jax.numpy as jnp
from functools import partial
import st6_sector as SS
import fncal_dmc as F

CSR = os.environ.get('ST6_CSR', '/project/theorie/a/A.Otaifi/chatty_stall6/csr')
sec = SS.Sector(CSR, os.path.join(F.DATA, 'psi0_6x6_table.npz'))
D = sec.D
BIG = np.int32(2 ** 31 - 1)


@partial(jax.jit, donate_argnums=(0,))
def prop(lab_new, lab, rbase, inc, idx, cod, s):
    row = rbase + jnp.cumsum(inc.astype(jnp.int32))
    ok = (cod > 0) & (row != idx) & (s[row] * s[idx] < 0)
    lr = lab[row]; lc = lab[idx]
    lab_new = lab_new.at[row].min(jnp.where(ok, lc, BIG))
    lab_new = lab_new.at[idx].min(jnp.where(ok, lr, BIG))
    return lab_new


pool = np.load(F.POOL)['states'].astype(np.uint64).reshape(-1)
pool_idx = np.asarray(SS.canon(sec.T, sec.reps, jnp.asarray(pool)))
out = {}
for g in ['vitvit', 'vitex', 'exvit']:
    reps, la, s = F.load_guide(g)
    s = jnp.asarray(s, jnp.float32)
    lab = jnp.arange(D, dtype=jnp.int32)
    t0 = time.time()
    for it in range(1, 2000):
        new = lab
        new = jnp.array(lab)
        for (rb, inc, idx, cod) in sec.chunks:
            new = prop(new, lab, rb, inc, idx, cod, s)
        ch = int(jnp.sum(new != lab)); lab = new
        if it % 10 == 0: SS.log(g, 'iter', it, 'changed', ch)
        if ch == 0: break
    # pointer jumping not needed: labels are min over component after convergence
    labn = np.asarray(lab)
    u, inv, cnt = np.unique(labn, return_inverse=True, return_counts=True)
    p0 = np.asarray(sec.p0); mass_a2 = np.exp(2 * la) * np.asarray(sec.n)
    mass_a2 = mass_a2 / mass_a2.sum()
    order = np.argsort(-cnt)[:8]
    poolc = np.bincount(inv[pool_idx], minlength=len(u)) / len(pool_idx)
    rows = []
    for k in order:
        m = inv == k
        rows.append(dict(size=int(cnt[k]), frac_orbits=float(cnt[k] / D), mass_psi0sq=float(p0[m].sum()), mass_guide_a2=float(mass_a2[m].sum()),
                         pool_frac=float(poolc[k])))
    out[g] = dict(n_components=int(len(u)), iters=it, sec=time.time() - t0, top=rows,
                  n_singletons=int((cnt == 1).sum()), mass_psi0sq_outside_top1=float(1 - rows[0]['mass_psi0sq']),
                  pool_frac_outside_top1=float(1 - rows[0]['pool_frac']))
    SS.log(g, json.dumps(out[g])[:1500])
    json.dump(out, open('pockets.json', 'w'), indent=1)
