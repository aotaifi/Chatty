"""Unit test of st6_sector on a small synthetic symmetric sector (CPU ok), against dense numpy:
H matvec (upper coded format), lattice-FN ground state, Krylov threshold step (brute force over cuts).
Optional: canonical-rep lookup on the real 6x6 table vs the independent numpy Psi0 class.
Usage: python test_st6_small.py [PSI0_TABLE]
"""
import sys
import numpy as np
import st6_sector as SS
import jax.numpy as jnp

rng = np.random.default_rng(1)
D = 400
n = rng.integers(1, 9, D).astype(float)
# random sparse symmetric H with positive off-diagonals (Heisenberg-like sign problem), random diagonal
Hd = np.zeros((D, D))
for _ in range(6 * D):
    i, j = rng.integers(0, D, 2)
    if i != j:
        v = rng.random(); Hd[i, j] += v; Hd[j, i] += v
Hd[np.arange(D), np.arange(D)] = rng.normal(0, 2, D)
ui, uc, rows = [], [], []
for i in range(D):
    for j in range(i, D):
        if Hd[i, j] != 0:
            ui.append(j); uc.append(8 * Hd[i, j] * np.sqrt(n[j] / n[i])); rows.append(i)
ui = np.array(ui, np.int32); uc = np.array(uc); rows = np.array(rows)
inc = np.diff(np.concatenate([[0], rows])).astype(np.uint8)
ew, ev = np.linalg.eigh(Hd)
v0 = ev[:, 0]
amp = v0 / np.sqrt(n)
S = SS.Sector.__new__(SS.Sector)
S._setup(n, ui, uc, inc, np.arange(D, dtype=np.uint64), amp, ew[0], ch=257)
x = rng.normal(size=D)
e1 = np.max(np.abs(np.asarray(S.H(jnp.asarray(x))) - Hd @ x))
print('matvec max err', e1); assert e1 < 1e-10
# FN: guide a = |v0| * exp(noise), s = sgn v0 with some flips
a = np.abs(amp) * np.exp(0.3 * rng.normal(size=D)); s = np.sign(v0); s[rng.random(D) < 0.15] *= -1
if s[0] < 0: s = -s
la = np.log(a)
E, u, info = S.fn_solve(jnp.asarray(la), jnp.asarray(s), tol=1e-12)
an = a / np.sqrt(np.sum(n * a * a)); w = np.maximum(an, 1e-15) * np.sqrt(n)
Hf = np.zeros((D, D))
for i in range(D):
    for j in range(D):
        if i == j: continue
        if s[i] != s[j]: Hf[i, j] = -Hd[i, j]
    Hf[i, i] = sum(Hd[i, j] * w[j] / w[i] for j in range(D) if s[j] == s[i])
ef, vf = np.linalg.eigh(Hf)
print('FN', E, ef[0], 'diff', E - ef[0], info['lobpcg_it'], 'bound E_FN <= <H>:', E <= float((s * w) @ Hd @ (s * w)) / (w @ w))
assert abs(E - ef[0]) < 1e-9
assert np.max(np.abs(np.abs(vf[:, 0]) - np.asarray(u))) < 1e-6
# Krylov: brute force over all cuts
v = np.abs(np.asarray(u))
sn, Eb, kinfo = S.krylov(jnp.asarray(v), jnp.asarray(s))
psi = v * s; r = (Hd @ psi) / psi
best = (np.inf, None)
for T in np.concatenate([[-np.inf], np.sort(r)]):
    sp = s * np.where(r <= T, 1, -1)
    Ec = (sp * v) @ Hd @ (sp * v)
    if Ec < best[0] - 1e-14: best = (Ec, sp)
sp = best[1] * (1 if best[1][0] > 0 else -1)
print('Krylov', Eb, best[0], 'diff', Eb - best[0], 'sign mismatch', int(np.sum(np.asarray(sn) != sp)))
assert abs(Eb - best[0]) < 1e-10 and np.all(np.asarray(sn) == sp)
print('score', S.score(jnp.asarray(la), jnp.asarray(s)))
print('SYNTHETIC OK')

if len(sys.argv) > 1:
    sys.path.insert(0, '.')
    from psi0_6x6 import Psi0
    P = Psi0(sys.argv[1])
    z = np.load(sys.argv[1])
    reps = jnp.asarray(z['reps']); T = jnp.asarray(SS.byte_tables(SS.space_group()))
    idx = rng.integers(0, len(z['reps']), 5000)
    g = rng.integers(0, 288, 5000); fl = rng.random(5000) < 0.5
    x = SS.image(T, reps[jnp.asarray(idx)], jnp.asarray(g), jnp.asarray(fl))
    back = np.asarray(SS.canon(T, reps, x))
    print('canon roundtrip exact:', np.all(back == idx))
    amp_np = P(np.asarray(x))
    print('lookup vs Psi0 class max diff', np.max(np.abs(amp_np - z['amp'][back])))
    # neighbours of images: compare lookups
    xb = np.asarray(x); bb = rng.integers(0, SS.NBOND, 5000)
    ok = ((xb >> SS.BI[bb]) ^ (xb >> SS.BJ[bb])) & np.uint64(1) == 1
    y = (xb ^ SS.MASKS[bb])[ok]
    iy = np.asarray(SS.canon(T, reps, jnp.asarray(y)))
    print('neighbour lookup vs Psi0 max diff', np.max(np.abs(P(y) - z['amp'][iy])))
