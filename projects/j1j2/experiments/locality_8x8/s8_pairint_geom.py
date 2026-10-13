"""8x8 bond geometry shared by s8_pairint.py / s8_elcone.py."""
import numpy as np
L = 8; N = 64; ONE = np.uint64(1)
q = lambda x, y: (x % L) + L * (y % L)
NN, NNN = [], []
for y in range(L):
    for x in range(L):
        i = q(x, y); NN += [(i, q(x + 1, y)), (i, q(x, y + 1))]; NNN += [(i, q(x + 1, y + 1)), (i, q(x + 1, y - 1))]
B = NN + NNN; NB = len(B)
BI = np.array([b[0] for b in B], np.uint64); BJ = np.array([b[1] for b in B], np.uint64)
MASKS = (ONE << BI) | (ONE << BJ)
JB = np.array([1.0] * len(NN) + [0.5] * len(NNN))


def sdist(p, r):
    dx = abs(p % L - r % L); dy = abs(p // L - r // L); dx = min(dx, L - dx); dy = min(dy, L - dy)
    return np.sqrt(dx * dx + dy * dy)
DB = np.array([[min(sdist(p, r) for p in B[a] for r in B[c]) for c in range(NB)] for a in range(NB)])
