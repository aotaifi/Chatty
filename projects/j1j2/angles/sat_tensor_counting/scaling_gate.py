#!/usr/bin/env python3
"""Compare *asymptotic exponent proxies*, not measured runtimes.

Conditional theorem (OpenAI Oct 2026 Theorem D.1): 2^O(s/sqrt(log s)).
Ordinary exact axis sweep of a local L x L x M space-time network:
    2^O(min(L*M, L*L)).
Unknown O(.) constants/pre-asymptotic overhead are suppressed.
"""
from math import log, sqrt

def audit(L: int, M: int):
    assert L >= 2 and M >= 1
    s = L * L * M
    oai = s / sqrt(log(s))
    sweep = min(L * M, L * L)
    return (s, oai, sweep, oai / sweep)

if __name__ == "__main__":
    print(f"{'L':>5} {'M':>6} {'wires proxy':>12} {'OAI proxy':>14} {'sweep proxy':>13} {'exponent ratio':>15}")
    for L, M in [(8, 1), (8, 8), (16, 1), (16, 16), (32, 32), (64, 64), (128, 128)]:
        s, oai, sweep, ratio = audit(L, M)
        print(f"{L:5d} {M:6d} {s:12d} {oai:14.1f} {sweep:13d} {ratio:15.1f}")
    # These are checks of the arithmetic only, not empirical QMC validation.
    for L in (16, 64, 256, 1024):
        for M in (1, L//2, L, 2*L):
            assert audit(L, M)[-1] > 1
    print("Arithmetic checks passed (not an algorithm performance benchmark).")
