# FN refresh -> projected sign-step scaling angle
**Opened:** 2026-09-30

## Why this is distinct from true K2
True K2 applies the next Krylov polynomial to the K1-distorted modulus and worsens the 6x6 fixed-amplitude energy.
The scalable candidate instead projects back onto an improved modulus:
[
s_0,a_0 \xrightarrow{K1} s_1
\xrightarrow{FN} a_1
\xrightarrow{\text{recompute }H[a_1s_1]/(a_1s_1)}
s_2.
]

## 6x6 status: already tested
Paderborn job 3466255 was exactly this closed-loop test.
The first FN handoff produced a refreshed amplitude, after which the label-free threshold moved
from T0=-14.9857997791 to T1=-14.6515439355.
Despite the threshold motion, the physical-weighted sign change s1 -> s2 was exactly 0.
The subsequent amplitude correction g2 was statistically consistent with replica noise.
Thus 6x6 reached a practical one-step node/amplitude fixed point in the tested M=128, beta=1.2 regime.
## 8x8 live test
First FN/K1 production job 3471544 completed successfully on 2026-09-30.
Its first four-bin amplitude correction is
g0^(8) = [0.14059278, 0.01262784, -0.08930914, -0.06367385].
Replica FN mean energies were -31.88701572 and -31.88383250.

A dedicated 8x8 closed-loop iteration-2 script was created:
- experiments/gfmc_8x8_closedloop_iter2.py
- experiments/8x8_closedloop_iter2_h100.sbatch

Paderborn job 3491791 was submitted to:
1. apply the measured 8x8 FN amplitude correction;
2. recompute the projected local-energy field;
3. infer the next threshold label-free;
4. measure physical-weighted K1 -> second-sign-step change;
5. run two FN replicas under the refreshed signs;
6. measure whether the next amplitude correction is signal or noise.

A durable Mac watcher is attached as Chatty job 20260930-065655-98000 and sends start/finish email.
