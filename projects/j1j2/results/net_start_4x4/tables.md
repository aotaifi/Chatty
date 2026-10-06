
### net H (no sign prior): loop runs

| run | iters | cpu_h | eps(it0) | eps best / final | E_FN eps final | w_s final | SR steps accepted |
|---|---|---|---|---|---|---|---|
| loopH_N1e4_s0 | 10 | 4.8 | 3.58e-04 | 2.89e-04 / 2.93e-04 | 2.34e-04 | 1.7e-06 | 25 |
| loopH_N1e4_s1 | 10 | 1.8 | 3.58e-04 | 2.64e-04 / 5.15e-04 | 2.45e-04 | 1.4e-06 | 22 |
| loopH_N1e5_s0 | 10 | 3.4 | 3.58e-04 | 2.54e-04 / 2.54e-04 | 2.17e-04 | 1.6e-06 | 20 |
| loopH_N1e5_sh1e-2_s0 | 10 | 12.1 | 3.58e-04 | 2.65e-04 / 2.82e-04 | 2.15e-04 | 1.3e-06 | 47 |
| loopH_N1e5_sh1e-3_s0 | 10 | 2.6 | 3.58e-04 | 2.60e-04 / 2.60e-04 | 2.22e-04 | 1.5e-06 | 9 |
| loopH_N1e6_s0 | 15 | 16.6 | 3.58e-04 | 1.89e-04 / 1.89e-04 | 1.68e-04 | 1.3e-06 | 72 |
| loopH_N1e6_sh1e-2_s0 | 10 | 13.1 | 3.58e-04 | 2.14e-04 / 2.14e-04 | 1.81e-04 | 1.3e-06 | 55 |
| loopH_N1e7_s0 | 15 | 34.6 | 3.58e-04 | 1.47e-04 / 1.47e-04 | 1.30e-04 | 1.3e-06 | 132 |
| exactloop_H | 40 | nan | 3.58e-04 | 4.05e-06 / 4.05e-06 | 3.98e-06 | 4.4e-07 | - |

### net G (Marshall sign rule): loop runs

| run | iters | cpu_h | eps(it0) | eps best / final | E_FN eps final | w_s final | SR steps accepted |
|---|---|---|---|---|---|---|---|
| loopG_N1e4_s0 | 10 | 1.6 | 2.03e-04 | 1.62e-04 / 2.15e-04 | 1.64e-04 | 7.9e-07 | 17 |
| loopG_N1e5_s0 | 10 | 3.5 | 2.03e-04 | 1.48e-04 / 1.48e-04 | 1.29e-04 | 8.3e-07 | 22 |
| exactloop_G | 40 | nan | 2.03e-04 | 4.89e-06 / 4.89e-06 | 4.68e-06 | 3.3e-07 | - |

### net J (short training): loop runs

| run | iters | cpu_h | eps(it0) | eps best / final | E_FN eps final | w_s final | SR steps accepted |
|---|---|---|---|---|---|---|---|
| loopJ_N1e5_s0 | 10 | 6.9 | 2.76e-03 | 2.15e-03 / 2.15e-03 | 1.84e-03 | 1.8e-04 | 61 |
| loopJ_N1e6_s0 | 10 | 29.6 | 2.76e-03 | 1.54e-03 / 1.54e-03 | 1.42e-03 | 1.7e-04 | 180 |
| exactloop_J | 40 | nan | 2.76e-03 | 1.24e-05 / 1.24e-05 | 1.21e-05 | 8.9e-07 | - |