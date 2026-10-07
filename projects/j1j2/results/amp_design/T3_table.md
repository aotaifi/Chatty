| stage | <H>_G - E_ViT (per site, paired) | <H>_G |
|---|---|---|
| start (|ViT|, s1) | +4.59e-06 +- 7.9e-06 | -0.503650(23) |
| after it1 (a1, s1) | -8.76e-06 +- 3.4e-05 | -0.503663(40) |
| sign step 2 (a1, s2) | -2.06e-06 +- 5.5e-06 | -0.503656(22) |
| after it2 (a2 = a1, s2) | -8.17e-06 +- 4.3e-06 | -0.503662(22) |

| iteration | step | N | best candidate | verified dE/site | SE | accepted |
|---|---|---|---|---|---|---|
| 1 | 1 | 8192 | gd | -5.67e-06 | 7.8e-06 | False |
| 1 | 2 | 16384 | gd | -8.25e-06 | 3.9e-06 | True |
| 1 | 3 | 16384 | gd | -1.24e-06 | 2.4e-06 | False |
| 2 | 1 | 8192 | gd | -5.65e-07 | 1.3e-06 | False |
| 2 | 2 | 16384 | gd | -1.27e-06 | 2.3e-06 | False |
| 2 | 3 | 16384 | sr_eps1 | -7.20e-07 | 2.2e-06 | False |

| sign | ED w_s |
|---|---|
| ViT | 1.3e-04 +- 2.0e-05 |
| K1 net + hop(|ViT|) (start) | 6.0e-05 +- 1.2e-05 |
| composite net N1 | 1.1e-04 +- 2.3e-05 |
| N1 + hop(a1) (it 2) | 2.0e-05 +- 1.0e-05 |
