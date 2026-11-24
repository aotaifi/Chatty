# j1j2 cleanup manifest 2026-11-24 (lead)

Quota at scan: /scratch 967.5 of 972.8 G soft (1 T hard); /project 471.7 of 475 G soft, 465,836 of 475,000 files.
Scan: docs/cleanup_scan_2026-11-24.tsv (top level), docs/cleanup_scan_2026-11-24_children.tsv (children of ansatz_runs, runs/j1j2, Chatty/runs, scratch tmp).
j1j2 holds ~140 G of /scratch; most of /scratch is other projects (rows added to /project/theorie/a/A.Otaifi/CLEANUP_ACTIONS.md).

## PI decides (Low; not in the script)
- runs/j1j2/r1prime_prediction, 69.4 G on /project, 5 d old: the sign angle's (cited PLAN_sign). Over the /project soft quota on its own; simulation output belongs on /scratch. Settle: ask the sign angle which files a result rests on.
- Chatty/runs/a40slice, 15.1 G, 46 d: holds venv_jax_0.8.2 that cluster/A40_SLICES_HOWTO.md needs for A40 slices. Keep unless A40 slices are retired.
- Chatty/runs/_packed, 3.9 G: packed tars (ed6x6, fnguides, k1rep, ll6), probably the only copies. Keep.
- ansatz_runs/c2_t0b_6x6, 6.8 G: its tables/ are inputs cited in PLAN_ansatz.md l.183, 442. Keep.
- ansatz_runs/pfaffian_* (~14 G, 15-27 d): result paths cited in PLAN.md / PLAN_pfaffian.md. Settle: prune large tables keeping the cited json, per directory.
- Chatty/runs/compress6, stall6, archive, signs8, fncal6, rbmpp6: cited in READMEs/PLAN files. Keep.
- ansatz_runs/consol_6x6 (38.7 G) and all of ref2hop: younger than 14 days, live campaign. Not candidates.

## Medium (script)
| path | GB | rule | regeneration | citation check |
|---|---|---|---|---|
| /project/theorie/a/A.Otaifi/Chatty/runs/fn6x6 | 1.69 | no job, 30.6 d, 0 citations in project docs | unknown (no run log line found) -> treat as Low unless the PI knows it | real_cites 0, foreign 0 |

No High rows: nothing in the j1j2 room meets every High criterion.
