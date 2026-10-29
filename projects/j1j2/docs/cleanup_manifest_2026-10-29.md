# j1j2 cleanup manifest 2026-10-29 (cleanup skill; scan docs/cleanup_scan_2026-10-29.tsv)

PI decides: nothing. **No j1j2 path qualifies for deletion under the rubric.**

| path | GB | confidence | rule | note |
|---|---|---|---|---|
| /scratch/a/A.Otaifi/ansatz_runs (all children) | 12.7 | not a candidate | newer than 14 days; R1 job live; c2_t0b_6x6 (6.9 GB, 2026-10-23) read by bond_split.py and cited in PLAN_ansatz.md | keep |
| /scratch/a/A.Otaifi/sign_prob | 1.5 | Low | June predecessor, cited in Chatty/CLAUDE.md | keep |
| /project/theorie/a/A.Otaifi/Chatty/runs | 51.2 | Low | 64 real citations, 370 foreign | /project, not the /scratch problem |
| tmp/b0dbg | 0.39 | Medium after 2026-11-05 | B0 closed (README), 14-day rule not yet met | revisit |

j1j2 owns ~25 GB of the 995 GB on /scratch; the quota problem sits in other projects' rooms (CLEANUP_ACTIONS.md rows).
No cleanup_run script is written: there is nothing to delete.
