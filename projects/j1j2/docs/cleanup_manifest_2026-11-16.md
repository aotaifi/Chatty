# Cleanup manifest 2026-11-16 (j1j2, /project room)

Trigger: the /project/theorie user quota hit its 500 GiB hard limit on 11-15 ~16:00 and crashed other campaigns' jobs (EDQUOT). It is now 460.6 GiB.
One of the two writers was ours: the GN tangent-share diagnosis wrote these shards at 15:42 (relayed by mcp-lead on the PI's behalf).

## PI decides (outside the skill's rubric: newer than 14 days, so this needs your explicit yes)
| path | GB | confidence | rule | regeneration | citation check |
|---|---|---|---|---|---|
| runs/j1j2/r1prime_fit/diag_ho/gn/c2/a_shard{0..3}of4_{Aex,Abd}.npy (8 files) | 10.9 | High apart from age | merged intermediate: c2/m_gn.json verify_ok true, diag_rel_dev 0.0 (merge job 18000483 COMPLETED); the merged products m_gn.json, m_deltas.npy and h_shard*.npz stay | ~3-11 core-h (acc array 17999276: 2 CPUs x ~20 min per task; launch_chain2.sh regenerates) | no PLAN, report or README cites them; only launch_chain*.sh (the producer) names them |
| /scratch/a/A.Otaifi/gn_shards/p{0,1}/a_shard{0..7}of8_{Aex,Abd}.npy (32 files) | 47 | High apart from age | merged intermediate: p0/m_gn.json and p1/m_gn.json verify_ok true; m_gn.json, m_deltas.npy, h_shard*.npz and a_shard*.npz stay | ~120 core-h (Pfaffian angle, board 11-16 09:30) | not cited; /scratch at ~954 of 973 GiB soft |

Kept: a_shard*of4.npz (5 MB, row indices), everything else under diag_ho (3 GB incl. chk1 2.6 GB, to be judged at the next scan).
Script: docs/cleanup_run_2026-11-16.sh (deletes exactly these 8 + 32 files, 57.9 GB in all, writes README_DELETED, logs to delta_today.md).
