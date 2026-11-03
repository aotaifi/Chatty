# j1j2 cleanup manifest 2026-11-03 (lead, /cleanup by the PI; scan docs/cleanup_scan_2026-11-03.tsv)
/scratch 961.3 of 972.8 GB soft (1 TB hard); /project/theorie 109 of 475 GB, files 333k of 475k.

## Result: no j1j2 data qualifies for deletion
- j1j2's own /scratch data is ~28 GB: ansatz_runs 26.1 GB (every run dir < 14 days old, cited in PLAN*.md, several live),
  vit10_runs (< 2 days, live). The rubric excludes everything under 14 days, so High = Medium = 0 rows.
- Oldest j1j2-related: /scratch/a/A.Otaifi/sign_prob (1.43 GB, 30 d, the June predecessor, cited by Chatty/CLAUDE.md): Low, PI decides.
- No cleanup_run script this time (nothing to delete).

## Where the space is (not j1j2; rows in /project/theorie/a/A.Otaifi/CLEANUP_ACTIONS.md)
- PI-pending scripts that already exist: z2_hunt_ruby/docs/cleanup_run_20261102.sh (manifest cleanup_manifest_20261102.md),
  ruby_offdiag_fm_qmc/docs/cleanup_run_2026-10-29.sh (20.9 GB, needs the PI's 14-day override).
- /scratch/a/A.Otaifi/tmp/old13 (5.1 GB, 10-30, owner unknown) and claude_scratchpad_0723aa75_20261016 (4.1 GB, row exists).
- ~100 top-level /scratch dirs from Jan-Jun 2026 (ruby/paratoric/bridge/ss_o1 smoke runs), owners: those projects' leads.
