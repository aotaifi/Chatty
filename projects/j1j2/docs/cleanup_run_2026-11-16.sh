#!/bin/bash
# Run by the PI after reading docs/cleanup_manifest_2026-11-16.md. Deletes the 8 merged GN shards only.
set -euo pipefail
ROOT=/project/theorie/a/A.Otaifi/runs/j1j2/r1prime_fit/diag_ho/gn/c2
DELTA=/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/delta_today.md
grep -q '"verify_ok": *true' $ROOT/m_gn.json || { echo "merge not verified; abort"; exit 1; }
n=0; for s in 0 1 2 3; do for k in Aex Abd; do f=$ROOT/a_shard${s}of4_${k}.npy
  case "$f" in /project/theorie/a/A.Otaifi/runs/j1j2/*) ;; *) echo "outside project: $f"; exit 1;; esac
  [ -f "$f" ] && { rm -f "$f"; n=$((n+1)); }; done; done
echo "Deleted $n merged GN shards (a_shard*of4_{Aex,Abd}.npy, ~10.9 GB) on $(date -Is); merged result in m_gn.json; regenerate with ../launch_chain2.sh (acc array, ~3-11 core-h). Manifest: Chatty/organize/projects/j1j2/docs/cleanup_manifest_2026-11-16.md" > $ROOT/README_DELETED
echo "$(date '+%Y-%m-%d %H:%M') CLEANUP (PI ran docs/cleanup_run_2026-11-16.sh): deleted $n merged GN shards, ~10.9 GB, runs/j1j2/r1prime_fit/diag_ho/gn/c2" >> $DELTA
for P in p0 p1; do SR=/scratch/a/A.Otaifi/gn_shards/$P
  grep -q '"verify_ok": *true' $SR/m_gn.json || { echo "$P merge not verified; abort"; exit 1; }
  m=0; for s in 0 1 2 3 4 5 6 7; do for k in Aex Abd; do f=$SR/a_shard${s}of8_${k}.npy; [ -f "$f" ] && { rm -f "$f"; m=$((m+1)); }; done; done
  echo "Deleted $m merged GN shards (a_shard*of8_{Aex,Abd}.npy) on $(date -Is); merged result in m_gn.json; manifest Chatty/organize/projects/j1j2/docs/cleanup_manifest_2026-11-16.md" > $SR/README_DELETED
  echo "$(date '+%Y-%m-%d %H:%M') CLEANUP (PI ran docs/cleanup_run_2026-11-16.sh): deleted $m merged GN shards on /scratch/a/A.Otaifi/gn_shards/$P" >> $DELTA; n=$((n+m)); done
echo "done: $n files"
