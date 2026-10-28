#!/bin/bash
# Test T2 launcher (PLAN_pfaffian.md "Test T2"): three cx_h1.py runs (s1..s3) in one Paderborn H100 job.  Default: STAGE ONLY.  --submit: stage, sbatch (gpu_h100, hpc-prf-ppdmrg, default QoS, --no-requeue, 40 min), arm durable watcher.
# Needs the VPN (paderborn/README.md).  Local /scratch/a/A.Otaifi/ansatz_runs/pfaffian_c/runs/t2_cplx; remote $P/runs/t2_cplx.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
P=/scratch/hpc-prf-ppdmrg/hpcalot/chatty_compress6; RUN=$P/runs/t2_cplx
LOC=/scratch/a/A.Otaifi/ansatz_runs/pfaffian_c/runs/t2_cplx
H1=/project/theorie/a/A.Otaifi/Chatty/runs/compress6/runs/h1_full
SSHO="ssh -o BatchMode=yes -o HostName=login4"
rem() { $SSHO paderborn "export PATH=/opt/software/slurm/25.11.8-1/bin:\$PATH; $1"; }
mkdir -p "$LOC"
# 1. code pinned: codeH1 on Paderborn must equal h1_full/code_sha256.txt (nothing copied)
cp "$H1/code_sha256.txt" "$LOC/code_sha256.txt"
rem "cd $P/codeH1 && sha256sum cx_h1.py cx_lib.py rbmpp_lib.py" | diff - "$LOC/code_sha256.txt" > /dev/null || { echo "codeH1 sha256 differs from h1_full"; exit 1; }
# 2. specs: h1_full spec with f1_table_B -> c1_cplx_s{k}/round_1/la_table.npy; la_table existence + sha256 recorded
: > "$LOC/la_table_sha256.txt"
python3 - "$H1/spec_h1.json" "$LOC" "$P" <<'PY'
import json, sys, os
src, loc, P = sys.argv[1:4]; sp = json.load(open(src))
for k in (1, 2, 3):
    s = dict(sp); s['f1_table_B'] = f'{P}/runs/c1_cplx_s{k}/round_1/la_table.npy'
    d = f'{loc}/s{k}'; os.makedirs(d, exist_ok=True); json.dump(s, open(f'{d}/spec_h1.json', 'w'), indent=1)
PY
for k in 1 2 3; do
  T=$P/runs/c1_cplx_s$k/round_1/la_table.npy
  rem "sha256sum $T" >> "$LOC/la_table_sha256.txt" || { echo "missing $T"; exit 1; }
  rem "mkdir -p $RUN/s$k"; rsync -a -e "$SSHO" "$LOC/s$k/spec_h1.json" paderborn:$RUN/s$k/
done
# 3. brief
cat > "$LOC/ANALYSIS_BRIEF.md" <<EOB
# Test T2 (remote $RUN, local $LOC); job id in $LOC/job_id.txt
Question: PLAN_pfaffian.md "Test T2": tax_2 of the complex copies c1_cplx_s{1,2,3} (the F1B slot of cx_h1.py, code sha b555f9ba, codeH1) vs the real F1 A copy (1.846e-6).
Files: $LOC/s{1,2,3}/cx_h1.json {control, summary{o2b_a2,F1A,F1B:{tax_1,tax_2,ratio}}, stop, done}. No MC chains (quotable.py n/a). la_table sha256: $LOC/la_table_sha256.txt.
Gate (per sub-dir; expected control.ok=False and control.F1B=False because cx_h1 compares the F1B slot with the REAL F1 B reference: ignore those two, they are not a failure):
  python3 - <<'PYG'
import json
for k in (1,2,3):
    d=json.load(open('$LOC/s%d/cx_h1.json'%k)); c=d['control']; s=d.get('summary',{})
    ok=d.get('done') and not d.get('stop') and c['phi1']['ok'] and c['o2b_a2']['ok'] and c['F1A'] is True
    r=lambda a,b: abs(a-b)/b<1e-6
    ok=ok and r(s['F1A']['tax_2'],1.846217e-6) and r(s['o2b_a2']['tax_2'],1.903333e-6)
    print(k,'GATE_OK' if ok else 'GATE_FAIL',d.get('stop'),'F1A tax_2',s['F1A']['tax_2'],'o2b_a2 tax_2',s['o2b_a2']['tax_2'],'COMPLEX tax_2',s['F1B']['tax_2'])
PYG
Gate failure on any sub-dir (control phi1/o2b_a2/F1A, or tax_2 reference to rel 1e-6): stop, report to the angle lead, no verdict.
Verdict (PLAN_pfaffian.md Test T2) from the three complex tax_2 (m = mean): PASS m <= 0.82e-6 and every seed <= 1.63e-6; PARTIAL m in (0.82e-6, 1.63e-6); KILL m >= 1.63e-6 (mean <= 0.82e-6 with a seed > 1.63e-6 is not covered by the contract: report the numbers and say so).
Next: report the three tax_2, mean, gate lines and verdict to the angle lead; the lead decides.
EOB
rsync -a -e "$SSHO" "$LOC/ANALYSIS_BRIEF.md" paderborn:$RUN/
echo "staged remote $RUN, local $LOC"; cat "$LOC/la_table_sha256.txt"
[ "${1:-}" = "--submit" ] || exit 0
rsync -a -e "$SSHO" "$HERE/t2.sbatch" paderborn:$P/t2.sbatch
OUTL=$(rem "sbatch --parsable $P/t2.sbatch" 2>/dev/null)
JOB=${OUTL%%;*}; [[ "$JOB" =~ ^[0-9]+$ ]] || { echo "sbatch failed: $OUTL"; exit 1; }
echo "$JOB" > "$LOC/job_id.txt"
# durable watcher: t2_pdb_watch.sh (c1_pdb_watch.sh keyed on cx_h1.json); wake none (worker)
C1_REMRUN=$RUN C1_SUBS="s1 s2 s3" setsid nohup "$HERE/t2_pdb_watch.sh" "$JOB" "$LOC" > "$LOC/watcher.log" 2>&1 < /dev/null &
echo "job=$JOB run=$RUN watcher_pid=$!"
rem "sacct -j $JOB -X -n -o State,Reason%30" 2>/dev/null
