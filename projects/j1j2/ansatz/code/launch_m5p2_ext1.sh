#!/bin/bash
# M5' RERUN, EXTENSION SEGMENT 1 (PLAN_pfaffian.md "Extension segment 1", lead GO 2026-11-05): stage / --test / --submit on the theorie A40 ONLY.  Derived from launch_m5p2_th.sh (not edited).  RUN ON th-ws-7010m51.
# Fit: warm from theta(1500) = pfaffian_m5p2/round/fit_th/round_1/theta.npy (sha pinned below), same cx_r1fit f1374bed and the SAME spec except max_steps 600 and save_theta_at 150/300/450/600 (min_steps 1500, seed 1, fit_sec 90000 kept).
# Walls: fit 600 steps x 31.2 s = 5.2 h + the fit's own final table ~0.7 h (round: DONE 49909 s = 13.9 h for 1500 steps) -> ~6 h, wall 12:00:00 (x2); tables 2 x ~42 min = 1.4 h, wall 03:00:00.
set -euo pipefail
case "$(hostname -s)" in th-ws-7010m51) ;; *) echo "run on th-ws-7010m51, not $(hostname -s)"; exit 1;; esac
HERE=$(cd "$(dirname "$0")" && pwd); MODE=${1:-check}
Q=/scratch/a/A.Otaifi/ansatz_runs/pfaffian_m5p2; E=$Q/ext1; RF=$Q/round/fit_th/round_1
PART=${TH_PART:-cip,inter}; GRES=${TH_GRES:-gpu:a40:1}; TMF=12:00:00; TMT=03:00:00
FT=$E/fit_th; TT=$E/tab_th; CD=$Q/code_th
chk() { h=$(sha256sum "$1" | cut -d' ' -f1); [ "$h" = "$2" ] || { echo "sha256 mismatch $1"; exit 1; }; echo "input ok ${h:0:12} $1"; }
chk $RF/theta.npy 325a948234504df6e72bd9b86ebe27df6200e89aff9a345f57e0660798b0a05a
chk $RF/theta_step1200.npy 749f37f56baf4597fdb1a88e406ed30c4362809d635bbc6732a108c236839611; chk $RF/theta_step1350.npy e20b5eb9b261ab35355dbbb669f24f779c6b5943b53f639f701e44ea48ef0a97
chk $Q/theta_rel.npy bff7cdc48555768b6af6984a914b6e696cca1130790c67a623dfe8b14f1d03b6
chk $CD/cx_r1fit.py f1374bed9c7529a53ea8d95177eb709f6a79c2fbd78906351a66c7f0319babc5; chk $CD/rbmpp_lib.py 97eade567b035835ba6d74d4b5f21a11612cdf264769152ee3d1533f812d82fb
chk $CD/cx_r1.py 5ebae9d83497d0805137ae0487a0878b53afcb8fbf93f79d6213bd50eb3ab19d; chk $CD/cx_lib.py f320a0954d53c6155cc462d33b91602de7478a5e0cfdc2ee74d5a125cf281a45
for f in cx_r1.py cx_r1fit.py cx_lib.py rbmpp_lib.py; do chk $HERE/$f $(grep " $f\$" $Q/code_th_sha256.txt | cut -d' ' -f1); done
chk /scratch/a/A.Otaifi/ansatz_runs/pfaffian_v/target/target_T5_b1_sG.npy 35ef133508403ddbdfd7e459fa690d348034e0af8ababaf28b9c92f9025c8e99
chk /scratch/a/A.Otaifi/ansatz_runs/pfaffian_lswo2/s_G.npy 94eb19c95063ea22d2bee540e551d99870c6bb815210656fe20cc2d01e87c46a
L=/project/theorie/a/A.Otaifi/Chatty/runs
chk $L/stall6/data/psi0_6x6_table.npz e2d0c838d7983f29e61805c9feef67d3aea24584058c2b703f9fbd8c5da1660d; chk $L/stall6/data/sym_tables.npz 1f8576537870f2737ad6200ea709da953fb214d3a24b65677efe8248254944d9
chk $L/stall6/csr/norb.npy 9f37ec985d777ec56bfe82e249d0b7a2601f1cee1d518fcfb08c0c8e5ecf5730; chk $L/compress6/runs/refresh_17409419/s2.npy fd4f1ef0638dc726f9652f81bbf5751eadc796601393fb4dddf5f1f031154451
for f in up_codes.npy up_inc.npy up_indices.npy meta.json; do [ -s $L/stall6/csr/$f ] || { echo "missing csr/$f"; exit 1; }; done
mkdir -p $FT $TT/tab_avg $TT/tab_s1800 $E/readout
python3 - "$Q" "$E" <<'PY'
import json, sys, os
Q, E = sys.argv[1:3]; a = json.load(open(f'{Q}/round/fit_th/spec_r1.json')); b = json.loads(json.dumps(a))
b['warm_theta'] = f'{Q}/round/fit_th/round_1/theta.npy'; b['max_steps'] = 600; b['fit_overrides'] = dict(a['fit_overrides'], save_theta_at=[150, 300, 450, 600])
assert b['fit_overrides'] == {'seed': 1, 'min_steps': 1500, 'save_theta_at': [150, 300, 450, 600]}
json.dump(b, open(f'{E}/fit_th/spec_r1.json', 'w'), indent=1)
for tag, th in (('avg', f'{E}/tab_th/theta_avg.npy'), ('s1800', f'{E}/fit_th/round_1/theta_step300.npy')):
    t = json.loads(json.dumps(b)); t.update(warm_theta=th, fit_sec=1.0, max_steps=0); t['fit_overrides'] = {'seed': 1}; json.dump(t, open(f'{E}/tab_th/tab_{tag}/spec_r1.json', 'w'), indent=1)
d = [k for k in set(a) | set(b) if a.get(k) != b.get(k)]; assert sorted(d) == ['fit_overrides', 'max_steps', 'warm_theta'], d
for k in ('target_la', 'sign_path', 'pp_params', 'pp_table', 'o2b_fit_spec', 'refresh'): assert os.path.exists(b[k]), (k, b[k])
print('specs ok: ext1 fit spec == round spec except warm_theta, max_steps, fit_overrides.save_theta_at; 2 table specs written')
PY
diff -u $Q/round/fit_th/spec_r1.json $FT/spec_r1.json > $E/spec_diff_vs_round_fit_th.txt || true
for D in $FT $TT; do printf '%s\n' "# M5' RERUN ext1 intermediate stage ($D); full brief $E/ANALYSIS_BRIEF.md." "This job is the fit (or table) stage of a chain; the registered gates run only after the CPU readout job (ext1/readout), so DO NOT run any gate here." "Reply exactly: 'VERDICT: stage COMPLETED, readout and gates follow in ext1/readout' and 'NEXT: wait for the ext1/readout VERDICT; nothing to submit'. Run no command." > $D/ANALYSIS_BRIEF.md; done
cp $E/ANALYSIS_BRIEF.md $E/readout/ANALYSIS_BRIEF.md; echo "theorie A40 only (lead 2026-11-05): no Paderborn copy $(date -Is)" > $E/pdb_skip
fitf="--parsable --no-requeue --partition=$PART --gres=$GRES --job-name=pf_m5p2e1fit --time=$TMF --output=$FT/slurm-%j.out"
tabf="--parsable --no-requeue --partition=$PART --gres=$GRES --job-name=pf_m5p2e1tab --time=$TMT --output=$TT/slurm-%j.out"
echo "staged ($PART $GRES): $FT $TT; walls $TMF / $TMT"
if [ "$MODE" = --test ]; then echo "--- test-only fit"; sbatch --test-only $fitf "$HERE/r6_th.sbatch" "$FT" "$CD" 2>&1; echo "--- test-only tab"; sbatch --test-only $tabf "$HERE/m5p2x_tab_th.sbatch" "$TT" "$FT" "$CD" 2>&1; fi
[ "$MODE" = --submit ] || exit 0
[ -e "$FT/job_id.txt" ] && { echo "already submitted"; exit 1; }
JOB=$(sbatch $fitf "$HERE/r6_th.sbatch" "$FT" "$CD"); [[ "$JOB" =~ ^[0-9]+$ ]] || { echo "sbatch failed (fit): $JOB"; exit 1; }
echo "$JOB" > $FT/job_id.txt; echo "ext1 fit_th $JOB $PART $GRES $(date -Is)" >> $Q/job_ids.txt
setsid nohup "$HERE/lswo_a40_watch.sh" "$JOB" "$FT" > "$FT/watcher.log" 2>&1 < /dev/null & echo $! > $FT/watcher_pid.txt
JT=$(sbatch $tabf --dependency=afterok:$JOB "$HERE/m5p2x_tab_th.sbatch" "$TT" "$FT" "$CD"); [[ "$JT" =~ ^[0-9]+$ ]] || { echo "sbatch failed (tab): $JT; fit $JOB already submitted"; exit 1; }
echo "$JT" > $TT/job_id.txt; echo "ext1 tab_th $JT afterok:$JOB $(date -Is)" >> $Q/job_ids.txt
R1_TSUBS="avg s1800" setsid nohup "$HERE/r5_tab_watch.sh" "$JT" "$TT" > "$TT/watcher.log" 2>&1 < /dev/null & echo $! > $TT/watcher_pid.txt
setsid nohup "$HERE/m5p2x_chain.sh" ext1 > "$E/readout/chain.log" 2>&1 < /dev/null & echo $! > $E/readout/chain_pid.txt
sleep 2
for f in $FT/watcher_pid.txt $TT/watcher_pid.txt $E/readout/chain_pid.txt; do echo "$f pid $(cat $f) ppid $(ps -o ppid= -p $(cat $f))"; done
sacct -u $USER -S now-10minutes -X -n -o JobID,JobName%20,State,Reason%25 | grep pf_m5p2e1
