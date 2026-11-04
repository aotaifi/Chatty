#!/bin/bash
# M5' RERUN (PLAN_pfaffian.md "M5' rerun"): stage / --test / --submit the theorie A40 round fit + table chain in pfaffian_m5p2.  THEORIE A40 ONLY (no Paderborn, no race watcher).
# Derived from launch_m5p_th.sh (not edited).  RUN ON th-ws-7010m51.  Fit: r6_th.sbatch unchanged, cip,inter gpu:a40:1, wall 26:00:00 (1500 steps x 31.2 s = 13.0 h measured on the cancelled run, = wall/2); fit_sec 90000 as the cancelled run.
# Table: m5p2_tab_th.sbatch (tab_t5 + tab_s1200 + tab_s1500, ~3 x 40 min), wall 04:00:00.  Chain: m5p2_chain.sh round (setsid) -> CPU readout m5p2_read.sbatch (m5p2_read.py final + m5p2_plateau.py).
# Specs were staged from the cancelled run's fit_th spec (only warm_theta path differs; diffs in $Q/spec_diff_vs_m5p_*.txt).  Modes: (none) check only; --test: + sbatch --test-only; --submit.
set -euo pipefail
case "$(hostname -s)" in th-ws-7010m51) ;; *) echo "run on th-ws-7010m51, not $(hostname -s)"; exit 1;; esac
HERE=$(cd "$(dirname "$0")" && pwd); MODE=${1:-check}
Q=/scratch/a/A.Otaifi/ansatz_runs/pfaffian_m5p2; O=/scratch/a/A.Otaifi/ansatz_runs/pfaffian_m5p; V=/scratch/a/A.Otaifi/ansatz_runs/pfaffian_v; R4=/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r4
PART=${TH_PART:-cip,inter}; GRES=${TH_GRES:-gpu:a40:1}; TMF=26:00:00; TMT=04:00:00
FT=$Q/round/fit_th; TT=$Q/round/tab_th; CD=$Q/code_th
chk() { h=$(sha256sum "$1" | cut -d' ' -f1); [ "$h" = "$2" ] || { echo "sha256 mismatch $1"; exit 1; }; echo "input ok ${h:0:12} $1"; }
chk $Q/theta_rel.npy bff7cdc48555768b6af6984a914b6e696cca1130790c67a623dfe8b14f1d03b6; chk $O/theta_rel.npy bff7cdc48555768b6af6984a914b6e696cca1130790c67a623dfe8b14f1d03b6
chk $V/target/target_T5_b1_sG.npy 35ef133508403ddbdfd7e459fa690d348034e0af8ababaf28b9c92f9025c8e99
chk /scratch/a/A.Otaifi/ansatz_runs/pfaffian_lswo2/s_G.npy 94eb19c95063ea22d2bee540e551d99870c6bb815210656fe20cc2d01e87c46a
chk $O/rel1/tab_th/tab_rel/round_1/la_table.npy f29e7a1ef381f4d6351feea300f27472d1ca7023fcc5dd346192db614ad790c0
chk $CD/cx_r1fit.py f1374bed9c7529a53ea8d95177eb709f6a79c2fbd78906351a66c7f0319babc5; chk $CD/rbmpp_lib.py 97eade567b035835ba6d74d4b5f21a11612cdf264769152ee3d1533f812d82fb
chk $CD/cx_r1.py 5ebae9d83497d0805137ae0487a0878b53afcb8fbf93f79d6213bd50eb3ab19d; chk $CD/cx_lib.py f320a0954d53c6155cc462d33b91602de7478a5e0cfdc2ee74d5a125cf281a45
for f in cx_r1.py cx_r1fit.py cx_lib.py rbmpp_lib.py; do chk $HERE/$f $(grep " $f\$" $O/code_th_sha256.txt | cut -d' ' -f1); done
L=/project/theorie/a/A.Otaifi/Chatty/runs
chk $L/compress6/runs/o2b_17431886/round_1/spec_fit.json 3a89451e312a5f55fd2413a69d4d62665743a45707ac6ec692ee0b067615ec12
chk $L/rbmpp6/runs/rpT1_pp_17163493/params_final.npz e765e8fada11f3cc147c9d864c3720207cc022bb95275bc43318e5d41886b424
chk $L/rbmpp6/runs/rpTabPP_17164048/sym_tables_PP.npz d8a4e96c9a03b7ac18c0ef5bbaa9f4bcfe7632b34e01400cc2e6fb2f7253a934
chk $L/stall6/data/psi0_6x6_table.npz e2d0c838d7983f29e61805c9feef67d3aea24584058c2b703f9fbd8c5da1660d; chk $L/stall6/data/sym_tables.npz 1f8576537870f2737ad6200ea709da953fb214d3a24b65677efe8248254944d9
chk $L/stall6/csr/norb.npy 9f37ec985d777ec56bfe82e249d0b7a2601f1cee1d518fcfb08c0c8e5ecf5730; chk $L/compress6/runs/refresh_17409419/s2.npy fd4f1ef0638dc726f9652f81bbf5751eadc796601393fb4dddf5f1f031154451
for f in up_codes.npy up_inc.npy up_indices.npy meta.json; do [ -s $L/stall6/csr/$f ] || { echo "missing csr/$f"; exit 1; }; done
# specs: fit spec differs from the cancelled run's ONLY in warm_theta; every table spec = fit spec + {warm_theta, fit_sec 1, max_steps 0, fit_overrides seed only}
python3 - "$O" "$Q" <<'PY'
import json, sys, os
O, Q = sys.argv[1:3]; a = json.load(open(f'{O}/round/fit_th/spec_r1.json')); b = json.load(open(f'{Q}/round/fit_th/spec_r1.json'))
d = [k for k in set(a) | set(b) if a.get(k) != b.get(k)]; assert d == ['warm_theta'], d
assert b['warm_theta'] == f'{Q}/theta_rel.npy' and b['max_steps'] == 1500 and b['fit_sec'] == 90000.0 and b['fit_overrides'] == {'seed': 1, 'min_steps': 1500, 'save_theta_at': [600, 750, 900, 1050, 1200, 1350]}
w = {'t5': f'{Q}/round/tab_th/theta_5.npy', 's1200': f'{Q}/round/fit_th/round_1/theta_step1200.npy', 's1500': f'{Q}/round/fit_th/round_1/theta.npy'}
for tag, th in w.items():
    t = json.load(open(f'{Q}/round/tab_th/tab_{tag}/spec_r1.json')); e = json.loads(json.dumps(b)); e.update(warm_theta=th, fit_sec=1.0, max_steps=0); e['fit_overrides'] = {'seed': 1}; assert t == e, tag
for k in ('target_la', 'sign_path', 'pp_params', 'pp_table', 'o2b_fit_spec', 'refresh'): assert os.path.exists(b[k]), (k, b[k])
bad = [k for k, v in b.items() if isinstance(v, str) and v.startswith('/scratch/hpc-prf')]; assert not bad, bad
print('specs ok: fit spec == cancelled A40 spec except warm_theta; 3 table specs consistent')
PY
diff -u $O/round/fit_th/spec_r1.json $FT/spec_r1.json > $Q/spec_diff_vs_m5p_fit_th.txt || true
for D in fit_th tab_th; do :; done
cp $Q/ANALYSIS_BRIEF.md $Q/round/fit_th/ANALYSIS_BRIEF.md.full 2>/dev/null || true
for D in $FT $TT; do printf '%s\n' "# M5' RERUN intermediate stage ($D); full brief $Q/ANALYSIS_BRIEF.md." "This job is the fit (or table) stage of a chain; the registered gate runs only after the CPU readout job (readout_round), so DO NOT run any gate here." "Reply exactly: 'VERDICT: stage COMPLETED, readout and gate follow in readout_round' and 'NEXT: wait for the readout_round VERDICT; nothing to submit'. Run no command." > $D/ANALYSIS_BRIEF.md; rm -f $D/ANALYSIS_BRIEF.md.full; done
cp $Q/ANALYSIS_BRIEF.md $Q/readout_round/ANALYSIS_BRIEF.md
touch $Q/round/pdb_skip; echo "theorie A40 only (lead brief 2026-11-04): no Paderborn copy $(date -Is)" > $Q/round/pdb_skip
fitf="--parsable --no-requeue --partition=$PART --gres=$GRES --job-name=pf_m5p2fit --time=$TMF --output=$FT/slurm-%j.out"
tabf="--parsable --no-requeue --partition=$PART --gres=$GRES --job-name=pf_m5p2tab --time=$TMT --output=$TT/slurm-%j.out"
echo "staged ($PART $GRES): $FT $TT; walls $TMF / $TMT"
if [ "$MODE" = --test ]; then echo "--- test-only fit"; sbatch --test-only $fitf "$HERE/r6_th.sbatch" "$FT" "$CD" 2>&1; echo "--- test-only tab"; sbatch --test-only $tabf "$HERE/m5p2_tab_th.sbatch" "$TT" "$FT" "$CD" 2>&1; fi
[ "$MODE" = --submit ] || exit 0
[ -e "$FT/job_id.txt" ] && { echo "already submitted"; exit 1; }
JOB=$(sbatch $fitf "$HERE/r6_th.sbatch" "$FT" "$CD"); [[ "$JOB" =~ ^[0-9]+$ ]] || { echo "sbatch failed (fit): $JOB"; exit 1; }
echo "$JOB" > $FT/job_id.txt; echo "round fit_th $JOB $PART $GRES $(date -Is)" >> $Q/job_ids.txt
setsid nohup "$HERE/lswo_a40_watch.sh" "$JOB" "$FT" > "$FT/watcher.log" 2>&1 < /dev/null & echo $! > $FT/watcher_pid.txt
JT=$(sbatch $tabf --dependency=afterok:$JOB "$HERE/m5p2_tab_th.sbatch" "$TT" "$FT" "$CD"); [[ "$JT" =~ ^[0-9]+$ ]] || { echo "sbatch failed (tab): $JT; fit $JOB already submitted"; exit 1; }
echo "$JT" > $TT/job_id.txt; echo "round tab_th $JT afterok:$JOB $(date -Is)" >> $Q/job_ids.txt
R1_TSUBS="t5 s1200 s1500" setsid nohup "$HERE/r5_tab_watch.sh" "$JT" "$TT" > "$TT/watcher.log" 2>&1 < /dev/null & echo $! > $TT/watcher_pid.txt
setsid nohup "$HERE/m5p2_chain.sh" round > "$Q/readout_round/chain.log" 2>&1 < /dev/null & echo $! > $Q/readout_round/chain_pid.txt
sleep 2
for f in $FT/watcher_pid.txt $TT/watcher_pid.txt $Q/readout_round/chain_pid.txt; do echo "$f pid $(cat $f) ppid $(ps -o ppid= -p $(cat $f))"; done
sacct -u $USER -S now-10minutes -X -n -o JobID,JobName%20,State,Reason%25 | grep pf_m5p2
