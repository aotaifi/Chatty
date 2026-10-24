#!/bin/bash
# Synthetic test of every branch of watch_gfmc8.sh (no ssh): a fake `ssh` prints the sacct/stat lines the watcher parses.
# Each scenario must end with the right verdict in watcher_verdict.txt and a VERDICT line in a scratch delta file; a fresh RUNNING job must keep waiting.
H=$(cd "$(dirname "$0")" && pwd); T=$(mktemp -d); export POLL=1 STALL=100 FAILSEC=3 GF_FIN=$T/fin.sh GF_DELTA=$T/delta.md GF_REMOTE_PREFIX= RSYNC_EVERY=1000 GF_REMRUN=$T/rem; mkdir -p $T/rem
cat > $T/fin.sh <<'EOF'
#!/bin/bash
# stand-in for tools/job_finish.sh: records the state it was called with
while [ $# -gt 0 ]; do case "$1" in --state) S=$2; shift 2;; --delta) D=$2; shift 2;; --job) J=$2; shift 2;; *) shift;; esac; done
echo "VERDICT job $J $S" >> $D
EOF
chmod +x $T/fin.sh
cat > $T/ssh.sh <<'EOF'
#!/bin/bash
# $1 = remote command (ignored); prints the lines the watcher parses, from files written by the test
cat $FAKEDIR/out 2>/dev/null; exit $(cat $FAKEDIR/rc 2>/dev/null || echo 0)
EOF
chmod +x $T/ssh.sh; export GF_SSH=$T/ssh.sh FAKEDIR=$T/fake; mkdir -p $T/fake
run() { # name state age done expect
  L=$T/loc_$1; mkdir -p $L; rm -f $L/watcher_verdict.txt
  printf 'STATE=%s\nNOW=1000\nREF=%s\nDONE=%s\nEND\n' "$2" "$((1000-$3))" "$4" > $T/fake/out; echo 0 > $T/fake/rc
  timeout 40 bash $H/watch_gfmc8.sh 4242 $L > /dev/null 2>&1
  v=$(awk '{print $1}' $L/watcher_verdict.txt 2>/dev/null); [ "$v" = "$5" ] && echo "PASS $1 -> $v" || echo "FAIL $1 -> '$v' (want $5)"; }
run completed COMPLETED 10 1 COMPLETED
run completed_noverdict COMPLETED 10 0 FAILED
run failed FAILED 10 0 FAILED
run oom OUT_OF_MEMORY 10 0 FAILED
run timeout TIMEOUT 10 0 TIMEOUT
run stalled RUNNING 5000 0 STALLED
# vanished job: sacct prints an empty state 5 times
run gone "" 0 0 DISAPPEARED
# ssh failing for longer than FAILSEC
L=$T/loc_sshfail; mkdir -p $L; echo 255 > $T/fake/rc; : > $T/fake/out; timeout 40 bash $H/watch_gfmc8.sh 4242 $L >/dev/null 2>&1; v=$(awk '{print $1}' $L/watcher_verdict.txt 2>/dev/null); [ "$v" = DISAPPEARED ] && echo "PASS sshfail -> $v" || echo "FAIL sshfail -> '$v'"
# a RUNNING job with a fresh heartbeat must NOT finish
L=$T/loc_run; mkdir -p $L; rm -f $L/watcher_verdict.txt; printf 'STATE=RUNNING\nNOW=1000\nREF=990\nDONE=0\nEND\n' > $T/fake/out; echo 0 > $T/fake/rc
timeout 6 bash $H/watch_gfmc8.sh 4242 $L >/dev/null 2>&1; [ -f $L/watcher_verdict.txt ] && echo "FAIL running finished" || echo "PASS running keeps waiting"
echo "delta lines:"; cat $T/delta.md
