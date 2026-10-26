#!/bin/bash
# Submit a hop-aware SR job on Paderborn (gpu_h100) and arm its durable watcher in the same action.
#   GO=1 submit_hopsr.sh SBATCH_FILE DONE_PATTERN BRIEF_FILE     (dry run without GO=1)
# BRIEF_FILE: ANALYSIS_BRIEF.md template; the string @JID@ is replaced by the job id.  Local copy of the run: /scratch/a/A.Otaifi/hopsr/pdb/<jid>.
set -u
GO=${GO:-0}; SB=$1; PAT=$2; BRIEF=$3
HERE=$(cd "$(dirname "$0")" && pwd); GF=$HERE/..; RB=$GF/../rbmpp_6x6
B=/scratch/hpc-prf-ppdmrg/hpcalot; SSH="ssh -o BatchMode=yes -o HostName=login4"; SL=/opt/software/slurm/25.11.8-1/bin
LOC=/scratch/a/A.Otaifi/hopsr/pdb
for f in $SB $BRIEF $HERE/hopsr_lib.py $HERE/hopsr_train.py $GF/vmc_moments.py $RB/rbmpp_lib.py; do [ -f "$f" ] || { echo "missing $f"; exit 1; }; done
if [ "$GO" != 1 ]; then echo "DRY: rsync code -> $B/hopsr/code; sbatch $(basename $SB); watcher (pattern $PAT) -> $LOC/<jid>"; exit 0; fi
$SSH paderborn "mkdir -p $B/hopsr/code/gfmc_fn_lattice/hopsr/specs $B/hopsr/code/rbmpp_6x6 $B/hopsr/runs $B/hopsr/data" || exit 1
rsync -a -e "$SSH" $HERE/*.py $HERE/*.sbatch paderborn:$B/hopsr/code/gfmc_fn_lattice/hopsr/ || exit 1
rsync -a -e "$SSH" $HERE/specs/ paderborn:$B/hopsr/code/gfmc_fn_lattice/hopsr/specs/ || exit 1
rsync -a -e "$SSH" $GF/vmc_moments.py paderborn:$B/hopsr/code/gfmc_fn_lattice/ || exit 1
rsync -a -e "$SSH" $RB/rbmpp_lib.py paderborn:$B/hopsr/code/rbmpp_6x6/ || exit 1
rsync -a -e "$SSH" /scratch/a/A.Otaifi/hopsr/init4_rbmonly_opt.npz paderborn:$B/hopsr/data/ || exit 1
jid=$($SSH -n paderborn "export PATH=$SL:\$PATH; cd $B/hopsr/runs; sbatch --parsable $B/hopsr/code/gfmc_fn_lattice/hopsr/$(basename $SB)")
[[ "$jid" =~ ^[0-9]+$ ]] || { echo "SUBMIT FAILED: '$jid'"; exit 1; }
L=$LOC/$jid; mkdir -p $L
sed "s/@JID@/$jid/g" $BRIEF > $L/ANALYSIS_BRIEF.md
name=$(basename $SB .sbatch | sed 's/_hopsr//')
GF_REMRUN=$B/hopsr/runs/${name}_$jid GF_DONE_PAT=$PAT STALL=${STALL:-2400} POLL=90 setsid nohup bash $GF/m13b_8x8/watch_gfmc8.sh $jid $L > $L/watcher.log 2>&1 < /dev/null &
wp=$!; sleep 1
echo "SUBMITTED job $jid; watcher pid $wp ppid $(ps -o ppid= -p $wp | tr -d ' '); local $L; remote $B/hopsr/runs/${name}_$jid"
$SSH -n paderborn "export PATH=$SL:\$PATH; squeue -j $jid -o '%i %T %r %q'"
