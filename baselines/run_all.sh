#!/usr/bin/env bash
# run_all.sh: train and evaluate the ablation rows (the paper's ablation table) with a simple GPU queue.
# Each job is one `scripts/reproduce_paper.sh ablation --row R --task T --seed S` (row a: `main --task T --seed S`, i.e.
# DexDuet itself), which trains, evaluates (100 lanes, evaluation seed 1042, frozen start state) and writes the run's
# summary; `scripts/reproduce_paper.sh table` then builds the table.
#
#   DRY_RUN=1 GPUS="0 1" bash baselines/run_all.sh                     # print the plan
#   CH_MANO_EXPORT=<MANO export> GPUS="0 1 2 3" PER_GPU=2 bash baselines/run_all.sh
#
# Options (environment variables):
#   GPUS="0 1"      GPU indices to use (default 0)
#   PER_GPU=2       jobs at the same time on one GPU (a training run needs about 26 GB at 12,000 environments)
#   ROWS="a b c d e f g"  rows (default b-g; a = DexDuet on Inspire with the same seeds)
#   SEEDS="15 42 113"; TASKS="Box-200 Ketchup-100 Ketchup-300 Mixer-170 Mixer-300 Notebook-300 Waffleiron-300"
#   DRY_RUN=1       print the jobs and the GPU each would use; run nothing
# The other variables of scripts/reproduce_paper.sh apply (DEXDUET_PYTHON, DEXDUET_OUT, CH_MANO_EXPORT).
# Starting it again is safe: a job whose summary exists returns at once. A run folder left by an interrupted training stops its
# job with a message (Genesis runs cannot resume); move that folder away and start the queue again.
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
GPUS=${GPUS:-0}; PER_GPU=${PER_GPU:-2}; ROWS=${ROWS:-"b c d e f g"}; SEEDS=${SEEDS:-"15 42 113"}
TASKS=${TASKS:-"Notebook-300 Waffleiron-300 Ketchup-300 Mixer-300 Box-200 Mixer-170 Ketchup-100"}   # long clips first
DRY_RUN=${DRY_RUN:-0}
LOGS=${DEXDUET_OUT:-$REPO/paper_repro}/logs/run_all; mkdir -p "$LOGS"

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGS/run_all.log"; }

job_cmd() {   # row task seed -> the reproduce_paper.sh arguments
  if [ "$1" = a ]; then echo "main --hand inspire_hand --task $2 --seed $3"; else echo "ablation --row $1 --task $2 --seed $3"; fi
}

run_one() {   # gpu row task seed
  local gpu=$1 name="row_$2_$3_seed$4"
  say "$name: start on GPU $gpu"
  if CUDA_VISIBLE_DEVICES=$gpu bash "$REPO/scripts/reproduce_paper.sh" $(job_cmd "$2" "$3" "$4") > "$LOGS/$name.log" 2>&1; then
    say "$name: done ($(tail -n 1 "$LOGS/$name.log" | cut -c1-160))"
  else
    say "$name: FAILED, see $LOGS/$name.log"
  fi
}

JOBS=()
for seed in $SEEDS; do for row in $ROWS; do for task in $TASKS; do JOBS+=("$row $task $seed"); done; done; done
SLOTS=(); for g in $GPUS; do for i in $(seq 1 "$PER_GPU"); do SLOTS+=("$g:$i"); done; done
say "run_all: ${#JOBS[@]} job(s) on ${#SLOTS[@]} slot(s) [GPUs $GPUS x $PER_GPU]; dry_run=$DRY_RUN"

declare -A SLOT_PID
n=0
for job in "${JOBS[@]}"; do
  set -- $job
  if [ "$DRY_RUN" = 1 ]; then
    echo "GPU ${SLOTS[$((n % ${#SLOTS[@]}))]%%:*}: bash scripts/reproduce_paper.sh $(job_cmd "$1" "$2" "$3")"
    n=$((n + 1)); continue
  fi
  free=""
  while [ -z "$free" ]; do
    for s in "${SLOTS[@]}"; do
      p=${SLOT_PID[$s]:-}
      if [ -z "$p" ] || ! kill -0 "$p" 2>/dev/null; then free=$s; break; fi
    done
    [ -n "$free" ] || wait -n 2>/dev/null || sleep 10
  done
  run_one "${free%%:*}" "$1" "$2" "$3" &
  SLOT_PID[$free]=$!
  sleep 20   # stagger the starts (Genesis compiles its kernels at start-up)
done
wait
say "run_all: finished; per-job logs in $LOGS"
