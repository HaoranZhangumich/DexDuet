#!/bin/bash
# Train DexMachina on its main benchmark with Inspire hands only: 7 clips x 5
# seeds = 35 training runs. It needs the retargeted Inspire demonstrations
# (examples/retarget_missing.sh, checked by the preflight below) and does not
# retarget.
#
# The hyperparameters below are those of the authors' released reference runs
# (dexmachina_main_results_data.zip, linked from dexmachina/eval/eval.md; the
# config.yaml inside each run folder). Several of them differ per clip group:
# early_reset_threshold, bc_rew_weight, wait_epochs, deque_len, dialback_ep_len,
# curr_rew_thres, max_epochs and num_zero_epoch.
set -euo pipefail
cd "$(dirname "$0")/.."
. scripts/activate_env.sh dexmachina

HAND=inspire_hand
read -r -a SEEDS <<< "${DEXMACHINA_SEEDS:-42 24 66 15 113}"
SKIP_COMPLETED="${DEXMACHINA_SKIP_COMPLETED:-1}"
NUM_ENVS="${DEXMACHINA_NUM_ENVS:-12000}"
MAX_PARALLEL="${DEXMACHINA_MAX_PARALLEL:-1}"

if [[ ! "$NUM_ENVS" =~ ^[1-9][0-9]*$ || ! "$MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]]; then
  echo "DEXMACHINA_NUM_ENVS and DEXMACHINA_MAX_PARALLEL must be positive integers" >&2
  exit 2
fi

python scripts/preflight.py --stage train --hands "$HAND"

run_train () {
  local clip=$1 seed=$2 max_epochs=$3 num_zero_epoch=$4 lower_kp=$5 imi=$6 con=$7 bc=$8 \
        wait_epochs=$9 deque_len=${10} dialback_ep_len=${11} thres_con=${12} thres_imi=${13} thres_bc=${14}
  local exp="reproduce_seed${seed}"
  # train_rl_games removes only the separator between object and start frame.
  local clip_dir="${clip/-/}"
  local run_dir="logs/rl_games/${HAND}/inspire-${exp}_${clip_dir}_B${NUM_ENVS}_hybrid_thres0.5_ho16_imi${imi}_con${con}.0_bc${bc}"
  local checkpoint="${run_dir}/nn/${HAND}.pth"
  local complete_marker="${run_dir}/.training_complete"

  if [[ "$SKIP_COMPLETED" == 1 && -f "$complete_marker" && -f "$checkpoint" ]]; then
    echo "=== already complete: clip=$clip hand=$HAND seed=$seed ==="
    return
  fi

  echo "=== clip=$clip hand=$HAND seed=$seed B=$NUM_ENVS max_epochs=$max_epochs lower_kp=$lower_kp imi=$imi con=$con bc=$bc ==="
  python dexmachina/rl/train_rl_games.py -B "$NUM_ENVS" -obf -obt --max_epochs "$max_epochs" --num_zero_epoch "$num_zero_epoch" \
    --actuate_object --retarget_name para --horizon 16 -imw 0.5 --gain_mode all \
    --curr_schedule uniform --wait_epochs "$wait_epochs" --learning_rate 0.0003 \
    --contact_beta 10 --upper_ratios 0.9 0.9 1 --lower_ratios "$lower_kp" "$lower_kp" 1 \
    --save_freq "$max_epochs" --group_collisions --fixed_mode uniform --uniform_mode slow \
    --action_penalty 0.01 --dialback_ep_len "$dialback_ep_len" --skip_grad --deque_len "$deque_len" \
    --task_rew_betas 10 1 5 --use_retarget_contact --aux_reset_thres 0 0 0 \
    --curr_rew_thres 0.6 "$thres_con" "$thres_imi" "$thres_bc" -am hybrid --hybrid_scales 0.1 1.0 \
    --kp_init 80 --kv_init 5 --clip "$clip" \
    -imi "$imi" -bc "$bc" -con "$con" -ert 0.5 --hand "$HAND" --seed "$seed" -exp "$exp"

  if [[ ! -f "$checkpoint" ]]; then
    echo "Training returned successfully but the expected checkpoint is missing: $checkpoint" >&2
    exit 1
  fi
  touch "$complete_marker"
}

declare -A ACTIVE_PIDS=()

wait_for_one () {
  local finished_pid status
  if wait -n -p finished_pid "${!ACTIVE_PIDS[@]}"; then
    status=0
  else
    status=$?
  fi
  unset "ACTIVE_PIDS[$finished_pid]"
  return "$status"
}

wait_for_all () {
  local failed=0
  while (( ${#ACTIVE_PIDS[@]} > 0 )); do
    wait_for_one || failed=$?
  done
  return "$failed"
}

schedule_train () {
  run_train "$@" &
  ACTIVE_PIDS[$!]=1
  if (( ${#ACTIVE_PIDS[@]} >= MAX_PARALLEL )); then
    wait_for_one
  fi
}

for seed in "${SEEDS[@]}"; do
  # Submit longest tasks first so parallel workers remain balanced. Group C:
  # max_epochs=13000, num_zero_epoch=1000 (zero_epoch=12000).
  for clip in ketchup-40-340-s01-u02 notebook-40-340-s02-u02 waffleiron-40-340-s01-u01; do
    schedule_train "$clip" "$seed" 13000 1000 0.5 0.3 3 0.3 100 40 30 0 0 0
  done

  # Group B: mixer-300 -- max_epochs=10000, zero_epoch=9000.
  schedule_train "mixer-40-340-s01-u01" "$seed" 10000 1000 0.6 0.2 2 0.2 200 40 30 0.01 0.01 0.01

  # Group A: short clips -- max_epochs=5000, zero_epoch=4500.
  for clip in ketchup-30-130-s01-u01 box-30-230-s01-u01 mixer-30-200-s01-u01; do
    schedule_train "$clip" "$seed" 5000 500 0.6 0.2 2 0.2 200 20 30 0 0 0
  done
done

if (( ${#ACTIVE_PIDS[@]} > 0 )); then
  wait_for_all
fi
