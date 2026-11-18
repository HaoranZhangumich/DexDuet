#!/bin/bash
# Train DexMachina on its main benchmark: 7 clips x 4 hands x 5 seeds.
# The hyperparameter groups are inferred from DexMachina's grouping config,
# dexmachina/eval/group_cfgs/dexmachina_main.yaml. That file leaves mixer-40-340
# (Mixer-300) with the non-Inspire hands open between the short-clip and the
# long-clip profile; this script uses the short-clip profile.
# examples/reproduce_inspire_train.sh takes the Inspire settings from the
# authors' released run configs instead (docs/REPRODUCE_DEXMACHINA.md, section 4).
set -euo pipefail
cd "$(dirname "$0")/.."
. scripts/activate_env.sh dexmachina

HANDS=(inspire_hand allegro_hand xhand schunk_hand)
SEEDS=(42 24 66 15 113)

# short clips + mixer-300: kp=0.6, imi=0.2, con=2 (same for all hands)
SHORT_CLIPS=(ketchup-30-130-s01-u01 box-30-230-s01-u01 mixer-30-200-s01-u01 mixer-40-340-s01-u01)
# remaining long clips: hyperparams differ by hand (inspire vs the rest)
LONG_CLIPS=(ketchup-40-340-s01-u02 notebook-40-340-s02-u02 waffleiron-40-340-s01-u01)

python scripts/preflight.py --stage train --hands "${HANDS[@]}"

run_train () {
  local clip=$1 hand=$2 seed=$3 lower_kp=$4 imi=$5 con=$6 imi_thres=$7
  echo "=== clip=$clip hand=$hand seed=$seed lower_kp=$lower_kp imi=$imi con=$con imi_thres=$imi_thres ==="
  python dexmachina/rl/train_rl_games.py -B 12000 -obf -obt --max_epochs 5000 \
    --actuate_object --retarget_name para --horizon 16 -imw 0.5 --gain_mode all \
    --curr_schedule uniform --wait_epochs 100 --learning_rate 0.0003 \
    --contact_beta 10 --upper_ratios 0.9 0.9 1 --lower_ratios "$lower_kp" "$lower_kp" 1 \
    --save_freq 5000 --group_collisions --fixed_mode uniform --uniform_mode slow \
    --action_penalty 0.01 --dialback_ep_len 80 --skip_grad --deque_len 30 \
    --task_rew_betas 10 1 5 --use_retarget_contact --aux_reset_thres 0 0 0 \
    --curr_rew_thres 0.6 "$imi_thres" 0.01 0.01 -am hybrid --hybrid_scales 0.1 1.0 \
    --kp_init 80 --kv_init 5 --clip "$clip" \
    -imi "$imi" -bc 0.3 -con "$con" -ert 0.6 --hand "$hand" --seed "$seed" -exp reproduce
}

for hand in "${HANDS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for clip in "${SHORT_CLIPS[@]}"; do
      run_train "$clip" "$hand" "$seed" 0.6 0.2 2 0.01
    done
    for clip in "${LONG_CLIPS[@]}"; do
      if [ "$hand" == "inspire_hand" ]; then
        run_train "$clip" "$hand" "$seed" 0.5 0.3 3 0.01
      else
        run_train "$clip" "$hand" "$seed" 0.5 0.2 2 0.0
      fi
    done
  done
done
