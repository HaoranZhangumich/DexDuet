#!/bin/bash
# Evaluate + score all Inspire Hand reproduce runs: 20 episodes per checkpoint
# (matching the paper's B.4 methodology), then group into ADD-AUC stats.
set -euo pipefail
cd "$(dirname "$0")/.."
. scripts/activate_env.sh dexmachina

shopt -s nullglob
ALL_CHECKPOINTS=(logs/rl_games/inspire_hand/*reproduce_*/nn/inspire_hand.pth)
CHECKPOINTS=()
for ckpt in "${ALL_CHECKPOINTS[@]}"; do
  run_dir="${ckpt%/nn/*}"
  if [[ -f "$run_dir/.training_complete" ]]; then
    CHECKPOINTS+=("$ckpt")
  else
    echo "=== ignoring incomplete run: $run_dir ==="
  fi
done
if [[ ${#CHECKPOINTS[@]} -eq 0 ]]; then
  echo "No completed Inspire Hand reproduction checkpoints found under logs/rl_games" >&2
  exit 1
fi

for ckpt in "${CHECKPOINTS[@]}"; do
  echo "=== evaluating $ckpt ==="
  python scripts/preflight.py --stage eval --checkpoint "$ckpt"
  python -m dexmachina.rl.eval_rl_games --checkpoint "$ckpt" --num_envs 20
  eval_dir="${ckpt%/nn/*}/$(basename "$ckpt" .pth)_eval"
  python -m dexmachina.eval.compute_add --input "$eval_dir/eval_ep0.npy"
done

python -m dexmachina.eval.group_results \
  --group_config dexmachina_main --config_path dexmachina/eval/group_cfgs \
  --input logs/rl_games/inspire_hand --pattern "**/eval_ep*.npy" \
  --use_auc --output_name inspire_hand_stats
