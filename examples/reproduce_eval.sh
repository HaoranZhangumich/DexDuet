#!/bin/bash
# Evaluate + score every checkpoint produced by reproduce_benchmark.sh,
# then group results into the paper's reported ADD-AUC stats.
set -euo pipefail
cd "$(dirname "$0")/.."
. scripts/activate_env.sh dexmachina

shopt -s nullglob
CHECKPOINTS=(logs/rl_games/*/reproduce_*/nn/*_hand.pth)
if [[ ${#CHECKPOINTS[@]} -eq 0 ]]; then
  echo "No reproduction checkpoints found under logs/rl_games" >&2
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
  --input logs/ --pattern "**/eval_ep*.npy" --use_auc --output_name my_repro_stats
