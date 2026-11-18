#!/usr/bin/env bash
# Recompute the published Inspire Hand ADD-AUC table from the authors' public
# result archive. This validates reporting independently of licensed ARCTIC data.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REFERENCE_ROOT="${DEXMACHINA_REFERENCE_ROOT:-$HOME/dexmachina_data/reference}"
ARCHIVE="$REFERENCE_ROOT/dexmachina_main_results_data.zip"
DATA_ROOT="$REFERENCE_ROOT/dexmachina_main_results_data"
OUTPUT_ROOT="$REFERENCE_ROOT/recomputed"
URL='https://drive.usercontent.google.com/download?id=1d8sMncXvPir-PdiUFYW4t-J5YlEf8NFW&export=download&confirm=t'

mkdir -p "$REFERENCE_ROOT" "$OUTPUT_ROOT"
if [[ ! -f "$ARCHIVE" ]]; then
  echo "Downloading the authors' 3.86 GB reference archive..."
  curl -L --fail --retry 3 --continue-at - --progress-bar "$URL" -o "$ARCHIVE"
fi

unzip -tq "$ARCHIVE"
unzip -q -o "$ARCHIVE" \
  'dexmachina_main_results_data/run_paths.json' \
  'dexmachina_main_results_data/runs/inspire-*/config.yaml' \
  'dexmachina_main_results_data/runs/inspire-*/add.npy' \
  'dexmachina_main_results_data/runs/inspire-*/logs/rl_games/inspire_hand/*/inspire_hand_eval/eval_ep0.npy' \
  -d "$REFERENCE_ROOT"

cd "$REPO_ROOT"
. scripts/activate_env.sh dexmachina
python -m dexmachina.eval.group_results \
  --run_paths_json "$DATA_ROOT/run_paths.json" \
  --run_paths_base "$DATA_ROOT" \
  --pattern '**/inspire_hand_eval/eval_ep*.npy' \
  --use_auc \
  --output_dir "$OUTPUT_ROOT" \
  --output_name inspire_hand_reference_recomputed \
  | tee "$OUTPUT_ROOT/group_results.log"

GENERATED="$OUTPUT_ROOT/inspire_hand_reference_recomputed_AUC.json"
TRACKED="$REPO_ROOT/dexmachina/eval/stats/dexmachina_main_stats_AUC.json"
python - "$GENERATED" "$TRACKED" <<'PY'
import json
import sys
from pathlib import Path

generated = json.loads(Path(sys.argv[1]).read_text())["avg_success_rates"]["method1"]["inspire_hand"]
reference = json.loads(Path(sys.argv[2]).read_text())["avg_success_rates"]["method1"]["inspire_hand"]
print("task\tADD-AUC\tstandard error\tseeds")
values = []
for task, result in generated.items():
    expected = reference[task]
    assert result["n_seeds"] == expected["n_seeds"] == 5
    assert abs(result["value"] - expected["value"]) < 1e-12
    assert abs(result["err"] - expected["err"]) < 1e-12
    values.append(result["value"])
    print(f'{task}\t{result["value"]:.6f}\t{result["err"]:.6f}\t{result["n_seeds"]}')
print(f"macro average (7 tasks)\t{sum(values) / len(values):.6f}")
print("Exact match with the published reference statistics.")
PY

echo "Generated: $GENERATED"
