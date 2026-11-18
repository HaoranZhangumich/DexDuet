#!/bin/bash
# Generate missing dexmachina/assets/contact_retarget/ data, needed by
# --use_retarget_contact during training. Distinct from retarget_missing.sh,
# which produces dexmachina/assets/retargeted/ (.pt) files instead.
# Skips any (hand, subject, obj, use_clip) combo whose output already exists
# AND is long enough (>= MIN_FRAMES) -- see retarget_missing.sh for why: this
# depends on retargeter_results/, which must itself be full-length (0-600),
# not sliced to a specific benchmark task's frame sub-range.
set -euo pipefail
cd "$(dirname "$0")/.."
. scripts/activate_env.sh dexmachina

MIN_FRAMES=600
read -r -a HANDS <<< "${DEXMACHINA_HANDS:-inspire_hand allegro_hand xhand schunk_hand}"
# (load_fname, obj, subject, use_clip) combos needed across all 7 benchmark tasks
CLIPS=(
  "assets/arctic/processed/s01/ketchup_use_01.npy ketchup s01 01"
  "assets/arctic/processed/s01/box_use_01.npy box s01 01"
  "assets/arctic/processed/s01/mixer_use_01.npy mixer s01 01"
  "assets/arctic/processed/s01/ketchup_use_02.npy ketchup s01 02"
  "assets/arctic/processed/s02/notebook_use_02.npy notebook s02 02"
  "assets/arctic/processed/s01/waffleiron_use_01.npy waffleiron s01 01"
)

CONTACT_DIR=dexmachina/assets/contact_retarget

is_long_enough () {
  local f=$1
  python -c "
import numpy as np, sys
try:
    d = np.load('$f', allow_pickle=True).item()
    n = d['left']['dexlink_contacts'].shape[0]
    sys.exit(0 if n >= $MIN_FRAMES else 1)
except Exception:
    sys.exit(1)
"
}

for hand in "${HANDS[@]}"; do
  for entry in "${CLIPS[@]}"; do
    read -r load_fname obj subject use_clip <<< "$entry"
    out_file="$CONTACT_DIR/$hand/$subject/${obj}_use_${use_clip}.npy"
    if [ -f "$out_file" ] && is_long_enough "$out_file"; then
      echo "=== skip (exists, >= $MIN_FRAMES frames): $out_file ==="
      continue
    fi
    # requires the retargeter_results/.../{obj}_use_{clip}_vector.npy input to
    # already exist AND be long enough (produced by retarget_missing.sh).
    retarget_input="dexmachina/assets/retargeter_results/$hand/$subject/${obj}_use_${use_clip}_vector.npy"
    if [ ! -f "$retarget_input" ]; then
      echo "!!! skipping $out_file: missing prerequisite $retarget_input (run retarget_missing.sh first) !!!"
      continue
    fi
    echo "=== mapping contacts load_fname=$load_fname hand=$hand -> $out_file ==="
    # map_contacts.py resolves some paths (e.g. --load_fname, retargeter_results
    # input) relative to cwd, so it must run from inside the inner dexmachina/
    # package dir, same as parallel_retarget.py.
    (cd dexmachina && python retargeting/map_contacts.py --hand "$hand" --load_fname "$load_fname")
  done
done

python scripts/preflight.py --stage train --hands "${HANDS[@]}"
