#!/bin/bash
# Generate retargeted demo data (.pt files) needed by reproduce_benchmark.sh,
# skipping any (hand, object, subject, use_clip) combo that's already retargeted
# AND already long enough (>= MIN_FRAMES).
#
# IMPORTANT: retarget over the FULL clip (0-600), NOT the specific benchmark
# task's frame sub-range (e.g. 30-130). load_genesis_retarget_data() in
# demo_data.py always re-slices [frame_start:frame_end] out of the loaded file
# at TRAINING time, assuming it spans the full clip from frame 0. Retargeting
# with the benchmark's own sub-range pre-slices the data, so training's own
# slice then double-slices it down to a too-short (and silently wrong) length.
# The retargeted files that DexMachina's authors provide (box, mixer, laptop)
# span 600 frames or more, and every raw clip of the benchmark has at least 605
# frames, so 0-600 covers every task.
#
# Note: mixer-30-200-s01-u01 and mixer-40-340-s01-u01 share the same retargeted
# file (filename only encodes obj/subject/use_clip, not frame range), so only
# one retarget run is needed to cover both the Mixer-170 and Mixer-300 tasks.
set -euo pipefail
cd "$(dirname "$0")/.."
. scripts/activate_env.sh dexmachina

python scripts/preflight.py --stage retarget

MIN_FRAMES=600
read -r -a HANDS <<< "${DEXMACHINA_HANDS:-inspire_hand allegro_hand xhand schunk_hand}"
# unique (obj, subject, use_clip) combos needed across all 7 benchmark tasks.
# Clip string uses a generic 0-600 frame range, NOT each task's actual sub-range.
CLIPS=(
  "ketchup-0-600-s01-u01 ketchup s01 01"
  "box-0-600-s01-u01 box s01 01"
  "mixer-0-600-s01-u01 mixer s01 01"
  "ketchup-0-600-s01-u02 ketchup s01 02"
  "notebook-0-600-s02-u02 notebook s02 02"
  "waffleiron-0-600-s01-u01 waffleiron s01 01"
)

RETARGET_DIR=dexmachina/assets/retargeted

is_long_enough () {
  local f=$1
  python -c "
import torch, sys
try:
    d = torch.load('$f', weights_only=False)
    tensors = [
        d['retarget_data'][side][key]
        for side in ('left', 'right')
        for key in ('kpt_pos', 'wrist_pose')
    ]
    n = tensors[0].shape[0]
    valid = n >= $MIN_FRAMES and all(torch.isfinite(x).all() for x in tensors)
    sys.exit(0 if valid else 1)
except Exception:
    sys.exit(1)
"
}

for hand in "${HANDS[@]}"; do
  for entry in "${CLIPS[@]}"; do
    read -r clip obj subject use_clip <<< "$entry"
    out_file="$RETARGET_DIR/$hand/$subject/${obj}_use_${use_clip}_vector_para.pt"
    if [ -f "$out_file" ] && is_long_enough "$out_file"; then
      echo "=== skip (exists, >= $MIN_FRAMES frames): $out_file ==="
      continue
    fi
    echo "=== retargeting clip=$clip hand=$hand -> $out_file ==="
    # parallel_retarget.py hardcodes robot_dir="assets/{hand}" (relative path),
    # so it must be run with cwd inside the inner dexmachina/ package dir.
    (cd dexmachina && python retargeting/parallel_retarget.py --clip "$clip" --hand "$hand" \
      --control_steps 2000 --save_name para --save -ow)
  done
done
