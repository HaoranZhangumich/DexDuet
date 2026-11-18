#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUBJECT=""
OBJECT=""
USE=""
OVERWRITE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --subject) SUBJECT="$2"; shift 2 ;;
    --object) OBJECT="$2"; shift 2 ;;
    --use) USE="$2"; shift 2 ;;
    --overwrite) OVERWRITE=1; shift ;;
    -h|--help)
      echo "Usage: $0 --subject s01 --object box --use 01 [--overwrite]"
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ -z "$SUBJECT" || -z "$OBJECT" || -z "$USE" ]]; then
  echo "--subject, --object, and --use are required" >&2
  exit 2
fi

RAW="$REPO_ROOT/arctic/data/arctic_data/data/raw_seqs/$SUBJECT/${OBJECT}_use_${USE}.mano.npy"
ARCTIC_OUTPUT="$REPO_ROOT/arctic/outputs/processed_verts/seqs/$SUBJECT/${OBJECT}_use_${USE}.npy"
DEX_OUTPUT="$REPO_ROOT/dexmachina/assets/arctic/processed/$SUBJECT/${OBJECT}_use_${USE}.npy"
if [[ ! -f "$RAW" ]]; then
  echo "Missing $RAW. Run scripts/download_arctic_data.sh first." >&2
  exit 1
fi
if [[ -f "$DEX_OUTPUT" && "$OVERWRITE" -eq 0 ]]; then
  echo "Already prepared: $DEX_OUTPUT"
  exit 0
fi

cd "$REPO_ROOT/arctic"
. "$REPO_ROOT/scripts/activate_env.sh" "${ARCTIC_ENV_NAME:-arctic_env}"
python "$REPO_ROOT/scripts/patch_arctic_device.py"
if command -v nvidia-smi >/dev/null 2>&1 && \
   nvidia-smi --query-gpu=compute_cap --format=csv,noheader | grep -Eq '^12\.'; then
  # The exact ARCTIC PyTorch 1.13/CUDA 11.6 pin has no Blackwell kernels.
  ARCTIC_DEVICE="cpu"
elif timeout 15s python -c \
  'import torch; x=torch.ones(1, device="cuda"); torch.cuda.synchronize()' \
  >/dev/null 2>&1; then
  ARCTIC_DEVICE="cuda:0"
else
  ARCTIC_DEVICE="cpu"
  echo "ARCTIC's pinned PyTorch cannot execute on this GPU; processing this clip on CPU."
fi
PYTHONPATH="$REPO_ROOT/arctic${PYTHONPATH:+:$PYTHONPATH}" \
  python "$REPO_ROOT/scripts/process_arctic_sequence.py" \
  --mano_p "$RAW" --device "$ARCTIC_DEVICE"
if [[ ! -f "$ARCTIC_OUTPUT" ]]; then
  echo "ARCTIC processing did not produce $ARCTIC_OUTPUT" >&2
  exit 1
fi

cd "$REPO_ROOT/dexmachina"
. "$REPO_ROOT/scripts/activate_env.sh" "${DEXMACHINA_ENV_NAME:-dexmachina}"
DEX_ARGS=(--processed_fname "$ARCTIC_OUTPUT" --farthest_sample --save --save_dir assets/arctic/processed)
if [[ "$OVERWRITE" -eq 1 ]]; then
  DEX_ARGS+=(--overwrite)
fi
python retargeting/process_arctic.py "${DEX_ARGS[@]}"
echo "Prepared DexMachina demonstration: $DEX_OUTPUT"
