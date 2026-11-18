#!/usr/bin/env bash
# reproduce_paper.sh: reproduce the DexDuet paper's DexMachina-suite results (REPRODUCE.md).
#
#   bash scripts/reproduce_paper.sh from-release [--suite main|ablation|all] [--hand ...] [--task ...] [--seed ...] [--row ...]
#   bash scripts/reproduce_paper.sh dexmachina --archive <dexmachina_main_results_data> [--hand ...] [--task ...]
#   bash scripts/reproduce_paper.sh main --task Box-200 [--hand inspire_hand] [--seed 15] [--smoke]
#   bash scripts/reproduce_paper.sh ablation --row b --task Box-200 --seed 15 [--smoke]
#   bash scripts/reproduce_paper.sh evaluate --run <run folder> --task box-200 --seed 15 [--label DexDuet] [--group dexduet]
#   bash scripts/reproduce_paper.sh contact-auc --run <run folder> --task box-200 --seed 15
#   bash scripts/reproduce_paper.sh kit-metrics --run <evaluated run folder>
#   bash scripts/reproduce_paper.sh table
#   bash scripts/reproduce_paper.sh --dry-run <command> ...      print every command without running it
#
# Environment variables:
#   DEXDUET_PYTHON        python of the dexmachina environment (default: python on PATH)
#   DEXDUET_OUT           outputs: summaries/, tables/, resets/, staged/, logs/, hold_test/ (default <repo>/paper_repro)
#   DEXDUET_CHECKPOINTS   where the released checkpoints are unpacked (default <repo>/checkpoints)
#   DEXDUET_CKPT_REPO     Hugging Face repository of the released checkpoints (download only; default oldTOM/dexduet-checkpoints)
#   CH_MANO_EXPORT        MANO export folder, needed for Contact-AUC (docs/SETUP_NEW_MACHINE.md, section 4.3)
#   CUDA_VISIBLE_DEVICES  the GPU to use (one)
# CPU thread pools default to 1 thread per process (OMP/MKL/OpenBLAS/NumExpr), unless already set.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${DEXDUET_PYTHON:-python}"
command -v "$PY" > /dev/null || { echo "reproduce_paper.sh: no python '$PY'; activate the dexmachina environment or set DEXDUET_PYTHON" >&2; exit 2; }
export DEXDUET_PYTHON="$PY"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}" NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
cd "$REPO"
exec "$PY" "$REPO/scripts/reproduce_paper.py" "$@"
