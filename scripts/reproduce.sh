#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMMAND="${1:-help}"
case "$COMMAND" in
  setup)
    "$REPO_ROOT/scripts/setup_arctic_env.sh"
    "$REPO_ROOT/scripts/setup_dexmachina_env.sh"
    ;;
  download-arctic) "$REPO_ROOT/scripts/download_arctic_data.sh" ;;
  prepare) "$REPO_ROOT/scripts/prepare_benchmark_data.sh" ;;
  preflight-inspire)
    . "$REPO_ROOT/scripts/activate_env.sh" dexmachina
    python "$REPO_ROOT/scripts/preflight.py" --stage train --hands inspire_hand
    ;;
  reference-inspire) "$REPO_ROOT/scripts/reproduce_reference_inspire.sh" ;;
  train-inspire) "$REPO_ROOT/examples/reproduce_inspire_train.sh" ;;
  eval-inspire) "$REPO_ROOT/examples/reproduce_inspire_eval.sh" ;;
  train-all) "$REPO_ROOT/examples/reproduce_benchmark.sh" ;;
  eval-all) "$REPO_ROOT/examples/reproduce_eval.sh" ;;
  help|-h|--help)
    echo "Usage: $0 {setup|download-arctic|prepare|preflight-inspire|reference-inspire|train-inspire|eval-inspire|train-all|eval-all}"
    ;;
  *) echo "Unknown command: $COMMAND" >&2; exit 2 ;;
esac
