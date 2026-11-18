#!/usr/bin/env bash
# Long-running, resumable Inspire reproduction: train the requested sweep, then
# evaluate every completed checkpoint and aggregate ADD-AUC statistics. Sweep
# size, batch size, and concurrency can be set through the DEXMACHINA_* values
# consumed by examples/reproduce_inspire_train.sh.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

./scripts/reproduce.sh train-inspire
./scripts/reproduce.sh eval-inspire
