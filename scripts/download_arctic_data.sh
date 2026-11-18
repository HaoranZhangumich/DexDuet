#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. "$REPO_ROOT/scripts/activate_env.sh" "${ARCTIC_ENV_NAME:-arctic_env}"
python "$REPO_ROOT/scripts/download_arctic_data.py" "$@"
