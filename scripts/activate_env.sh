#!/usr/bin/env bash
# Source this file: `. scripts/activate_env.sh [dexmachina|arctic_env]`.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "This script must be sourced: . scripts/activate_env.sh [environment]" >&2
  exit 2
fi

DEXMACHINA_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEXMACHINA_REQUESTED_ENV="${1:-${DEXMACHINA_ENV_NAME:-dexmachina}}"

if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
  DEXMACHINA_CONDA_EXE="$CONDA_EXE"
elif command -v conda >/dev/null 2>&1; then
  DEXMACHINA_CONDA_EXE="$(command -v conda)"
elif [[ -x "$HOME/miniconda3/bin/conda" ]]; then
  DEXMACHINA_CONDA_EXE="$HOME/miniconda3/bin/conda"
elif [[ -x "$HOME/anaconda3/bin/conda" ]]; then
  DEXMACHINA_CONDA_EXE="$HOME/anaconda3/bin/conda"
else
  echo "Conda was not found. Set CONDA_EXE or install Miniconda." >&2
  return 1
fi

eval "$("$DEXMACHINA_CONDA_EXE" shell.bash hook)"
conda activate "$DEXMACHINA_REQUESTED_ENV"

export PYTHONNOUSERSITE=1
export SETUPTOOLS_USE_DISTUTILS="${SETUPTOOLS_USE_DISTUTILS:-stdlib}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$DEXMACHINA_REPO_ROOT/.cache/matplotlib}"
mkdir -p "$MPLCONFIGDIR"
if [[ "$DEXMACHINA_REQUESTED_ENV" == "dexmachina" ]]; then
  export WANDB_MODE="${WANDB_MODE:-offline}"
fi
