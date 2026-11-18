#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${DEXMACHINA_ENV_NAME:-dexmachina}"
GENESIS_REV="${GENESIS_REV:-e9df2d8f7d30595d65cabdc618c6b5ca8cc657e2}"
# Genesis broad-phase overflow fix, applied on top of GENESIS_REV (docs/METHOD.md, section 4). Applying it with the
# committer and date below recreates GENESIS_FIXED_REV exactly, so the result can be checked by its SHA.
GENESIS_PATCH="$REPO_ROOT/environment/patches/genesis/0001-collider-bound-broad-phase-pair-buffer-guard-pair-co.patch"
GENESIS_FIXED_REV="${GENESIS_FIXED_REV:-fca4168f562b0762130932cda68ba09d011336ae}"
RL_GAMES_REV="${RL_GAMES_REV:-2c1a7718a3eed71e7c230b6bb8fb2897b0b1d0c4}"

if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
  CONDA_BIN="$CONDA_EXE"
elif command -v conda >/dev/null 2>&1; then
  CONDA_BIN="$(command -v conda)"
elif [[ -x "$HOME/miniconda3/bin/conda" ]]; then
  CONDA_BIN="$HOME/miniconda3/bin/conda"
elif [[ -x "$HOME/anaconda3/bin/conda" ]]; then
  CONDA_BIN="$HOME/anaconda3/bin/conda"
else
  echo "Conda was not found. Set CONDA_EXE or install Miniconda." >&2
  exit 1
fi

if [[ ! -d "$REPO_ROOT/Genesis/.git" ]]; then
  git clone https://github.com/MandiZhao/Genesis.git "$REPO_ROOT/Genesis"
fi
if [[ "$(git -C "$REPO_ROOT/Genesis" rev-parse HEAD)" != "$GENESIS_FIXED_REV" ]]; then
  if ! git -C "$REPO_ROOT/Genesis" diff --quiet || \
     ! git -C "$REPO_ROOT/Genesis" diff --cached --quiet; then
    echo "Genesis has tracked local changes; cannot switch safely to $GENESIS_REV" >&2
    exit 1
  fi
  git -C "$REPO_ROOT/Genesis" switch --detach "$GENESIS_REV"
  if ! GIT_COMMITTER_NAME="DexDuet Authors" GIT_COMMITTER_EMAIL="dexduet@users.noreply.github.com" \
       git -C "$REPO_ROOT/Genesis" -c commit.gpgSign=false \
       am --whitespace=nowarn --committer-date-is-author-date "$GENESIS_PATCH"; then
    git -C "$REPO_ROOT/Genesis" am --abort || true
    echo "Could not apply $GENESIS_PATCH to Genesis $GENESIS_REV" >&2
    exit 1
  fi
  if [[ "$(git -C "$REPO_ROOT/Genesis" rev-parse HEAD)" != "$GENESIS_FIXED_REV" ]]; then
    echo "Patched Genesis is $(git -C "$REPO_ROOT/Genesis" rev-parse HEAD); expected $GENESIS_FIXED_REV" >&2
    exit 1
  fi
fi
if [[ ! -d "$REPO_ROOT/Genesis/rl_games/.git" ]]; then
  git clone https://github.com/MandiZhao/rl_games.git "$REPO_ROOT/Genesis/rl_games"
  git -C "$REPO_ROOT/Genesis/rl_games" checkout "$RL_GAMES_REV"
fi

if ! "$CONDA_BIN" env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  "$CONDA_BIN" env create -n "$ENV_NAME" -f "$REPO_ROOT/dexmachina.yaml"
fi

ENV_PYTHON="$("$CONDA_BIN" run -n "$ENV_NAME" python -c 'import sys; print(sys.executable)')"
DEX_PY=("$ENV_PYTHON")
# RTX 50-series GPUs require CUDA 12.8+ binaries. Keep the paper's 2.5.1 pin
# on older hardware, but use the nearest compatible stable release on Blackwell.
if command -v nvidia-smi >/dev/null 2>&1 && \
   nvidia-smi --query-gpu=compute_cap --format=csv,noheader | grep -Eq '^12\.'; then
  "${DEX_PY[@]}" -m pip install torch==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu128
fi
"${DEX_PY[@]}" -m pip install -e "$REPO_ROOT/Genesis"
"${DEX_PY[@]}" -m pip install -e "$REPO_ROOT/Genesis/rl_games"
"${DEX_PY[@]}" -m pip install --no-deps -e "$REPO_ROOT"
# Restore the exact runtime pins required by the paper's Genesis 0.2.1 fork.
"${DEX_PY[@]}" -m pip install \
  "mujoco==3.2.5" "pydantic==2.7.1" "typing-extensions==4.13.2" \
  "scikit-learn==1.7.0" "libigl==2.5.1" "dash==3.2.0" \
  "typing-inspection==0.4.1"
"${DEX_PY[@]}" -m pip check

"${DEX_PY[@]}" - <<PY
from pathlib import Path
import importlib.metadata as md
import dexmachina, genesis, torch
root = Path(r"$REPO_ROOT").resolve()
assert Path(dexmachina.__file__).resolve().is_relative_to(root)
assert Path(genesis.__file__).resolve().is_relative_to(root / "Genesis")
from genesis.engine.solvers.rigid.collider_decomp import Collider
assert hasattr(Collider, "check_overflow"), "Genesis lacks the broad-phase overflow fix"
print("DexMachina environment ready")
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    test = torch.ones(1, device="cuda")
    torch.cuda.synchronize()
    print("CUDA tensor test", test.item(), torch.cuda.get_device_name(0))
print("genesis-world", md.version("genesis-world"), genesis.__file__)
print("rl-games", md.version("rl-games"))
PY
