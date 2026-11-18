#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ARCTIC_ENV_NAME:-arctic_env}"
REV="${ARCTIC_REV:-49f3eb4b7f01663172576357bb3d95695d7a7689}"

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

if [[ ! -d "$REPO_ROOT/arctic/.git" ]]; then
  git clone https://github.com/zc-alexfan/arctic.git "$REPO_ROOT/arctic"
  git -C "$REPO_ROOT/arctic" checkout "$REV"
fi

if ! "$CONDA_BIN" env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  "$CONDA_BIN" create -y -n "$ENV_NAME" python=3.10 pip
fi

"$CONDA_BIN" install -y -n "$ENV_NAME" \
  pytorch==1.13.0 torchvision==0.14.0 pytorch-cuda=11.6 \
  -c pytorch -c nvidia
"$CONDA_BIN" install -y -n "$ENV_NAME" pytorch3d==0.7.3 -c pytorch3d
# PyTorch 1.13 fails with newer MKL builds (`iJIT_NotifyEvent` undefined).
"$CONDA_BIN" install -y -n "$ENV_NAME" mkl=2023.1.0

ENV_PYTHON="$("$CONDA_BIN" run -n "$ENV_NAME" python -c 'import sys; print(sys.executable)')"
ARCTIC_PY=("$ENV_PYTHON")
"${ARCTIC_PY[@]}" -m pip install "setuptools==69.5.1" wheel
"${ARCTIC_PY[@]}" -m pip install --no-build-isolation \
  "git+https://github.com/mattloper/chumpy.git@580566eafc9ac68b2614b64d6f7aaa84eebb70da"
"${ARCTIC_PY[@]}" -m pip install \
  -c "$REPO_ROOT/environment/arctic-constraints.txt" \
  -r "$REPO_ROOT/arctic/requirements.txt"
"${ARCTIC_PY[@]}" "$REPO_ROOT/scripts/patch_smplx.py"
"${ARCTIC_PY[@]}" "$REPO_ROOT/scripts/patch_arctic_device.py"

"${ARCTIC_PY[@]}" - <<'PY'
import aitviewer, numpy, pytorch3d, pytorch_lightning, torch, torchvision
print("ARCTIC environment ready")
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("numpy", numpy.__version__, "pytorch3d", pytorch3d.__version__)
print("pytorch-lightning", pytorch_lightning.__version__, "aitviewer", aitviewer.__version__)
PY
