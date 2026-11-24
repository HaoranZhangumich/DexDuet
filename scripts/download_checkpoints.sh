#!/usr/bin/env bash
# download_checkpoints.sh: download the released DexDuet checkpoints from Hugging Face, check them against
# CHECKPOINTS.sha256 and unpack them.
#
#   bash scripts/download_checkpoints.sh                         # every package
#   bash scripts/download_checkpoints.sh dexduet_ablations.zip   # one package
#
# Packages (each unpacks into $DEXDUET_CHECKPOINTS):
#   dexduet_dexmachina_suite.zip         dexduet/<hand>/<task>/seed<S>/                        (main table, four hands)
#   dexduet_ablations.zip                ablations/row_<b..g>/inspire_hand/<task>/seed<S>/      (6 rows x 7 tasks x 3 seeds;
#                                        row (a) is the suite's Inspire runs)
# Each run folder holds <hand>.pth, config/{agent.yaml,env.yaml}, RUN_INFO.json and, for runs with the per-hand policy
# (all but ablation row (e)), config/initialization.json and final_checkpoint.json. The ARCTIC-derived demonstration
# data are not included (licence): evaluation rebuilds them from your own ARCTIC download (scripts/stage_release_run.py).
#
# Environment variables:
#   DEXDUET_CKPT_REPO     Hugging Face model repository (default oldTOM/dexduet-checkpoints)
#   DEXDUET_CHECKPOINTS   where to unpack (default <repo>/checkpoints); the zips are kept in its downloads/ folder
#   DEXDUET_ALLOW_UNVERIFIED=1   unpack a package that CHECKPOINTS.sha256 does not list yet (prints a warning)
# Uses `hf download` (huggingface_hub) when available, else curl on https://huggingface.co/<repo>/resolve/main/<file>.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${DEXDUET_CHECKPOINTS:-$REPO/checkpoints}"
SUMS="$REPO/CHECKPOINTS.sha256"
PACKAGES=("$@")
[ ${#PACKAGES[@]} -gt 0 ] || PACKAGES=(dexduet_dexmachina_suite.zip dexduet_ablations.zip)

DEXDUET_CKPT_REPO="${DEXDUET_CKPT_REPO:-oldTOM/dexduet-checkpoints}"
mkdir -p "$DEST/downloads"

fetch() {   # file -> $DEST/downloads/file; returns 1 if the download fails
  local file=$1 out="$DEST/downloads/$1"
  if [ -s "$out" ]; then echo "have $out"; return 0; fi
  if command -v hf > /dev/null 2>&1; then
    hf download "$DEXDUET_CKPT_REPO" "$file" --repo-type model --local-dir "$DEST/downloads" || return 1
  elif command -v huggingface-cli > /dev/null 2>&1; then
    huggingface-cli download "$DEXDUET_CKPT_REPO" "$file" --repo-type model --local-dir "$DEST/downloads" || return 1
  else
    curl -L --fail --retry 3 --continue-at - -o "$out.part" "https://huggingface.co/$DEXDUET_CKPT_REPO/resolve/main/$file" \
      || { rm -f "$out.part"; return 1; }
    mv "$out.part" "$out"
  fi
  [ -s "$out" ]
}

verify() {   # file: check its sha256 against CHECKPOINTS.sha256
  local file=$1 expected actual
  expected=$(awk -v f="$file" '$1 !~ /^#/ && $2 == f {print $1}' "$SUMS" 2>/dev/null | head -1)
  actual=$(sha256sum "$DEST/downloads/$file" | awk '{print $1}')
  if [ -z "$expected" ]; then
    if [ "${DEXDUET_ALLOW_UNVERIFIED:-0}" = 1 ]; then
      echo "WARNING: CHECKPOINTS.sha256 has no entry for $file; unpacking unverified (sha256 $actual)" >&2
      return 0
    fi
    echo "CHECKPOINTS.sha256 has no entry for $file (sha256 of the download: $actual)." >&2
    echo "Update the repository (git pull) or, if you trust the source, run again with DEXDUET_ALLOW_UNVERIFIED=1." >&2
    exit 1
  fi
  if [ "$expected" != "$actual" ]; then
    echo "sha256 mismatch for $file: expected $expected, got $actual; delete $DEST/downloads/$file and retry" >&2
    exit 1
  fi
  echo "sha256 OK: $file"
}

for file in "${PACKAGES[@]}"; do
  case "$file" in
    dexduet_dexmachina_suite.zip|dexduet_ablations.zip) ;;
    *) echo "unknown package $file (dexduet_dexmachina_suite.zip, dexduet_ablations.zip)" >&2
       exit 2 ;;
  esac
  fetch "$file" || { echo "download of $file failed" >&2; exit 1; }
  verify "$file"
  unzip -q -n "$DEST/downloads/$file" -d "$DEST"
  echo "unpacked $file into $DEST"
done
