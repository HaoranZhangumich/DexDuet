#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLIPS=(
  "s01 ketchup 01"
  "s01 box 01"
  "s01 mixer 01"
  "s01 ketchup 02"
  "s02 notebook 02"
  "s01 waffleiron 01"
)
for entry in "${CLIPS[@]}"; do
  read -r subject object use <<< "$entry"
  "$REPO_ROOT/scripts/prepare_arctic_clip.sh" \
    --subject "$subject" --object "$object" --use "$use"
done

"$REPO_ROOT/examples/retarget_missing.sh"
"$REPO_ROOT/examples/contact_retarget_missing.sh"
