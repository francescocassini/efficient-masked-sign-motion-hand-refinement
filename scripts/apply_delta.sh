#!/usr/bin/env bash
set -euo pipefail

EXPECTED_SOKE_COMMIT="5cbc55d84b5a7cbf05a9cf020c468052e8d94d00"

usage() {
  echo "Usage: bash scripts/apply_delta.sh /path/to/SOKE" >&2
}

if [[ $# -ne 1 ]]; then
  usage
  exit 2
fi

TARGET="$1"
DELTA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -d "$TARGET/.git" ]]; then
  echo "[error] Target is not a Git checkout: $TARGET" >&2
  exit 1
fi

current_commit="$(git -C "$TARGET" rev-parse HEAD)"
if [[ "$current_commit" != "$EXPECTED_SOKE_COMMIT" ]]; then
  echo "[warn] Target commit is $current_commit" >&2
  echo "[warn] Expected SOKE commit is $EXPECTED_SOKE_COMMIT" >&2
  echo "[warn] The patch may still apply, but the pinned commit is recommended." >&2
fi

if ! git -C "$TARGET" diff --quiet; then
  echo "[error] Target checkout has uncommitted changes. Commit or stash them first." >&2
  exit 1
fi

echo "[delta] Applying SOKE integration patch"
git -C "$TARGET" apply --check --whitespace=nowarn "$DELTA_ROOT/patches/soke-integration.patch"
git -C "$TARGET" apply --whitespace=nowarn "$DELTA_ROOT/patches/soke-integration.patch"

echo "[delta] Copying overlay files"
cp -a "$DELTA_ROOT/overlay/." "$TARGET/"

echo "[delta] Done"
echo "[delta] Review with: git -C \"$TARGET\" status --short"
