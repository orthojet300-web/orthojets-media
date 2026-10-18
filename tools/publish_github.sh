#!/usr/bin/env bash
# usage: tools/publish_github.sh TAG FILE "Title"
# Creates (or reuses) a GitHub Release TAG in this repo, uploads FILE, prints the public download URL.
set -euo pipefail
TAG="$1"; FILE="$2"; TITLE="${3:-$1}"
cd "$(dirname "$0")/.."
REPO="$(git remote get-url origin | sed -E 's#.*github.com[:/]##; s#\.git$##')"
if ! gh release view "$TAG" --repo "$REPO" >/dev/null 2>&1; then
  gh release create "$TAG" --repo "$REPO" --title "$TITLE" --notes "Orthojets episode $TAG" >/dev/null
fi
gh release upload "$TAG" "$FILE" --repo "$REPO" --clobber >/dev/null
URL="https://github.com/$REPO/releases/download/$TAG/$(basename "$FILE")"
# confirm it is publicly downloadable
code=$(curl -sS -L -o /dev/null -w '%{http_code}' -r 0-1023 "$URL" || true)
echo "$URL"
[ "$code" = "206" ] || [ "$code" = "200" ] || { echo "WARNING: download check returned $code" >&2; exit 1; }
