#!/usr/bin/env bash
# usage: tools/publish_github.sh SLUG FILE
# Publishes FILE on the orphan branch `media` (which only ever holds the current episode's video,
# force-pushed each time so the repo doesn't grow) and prints its public raw URL for Metricool.
# (GitHub Releases are not permitted for Claude sessions, so a branch is used instead.)
set -euo pipefail
SLUG="$1"; FILE="$(realpath "$2")"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE="$(git -C "$ROOT" remote get-url origin)"
REPO="$(echo "$REMOTE" | sed -E 's#.*github.com[:/]##; s#\.git$##')"
TMP="$(mktemp -d)"
git -C "$TMP" init -q
git -C "$TMP" checkout -q --orphan media
cp "$FILE" "$TMP/$SLUG.mp4"
printf '# Orthojets media\nCurrent episode video for scheduling. Overwritten on every publish.\n' > "$TMP/README.md"
git -C "$TMP" add -A
git -C "$TMP" -c user.name="orthojets-bot" -c user.email="orthojet300@gmail.com" commit -q -m "media: $SLUG"
git -C "$TMP" push -q -f "$REMOTE" media:media
echo "https://raw.githubusercontent.com/$REPO/media/$SLUG.mp4"
