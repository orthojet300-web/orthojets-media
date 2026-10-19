#!/usr/bin/env bash
# usage: tools/publish_dir.sh PREFIX DIR
# Like publish_github.sh, but for a folder of images (the weekly carousel). Force-pushes DIR/* to the orphan
# `media` branch under PREFIX/ and prints one public raw URL per file (sorted). Metricool copies the files
# to its own storage when the post is created, so overwriting `media` later is safe.
set -euo pipefail
PREFIX="$1"; DIR="$(realpath "$2")"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE="$(git -C "$ROOT" remote get-url origin)"
REPO="$(echo "$REMOTE" | sed -E 's#.*github.com[:/]##; s#\.git$##')"
TMP="$(mktemp -d)"
git -C "$TMP" init -q
git -C "$TMP" checkout -q --orphan media
mkdir -p "$TMP/$PREFIX"; cp "$DIR"/* "$TMP/$PREFIX/"
printf '# Orthojets media\nCurrent upload for scheduling. Overwritten on every publish.\n' > "$TMP/README.md"
git -C "$TMP" add -A
git -C "$TMP" -c user.name="orthojets-bot" -c user.email="orthojet300@gmail.com" commit -q -m "media: $PREFIX"
git -C "$TMP" push -q -f "$REMOTE" media:media 2>/dev/null || git -C "$TMP" push -q -f "$REMOTE" media:media
for f in $(ls "$DIR" | sort); do echo "https://raw.githubusercontent.com/$REPO/media/$PREFIX/$f"; done
