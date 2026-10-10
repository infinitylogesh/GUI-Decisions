#!/usr/bin/env bash
# Assemble the Space (app + the slotread package) in a build folder and upload it.
#   HF_TOKEN=... bash space/deploy.sh [infinitylogesh/GUI-Decisions]
set -euo pipefail
REPO=${1:-infinitylogesh/GUI-Decisions}
HERE=$(cd "$(dirname "$0")" && pwd); B=$(mktemp -d)
cp -r "$HERE"/{app.py,intro.html,ambient.css,how.md,requirements.txt,README.md,examples} "$B"/
rsync -a --exclude __pycache__ "$HERE/../slotread" "$B"/
hf upload "$REPO" "$B" . --repo-type space --commit-message "${MSG:-update}"
echo "uploaded $B -> https://huggingface.co/spaces/$REPO"
