#!/usr/bin/env bash
# Download a pushed adapter into the run layout the readers expect: runs/<name>/{config.json, adapter/}.
#   bash scripts/get_adapter.sh infinitylogesh/gemma4-31b-j-E3-s0-lora runs/j_E3
set -euo pipefail
repo=$1; out=$2
hf download "$repo" --exclude "checkpoints/*" --local-dir "$out/adapter"
mv "$out/adapter/config.json" "$out/config.json"
echo "$out ready: serve $out/adapter, read with name=$out"
