#!/usr/bin/env bash
# GUI-Owl-1.5-2B-Instruct (Qwen3-VL) with its slot adapter: the base model as `owl` (vanilla GUI-Owl,
# scripts/gui_owl.py) and the adapter beside it, on one server.
#   bash serving/serve_gui_owl.sh owl-E=runs/owl_E/adapter
MODEL=mPLUG/GUI-Owl-1.5-2B-Instruct NAME=owl PORT=${PORT:-8002} GPU_UTIL=${GPU_UTIL:-0.6} MAX_LEN=16384 \
  EXTRA_ARGS="--limit-mm-per-prompt {\"image\":1}" exec bash "$(dirname "$0")/serve_vllm.sh" "$@"
