#!/usr/bin/env bash
# Serve the base model with slot adapters as LoRAs (vLLM 0.30). The base model is served as `gemma4`
# (the vanilla baseline, and the writer of text steps); each adapter under its own name.
#
#   bash serving/patch_vllm.sh                       # once: 256 label ids per request
#   bash serving/serve_vllm.sh j-E=runs/j_E/adapter j-E3=runs/j_E3/adapter
#
# Read-path settings (the blog's "other changes that made it read fast"):
#   --api-server-count 8      a single API-server process serializes the parallel per-slot requests
#   --max-logprobs 300        read_one's prompt log-probabilities
#   --mm-processor-cache-gb 0 concurrent same-image requests raced in the media cache on vLLM 0.30
#   no --async-scheduling     on vLLM 0.30 it crashed in _fill_logprob_token_ids_kernel (ASYNC=1 restores it)
# Env: MODEL (default nvidia/Gemma-4-31B-IT-NVFP4), NAME (gemma4), PORT (8001), GPU_UTIL (0.85),
#      API_SERVERS (8), MAX_LEN (8192), EXTRA_ARGS (more vllm serve flags).
set -euo pipefail
ARGS=()
[ $# -gt 0 ] && ARGS=(--enable-lora --max-lora-rank 16 --max-loras $# --lora-modules "$@")
exec vllm serve "${MODEL:-nvidia/Gemma-4-31B-IT-NVFP4}" --served-model-name "${NAME:-gemma4}" \
  --host 127.0.0.1 --port "${PORT:-8001}" --max-model-len "${MAX_LEN:-8192}" \
  --gpu-memory-utilization "${GPU_UTIL:-0.85}" --max-logprobs 300 --enable-prefix-caching \
  --mm-processor-cache-gb 0 --api-server-count "${API_SERVERS:-8}" \
  $([ "${ASYNC:-0}" = 1 ] && echo --async-scheduling) ${EXTRA_ARGS:-} "${ARGS[@]}"
