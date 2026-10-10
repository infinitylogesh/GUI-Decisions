#!/usr/bin/env bash
# Raise vLLM's per-request logprob_token_ids cap (MAX_LOGPROB_TOKEN_IDS) from 128 to 256, so a
# 256-bin slot is read in one request instead of two. The constant validates requests and sizes a
# [max_num_reqs x cap] int32 buffer (~256 KB at 256); the fill kernel pads to the next power of two
# of the ids actually requested, so nothing else changes. Idempotent; restart vLLM afterwards and
# read with MAX_LABEL_IDS=256.
#   bash serving/patch_vllm.sh [venv]        (default: the active python's vLLM)
set -euo pipefail
PY=${1:+$1/bin/python}
PY=${PY:-python}
F=$("$PY" -c "import vllm.sampling_params as s; print(s.__file__)")
CAP=${CAP:-256}
if grep -q "^MAX_LOGPROB_TOKEN_IDS = $CAP$" "$F"; then
  echo "already patched: MAX_LOGPROB_TOKEN_IDS = $CAP ($F)"
else
  sed -i "s/^MAX_LOGPROB_TOKEN_IDS = [0-9]*$/MAX_LOGPROB_TOKEN_IDS = $CAP/" "$F"
  grep -q "^MAX_LOGPROB_TOKEN_IDS = $CAP$" "$F" && echo "patched: MAX_LOGPROB_TOKEN_IDS = $CAP ($F)" \
    || { echo "patch FAILED: constant not found in $F"; exit 1; }
fi
