#!/usr/bin/env bash
# Optional: Gemma 4 31B NVFP4 on SGLang (0.5.21) with slot adapters, for one-request slot reads
# (slotread.client.SlotModel.read_sgl). On our box it read no faster than vLLM's parallel requests.
#
#   pip install "sglang[all]"; python serving/sglang/patch_sglang.py         # in SGLang's environment
#   python serving/sglang/convert_adapter.py runs/j_E3/adapter runs/j_E3/adapter_sgl
#   bash serving/sglang/serve_sglang.sh j-E3=runs/j_E3/adapter_sgl
#
# Image budget: SGLang takes it from the model folder's processor_config.json (max_soft_tokens), not
# per request; serve from a copy of the folder with that value edited for 560 / 1,120 tokens.
exec python -m sglang.launch_server --model-path "${MODEL:-nvidia/Gemma-4-31B-IT-NVFP4}" \
  --host 127.0.0.1 --port "${PORT:-30000}" --mem-fraction-static "${MEM:-0.8}" --context-length 8192 \
  --enable-lora --max-lora-rank 16 --lora-paths "$@"
