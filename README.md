# GUI-Decisions

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-md.svg)](https://huggingface.co/spaces/infinitylogesh/GUI-Decisions)
[![GUI-Decisions-31B-LoRA](https://img.shields.io/badge/%F0%9F%A4%97%20Model-GUI--Decisions--31B--LoRA-ff5a14)](https://huggingface.co/infinitylogesh/GUI-Decisions-31B-LoRA)
[![GUI-Decisions-31B](https://img.shields.io/badge/%F0%9F%A4%97%20Model-GUI--Decisions--31B-ff5a14)](https://huggingface.co/infinitylogesh/GUI-Decisions-31B)
[![GUI-Decisions-2B-LoRA](https://img.shields.io/badge/%F0%9F%A4%97%20Model-GUI--Decisions--2B--LoRA-3a2013)](https://huggingface.co/infinitylogesh/GUI-Decisions-2B-LoRA)
[![GUI-Decisions-2B](https://img.shields.io/badge/%F0%9F%A4%97%20Model-GUI--Decisions--2B-3a2013)](https://huggingface.co/infinitylogesh/GUI-Decisions-2B)

[![Slot read vs writing JSON: the same model on the same Wikipedia task, 4 decisions in 0.63 s against 2.08 s](assets/hero.gif)](https://huggingface.co/spaces/infinitylogesh/GUI-Decisions)

*Same model, same screenshots: reading slots (left, 158 ms per decision) against writing the step as JSON
(right, 520 ms). **[Try it in the Space](https://huggingface.co/spaces/infinitylogesh/GUI-Decisions)** with your own screenshots.*

Code for training a System-one decision model for GUI grounding tasks. This improves the latency of GUI grounding step by 3x. It uses a fixed template of slots to predict the next-token distribution for each slot, and a small LoRA to teach the model to put its probability on the right labels.

Code is associated with the blog post **[GUI-Decisions - Stop Decoding Coordinates for 3x Faster Computer-Use Grounding](https://huggingface.co/blog/infinitylogesh/gui-decisions-stop-decoding-coordinates)**.

A computer-use step needs an action and usually a point on the screen. Instead of having the model *write*
`{"action": "click", "point": [412, 288]}` token by token, the assistant turn is a fixed template of
**slots**, each ending in a placeholder:

```
action: ___ x: ___ y: ___ x2: ___ y2: ___ key: ___ sdir: ___ samt: ___
```

Each slot's answer is the model's next-token distribution at the token before its placeholder, restricted
to that slot's labels: action words for `action`, and **256 bins** for each coordinate, every bin a single
byte-like token (`!`, `"`, … `ab`, … `gs`). Nothing is decoded. A small LoRA teaches the model to put its
probability on the right labels (cross-entropy against soft ordinal targets), and a read takes ~146 ms per
step on Gemma 4 31B against ~470–550 ms for the same model writing the step in its native JSON format.

The zero-shot and categorical decision models of section 1 of the post live in
[SystemOne](https://github.com/infinitylogesh/systemone). This repository covers sections 2–3, the latency
work, the small-model results and the appendix (boxes, masks, SAM 2).

## Quickstart: serve with vLLM

| model | what | vLLM step, median (60 cold screenshots, one RTX PRO 6000) |
|---|---|---|
| [GUI-Decisions-31B-LoRA](https://huggingface.co/infinitylogesh/GUI-Decisions-31B-LoRA) | the E3 adapter for Gemma 4 31B: computer use, boxes, masks | **152 ms** on the NVFP4 base |
| [GUI-Decisions-31B](https://huggingface.co/infinitylogesh/GUI-Decisions-31B) | the same, merged (bf16) | 190 ms |
| [GUI-Decisions-2B-LoRA](https://huggingface.co/infinitylogesh/GUI-Decisions-2B-LoRA) | computer use on GUI-Owl-1.5-2B | 87 ms |
| [GUI-Decisions-2B](https://huggingface.co/infinitylogesh/GUI-Decisions-2B) | the same, merged (bf16) | 83 ms |

Try it in the browser first: [GUI-Decisions Space](https://huggingface.co/spaces/infinitylogesh/GUI-Decisions).

**1. Install** vLLM 0.30 and this package:

```bash
git clone https://github.com/infinitylogesh/gui-decisions && cd gui-decisions
pip install "vllm==0.30.*" -e .
bash serving/patch_vllm.sh        # 256 label ids per request (else read with MAX_LABEL_IDS=128)
```

**2. Serve** the NVFP4 base with the adapter; the base model stays available as `gemma4` and writes the text
of type / answer steps:

```bash
bash serving/serve_vllm.sh gui-decisions=infinitylogesh/GUI-Decisions-31B-LoRA
```

which runs

```bash
vllm serve nvidia/Gemma-4-31B-IT-NVFP4 --served-model-name gemma4 --host 127.0.0.1 --port 8001 \
  --enable-lora --max-lora-rank 16 --max-loras 1 --lora-modules gui-decisions=infinitylogesh/GUI-Decisions-31B-LoRA \
  --max-model-len 8192 --gpu-memory-utilization 0.85 --max-logprobs 300 --enable-prefix-caching \
  --mm-processor-cache-gb 0 --api-server-count 8
```

The merged model: `MODEL=infinitylogesh/GUI-Decisions-31B NAME=gui-decisions bash serving/serve_vllm.sh`.
The 2B model: `PORT=8001 bash serving/serve_gui_owl.sh gui-decisions-2b=infinitylogesh/GUI-Decisions-2B-LoRA`,
or merged: `MODEL=infinitylogesh/GUI-Decisions-2B NAME=gui-decisions-2b MAX_LEN=16384 GPU_UTIL=0.6 EXTRA_ARGS='--limit-mm-per-prompt {"image":1}' bash serving/serve_vllm.sh`.
The first start compiles FlashInfer's FP4 kernels for your GPU (a few minutes, then cached); if it dies with
exit code 137 the compiler ran out of host memory, so start again with `MAX_JOBS=4`.

**3. Read** steps and boxes:

```python
from PIL import Image
from slotread.client import SlotModel

m = SlotModel("gui-decisions", "infinitylogesh/GUI-Decisions-31B-LoRA", text_model="gemma4")   # VLLM_URL, default http://127.0.0.1:8001
shot = Image.open("screenshot.png")
step = m.decide(shot, "turn on dark mode")          # {'action': 'click', 'x': 0.921, 'y': 0.247, 'ms': 152.0}
if step["action"] in ("type", "answer", "open"):
    step["text"] = m.write_text(shot, "turn on dark mode", step["action"])   # the base model writes it
box = m.box(Image.open("photo.jpg"), "the red frisbee")   # {'box': [x1, y1, x2, y2], 'ms': ...}, in [0, 1]
```

`decide` reads `action`, `x` and `y` first, then only the slots the action needs (`key`, `sdir` / `samt`,
or `x2` / `y2`). `x` and `y` are fractions of the screenshot; `previous` takes earlier steps as numbered
lines. `m.read(task, url, user)` returns the raw slot distributions. See [the read path](#the-read-path)
for what a read is and why these flags.


## Setup

One GPU box (the post used one RTX PRO 6000, 96 GB), Python 3.12, [uv](https://github.com/astral-sh/uv):

```bash
bash recipes/setup.sh                 # .venv-train (training, clients) and .venv-serve (vLLM 0.30, patched)
bash recipes/fetch_refcoco.sh         # ~20k COCO images RefCOCO uses -> data/coco_ref
bash recipes/fetch_coco17.sh          # COCO val2017 -> data/coco17 (outside-RefCOCO tests)
export COCO_DIR=$PWD/data/coco_ref COCO17_DIR=$PWD/data/coco17
```

Training needs the bf16 weights (`google/gemma-4-31B-it`, accept the license on Hugging Face); serving uses
`nvidia/Gemma-4-31B-IT-NVFP4`. Datasets are cached under `~/.cache/slotread` (`SLOTREAD_CACHE`).

## Use your own run

```bash
bash scripts/get_adapter.sh <hf-repo> runs/j_E3          # runs/j_E3/{config.json, adapter/}
source .venv-serve/bin/activate && bash serving/serve_vllm.sh j-E3=runs/j_E3/adapter &
```

```python
from PIL import Image
from slotread.client import SlotModel, agent_user
from slotread.images import data_url

m = SlotModel("j-E3", "runs/j_E3", text_model="gemma4")   # name as served; run folder, released folder or HF repo id
print(m.decide(Image.open("screenshot.png"), "open the settings"))   # {'action': 'click', 'x': 0.91, 'y': 0.06, 'ms': ...}
lp = m.read("agent", data_url(Image.open("screenshot.png")), agent_user("open the settings"))   # raw distributions
```

`config.json` (`slotread_config.json` in the released models) carries each task's system prompt, slot order
and label ids, so a reader needs nothing else.

Without a server (transformers, one forward pass reads every slot; what the Space uses):

```python
from slotread.local import LocalSlotModel

m = LocalSlotModel.from_pretrained("infinitylogesh/GUI-Decisions-31B")      # merged E3
# or the LoRA on the base model, which can also write text steps with the adapter off:
m = LocalSlotModel.from_pretrained("google/gemma-4-31B-it", adapter="infinitylogesh/GUI-Decisions-31B-LoRA",
                                   processor="infinitylogesh/GUI-Decisions-31B-LoRA")
step = m.decide(Image.open("screenshot.png"), "open the settings")
```

`scripts/merge_adapter.py` merges a run into its base model (checking merged against unmerged reads), and
`scripts/check_local.py` evaluates an in-process model on the post's sets, paired case by case with a
published evaluation. `space/` is the Hugging Face Space (`space/deploy.sh`).
The browser agent: `python -m slotread.agent --model j-E3=runs/j_E3 --url https://en.wikipedia.org --task "..."`.

## The read path

Each slot is one chat request whose assistant turn is prefilled up to that slot (`action: ___ x:`), with
`continue_final_message: true`, `max_tokens: 1` and `logprob_token_ids` set to the slot's label ids from
the config: vLLM returns the log-probabilities of exactly those labels and generates nothing else. A
step's requests run in parallel and share the screenshot through the prefix cache. The answer is the
softmax over the labels: the argmax for words, and for a coordinate the argmax bin refined by its ±2
neighbours (bin *k* of 256 is (*k* + 0.5) / 256 of the width or height).

Three serving changes took a step from ~240 ms to ~146 ms:

- **256 label ids per request.** vLLM 0.30 caps `logprob_token_ids` at 128, so a 256-bin slot took two
  requests; `serving/patch_vllm.sh` raises the cap (`MAX_LABEL_IDS=256`, the client's default: set it to
  128 on an unpatched server).
- **`--api-server-count 8`.** A single API-server process serialized the parallel slot requests.
- **JPEG at the model's input size** (`IMG_FMT=jpeg`, `IMG_MAX_PX=802816`, the serving default) instead of
  full-size PNG (see the note on image transport below).

One request per step (`read_one`, or SGLang's `read_sgl`) is possible but was not faster here: vLLM batches
the parallel slot requests well, and SGLang's NVFP4 + LoRA prefill was slower (see `serving/sglang/`).

## Citation

```bibtex
@article{umapathi2026slotreading,
  title   = "GUI-Decisions - Stop Decoding Coordinates for 3x Faster Computer-Use Grounding",
  author  = "Umapathi, Logesh Kumar",
  journal = "logeshumapathi.com",
  year    = "2026",
  month   = "Oct",
  url     = "https://huggingface.co/blog/infinitylogesh/gui-decisions-stop-decoding-coordinates"
}
```
