"""Controlled latency of one computer-use step: slot adapters against the base model writing the step.

Fair by construction:
  - cold reads: every condition sees each screenshot once, and each screenshot is used by one
    condition only (vLLM's prefix cache is per adapter, and vanilla variants share the base model's), as
    in a real step where every screenshot is new
  - the read mode rotates across screenshots and the adapter order rotates within each
  - one request at a time, nothing else on the server

Slot reads, per adapter:
  slots/128    one request per slot, label ids split at stock vLLM's 128 cap (two requests per 256-bin slot)
  slots/256    the same with serving/patch_vllm.sh's 256 cap (one request per slot)
  one_request  every slot from one request's prompt log-probabilities (top 20)
Vanilla (the base model, served as gemma4):
  json_xy      the step as JSON with x, y as fractions of the screen (~30 tokens; the blog's "pixel JSON" row)
  json_yx      the step as JSON with "point": [y, x] on 0-1000, Gemma 4's own convention (~20 tokens)

  python scripts/latency_bench.py 80 j-E=runs/j_E j-A=runs/j_A j-F=runs/j_F j-Bplus=runs/j_Bplus
"""

import hashlib, json, os, statistics, sys, time

from slotread import client as C
from slotread.client import SlotModel, agent_user, generate
from slotread.data.evalsets import screenspot
from slotread.images import data_url

SLOT_MODES = ("slots/128", "slots/256", "one_request")
JSON_XY = ('Answer with the next action only, as JSON: {"action": "click", "x": <0-1>, "y": <0-1>} '
           "with x and y as fractions of the screenshot width and height.")
JSON_YX = ('Answer with the next action only, as JSON: {"action": <one of click, type, scroll, swipe, press, open, back, '
           'home, wait, answer, select>, "point": [y, x] normalized to 0-1000 (y first)}.')


def distinct_screenshots(n, skip=20):
    seen, out = set(), []
    for c in screenspot(0)[skip:]:
        h = hashlib.md5(c["image"].resize((64, 64)).tobytes()).hexdigest()
        if h not in seen:
            seen.add(h)
            out.append(c)
        if len(out) >= n:
            break
    return out


def slot_read(m, mode, url, c):
    C.MAX_IDS = 128 if mode == "slots/128" else 256
    t = time.perf_counter()
    if mode == "one_request":
        C.PL_TOPK = 20
        m.read_one("agent", url, agent_user(c["instruction"]))
    else:
        m.read("agent", url, agent_user(c["instruction"]))
    return (time.perf_counter() - t) * 1000, 0


def vanilla(kind, url, c):
    text, ms, usage = generate("gemma4", url, agent_user(c["instruction"]) + "\n\n" + (JSON_XY if kind == "json_xy" else JSON_YX),
                               max_tokens=48)
    return ms, usage.get("completion_tokens", 0)


def main():
    n = int(sys.argv[1])
    models = [SlotModel(*a.split("=", 1)) for a in sys.argv[2:]]
    conds = [(m.name, mode) for m in models for mode in SLOT_MODES] + [("gemma4", "json_xy"), ("gemma4", "json_yx")]
    shots = distinct_screenshots(n * len(conds) + 2)
    for c in shots[:2]:  # warm-up, kept out of the timings
        url = data_url(c["image"])
        for _ in range(4):
            for name, mode in conds:
                (slot_read(next(m for m in models if m.name == name), mode, url, c) if name != "gemma4" else vanilla(mode, url, c))
    rows = []
    for i, c in enumerate(shots[2:]):
        name, mode = conds[i % len(conds)]  # each screenshot: one condition (cold)
        url = data_url(c["image"])
        ms, toks = (slot_read(next(m for m in models if m.name == name), mode, url, c) if name != "gemma4"
                    else vanilla(mode, url, c))
        rows.append(dict(model=name, mode=mode, ms=ms, tokens=toks))
    print("\ncold reads, one request at a time; ms median (p90), n per cell\n")
    print("| model | mode | ms | output tokens | n |\n|---|---|---|---|---|")
    summary = {}
    for name, mode in conds:
        xs = sorted(r["ms"] for r in rows if r["model"] == name and r["mode"] == mode)
        toks = [r["tokens"] for r in rows if r["model"] == name and r["mode"] == mode]
        summary[f"{name}/{mode}"] = dict(median=statistics.median(xs), p90=xs[int(0.9 * (len(xs) - 1))], n=len(xs))
        print(f"| {name} | {mode} | {statistics.median(xs):.0f} ({xs[int(0.9 * (len(xs) - 1))]:.0f}) | "
              f"{statistics.median(toks):.0f} | {len(xs)} |")
    os.makedirs("results", exist_ok=True)
    json.dump(dict(n=n, summary=summary, rows=rows), open(f"results/latency_bench_{n}.json", "w"))


if __name__ == "__main__":
    main()
