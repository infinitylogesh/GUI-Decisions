"""Evaluate an in-process (transformers) slot model on the post's sets with slotread.evaluate's cases and
scoring, and compare it case by case with a published evaluation of the served adapter.

  python scripts/check_local.py click merged/GUI-Decisions-31B --against results/p3_click.json:j-E3
  python scripts/check_local.py agent_test google/gemma-4-31B-it --adapter runs/j_E3/adapter --config runs/j_E3/config.json

Images go through the same full-size PNG data URL as the served evaluation (lossless), so the only
differences are the weights (e.g. merged bf16 against NVFP4 + LoRA) and the runtime.
"""
import argparse, base64, io, json, os, threading, time

from PIL import Image

from slotread import evaluate as E
from slotread.local import LocalSlotModel


class Local:
    """slotread.client.SlotModel's read interface on a LocalSlotModel (one forward reads every slot)."""

    def __init__(self, name, m):
        self.name, self.path, self.m, self.arm, self.tasks = name, name, m, m.arm, m.tasks
        self.lock, self.ms = threading.Lock(), []

    def read(self, task, url, user, slots=None, lazy=False):
        im = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB")
        with self.lock:
            lp, ms = self.m.read(task, im, user)
        self.ms.append(ms)
        return {s: v for s, v in lp.items() if slots is None or s in slots}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which")
    ap.add_argument("model")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--adapter")
    ap.add_argument("--config")
    ap.add_argument("--processor")
    ap.add_argument("--name", default="local")
    ap.add_argument("--against", help="published eval JSON:arm name, compared case by case")
    a = ap.parse_args()
    model = Local(a.name, LocalSlotModel.from_pretrained(a.model, adapter=a.adapter, processor=a.processor, config=a.config))
    cs = E.cases(a.which, a.n)
    print(f"{a.which}: {len(cs)} cases", flush=True)
    rows, t = [], time.time()
    for i, c in enumerate(cs):
        rows.append(dict(id=i, kind=c["kind"], group=list(c["ex"].get("group", [])), arms={a.name: E.evaluate_case(model, c)}))
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(cs)} ({time.time() - t:.0f}s)", flush=True)
    names = [a.name]
    if a.against:
        path, arm = a.against.rsplit(":", 1)
        pub = {r["id"]: r for r in json.load(open(path))["rows"]}
        assert all(r["id"] in pub and pub[r["id"]]["group"] == r["group"] for r in rows), "cases differ from the published run"
        for r in rows:
            r["arms"]["published"] = pub[r["id"]]["arms"][arm]
        names.append("published")
    E.summarize(a.which, rows, names)
    ms = sorted(model.ms)
    print(f"\nforward pass: median {ms[len(ms) // 2]:.0f} ms over {len(ms)} reads", flush=True)
    os.makedirs(E.OUT, exist_ok=True)
    out = f"{E.OUT}/local_{a.name}_{a.which}_{a.n}.json"
    json.dump(dict(which=a.which, n=len(rows), arms={a.name: a.model}, rows=rows), open(out, "w"))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
