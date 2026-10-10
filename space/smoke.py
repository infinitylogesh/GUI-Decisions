"""Run the Space's functions on every example without the UI (on a GPU box, before deploying):
  BASE=google/gemma-4-31B-it ADAPTER=/path/to/GUI-Decisions-31B-LoRA python space/smoke.py out_dir
"""
import os, sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import app  # noqa: E402  (loads the model)

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
for k, (f, ins, prev) in enumerate(app.rows("agent.tsv")):
    im, md, lab = app.decide(Image.open(app.ex(f)), ins, prev.replace("\\n", "\n"), True)
    im.save(f"{out}/agent_{k}.jpg")
    print(f"--- {f} | {ins}\n{md}\n{sorted(lab.items(), key=lambda x: -x[1])[:3]}\n", flush=True)
for k, (f, t) in enumerate(app.rows("box.tsv")):
    im, md = app.box(Image.open(app.ex(f)), t)
    im.save(f"{out}/box_{k}.jpg")
    print(f"--- {f} | {t}\n{md}\n", flush=True)
