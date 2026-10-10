"""Merge a slot LoRA into its base model, check the merge, and save a model that carries its own slot config.

  python scripts/merge_adapter.py --base google/gemma-4-31B-it --run runs/j_E3 --out merged/GUI-Decisions-31B

Checks before saving: the merged model's slot reads equal the unmerged (base + LoRA) reads on N screenshots
(argmax per slot, and the largest probability difference). Writes the merged weights (safetensors), the
processor, and slotread_config.json (the run's config.json: system prompts, slot order, label ids).
"""

import argparse, os, shutil

import torch

from slotread.client import CONFIG_NAME, agent_user
from slotread.data.evalsets import screenspot
from slotread.local import LocalSlotModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--run", required=True, help="run folder: config.json and adapter/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--processor", default="nvidia/Gemma-4-31B-IT-NVFP4", help="the processor training used (its chat template)")
    ap.add_argument("--check", type=int, default=40, help="screenshots to compare before and after merging")
    a = ap.parse_args()

    m = LocalSlotModel.from_pretrained(a.base, adapter=os.path.join(a.run, "adapter"), processor=a.processor,
                                       config=os.path.join(a.run, "config.json"))
    cases = screenspot(a.check)
    before = [m.read("agent", c["image"], agent_user(c["instruction"]))[0] for c in cases]
    m.model = m.model.merge_and_unload()
    after = [m.read("agent", c["image"], agent_user(c["instruction"]))[0] for c in cases]
    agree, dmax = {}, 0.0
    for b, f in zip(before, after):
        for s in b:
            agree.setdefault(s, []).append(int(b[s].argmax()) == int(f[s].argmax()))
            dmax = max(dmax, float((torch.softmax(b[s], -1) - torch.softmax(f[s], -1)).abs().max()))
    print("argmax agreement, unmerged vs merged: " + ", ".join(f"{s} {sum(v)}/{len(v)}" for s, v in agree.items()))
    print(f"largest probability difference: {dmax:.4f}", flush=True)

    os.makedirs(a.out, exist_ok=True)
    m.model.save_pretrained(a.out, safe_serialization=True, max_shard_size="5GB")
    m.proc.save_pretrained(a.out)
    shutil.copy(os.path.join(a.run, "config.json"), os.path.join(a.out, CONFIG_NAME))
    print(f"saved {a.out}", flush=True)


if __name__ == "__main__":
    main()
