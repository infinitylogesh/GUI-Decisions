"""Push a run's adapter, config.json and checkpoints to a (private) Hugging Face repo.

  HF_TOKEN=... python scripts/push_adapter.py runs/j_E3 <user>/gemma4-31b-j-E3-lora
"""
import json, os, sys

from huggingface_hub import HfApi

run, repo = sys.argv[1], sys.argv[2]
cfg = json.load(open(os.path.join(run, "config.json")))
base = cfg.get("model", "google/gemma-4-31B-it")
api = HfApi()
api.create_repo(repo, private=True, exist_ok=True)
api.upload_folder(repo_id=repo, folder_path=os.path.join(run, "adapter"))
api.upload_file(repo_id=repo, path_or_fileobj=os.path.join(run, "config.json"), path_in_repo="config.json")
card = (f"---\nbase_model: {base}\nlibrary_name: peft\n---\n# {os.path.basename(run)}\n\n"
        f"A slot-reading LoRA (arm {cfg['arm']}: {cfg['bins']} bins, {cfg['tokens']} labels, coarse slot: {cfg['coarse']}) "
        f"for {', '.join(cfg['tasks'])}. `config.json` holds each task's system prompt, slot order and label ids; "
        "read it with slotread.client.SlotModel.\n")
api.upload_file(repo_id=repo, path_or_fileobj=card.encode(), path_in_repo="README.md")
for d in sorted(d for d in os.listdir(run) if d.startswith("adapter-")):
    api.upload_folder(repo_id=repo, folder_path=os.path.join(run, d), path_in_repo=f"checkpoints/{d}")
print(f"pushed {run} -> https://huggingface.co/{repo}")
