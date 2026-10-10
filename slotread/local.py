"""Reading slots in-process with transformers: no server. One forward pass computes every slot of a layout
(only the logits at the slot positions, logits_to_keep), exactly as training reads them. Used for the merged
models and the Hugging Face Space (ZeroGPU runs PyTorch, not vLLM).

  from slotread.local import LocalSlotModel
  m = LocalSlotModel.from_pretrained("infinitylogesh/GUI-Decisions-31B")          # merged model
  m = LocalSlotModel.from_pretrained("google/gemma-4-31B-it", adapter="infinitylogesh/GUI-Decisions-31B-LoRA")
  pred = m.decide(Image.open("shot.png"), "open the settings")    # {'action': 'click', 'x': ..., 'y': ...}

The slot template and label ids come from slotread_config.json in the model (or adapter) repo: the
config.json that slotread.train writes, renamed so it does not collide with the model's own config.json.
"""

import json, time

import torch

from . import labels as L
from .client import agent_user, load_config
from .images import cap_image
from .labels import Arm
from .tasks import AgentTask, BoxTask, SegTask

class LocalSlotModel:
    def __init__(self, model, processor, cfg):
        self.model, self.proc, self.cfg = model, processor, cfg
        self.arm = Arm(cfg["arm"], cfg["coarse"], cfg["bins"], cfg["tokens"], cfg["fine"])
        self.tasks = {"agent": AgentTask(), "box": BoxTask(), "seg": SegTask(cfg.get("rays", 24))}
        L.PH = cfg["placeholder"]  # the placeholder id of this model's tokenizer

    @classmethod
    def from_pretrained(cls, model_id, adapter=None, processor=None, config=None, dtype=torch.bfloat16, device_map=None):
        """config: a slotread config file; default: slotread_config.json in the adapter or model repo."""
        from transformers import AutoModelForImageTextToText, AutoProcessor
        model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=dtype, device_map=device_map or {"": 0},
                                                            attn_implementation="sdpa")
        if adapter:
            from peft import PeftModel
            # on the model's device: under ZeroGPU CUDA looks available before a GPU is attached
            kw = {"torch_device": "cpu"} if device_map == "cpu" else {}
            model = PeftModel.from_pretrained(model, adapter, **kw)
        model.eval()
        cfg = json.load(open(config)) if config else load_config(adapter or model_id)
        return cls(model, AutoProcessor.from_pretrained(processor or model_id), cfg)

    @torch.no_grad()
    def read(self, task, image, user):
        """{slot: log-probabilities over its labels} for the task's whole layout, one forward pass;
        also the forward's milliseconds."""
        t = self.tasks[task]
        c = self.cfg["tasks"][task]
        image = cap_image(image.convert("RGB"), self.cfg.get("image_cap") or 0)  # as trained (GUI-Owl: 1 MP)
        ex = {"agent": dict(image=image, user=user), "box": dict(image=image, instruction=user),
              "seg": dict(image=image, sentence=user)}[task]
        msgs = [{"role": "system", "content": [{"type": "text", "text": c["system"]}]},
                {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": t.user(ex)}]},
                {"role": "assistant", "content": [{"type": "text", "text": " ".join(f"{s}: ___" for s in c["slots"])}]}]
        enc = self.proc.apply_chat_template(msgs, tokenize=True, return_dict=True, return_tensors="pt",
                                           add_generation_prompt=False, continue_final_message=True, enable_thinking=False)
        ids = enc["input_ids"][0]
        keep = (ids == self.cfg["placeholder"]).nonzero().flatten()[-len(c["slots"]):] - 1
        dev = next(self.model.parameters()).device
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = self.model(**{k: v.to(dev) for k, v in enc.items()}, logits_to_keep=keep.to(dev), use_cache=False)
        lv = torch.log_softmax(out.logits[0].float(), -1)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        return {s: lv[i, c["labels"][s]].cpu() for i, s in enumerate(c["slots"])}, ms

    def decide(self, image, instruction, previous="None"):
        """A computer-use step from one forward pass: action and point, plus the swipe end, key and scroll
        slots (meaningful only for their actions); `ms` is the forward pass."""
        from .data.aguvis import KEYS, SDIR
        lp, ms = self.read("agent", image, agent_user(instruction, previous))
        d = self.tasks["agent"].decode(lp, self.arm)
        d.update(key=KEYS[int(lp["key"].argmax())], sdir=SDIR[int(lp["sdir"].argmax())], samt=int(lp["samt"].argmax()))
        return dict(d, ms=round(ms, 1))

    def box(self, image, description):
        lp, ms = self.read("box", image, description)
        return dict(box=self.tasks["box"].decode(lp, self.arm), ms=round(ms, 1))

    @torch.no_grad()
    def write_text(self, image, instruction, action, previous="None", max_new_tokens=24):
        """The text a type / answer / open step needs, written by the base model: the adapter is switched
        off, so this needs base + adapter (a merged model cannot switch it off)."""
        from .agent import TEXT_ASK
        if not hasattr(self.model, "disable_adapter"):
            raise RuntimeError("text steps need the base model: load with adapter=...")
        image = cap_image(image.convert("RGB"), self.cfg.get("image_cap") or 0)
        msgs = [{"role": "user", "content": [{"type": "image", "image": image},
                                             {"type": "text", "text": agent_user(instruction, previous) + "\n\n" + TEXT_ASK[action]}]}]
        enc = self.proc.apply_chat_template(msgs, tokenize=True, return_dict=True, return_tensors="pt",
                                           add_generation_prompt=True, enable_thinking=False)
        dev = next(self.model.parameters()).device
        enc = {k: v.to(dev) for k, v in enc.items()}
        with self.model.disable_adapter():
            out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        t = self.proc.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return t.strip().strip('"').split("\n")[0][:120]

    @torch.no_grad()
    def vanilla_decide(self, image, instruction, previous="None", max_new_tokens=96):
        """The base model (adapter off) writing the step as JSON in Gemma 4's native [y, x] format, as in
        the post's comparisons: (action, point, args, text, raw output, ms for the whole generation)."""
        from .agent import json_system, parse_step
        if not hasattr(self.model, "disable_adapter"):
            raise RuntimeError("the vanilla comparison needs the base model: load with adapter=...")
        image = cap_image(image.convert("RGB"), self.cfg.get("image_cap") or 0)
        msgs = [{"role": "system", "content": [{"type": "text", "text": json_system(True)}]},
                {"role": "user", "content": [{"type": "image", "image": image},
                                             {"type": "text", "text": agent_user(instruction, previous)}]}]
        enc = self.proc.apply_chat_template(msgs, tokenize=True, return_dict=True, return_tensors="pt",
                                           add_generation_prompt=True, enable_thinking=False)
        dev = next(self.model.parameters()).device
        enc = {k: v.to(dev) for k, v in enc.items()}
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with self.model.disable_adapter():
            out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        txt = self.proc.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return parse_step(txt, True) + (txt, round(ms, 1))
