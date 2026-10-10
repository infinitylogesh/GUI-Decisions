"""Train one LoRA adapter for slot reading, jointly across tasks.

Every slot's answer is the full-vocabulary log-probability of its labels at the token before its
placeholder; the loss is cross-entropy against soft ordinal targets, normalized per number (a coarse
slot adds to its number's loss rather than diluting it). Only the logits at the slot positions are
computed (logits_to_keep), so a whole layout costs one forward pass.

  # the blog's arms (Gemma 4 31B, all three tasks)
  python -m slotread.train --arm E --out runs/j_E --run-name j-E
  python -m slotread.train --arm A --out runs/j_A        # also F, Bplus (and C, Bplusw)
  # E3: general-object boxes, warm-started from E
  python -m slotread.train --arm E --init runs/j_E/adapter --lr 1e-4 --n-agent 5000 --n-click 1000 \\
      --n-box 4000 --n-box-photo 5000 --n-seg 6000 --out runs/j_E3
  # a smaller model, computer use only
  python -m slotread.train --arm E --model mPLUG/GUI-Owl-1.5-2B-Instruct --processor mPLUG/GUI-Owl-1.5-2B-Instruct \\
      --tasks agent --n-agent 5000 --n-click 2000 --image-cap 1048576 --out runs/owl_E

Writes <out>/config.json (each task's system prompt, slot order and label ids: what a reader needs),
<out>/adapter and <out>/adapter-<n> checkpoints.
"""

import argparse, json, math, os, random, time
from multiprocessing import Pool

import torch

from . import images
from . import labels as L
from .geometry import poly_mask, radial_encode
from .numbers import LOG_MIN
from .tasks import AgentTask, BoxTask, SegTask

LORA_TARGETS = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"


def encode(proc, task, ex, arm):
    """The chat-templated input and the positions whose logits predict each slot (the token before
    each placeholder)."""
    enc = proc.apply_chat_template(task.messages(ex, arm), tokenize=True, return_dict=True, return_tensors="pt",
                                   add_generation_prompt=False, continue_final_message=True, enable_thinking=False)
    ids = enc["input_ids"][0]
    n = len(task.layout(arm))
    pos = (ids == L.PH).nonzero().flatten()[-n:]
    assert len(pos) == n, f"{task.name}: {len(pos)} placeholders for {n} slots"
    return enc, pos - 1


def read(model, enc, keep, slots, labels):
    """Full-vocabulary log-probabilities of each slot's labels (a loss on them also moves mass onto
    the labels; readouts renormalize over the labels)."""
    out = model(**{k: v.to(model.device) for k, v in enc.items()}, logits_to_keep=keep.to(model.device),
                use_cache=False)
    lv = torch.log_softmax(out.logits[0].float(), -1)
    return {s: lv[i, labels[s]] for i, s in enumerate(slots)}


def _radial(args):
    polys, size, K = args
    return radial_encode(poly_mask(polys, *size), K)


def aggregate(name, rows):
    keys = {k for r in rows for k, v in r.items() if v is not None}
    m = {k: sum(r[k] for r in rows if r.get(k) is not None) / max(1, sum(r.get(k) is not None for r in rows))
         for k in keys if k not in ("inter", "union")}
    if name == "seg":
        m["ciou"] = sum(r["inter"] for r in rows) / max(1, sum(r["union"] for r in rows))
        m["miou"] = m.pop("iou")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list(L.ARMS), required=True)
    ap.add_argument("--tasks", default="agent,box,seg")
    ap.add_argument("--n-agent", type=int, default=7000, help="AGUVIS training steps")
    ap.add_argument("--n-click", type=int, default=2000, help="Wave-UI clicks in the agent layout")
    ap.add_argument("--n-box", type=int, default=6000, help="Wave-UI element boxes")
    ap.add_argument("--n-box-photo", type=int, default=0, help="RefCOCO referent boxes (general objects)")
    ap.add_argument("--n-seg", type=int, default=9000, help="RefCOCO masks")
    ap.add_argument("--n-val", type=int, default=100, help="per task, for the in-run evals")
    ap.add_argument("--rays", type=int, default=24)
    ap.add_argument("--init", default="", help="continue this adapter (warm start) instead of a fresh LoRA")
    ap.add_argument("--max-train", type=int, default=0, help="cap the shuffled training list (pilots)")
    ap.add_argument("--model", default="google/gemma-4-31B-it")
    ap.add_argument("--processor", default="nvidia/Gemma-4-31B-IT-NVFP4", help="the served model's processor (template)")
    ap.add_argument("--image-cap", type=int, default=0, help="max pixels per image, as served; 0: as given")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=4000)
    ap.add_argument("--clip", type=float, default=1.0, help="gradient-norm clip (the norm before clipping is logged)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb", default="", help="wandb project (needs WANDB_API_KEY); empty: stdout only")
    ap.add_argument("--run-name", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    random.seed(a.seed); torch.manual_seed(a.seed)

    from transformers import AutoModelForImageTextToText, AutoProcessor
    from peft import LoraConfig, PeftModel, get_peft_model

    proc = AutoProcessor.from_pretrained(a.processor)
    tok = proc.tokenizer
    if "gemma" not in a.model.lower():  # placeholder and digit ids from this tokenizer
        L.configure_tokenizer(tok)
    images.IMAGE_CAP = a.image_cap
    arm = L.make_arm(a.arm, tok)
    make = {"agent": AgentTask, "box": BoxTask, "seg": lambda: SegTask(a.rays)}
    tasks = {n: make[n]() for n in a.tasks.split(",")}
    labels = {n: t.labels(arm, tok) for n, t in tasks.items()}
    for n, t in tasks.items():
        print(f"{n}: {len(t.layout(arm))} slots | {' '.join(t.layout(arm))[:160]}", flush=True)
    json.dump(dict(arm=arm.name, coarse=arm.coarse, bins=arm.bins, tokens=arm.tokens, fine=arm.fine,
                   log_min=LOG_MIN, rays=a.rays, placeholder=L.PH, seed=a.seed, run_name=a.run_name,
                   model=a.model, image_cap=a.image_cap, init=a.init,
                   counts=dict(agent=a.n_agent, click=a.n_click, box=a.n_box, box_photo=a.n_box_photo, seg=a.n_seg),
                   tasks={n: t.config(arm, tok) for n, t in tasks.items()}),
              open(f"{a.out}/config.json", "w"), indent=1)

    t0 = time.time()
    train, val = [], {}
    if "agent" in tasks:
        tr, va = tasks["agent"].examples(a.n_agent, a.n_click, 200)
        train += [("agent", ex) for ex in tr]
        val["agent"] = va[:a.n_val]
    if "box" in tasks:
        tr, va, va_photo = tasks["box"].examples(a.n_box, a.n_val, a.n_box_photo, a.seed)
        train += [("box", ex) for ex in tr]
        val["box"] = va
        if va_photo:
            val["box_photo"] = va_photo
    if "seg" in tasks:
        tr, va = tasks["seg"].examples(a.n_seg, a.n_val, seed=a.seed)
        with Pool(16) as p:  # the radial encodings, once, off the training loop
            for ex, enc in zip(tr + va, p.map(_radial, [(e["polys"], e["size"], a.rays) for e in tr + va], chunksize=16)):
                ex["radial"] = enc
        train += [("seg", ex) for ex in tr]
        val["seg"] = va
    random.Random(a.seed).shuffle(train)
    if a.max_train:
        train = train[:a.max_train]
    counts = {n: sum(t == n for t, _ in train) for n in tasks}
    if a.n_box_photo:
        counts["box_photo"] = sum(t == "box" and ex.get("source") == "refcoco" for t, ex in train)
    print(f"data: {len(train)} train {counts}, val {({n: len(v) for n, v in val.items()})} in {time.time() - t0:.0f}s",
          flush=True)

    model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0},
                                                        attn_implementation="sdpa")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    if a.init:
        model = PeftModel.from_pretrained(model, a.init, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=a.r, lora_alpha=2 * a.r, lora_dropout=0.05, task_type="CAUSAL_LM",
                                                 target_modules=LORA_TARGETS))
    model.print_trainable_parameters()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    steps = math.ceil(len(train) / a.accum)
    sched = torch.optim.lr_scheduler.LambdaLR(  # 30 warm-up steps, then cosine to zero
        opt, lambda s: min(1, (s + 1) / 30) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))

    wb = None
    if a.wandb and os.environ.get("WANDB_API_KEY"):
        import wandb
        wb = wandb.init(project=a.wandb, name=a.run_name or os.path.basename(a.out), dir=a.out,
                        config=dict(vars(a), bins=arm.bins, coarse=arm.coarse, tokens=arm.tokens, counts=counts))
    elif a.wandb:
        print("no WANDB_API_KEY: logging to stdout only", flush=True)

    def evaluate(tag, step):
        model.eval()
        out = {}
        with torch.no_grad():
            for n, exs in val.items():
                tn = "box" if n == "box_photo" else n
                task, slots = tasks[tn], tasks[tn].layout(arm)
                rows = []
                for ex in exs:
                    enc, keep = encode(proc, task, ex, arm)
                    rows.append(task.score(task.decode(read(model, enc, keep, slots, labels[tn]), arm), ex))
                out[n] = aggregate(n, rows)
        model.train()
        show = {"agent": ("step", "action", "point"), "box": ("iou", "iou75", "center_in"),
                "box_photo": ("iou", "iou50", "iou75"), "seg": ("miou", "ciou", "rec50")}
        print(f"EVAL {tag}: " + " | ".join(f"{n} " + " ".join(f"{k}={out[n][k]:.3f}" for k in show[n]
                                                              if out[n].get(k) is not None) for n in out), flush=True)
        if wb:
            wb.log({f"val/{n}/{k}": v for n, m in out.items() for k, v in m.items()}, step=step)
        return out

    evaluate("start", 0)
    model.train()
    t1, run, seen, norms, skipped = time.time(), {}, 0, [], 0
    for n, ex in train:
        task = tasks[n]
        try:
            enc, keep = encode(proc, task, ex, arm)
        except (OSError, ValueError) as e:  # an unreadable image: skip it rather than lose the run
            skipped += 1
            print(f"SKIP {n} example ({skipped} so far): {repr(e)[:160]}", flush=True)
            enc = None
        if enc is not None:
            lp = read(model, enc, keep, task.layout(arm), labels[n])
            tg = task.targets(ex, arm)
            losses = {s: -(t.to(lp[s].device) * lp[s]).sum() for s, t in tg.items()}
            numbers = {task.number_of(s) for s in tg}
            loss = sum(losses.values()) / len(numbers)  # per number: coarse slots add, they do not dilute
            (loss / a.accum).backward()
            mass = sum(float(lp[s].detach().exp().sum()) for s in tg) / len(tg)  # probability on the labels
            run[f"loss_{n}"] = 0.98 * run.get(f"loss_{n}", float(loss.detach())) + 0.02 * float(loss.detach())
            run[f"mass_{n}"] = 0.98 * run.get(f"mass_{n}", mass) + 0.02 * mass
        seen += 1
        if seen % a.accum == 0 or seen == len(train):
            gn = float(torch.nn.utils.clip_grad_norm_(params, a.clip))
            norms.append(gn)
            run["grad_norm"] = 0.9 * run.get("grad_norm", gn) + 0.1 * gn  # before clipping
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        if seen % 50 == 0:
            el = time.time() - t1
            print(f"[{seen}/{len(train)}] {el / seen:.2f}s/ex eta {el / seen * (len(train) - seen) / 60:.0f}min | "
                  + " ".join(f"{k}={v:.3f}" for k, v in sorted(run.items())), flush=True)
            if wb:
                wb.log({**{f"train/{k}": v for k, v in run.items()}, "train/sec_per_example": el / seen,
                        "train/lr": sched.get_last_lr()[0]}, step=seen)
        if seen % a.eval_every == 0 and seen < len(train):
            evaluate(f"after {seen}", seen)
            model.save_pretrained(f"{a.out}/adapter-{seen}")
    model.save_pretrained(f"{a.out}/adapter")
    print(f"skipped {skipped} unreadable examples", flush=True)
    if norms:
        json.dump(norms, open(f"{a.out}/grad_norms.json", "w"))
    evaluate("end", seen)
    if wb:
        wb.finish()
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
