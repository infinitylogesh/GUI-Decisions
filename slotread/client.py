"""Reading slots from a served adapter (vLLM; SGLang optional).

A served adapter is described by its run's config.json (written by slotread.train): each task's system
prompt, slot order and label ids. SlotModel turns that into reads.

  read        one request per slot, in parallel: each request's assistant text ends right before that
              slot's placeholder ("action: ___ x:"), max_tokens=1, and vLLM returns the log-probabilities
              of exactly that slot's label ids (logprob_token_ids). The default.
  read(..., lazy=True)  computer use only: action, x and y first (all a click needs); key, scroll or
              swipe-end slots only when the action calls for them
  read_one    every slot from one request's prompt log-probabilities (top-k per position; slower)
  read_sgl    every slot from one SGLang /generate request (logprob_start_len + token_ids_logprob)

Serving (serving/serve_vllm.sh): vLLM caps logprob_token_ids at 128 per request; serving/patch_vllm.sh
raises it to 256 so a 256-bin slot is one request (MAX_LABEL_IDS=256). Run vLLM with
--api-server-count 8: a single API-server process serializes the parallel slot requests.
"""

import base64, io, json, math, os, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor

import torch

from .numbers import slots_for
from .labels import Arm, PLACEHOLDER
from .tasks import AgentTask, BoxTask, SegTask

VLLM_URL = os.environ.get("VLLM_URL", f"http://127.0.0.1:{os.environ.get('PORT', 8001)}")
MAX_IDS = int(os.environ.get("MAX_LABEL_IDS", 256))  # 128 on stock vLLM (see patch_vllm.sh)
PL_TOPK = int(os.environ.get("PL_TOPK", 50))  # read_one: prompt log-probabilities kept per position
EXTRA_BODY = json.loads(os.environ.get("EXTRA_BODY", "{}"))  # e.g. {"mm_processor_kwargs": {"max_soft_tokens": 1120}}
SGL_URL = os.environ.get("SGL_URL", "http://127.0.0.1:30000")


def post(body, url=None, path="/v1/chat/completions", tries=2):
    """POST JSON; returns (response dict, ms). HTTP errors come back as {"error": ...}."""
    t = time.perf_counter()
    d = {}
    for _ in range(tries):
        r = urllib.request.Request((url or VLLM_URL) + path, json.dumps(body).encode(),
                                   {"content-type": "application/json"})
        try:
            d = json.load(urllib.request.urlopen(r, timeout=300))
            break
        except urllib.error.HTTPError as e:
            d = {"error": e.read().decode()[:300]}
            if e.code != 500:
                break
    return d, (time.perf_counter() - t) * 1000


def generate(model, url, text, max_tokens=64, think=False, system=None, **extra):
    """A plain chat completion with one image (the vanilla baselines). Returns (text, ms, usage)."""
    msgs = ([{"role": "system", "content": [{"type": "text", "text": system}]}] if system else []) + \
           [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                         {"type": "text", "text": text}]}]
    d, ms = post(dict(model=model, messages=msgs, max_tokens=max_tokens, temperature=0,
                      chat_template_kwargs={"enable_thinking": think}, **extra))
    if "error" in d:
        return "", ms, {}
    return d["choices"][0]["message"].get("content") or "", ms, d.get("usage", {})


CONFIG_NAME = "slotread_config.json"


def load_config(path):
    """A model's slot config: a run folder (config.json, as slotread.train writes it), a folder with
    slotread_config.json (the released models), or a Hugging Face repo id holding slotread_config.json."""
    if os.path.isdir(path):
        for name in (CONFIG_NAME, "config.json"):
            p = os.path.join(path, name)
            if os.path.exists(p):
                c = json.load(open(p))
                if "tasks" in c:
                    return c
        raise FileNotFoundError(f"no {CONFIG_NAME} in {path}")
    from huggingface_hub import hf_hub_download
    return json.load(open(hf_hub_download(path, CONFIG_NAME)))


class SlotModel:
    """A served slot model: `name` as vLLM serves it (the adapter's --lora-modules name, or the merged
    model's --served-model-name), `path` its slot config (run folder, released model folder or HF repo id).
    `text_model`: the served base model that writes text steps (what to type, the answer)."""

    def __init__(self, name, path, text_model=None):
        self.name, self.path, self.text_model = name, path, text_model
        c = load_config(path)
        self.cfg = c
        self.arm = Arm(c["arm"], c["coarse"], c["bins"], c["tokens"], c["fine"])
        self.tasks = {"agent": AgentTask(), "box": BoxTask(), "seg": SegTask(c["rays"])}

    def task_cfg(self, task):
        return self.cfg["tasks"][task]

    def body(self, task, url, user, assistant, **extra):
        msgs = [{"role": "system", "content": [{"type": "text", "text": self.task_cfg(task)["system"]}]},
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                             {"type": "text", "text": user}]},
                {"role": "assistant", "content": [{"type": "text", "text": assistant}]}]
        return dict(model=self.name, messages=msgs, add_generation_prompt=False, continue_final_message=True,
                    chat_template_kwargs={"enable_thinking": False}, max_tokens=1, temperature=0,
                    logprobs=True, return_tokens_as_token_ids=True, **EXTRA_BODY, **extra)

    def read(self, task, url, user, slots=None, lazy=False):
        """{slot: log-probabilities over its labels} from parallel per-slot requests."""
        if lazy and task == "agent":
            from .data.aguvis import ACTIONS
            lp = self.read(task, url, user, ["action", *slots_for("x", self.arm), *slots_for("y", self.arm)])
            more = {"press": ["key"], "scroll": ["sdir", "samt"],
                    "swipe": slots_for("x2", self.arm) + slots_for("y2", self.arm)}.get(ACTIONS[int(lp["action"].argmax())])
            if more:
                lp.update(self.read(task, url, user, more))
            return lp
        c = self.task_cfg(task)
        order, lab = c["slots"], c["labels"]
        slots = slots or order
        jobs = [(s, lab[s][i:i + MAX_IDS]) for s in slots for i in range(0, len(lab[s]), MAX_IDS)]

        def one(job):
            s, ids = job
            k = order.index(s)
            prefix = " ".join(f"{x}: ___" for x in order[:k]) + (" " if k else "") + f"{s}:"
            d, _ = post(self.body(task, url, user, prefix, top_logprobs=1, logprob_token_ids=ids))
            if "error" in d:
                raise RuntimeError(f"{self.name}/{task}/{s}: {d['error']}")
            top = d["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
            return s, {int(t["token"].split(":")[1]): t["logprob"] for t in top}

        with ThreadPoolExecutor(len(jobs)) as p:
            res = list(p.map(one, jobs))
        got = {}
        for s, v in res:
            got.setdefault(s, {}).update(v)
        return {s: torch.tensor([got[s].get(i, -1e4) for i in lab[s]]) for s in slots}

    def read_one(self, task, url, user):
        """Every slot from one request's prompt log-probabilities: each slot's answer is its own
        placeholder's row. Labels outside the top PL_TOPK get 5 nats below the lowest returned."""
        c = self.task_cfg(task)
        order, lab = c["slots"], c["labels"]
        d, _ = post(self.body(task, url, user, " ".join(f"{s}: ___" for s in order), top_logprobs=1,
                              prompt_logprobs=PL_TOPK))
        if "error" in d:
            raise RuntimeError(f"{self.name}/{task}: {d['error']}")
        pl = d.get("prompt_logprobs") or []
        marks = [i for i, p in enumerate(pl) if p and any(v.get("decoded_token") == PLACEHOLDER for v in p.values())]
        marks = marks[-len(order):]
        if len(marks) != len(order):
            raise RuntimeError(f"{self.name}/{task}: {len(marks)} placeholders for {len(order)} slots")
        out = {}
        for s, m in zip(order, marks):
            vals = {int(k): v["logprob"] for k, v in (pl[m] or {}).items()}
            floor = (min(vals.values()) - 5) if vals else -20.0
            out[s] = torch.tensor([vals.get(i, floor) for i in lab[s]])
        return out

    def read_sgl(self, task, url, user, slots=None, budget=280, processor="nvidia/Gemma-4-31B-IT-NVFP4"):
        """Every requested slot from ONE SGLang /generate request (serving/sglang): the prompt carries
        the whole layout, and SGLang returns, for the last positions only (logprob_start_len), the
        log-probabilities of our label ids only (token_ids_logprob). Gemma 4 only (its image token
        count is computed client-side to find where the slots start)."""
        from PIL import Image
        proc = _processor(processor)
        c = self.task_cfg(task)
        order, lab = c["slots"], c["labels"]
        slots = slots or order
        layout = " ".join(f"{x}: ___" for x in order)
        msgs = [{"role": "system", "content": [{"type": "text", "text": c["system"]}]},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user}]},
                {"role": "assistant", "content": [{"type": "text", "text": layout}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False,
                                        continue_final_message=True, enable_thinking=False)
        W, H = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).size
        # Gemma 4 fits the image to the budget in 48 x 48 pixel cells (16 px patches, 3 x 3 pooling),
        # one token per cell, plus 2 boundary tokens
        f = math.sqrt(budget * 9 * 256 / (W * H))
        soft = int(f * H // 48) * int(f * W // 48)
        n_text = len(proc.tokenizer(text, add_special_tokens=False)["input_ids"])
        tail = len(proc.tokenizer(layout, add_special_tokens=False)["input_ids"])
        start = max(0, n_text - 1 + soft + 2 - tail - 4)  # a few positions early is harmless
        ph = proc.tokenizer("x: ___", add_special_tokens=False)["input_ids"][-1]
        body = dict(text=text, image_data=[url], lora_path=self.name, return_logprob=True, logprob_start_len=start,
                    top_logprobs_num=0, token_ids_logprob=sorted({i for x in slots for i in lab[x]}),
                    sampling_params=dict(max_new_tokens=1, temperature=0))
        d, _ = post(body, url=SGL_URL, path="/generate")
        if "error" in d:
            raise RuntimeError(f"{self.name}/{task}: {d['error']}")
        mi = d["meta_info"]
        marks = [i for i, t in enumerate(mi["input_token_logprobs"]) if t[1] == ph][-len(order):]
        if len(marks) != len(order):
            raise RuntimeError(f"{self.name}/{task}: {len(marks)} placeholders for {len(order)} slots")
        out = {}
        for x in slots:
            row = {t[1]: t[0] for t in mi["input_token_ids_logprobs"][marks[order.index(x)]]}
            out[x] = torch.tensor([row.get(i, -1e4) for i in lab[x]])
        return out


    def image_url(self, image):
        """A PIL image as the data URL sent to the server: JPEG at the model's input size by default (the
        fast path; IMG_FMT / IMG_MAX_PX), capped as in training when the config says so (GUI-Owl: 1 MP).
        A string is taken as a URL already."""
        if isinstance(image, str):
            return image
        from .images import cap_image, data_url
        cap = self.cfg.get("image_cap")
        return data_url(cap_image(image.convert("RGB"), cap), max_px=cap) if cap else data_url(image)

    def decide(self, image, instruction, previous="None"):
        """A computer-use step: action and point; the key, scroll or swipe end when the action needs them.
        Reads action, x and y first, then the action's other slots (lazy)."""
        from .data.aguvis import KEYS, SDIR
        t = time.perf_counter()
        lp = self.read("agent", self.image_url(image), agent_user(instruction, previous), lazy=True)
        d = self.tasks["agent"].decode(lp, self.arm)
        if "key" in lp:
            d["key"] = KEYS[int(lp["key"].argmax())]
        if "sdir" in lp:
            d.update(sdir=SDIR[int(lp["sdir"].argmax())], samt=int(lp["samt"].argmax()))
        return dict(d, ms=round((time.perf_counter() - t) * 1000, 1))

    def box(self, image, description):
        """An element's or object's box [x1, y1, x2, y2] in [0, 1]."""
        t = time.perf_counter()
        lp = self.read("box", self.image_url(image), self.tasks["box"].user({"instruction": description}))
        return dict(box=self.tasks["box"].decode(lp, self.arm), ms=round((time.perf_counter() - t) * 1000, 1))

    def write_text(self, image, instruction, action, previous="None", max_tokens=24):
        """The text a type / answer / open step needs, written by the served base model (text_model)."""
        from .agent import TEXT_ASK
        if not self.text_model:
            raise RuntimeError("text steps need the base model: serve it and pass text_model=<its served name>")
        text, _, _ = generate(self.text_model, self.image_url(image),
                              agent_user(instruction, previous) + "\n\n" + TEXT_ASK[action], max_tokens=max_tokens)
        return text.strip().strip('"').split("\n")[0][:120]


_PROC = {}


def _processor(name):
    if name not in _PROC:
        from transformers import AutoProcessor
        _PROC[name] = AutoProcessor.from_pretrained(name)
    return _PROC[name]


def agent_user(instruction, previous="None"):
    """The agent layout's user text (AGUVIS's format)."""
    return ("Please generate the next move according to the UI screenshot, instruction and previous actions.\n\n"
            f"Instruction: {instruction}\n\nPrevious actions:\n{previous}")
