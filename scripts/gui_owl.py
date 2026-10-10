"""Slot reading on a 2B GUI model: vanilla GUI-Owl-1.5-2B-Instruct against our slot adapter on the same
model, served together (serving/serve_gui_owl.sh: the base model as `owl`, the adapter as e.g. `owl-E`).

Vanilla runs as its authors ship it (Mobile-Agent-v3.5, mobile_use/utils.py): their `mobile_use` tool
system prompt (gui_owl_system_prompt.txt), their user text (date + instruction + previous actions, then
the screenshot), and its native output: "Action: <one sentence>" then a <tool_call> JSON with
coordinates on a 1000 x 1000 grid. The tool call is mapped onto our action set and scored with the same
rules as the slot reads.

  python scripts/gui_owl.py eval click 0          # ScreenSpot clicks, capped and native images
  python scripts/gui_owl.py eval agent_test 0     # the clean AGUVIS test set
  python scripts/gui_owl.py bench 80 owl-E=runs/owl_E      # cold latency: slots (full template, lazy) vs vanilla

"capped" images are at most 1,048,576 pixels (what the adapter trained on); "native" is the screenshot
as given. Accuracy for the slot adapter itself: python -m slotread.evaluate click 0 owl-E=runs/owl_E
(with IMAGE_CAP=1048576).
"""

import datetime, hashlib, json, os, re, statistics, sys, time
from concurrent.futures import ThreadPoolExecutor

from slotread import client as C
from slotread.client import SlotModel, agent_user, post
from slotread.data import aguvis, evalsets
from slotread.images import cap_image, data_url

CAP = 1048576
SYSTEM = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gui_owl_system_prompt.txt")).read()
SETTINGS = ("capped", "native")


def user_text(instruction, previous="None"):
    today = datetime.date.today()
    return ("Please generate the next move according to the UI screenshot, instruction and previous actions.\n\n"
            f"Instruction: Today's date is: {today:%Y-%m-%d} {today:%A}.{instruction}\n\nPrevious actions:\n{previous}")


def split_user(user):
    """AGUVIS user text -> (instruction, previous actions), to rebuild it in GUI-Owl's form."""
    m = re.search(r"Instruction:\s*(.*?)\n\nPrevious actions:\n(.*)", user, re.S)
    return (m.group(1).strip(), m.group(2).strip() or "None") if m else (user, "None")


def vanilla(url, instruction, previous="None"):
    d, ms = post(dict(model="owl", temperature=0, max_tokens=160,
                      messages=[{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
                                {"role": "user", "content": [{"type": "text", "text": user_text(instruction, previous)},
                                                             {"type": "image_url", "image_url": {"url": url}}]}]))
    if "error" in d:
        return None, ms, {}
    return d["choices"][0]["message"]["content"], ms, d.get("usage", {})


def parse(text):
    """GUI-Owl's tool call -> (our action, {x, y, x2, y2}) in [0, 1]."""
    m = re.search(r"<tool_call>\s*(\{.*\})\s*</tool_call>", text or "", re.S)
    if not m:
        return None, {}
    try:
        args = json.loads(m.group(1)).get("arguments", {})
    except json.JSONDecodeError:
        return None, {}
    a, c, c2 = args.get("action"), args.get("coordinate"), args.get("coordinate2")
    pt = {}
    if isinstance(c, list) and len(c) == 2:
        pt.update(x=c[0] / 1000, y=c[1] / 1000)
    if isinstance(c2, list) and len(c2) == 2:
        pt.update(x2=c2[0] / 1000, y2=c2[1] / 1000)
    if a in ("click", "long_press"):
        return "click", pt
    if a == "swipe":
        return "swipe", pt
    if a == "system_button":
        return {"Back": "back", "Home": "home", "Enter": "press"}.get(args.get("button"), "press"), pt
    return {"type": "type", "open": "open", "wait": "wait", "answer": "answer", "terminate": "answer",
            "key": "press", "interact": "answer"}.get(a), pt


def score_agent(act, pt, ex, thresh=0.14):
    """The slot reads' rules; GUI-Owl scrolls by swiping, so swipe and scroll count as one action."""
    gold = ex["action"]
    ok_a = act == gold or {act, gold} == {"swipe", "scroll"}
    has_pt = gold in ("click", "select", "swipe")
    ok_p = (not has_pt) or ("x" in pt and abs(pt["x"] - ex["args"]["x"]) <= thresh and abs(pt["y"] - ex["args"]["y"]) <= thresh)
    return dict(action=float(ok_a), point=float(ok_p) if has_pt else None, step=float(ok_a and ok_p))


def url_for(im, setting, fmt="png"):
    """Accuracy runs: PNG (as the post measured); the latency bench: JPEG (the fast path)."""
    return data_url(cap_image(im.convert("RGB"), CAP if setting == "capped" else 0), fmt=fmt, max_px=0)


def evaluate(which, n):
    cs = evalsets.screenspot(n) if which == "click" else aguvis.test_steps(n)

    def one(ex):
        row = {}
        for setting in SETTINGS:
            url = url_for(ex["image"], setting)
            if which == "click":
                text, ms, usage = vanilla(url, ex["instruction"])
                act, pt = parse(text)
                b = ex["box"]
                r = dict(hit=float(act == "click" and "x" in pt and b[0] <= pt["x"] <= b[2] and b[1] <= pt["y"] <= b[3]))
            else:
                text, ms, usage = vanilla(url, *split_user(ex["user"]))
                r = score_agent(*parse(text), ex)
            row[setting] = dict(r, ms=ms, completion_tokens=usage.get("completion_tokens"))
        return row

    with ThreadPoolExecutor(8) as p:
        rows = list(p.map(one, cs))
    key = "hit" if which == "click" else "step"
    for setting in SETTINGS:
        rs = [r[setting] for r in rows]
        print(f"vanilla GUI-Owl, {setting} images: {key} {statistics.mean(r[key] for r in rs):.3f} "
              f"({len(rs)} cases, {statistics.median(r['completion_tokens'] or 0 for r in rs):.0f} output tokens)")
    os.makedirs("results", exist_ok=True)
    json.dump(dict(which=which, n=len(cs), rows=rows), open(f"results/gui_owl_vanilla_{which}_{len(cs)}.json", "w"))


def bench(n, slot):
    """Cold, one request at a time, condition order rotating. Slot reads come in two modes, sharing the
    adapter's prefix cache, so each screenshot gets one of them (alternating): "slots" reads the whole
    template (8 parallel requests), "slots lazy" reads action, x and y and then only the slots the action
    needs (what SlotModel.decide and the browser agent do). The base model (its own prefix cache) reads
    every screenshot at both image sizes. n screenshots per slot mode."""
    m = SlotModel(*slot.split("=", 1))
    C.MAX_IDS = 256
    seen, shots = set(), []
    for c in evalsets.screenspot(0)[20:]:
        h = hashlib.md5(c["image"].resize((64, 64)).tobytes()).hexdigest()
        if h not in seen:
            seen.add(h)
            shots.append(c)
        if len(shots) >= 2 * n + 10:
            break

    def slot_read(lazy):
        def f(c):
            url = url_for(c["image"], "capped", "jpeg")
            t = time.perf_counter()
            m.read("agent", url, agent_user(c["instruction"]), lazy=lazy)
            return (time.perf_counter() - t) * 1000, {}
        return f

    slot_modes = {"slots": slot_read(False), "slots lazy": slot_read(True)}
    conds = {**slot_modes, **{f"vanilla {s}": (lambda s: lambda c: vanilla(url_for(c["image"], s, "jpeg"), c["instruction"])[1:])(s)
                              for s in SETTINGS}}
    for c in shots[:10]:  # warm-up: reach every API-server process
        for f in conds.values():
            f(c)
    rows = []
    names = list(conds)
    for i, c in enumerate(shots[10:]):
        mode = list(slot_modes)[i % 2]
        run = [k for k in names if k == mode or k not in slot_modes]
        for k in run[i % len(run):] + run[:i % len(run)]:
            ms, usage = conds[k](c)
            rows.append(dict(cond=k, ms=ms, out=(usage or {}).get("completion_tokens")))
    print("| condition | ms median (p90) | output tokens | screenshots |\n|---|---|---|---|")
    for k in names:
        xs = sorted(r["ms"] for r in rows if r["cond"] == k)
        outs = [r["out"] for r in rows if r["cond"] == k and r["out"]]
        print(f"| {k} | {statistics.median(xs):.0f} ({xs[int(0.9 * (len(xs) - 1))]:.0f}) | "
              f"{statistics.median(outs) if outs else 0:.0f} | {len(xs)} |")
    os.makedirs("results", exist_ok=True)
    json.dump(dict(n=n, rows=rows), open(f"results/gui_owl_bench_{n}.json", "w"))


if __name__ == "__main__":
    if sys.argv[1] == "eval":
        evaluate(sys.argv[2], int(sys.argv[3]))
    else:
        bench(int(sys.argv[2]), sys.argv[3])
