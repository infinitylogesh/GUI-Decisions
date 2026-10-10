"""The vanilla baseline at its best: which output format does the untouched Gemma 4 31B ground best in,
with and without thinking? Accuracy (concurrent) and cold single-request latency per variant.

  json_px        the step as JSON with pixel x, y (badly undersells the model: 4% of clicks)
  json_yx        the step as JSON with "point": [y, x] on 0-1000, y first: Gemma 4's own convention
  json_yx_think  the same with thinking on
  box2d          native grounding: "What's the bounding box for X?" -> box_2d, click its centre (clicks only)

  python scripts/vanilla_formats.py click 300         # ScreenSpot clicks
  python scripts/vanilla_formats.py agent_test 300    # clean AGUVIS steps
"""

import json, os, re, statistics, sys
from concurrent.futures import ThreadPoolExecutor

from slotread.agent import JSON_SYSTEM, TEXT_RULE
from slotread.data import aguvis, evalsets
from slotread.images import data_url, image_of
from slotread.client import generate

SYS_PX = JSON_SYSTEM.replace('"x": 0-1 from the left, "y": 0-1 from the top', '"x", "y": pixel coordinates in the {W}x{H} screenshot')
SYS_YX = JSON_SYSTEM.replace('"x": 0-1 from the left, "y": 0-1 from the top, "x2", "y2": a swipe\'s end',
                             '"point": [y, x] normalized to 0-1000 (y first), "point2": a swipe\'s end the same way')


def last_json(text):
    """The last JSON object or list in the text (after any thinking)."""
    for m in reversed(list(re.finditer(r"(\{.*?\}|\[\s*\{.*?\}\s*\])", text or "", re.S))):
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
    return None


def split_user(user):
    """AGUVIS user text -> (instruction, previous actions)."""
    m = re.search(r"Instruction:\s*(.*?)\n\nPrevious actions:\n(.*)", user, re.S)
    return (m.group(1).strip(), m.group(2).strip() or "None") if m else (user, "None")


def run_variant(name, url, size, instruction, previous):
    user = ("Please generate the next move according to the UI screenshot, instruction and previous actions.\n\n"
            f"Instruction: {instruction}\n\nPrevious actions:\n{previous}")
    if name == "box2d":
        text, ms, u = generate("gemma4", url, f'What\'s the bounding box for the "{instruction}" element in the image?',
                               max_tokens=64)
        o = last_json(text)
        o = o[0] if isinstance(o, list) and o else o
        b = o.get("box_2d") if isinstance(o, dict) else None
        if isinstance(b, list) and len(b) == 4:
            return "click", {"x": (b[1] + b[3]) / 2000, "y": (b[0] + b[2]) / 2000}, ms, u
        return None, {}, ms, u
    if name == "json_px":
        system = SYS_PX.replace("{W}", str(size[0])).replace("{H}", str(size[1])) + TEXT_RULE
        text, ms, u = generate("gemma4", url, user, max_tokens=96, system=system)
        o = last_json(text) or {}
        ok = isinstance(o.get("x"), (int, float)) and isinstance(o.get("y"), (int, float))
        return o.get("action"), ({"x": o["x"] / size[0], "y": o["y"] / size[1]} if ok else {}), ms, u
    think = name == "json_yx_think"
    text, ms, u = generate("gemma4", url, user, max_tokens=2048 if think else 96, think=think, system=SYS_YX + TEXT_RULE)
    o = last_json(text) or {}
    p = o.get("point")
    ok = isinstance(p, list) and len(p) == 2 and all(isinstance(v, (int, float)) for v in p)
    return o.get("action"), ({"x": p[1] / 1000, "y": p[0] / 1000} if ok else {}), ms, u


def score_agent(act, pt, ex, thresh=0.14):
    """AgentTask.score's rules; GUI agents often scroll by swiping, so swipe and scroll count as one."""
    gold = ex["action"]
    ok_a = act == gold or {act, gold} == {"swipe", "scroll"}
    has_pt = gold in ("click", "select", "swipe")
    ok_p = (not has_pt) or ("x" in pt and abs(pt["x"] - ex["args"]["x"]) <= thresh and abs(pt["y"] - ex["args"]["y"]) <= thresh)
    return dict(action=float(ok_a), point=float(ok_p) if has_pt else None, step=float(ok_a and ok_p))


def main():
    which, n = sys.argv[1], int(sys.argv[2])
    variants = ["json_px", "json_yx", "json_yx_think"] + (["box2d"] if which == "click" else [])
    cs = evalsets.screenspot(n) if which == "click" else aguvis.test_steps(n)

    def score(ex, v):
        im = image_of(ex)
        url = data_url(im)
        if which == "click":
            act, pt, ms, u = run_variant(v, url, im.size, ex["instruction"], "None")
            b = ex["box"]
            return dict(hit=float(act == "click" and "x" in pt and b[0] <= pt["x"] <= b[2] and b[1] <= pt["y"] <= b[3]),
                        ms=ms, out=(u or {}).get("completion_tokens"))
        act, pt, ms, u = run_variant(v, url, im.size, *split_user(ex["user"]))
        return dict(score_agent(act, pt, ex), ms=ms, out=(u or {}).get("completion_tokens"))

    key = "hit" if which == "click" else "step"
    print(f"{which}: {len(cs)} cases\n\n| variant | {key} | latency ms, alone (median of 20) | output tokens |\n|---|---|---|---|")
    summary = {}
    for v in variants:
        with ThreadPoolExecutor(4) as p:
            rows = list(p.map(lambda ex: score(ex, v), cs))
        acc = statistics.mean(r[key] for r in rows)
        lat = [score(ex, v)["ms"] for ex in cs[:20]]  # alone: one request at a time
        out = statistics.median([r["out"] for r in rows if r.get("out")] or [0])
        summary[v] = dict(acc=acc, latency=statistics.median(lat), out_tokens=out)
        print(f"| {v} | {acc:.3f} | {statistics.median(lat):.0f} | {out:.0f} |", flush=True)
    os.makedirs("results", exist_ok=True)
    json.dump(dict(which=which, n=len(cs), summary=summary), open(f"results/vanilla_formats_{which}_{len(cs)}.json", "w"))


if __name__ == "__main__":
    main()
