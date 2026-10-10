"""Evaluate served slot adapters on the same cases, side by side, with paired statistics.

  python -m slotread.evaluate <set> <n> name=run_dir [name=run_dir ...]      (n=0: the whole set)

  agent_test    1,600 AGUVIS steps no run trained on (rows 6,000+ of every source)
  click         ScreenSpot clicks in the agent layout (1,272)
  box           ScreenSpot element boxes (1,272)
  box_refcoco   RefCOCO val referents as boxes (general objects; 3,811)
  box_coco17    COCO val2017 objects by category name (photos RefCOCO never used)
  box_fashion   Fashionpedia garments (a new domain)
  seg_val       RefCOCO val masks (also seg_testA, seg_testB); SEG_EXPR=one: one expression per object

  SHARD=0/2 python -m slotread.evaluate seg_val 0 ...      every 2nd case from 0 (split across servers)
  python -m slotread.evaluate merge <out.json> <shard.json> ...

Every case is read by every adapter (agent, click and box: per-slot requests; seg: one request with
prompt log-probabilities), scored under three readouts (default, peak, expected), and the adapters
are compared case by case: a paired bootstrap interval on the difference, and McNemar's exact test for
right/wrong metrics. A difference counts only when its interval excludes zero.
"""

import itertools, json, math, os, sys, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .client import SlotModel, agent_user
from .data import aguvis, evalsets, refcoco
from .images import data_url, image_of
from .numbers import slots_for

OUT = os.environ.get("SLOTREAD_RESULTS", "results")
CASE_WORKERS = int(os.environ.get("CASE_WORKERS", 6))
HOWS = ("default", "peak", "expected")
PRIMARY = {"agent": "step", "click": "hit", "box": "iou", "seg": "iou"}
BOOT = 10000


def mcnemar(a, b):
    """Two-sided exact McNemar p-value from paired booleans; also the discordant counts."""
    n01, n10 = int(np.sum(a & ~b)), int(np.sum(~a & b))
    n = n01 + n10
    if n == 0:
        return 1.0, n01, n10
    p = 2 * sum(math.comb(n, i) for i in range(min(n01, n10) + 1)) / 2 ** n
    return min(1.0, p), n01, n10


def boot(diff, rng):
    """A paired bootstrap 95% interval on the mean difference."""
    idx = rng.integers(0, len(diff), (BOOT, len(diff)))
    m = diff[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def cases(which, n):
    if which == "agent_test":
        return [dict(kind="agent", ex=ex) for ex in aguvis.test_steps(n)]
    if which in ("click", "box"):
        return [dict(kind=which, ex=ex) for ex in evalsets.screenspot(n)]
    if which == "box_coco17":
        return [dict(kind="box", ex=ex) for ex in evalsets.coco17(n)]
    if which == "box_fashion":
        return [dict(kind="box", ex=ex) for ex in evalsets.fashion(n or 1000)]
    if which == "box_refcoco":
        return [dict(kind="box", ex=ex) for ex in refcoco.photo_boxes(n, "validation", seed=0)]
    split = {"seg_val": "validation", "seg_testA": "test", "seg_testB": "testB"}[which]
    one = os.environ.get("SEG_EXPR", "all") == "one"
    return [dict(kind="seg", ex=ex) for ex in refcoco.examples(refcoco.rows(split), n, seed=0, one_sentence=one)]


# accuracy runs send screenshots as the post measured them: full-size PNG (EVAL_IMG_FMT / EVAL_IMG_MAX_PX to
# change). Serving and latency use the faster model-sized JPEG (slotread.images); on ScreenSpot clicks one
# run measured 0.825 with it against 0.860 with full-size PNG, while agent steps were unchanged.
EVAL_FMT = os.environ.get("EVAL_IMG_FMT", "png")
EVAL_MAX_PX = int(os.environ.get("EVAL_IMG_MAX_PX", 0))


def evaluate_case(model, case):
    kind, ex = case["kind"], case["ex"]
    url = data_url(image_of(ex), fmt=EVAL_FMT, max_px=EVAL_MAX_PX)
    a, out = model.arm, {}
    if kind in ("agent", "click"):
        task = model.tasks["agent"]
        user = ex["user"] if kind == "agent" else agent_user(ex["instruction"])
        lp = model.read("agent", url, user, ["action"] + slots_for("x", a) + slots_for("y", a))
        for how in HOWS:
            p = task.decode(lp, a, how)
            if kind == "agent":
                out[how] = task.score(p, ex)
            else:
                b = ex["box"]
                out[how] = dict(hit=float(b[0] <= p["x"] <= b[2] and b[1] <= p["y"] <= b[3]),
                                action_click=float(p["action"] == "click"))
    elif kind == "box":
        task = model.tasks["box"]
        lp = model.read("box", url, task.user(ex), task.layout(a))
        for how in HOWS:
            out[how] = task.score(task.decode(lp, a, how), ex)
    else:
        task = model.tasks["seg"]
        lp = model.read_one("seg", url, task.user(ex))
        for how in HOWS:
            out[how] = task.score(task.decode(lp, a, how), ex)
    return out


def summarize(which, rows, names):
    kind = rows[0]["kind"]
    key = PRIMARY[kind]
    cols = [k for k in sorted(rows[0]["arms"][names[0]]["default"]) if k not in ("inter", "union")]
    cols += ["ciou"] if kind == "seg" else []
    print(f"\n{which}: {len(rows)} cases\n\n| adapter | readout | " + " | ".join(cols) + " |\n|---|---|" + "---|" * len(cols))
    summary = {}
    for name in names:
        for how in HOWS:
            vals = [r["arms"][name][how] for r in rows]
            m = {}
            for k in cols:
                if k == "ciou":
                    m[k] = sum(v["inter"] for v in vals) / max(1, sum(v["union"] for v in vals))
                    continue
                xs = [v[k] for v in vals if v.get(k) is not None]
                m[k] = sum(xs) / len(xs) if xs else None
            summary[f"{name}/{how}"] = m
            print(f"| {name} | {how} | " + " | ".join("" if m[k] is None else f"{m[k]:.3f}" for k in cols) + " |",
                  flush=True)
    rng = np.random.default_rng(0)
    print(f"\nPaired, default readouts, on {key}:\n")
    for x, y in itertools.combinations(names, 2):
        vx = np.array([r["arms"][x]["default"][key] or 0.0 for r in rows], dtype=float)
        vy = np.array([r["arms"][y]["default"][key] or 0.0 for r in rows], dtype=float)
        lo, hi = boot(vx - vy, rng)
        extra = ""
        if kind in ("agent", "click"):
            p, n01, n10 = mcnemar(vx > 0.5, vy > 0.5)
            extra = f" | McNemar p {p:.3f} ({n01} / {n10})"
        verdict = "tie" if lo <= 0 <= hi else (f"{x} ahead" if lo > 0 else f"{y} ahead")
        print(f"- {x} {vx.mean():.3f} vs {y} {vy.mean():.3f}: diff {vx.mean() - vy.mean():+.3f} "
              f"[{lo:+.3f}, {hi:+.3f}]{extra} -> {verdict}", flush=True)
    return summary


def merge(out, parts):
    """Rejoin shards of one set (every row carries every adapter, so pairing survives the split)."""
    ds = [json.load(open(p)) for p in parts]
    rows = [r for d in ds for r in d["rows"]]
    summary = summarize(ds[0]["which"], rows, list(ds[0]["arms"]))
    json.dump(dict(which=ds[0]["which"], n=len(rows), arms=ds[0]["arms"], summary=summary, rows=rows, shards=parts),
              open(out, "w"))
    print(f"\nmerged {len(parts)} shards, {len(rows)} cases -> {out}", flush=True)


def main(args=None):
    args = args or sys.argv[1:]
    if args[0] == "merge":
        return merge(args[1], args[2:])
    which, n = args[0], int(args[1])
    models = [SlotModel(*a.split("=", 1)) for a in args[2:]]
    cs = cases(which, n)
    shard = os.environ.get("SHARD")
    if shard:  # "i/k": every k-th case starting at i
        i, k = map(int, shard.split("/"))
        cs = cs[i::k]
    print(f"{which}: {len(cs)} cases{f' (shard {shard})' if shard else ''}, adapters {', '.join(m.name for m in models)}",
          flush=True)

    def one(ic):
        i, c = ic
        return dict(id=i, kind=c["kind"], group=list(c["ex"].get("group", [])),
                    arms={m.name: evaluate_case(m, c) for m in models})

    rows, t = [], time.time()
    with ThreadPoolExecutor(CASE_WORKERS) as p:
        for k, row in enumerate(p.map(one, enumerate(cs))):
            rows.append(row)
            if (k + 1) % 200 == 0:
                print(f"  {k + 1}/{len(cs)} ({time.time() - t:.0f}s)", flush=True)
    summary = summarize(which, rows, [m.name for m in models])
    os.makedirs(OUT, exist_ok=True)
    path = f"{OUT}/eval_{which}_{n}{'_s' + shard.replace('/', 'of') if shard else ''}.json"
    json.dump(dict(which=which, n=len(rows), arms={m.name: m.path for m in models}, summary=summary, rows=rows),
              open(path, "w"))
    print(f"\nwrote {path}", flush=True)


if __name__ == "__main__":
    main()
