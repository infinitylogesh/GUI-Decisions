"""Tasks: what the model sees and answers. A task gives its examples, its slot layout for an arm,
the soft targets, and how a read turns back into a prediction and a score.

  agent   computer use: action + point (+ swipe end, key, scroll), AGUVIS steps + Wave-UI clicks
  box     an element's or object's box as centre + log size, Wave-UI elements + RefCOCO referents
  seg     a referred object's mask as a centre + K ray lengths, RefCOCO

The assistant turn is the slot layout, each slot ending in the placeholder:

  action: ___ x: ___ y: ___ x2: ___ y2: ___ key: ___ sdir: ___ samt: ___

and a slot's answer is the next-token distribution at the token before its placeholder.
"""

import math

from . import labels as L
from .data import aguvis, refcoco, waveui
from .geometry import box_iou, mask_iou, poly_mask, radial_decode, radial_encode
from .images import image_of, image_size
from .numbers import LOG_MIN, number_targets, one_hot, read_number, slots_for, unwarp, warp


def coarse_desc(arm, what):
    """How a number's slots are described in a system prompt."""
    return f"the tens digit (00-99) and then the fine code of {what}" if arm.coarse else f"the code of {what}"


class Task:
    name = ""
    numbers = {}  # number -> "lin" | "log"

    def layout(self, arm):
        raise NotImplementedError

    def system(self, arm):
        raise NotImplementedError

    def user(self, ex):
        raise NotImplementedError

    def number_of(self, slot):
        return slot[:-1] if slot.endswith("c") and slot[:-1] in self.numbers else slot

    def messages(self, ex, arm, assistant=None):
        return [{"role": "system", "content": [{"type": "text", "text": self.system(arm)}]},
                {"role": "user", "content": [{"type": "image", "image": image_of(ex)},
                                             {"type": "text", "text": self.user(ex)}]},
                {"role": "assistant", "content": [{"type": "text", "text": assistant if assistant is not None
                                                   else " ".join(f"{s}: ___" for s in self.layout(arm))}]}]

    def labels(self, arm, tok):
        out = {}
        for s in self.layout(arm):
            n = self.number_of(s)
            out[s] = L.DIGITS if (n in self.numbers and s.endswith("c") and n != s) else (
                arm.fine if n in self.numbers else self.categorical_labels(s, tok))
        return out

    def categorical_labels(self, slot, tok):
        raise KeyError(slot)

    def config(self, arm, tok):
        """Everything a reader needs to rebuild this task's prompt and labels (saved in config.json)."""
        return dict(slots=self.layout(arm), system=self.system(arm), numbers=self.numbers,
                    labels=self.labels(arm, tok))


class AgentTask(Task):
    """Computer use. Only the slots an action uses are trained on that example (MASK), so x2 is free
    to mean nothing on a click. Text arguments (what to type, the final answer) are not slots: the base
    model writes them."""
    name = "agent"
    numbers = {"x": "lin", "y": "lin", "x2": "lin", "y2": "lin"}
    MASK = {"click": ["x", "y"], "select": ["x", "y"], "swipe": ["x", "y", "x2", "y2"],
            "scroll": ["sdir", "samt"], "press": ["key"]}

    def layout(self, arm):
        return (["action"] + slots_for("x", arm) + slots_for("y", arm) + slots_for("x2", arm)
                + slots_for("y2", arm) + ["key", "sdir", "samt"])

    def system(self, arm):
        return ("You are a GUI agent. Given a screenshot, the task and the previous actions, answer the next step. "
                "action: " + ", ".join(aguvis.ACTIONS) + ". "
                f"For x, {coarse_desc(arm, 'the horizontal position of the point, from the left edge to the right edge')}; "
                f"for y, the same vertically from the top to the bottom. x2 and y2: the end point of a swipe, written "
                f"the same way. key: the key to press. sdir and samt: scroll direction and amount.")

    def user(self, ex):
        return ex["user"]

    def categorical_labels(self, slot, tok):
        return {"action": L.word_label_ids(tok, "action", aguvis.ACTIONS),
                "key": L.word_label_ids(tok, "key", aguvis.KEYS),
                "sdir": L.word_label_ids(tok, "sdir", aguvis.SDIR), "samt": L.DIGITS}[slot]

    def targets(self, ex, arm):
        a, g = ex["action"], ex["args"]
        tg = {"action": one_hot(len(aguvis.ACTIONS), aguvis.ACTIONS.index(a))}
        for n in self.MASK.get(a, []):
            if n in self.numbers:
                if g.get(n) is None:
                    continue
                w = g.get("box" if n.startswith("x") else "boy", 0.02)  # the target's width / height, if known
                tg.update(number_targets(n, g[n], 0.25 * w, arm))
            elif n == "key":
                tg["key"] = one_hot(len(aguvis.KEYS), aguvis.KEYS.index(g.get("key", "other")))
            elif n == "sdir" and g.get("sdir") in aguvis.SDIR:
                tg["sdir"] = one_hot(len(aguvis.SDIR), aguvis.SDIR.index(g["sdir"]))
            elif n == "samt" and g.get("samt") is not None:
                tg["samt"] = one_hot(10, int(g["samt"]))
        return tg

    def decode(self, lp, arm, how="default"):
        out = dict(action=aguvis.ACTIONS[int(lp["action"].argmax())],
                   x=read_number(lp, "x", arm, how), y=read_number(lp, "y", arm, how))
        if "x2" in lp:
            out.update(x2=read_number(lp, "x2", arm, how), y2=read_number(lp, "y2", arm, how))
        return out

    def score(self, pred, ex, thresh=0.14):
        """A step is right when the action is, and (for a click, select or swipe) the point is within
        0.14 of the target on each axis."""
        ok_a = pred["action"] == ex["action"]
        ok_p = True
        if ex["action"] in ("click", "select", "swipe"):
            ok_p = abs(pred["x"] - ex["args"]["x"]) <= thresh and abs(pred["y"] - ex["args"]["y"]) <= thresh
        return dict(action=float(ok_a), point=float(ok_p) if ex["action"] in ("click", "select", "swipe") else None,
                    step=float(ok_a and ok_p))

    def examples(self, n_agent, n_click, n_val):
        """(train, val): AGUVIS steps + Wave-UI clicks; val is the first n_val steps of the same pool
        (held out from this run; use aguvis.test_steps() to compare runs)."""
        pool = aguvis.steps(n_agent + n_val)
        val, train = pool[:n_val], pool[n_val:]
        return train + (waveui.clicks(n_click) if n_click else []), val


class BoxTask(Task):
    """An element's or object's box as centre + size; sizes log-warped."""
    name = "box"
    numbers = {"cx": "lin", "cy": "lin", "w": "log", "h": "log"}

    def layout(self, arm):
        return slots_for("cx", arm) + slots_for("cy", arm) + slots_for("w", arm) + slots_for("h", arm)

    def system(self, arm):
        return ("You locate an element or object in an image and give its bounding box as a centre and a size. "
                f"For cx, {coarse_desc(arm, 'the horizontal position of the centre, from the left edge to the right edge')}; "
                "for cy, the same vertically from the top to the bottom. "
                f"For w, {coarse_desc(arm, 'the box width on a scale from tiny to the whole image')}; "
                "for h, the same for the height.")

    def user(self, ex):
        return f"Element: {ex['instruction']}"

    def norm_box(self, ex):
        W, H = image_size(ex)
        x1, y1, x2, y2 = ex["box"]
        return [x1 / W, y1 / H, x2 / W, y2 / H] if not ex.get("norm") else list(ex["box"])

    def targets(self, ex, arm):
        x1, y1, x2, y2 = self.norm_box(ex)
        w, h = max(x2 - x1, LOG_MIN), max(y2 - y1, LOG_MIN)
        tg = {}
        tg.update(number_targets("cx", (x1 + x2) / 2, 0.12 * w, arm))
        tg.update(number_targets("cy", (y1 + y2) / 2, 0.12 * h, arm))
        tg.update(number_targets("w", warp(w, "log"), 0.02, arm))
        tg.update(number_targets("h", warp(h, "log"), 0.02, arm))
        return tg

    def decode(self, lp, arm, how="default"):
        cx, cy = read_number(lp, "cx", arm, how), read_number(lp, "cy", arm, how)
        w, h = unwarp(read_number(lp, "w", arm, how), "log"), unwarp(read_number(lp, "h", arm, how), "log")
        return [max(0.0, cx - w / 2), max(0.0, cy - h / 2), min(1.0, cx + w / 2), min(1.0, cy + h / 2)]

    def score(self, pred, ex):
        b = self.norm_box(ex)
        i = box_iou(pred, b)
        cx, cy = (pred[0] + pred[2]) / 2, (pred[1] + pred[3]) / 2
        return dict(iou=i, iou50=float(i >= 0.5), iou75=float(i >= 0.75),
                    center_in=float(b[0] <= cx <= b[2] and b[1] <= cy <= b[3]))

    def examples(self, n_train, n_val, n_photo=0, seed=0):
        """(train, val): Wave-UI element boxes, plus n_photo RefCOCO referent boxes (general objects).
        val: Wave-UI validation boxes, and val_photo: RefCOCO val referents (None without photos)."""
        train = waveui.boxes(n_train)
        val = waveui.val_boxes(n_val)
        val_photo = None
        if n_photo:
            train += refcoco.photo_boxes(n_photo, "train", seed + 100)
            val_photo = refcoco.photo_boxes(n_val, "validation", seed + 100)
        return train, val, val_photo


class SegTask(Task):
    """A referred object's mask as a centre + K ray lengths; rays are fractions of the image diagonal,
    log-warped."""
    name = "seg"

    def __init__(self, K=24):
        self.K = K
        self.numbers = {"cx": "lin", "cy": "lin", **{f"r{k}": "log" for k in range(K)}}

    def layout(self, arm):
        out = slots_for("cx", arm) + slots_for("cy", arm)
        for k in range(self.K):
            out += slots_for(f"r{k}", arm)
        return out

    def system(self, arm):
        K = self.K
        return ("You outline an object in an image. cx and cy: a point well inside the object; "
                f"r0 to r{K - 1}: how far the object reaches from that point along {K} directions, r0 pointing right "
                f"and the directions turning clockwise in equal steps. For cx, "
                f"{coarse_desc(arm, 'the horizontal position of the point, from the left edge to the right edge')}; "
                "for cy, the same vertically from the top to the bottom. For each r, "
                f"{coarse_desc(arm, 'the distance on a scale from tiny to the whole image diagonal')}.")

    def user(self, ex):
        return f"Object: {ex['sentence']}"

    def encoded(self, ex):
        """(cx, cy, d) in pixels, computed once per example."""
        if "radial" not in ex:
            W, H = ex["size"]
            ex["radial"] = radial_encode(poly_mask(ex["polys"], W, H), self.K)
        return ex["radial"]

    def targets(self, ex, arm):
        W, H = ex["size"]
        cx, cy, d = self.encoded(ex)
        bx1, by1, bx2, by2 = ex["bbox"]
        diag = math.hypot(W, H)
        tg = {}
        tg.update(number_targets("cx", cx / W, 0.12 * (bx2 - bx1) / W, arm))
        tg.update(number_targets("cy", cy / H, 0.12 * (by2 - by1) / H, arm))
        for k in range(self.K):
            tg.update(number_targets(f"r{k}", warp(d[k] / diag, "log"), 0.03, arm))
        return tg

    def decode(self, lp, arm, how="default"):
        cx, cy = read_number(lp, "cx", arm, how), read_number(lp, "cy", arm, how)
        r = [unwarp(read_number(lp, f"r{k}", arm, how), "log") for k in range(self.K)]
        return dict(cx=cx, cy=cy, r=r)

    def score(self, pred, ex):
        W, H = ex["size"]
        diag = math.hypot(W, H)
        m, pts = radial_decode(pred["cx"] * W, pred["cy"] * H, [r * diag for r in pred["r"]], W, H)
        gt = poly_mask(ex["polys"], W, H)
        iou, inter, union = mask_iou(m, gt)
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        pb = [max(0, min(xs)), max(0, min(ys)), min(W, max(xs)), min(H, max(ys))]
        rec = box_iou([pb[0] / W, pb[1] / H, pb[2] / W, pb[3] / H],
                      [ex["bbox"][0] / W, ex["bbox"][1] / H, ex["bbox"][2] / W, ex["bbox"][3] / H])
        return dict(iou=float(iou), inter=inter, union=union, rec50=float(rec >= 0.5))

    def examples(self, n_train, n_val, seed=0):
        train = refcoco.examples(refcoco.rows("train"), n_train, seed)
        val = refcoco.examples(refcoco.rows("validation"), n_val, seed + 1, one_sentence=False)
        return train, val


TASKS = {"agent": AgentTask, "box": BoxTask, "seg": SegTask}
