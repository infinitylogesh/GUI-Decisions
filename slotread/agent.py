"""A computer-use agent in a real browser (Playwright), deciding each step one of two ways:

  slots    a served slot adapter (e.g. j-E3): action + point from slot reads, no decoding. Text steps
           (what to type, the final answer) are written by the base model (TEXT_MODEL, default gemma4).
  vanilla  the base model writes the whole step as JSON:
             gemma4-native        "point": [y, x] on a 0-1000 grid, y first (Gemma 4's own convention)
             gemma4-native-think  the same with thinking on
             gemma4-json          pixel x, y (undersells the model: kept for comparison)

  python -m slotread.agent --url https://en.wikipedia.org --task "..." --model j-E3=runs/j_E3 --steps 12
  python -m slotread.agent --url https://en.wikipedia.org --task "..." --model gemma4-native

run() yields one event per step (screenshot, decision, timings), which scripts/race.py records.
Safety: navigation is confined to the starting site, and anything that looks like a credential field
is refused.
"""

import argparse, base64, json, os, re, time
from urllib.parse import urlparse


from .client import SlotModel, post
from .data.aguvis import ACTIONS, KEYS, SDIR

TEXT_MODEL = os.environ.get("TEXT_MODEL", "gemma4")  # the base model, served beside the adapters
TEXT_ASK = {"type": "What exact text should be typed for this step? Answer with the text only, no quotes.",
            "answer": "Give the final answer for this task. Answer with the text only.",
            "open": "Which site should be opened? Answer with a domain only."}
JSON_SYSTEM = (
    "You are a GUI agent. Given a screenshot, the task and the previous actions, answer the next step "
    "as one JSON object and nothing else: {\"action\": one of [click, type, scroll, swipe, press, open, "
    "back, home, wait, answer, select], \"x\": 0-1 from the left, \"y\": 0-1 from the top, \"x2\", \"y2\": "
    "a swipe's end, \"key\": the key to press, \"sdir\": scroll direction, \"samt\": scroll amount 0-9}. "
    "Include only the fields the action needs.")
TEXT_RULE = ' For type, include "text": the text to type; for answer, include "text": the final answer.'
SECRET = re.compile(r"pass|card|cvv|ssn|secret|token|otp|pin\b", re.I)
KEYMAP = {"enter": "Enter", "tab": "Tab", "escape": "Escape", "space": " ", "delete": "Delete",
          "up": "ArrowUp", "down": "ArrowDown", "left": "ArrowLeft", "right": "ArrowRight"}
VANILLA = ("gemma4-native", "gemma4-native-think", "gemma4-json")


def user_text(task, done):
    history = "\n".join(f"{i + 1}. {d}" for i, d in enumerate(done)) or "None"
    return ("Please generate the next move according to the UI screenshot, instruction and previous actions.\n\n"
            f"Instruction: {task}\n\nPrevious actions:\n{history}")


def write_text(img, user, action):
    """The base model writes a text argument (what to type, the answer, the site to open)."""
    msgs = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": img}},
                                         {"type": "text", "text": user + "\n\n" + TEXT_ASK[action]}]}]
    d, _ = post(dict(model=TEXT_MODEL, messages=msgs, max_tokens=24, temperature=0,
                     chat_template_kwargs={"enable_thinking": False}))
    t = ((d.get("choices") or [{}])[0].get("message", {}) or {}).get("content") or ""
    return t.strip().strip('"').split("\n")[0][:120]


def decide(model, img, user, size=(1280, 800), lazy=True):
    """One step: (action, point {x, y} or None, args, text, raw output). `model` is a SlotModel or one
    of VANILLA. Slot models read only the slots the action needs (lazy)."""
    if isinstance(model, SlotModel):
        lp = model.read("agent", img, user, lazy=lazy)
        d = model.tasks["agent"].decode(lp, model.arm)
        top = lambda s, names, default: names[int(lp[s].argmax())] if s in lp else default
        args = {"x2": d.get("x2", d["x"]), "y2": d.get("y2", d["y"]), "key": top("key", KEYS, "enter"),
                "sdir": top("sdir", SDIR, "down"), "samt": int(lp["samt"].argmax()) if "samt" in lp else 5}
        text = write_text(img, user, d["action"]) if d["action"] in TEXT_ASK else None
        return d["action"], {"x": d["x"], "y": d["y"]}, args, text, None
    native, think = model.startswith("gemma4-native"), model.endswith("-think")
    msgs = [{"role": "system", "content": [{"type": "text", "text": json_system(native, size)}]},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": img}}, {"type": "text", "text": user}]}]
    d, _ = post(dict(model=TEXT_MODEL, messages=msgs, max_tokens=2048 if think else 96, temperature=0,
                     chat_template_kwargs={"enable_thinking": think}))
    txt = ((d.get("choices") or [{}])[0].get("message", {}) or {}).get("content") or ""
    return parse_step(txt, native, size) + (txt,)


def json_system(native, size=(1280, 800)):
    """The vanilla system prompt: Gemma 4's native [y, x] on 0-1000, or pixels."""
    if native:
        system = JSON_SYSTEM.replace('"x": 0-1 from the left, "y": 0-1 from the top, "x2", "y2": a swipe\'s end',
                                     '"point": [y, x] normalized to 0-1000 (y first), "point2": a swipe\'s end the same way')
    else:  # the base model answers in pixels when asked for fractions, so ask for pixels outright
        system = JSON_SYSTEM.replace('"x": 0-1 from the left, "y": 0-1 from the top',
                                     f'"x", "y": pixel coordinates in the {size[0]}x{size[1]} screenshot')
    return system + TEXT_RULE


def parse_step(txt, native, size=(1280, 800)):
    """A vanilla answer -> (action, point {x, y} in [0, 1] or None, args, text)."""
    o = {}
    for m in reversed(list(re.finditer(r"\{[^{}]*\}", txt))):  # the last JSON object (after any thinking)
        try:
            o = json.loads(m.group(0))
            break
        except Exception:
            continue
    num = lambda v: float(v) if isinstance(v, (int, float)) else None
    if native:
        yx = lambda v: ({"x": v[1] / 1000, "y": v[0] / 1000} if isinstance(v, list) and len(v) == 2
                        and all(num(c) is not None for c in v) else None)
        pt, end = yx(o.get("point")), yx(o.get("point2")) or {"x": 0.5, "y": 0.5}
    else:
        pt = ({"x": num(o["x"]) / size[0], "y": num(o["y"]) / size[1]}
              if num(o.get("x")) is not None and num(o.get("y")) is not None else None)
        end = {"x": (num(o.get("x2")) or size[0] / 2) / size[0], "y": (num(o.get("y2")) or size[1] / 2) / size[1]}
    action = o.get("action") if o.get("action") in ACTIONS else "wait"
    args = {"x2": end["x"], "y2": end["y"], "key": o.get("key", "enter"), "sdir": o.get("sdir", "down"),
            "samt": int(o["samt"]) if num(o.get("samt")) is not None else 5}
    return action, pt, args, o.get("text")


def shot_url(page, tries=3):
    """Chromium occasionally fails a capture right after a click; retry rather than end the run."""
    for k in range(tries):
        try:
            b = page.screenshot(type="jpeg", quality=80, timeout=10000)
            return "data:image/jpeg;base64," + base64.b64encode(b).decode()
        except Exception:
            if k == tries - 1:
                raise
            page.wait_for_timeout(400)


def field_at(page, x, y):
    """What sits under the point, so typing into a password box can be refused."""
    try:
        return page.evaluate("""([x, y]) => { const e = document.elementFromPoint(x, y); if (!e) return {};
            return { tag: e.tagName, type: e.type || "", name: e.name || "", ph: e.placeholder || "",
                     aria: e.getAttribute("aria-label") || "" }; }""", [x, y]) or {}
    except Exception:
        return {}


def act(page, action, pt, args, text, size):
    """Execute one action. Returns the history line, or raises."""
    W, H = size
    x, y = (pt["x"] * W, pt["y"] * H) if pt else (0, 0)
    if action in ("click", "select"):
        f = field_at(page, x, y)
        if f.get("type") == "password" or SECRET.search(" ".join(str(v) for v in f.values())):
            raise PermissionError("refusing to interact with a credential field")
        page.mouse.click(x, y)
        return f"click(x={pt['x']:.3f}, y={pt['y']:.3f})"
    if action == "type":
        f = field_at(page, x, y) if pt else {}
        if f.get("type") == "password" or SECRET.search(" ".join(str(v) for v in f.values()) + " " + (text or "")):
            raise PermissionError("refusing to type into a credential field")
        # keystrokes go to whatever has focus, and in the training data a click always came first:
        # focus the field under the point unless it already has focus
        if pt and f.get("tag") in ("INPUT", "TEXTAREA") and not page.evaluate(
                "([x, y]) => document.activeElement === document.elementFromPoint(x, y)", [x, y]):
            page.mouse.click(x, y)
            page.wait_for_timeout(120)
        page.keyboard.type(text or "", delay=12)
        return f'type("{text}")'
    if action == "scroll":
        amt = (args.get("samt", 5) + 1) * 220
        dx, dy = {"up": (0, -amt), "down": (0, amt), "left": (-amt, 0), "right": (amt, 0)}.get(args.get("sdir", "down"), (0, amt))
        page.mouse.wheel(dx, dy)
        return f"scroll(direction='{args.get('sdir', 'down')}', amount={args.get('samt', 5)})"
    if action == "swipe":
        page.mouse.move(x, y); page.mouse.down()
        page.mouse.move(args["x2"] * W, args["y2"] * H, steps=12); page.mouse.up()
        return f"swipe(from=({pt['x']:.2f},{pt['y']:.2f}), to=({args['x2']:.2f},{args['y2']:.2f}))"
    if action == "press":
        page.keyboard.press(KEYMAP.get(args.get("key", "enter"), "Enter"))
        return f"press(keys='{args.get('key', 'enter')}')"
    if action == "back":
        page.go_back(wait_until="domcontentloaded")
        return "navigate_back()"
    if action in ("home", "open"):
        page.goto(f"https://{urlparse(page.url).netloc}/", wait_until="domcontentloaded")
        return "navigate_home()"
    if action == "wait":
        page.wait_for_timeout(800)
        return "wait()"
    raise ValueError(action)


def run(task, url, steps, model, size=(1280, 800), pace=0.4, headless=True):
    """Yields step events with timestamps relative to the run's start (t_shot, t_decided, t_acted)."""
    from playwright.sync_api import sync_playwright
    host = urlparse(url).netloc
    done, last, repeats = [], None, 0
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        page = browser.new_page(viewport={"width": size[0], "height": size[1]})
        page.set_default_timeout(15000)
        page.goto(url, wait_until="domcontentloaded")
        t0 = time.perf_counter()
        rel = lambda: round(time.perf_counter() - t0, 3)
        yield {"type": "start", "url": page.url, "task": task, "size": list(size)}
        for i in range(steps):
            page.wait_for_timeout(int(pace * 1000))
            t_shot = rel()
            img = shot_url(page)
            ut = user_text(task, done)
            t = time.perf_counter()
            action, pt, args, text, raw = decide(model, img, ut, size)
            ev = {"type": "step", "i": i + 1, "image": img, "action": action, "point": pt, "args": args, "text": text,
                  "ms": round((time.perf_counter() - t) * 1000), "url": page.url, "title": page.title()[:80],
                  "t_shot": t_shot, "t_decided": rel(), "raw": raw}
            if action == "answer":
                ev.update(done=True, t_acted=ev["t_decided"])
                yield ev
                break
            # a model stuck on one action gets one nudge: the page not changing is the tell
            sig = (action, page.url, ev["title"])
            repeats = repeats + 1 if sig == last else 0
            last = sig
            if repeats >= 2:
                ev["note"] = "the same action three times with no change: " + ("enter" if action == "type" else "scrolling")
                action, args = ("press", {**args, "key": "enter"}) if action == "type" else ("scroll", {**args, "sdir": "down", "samt": 4})
            try:
                desc = act(page, action, pt, args, text, size)
            except Exception as e:
                ev["error"] = repr(e)[:160]
                yield ev
                break
            if urlparse(page.url).netloc != host:  # stay on the site we started on
                ev["note"] = f"navigation to {urlparse(page.url).netloc} blocked; going back"
                page.go_back(wait_until="domcontentloaded")
            done.append(desc)
            ev.update(executed=desc, t_acted=rel())
            yield ev
        else:
            yield {"type": "end", "reason": "step limit", "history": done, "t": rel(), "image": shot_url(page)}
            browser.close()
            return
        yield {"type": "end", "reason": "finished", "history": done, "t": rel(), "image": shot_url(page)}
        browser.close()


def model_of(spec):
    """"name=run_dir" -> SlotModel; a VANILLA name -> itself."""
    return SlotModel(*spec.split("=", 1)) if "=" in spec else spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--model", required=True, help="name=run_dir for a slot adapter, or " + " / ".join(VANILLA))
    a = ap.parse_args()
    for ev in run(a.task, a.url, a.steps, model_of(a.model)):
        if ev["type"] == "step":
            pt = ev.get("point")
            where = f"at {pt['x']:.3f},{pt['y']:.3f}" if pt else "(no point)"
            print(f"[{ev['i']}] {ev['action']} {(ev.get('text') or '')[:60]} {where} {ev['ms']} ms -> "
                  f"{ev.get('executed') or ev.get('error') or ''} | {ev['title']}", flush=True)
        elif ev["type"] == "end":
            print("end:", ev["reason"], flush=True)


if __name__ == "__main__":
    main()
