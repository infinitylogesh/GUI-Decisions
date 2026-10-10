"""Side-by-side race: the same Gemma 4 31B on the same browser task, reading its steps from slots (a
slot adapter, e.g. E3) or writing them as JSON (the untouched base model).

Each run is recorded alone on an idle server (slotread.agent.run timestamps every step), then the two
recordings are replayed side by side on their real timelines, so the gap on screen is the measured one
and neither run slowed the other. The replay page renders to an MP4 with Playwright and ffmpeg.

  python scripts/race.py record --task "..." --url https://en.wikipedia.org --slots j-E3=runs/j_E3 \
      [--vanilla gemma4-native | gemma4-native-think] [--steps 12] [--out results/race.json]
  python scripts/race.py decide results/race.json results/decide.json --slots j-E3=runs/j_E3 [--skip 1]
      # decisions only: both models decide every step of one recorded trajectory, same screenshots
  python scripts/race.py html results/race.json results/race.html
  python scripts/race.py video results/race.html results/race.mp4
"""

import argparse, base64, io, json, os, subprocess, time

from slotread import agent as B

VANILLA = {"gemma4-json": ("Writes JSON", "base model · generates the step as text (pixel x, y)"),
           "gemma4-native": ("Writes JSON", "base model · generates the step in its native [y, x] format"),
           "gemma4-native-think": ("Thinks, then writes JSON", "base model · reasons, then writes the step ([y, x])")}


def agents(slots, vanilla):
    name = slots.split("=", 1)[0]
    return [(B.model_of(slots), name, "Slot read", f"{name} adapter · one read per step, no decoding"),
            (vanilla, vanilla, *VANILLA[vanilla])]


def record(task, url, steps, out, pace, slots, vanilla="gemma4-native"):
    runs = []
    for model, key, name, sub in agents(slots, vanilla):
        evs = []
        for ev in B.run(task, url, steps, model, pace=pace):
            evs.append({k: v for k, v in ev.items() if k not in ("history", "args")})
            if ev["type"] == "step":
                print(f"  {name} [{ev['i']}] {ev['action']} {(ev.get('text') or '')[:40]!r} model {ev['ms']} ms "
                      f"| {ev.get('executed') or ev.get('error') or ''}", flush=True)
        end = evs[-1]
        print(f"{name}: {end.get('reason')} in {end.get('t')} s, {sum(e['type'] == 'step' for e in evs)} steps", flush=True)
        runs.append(dict(model=key, name=name, sub=sub, events=evs))
    json.dump(dict(task=task, url=url, pace=pace, runs=runs, recorded=time.strftime("%Y-%m-%d %H:%M")), open(out, "w"))
    print(f"wrote {out}", flush=True)


def decide(src, run_idx, out, slots, vanilla="gemma4-native", skip=0, clicks_only=False):
    """Decision-only race: both models decide every step of one recorded trajectory, on the same
    screenshot and history, one request at a time (each model's first look at each screenshot),
    alternating which goes first. No browser time: a pane moves on when its decision is in, so the
    gap on screen is the deciding alone. skip: wasted opening steps (kept in the history)."""
    from PIL import Image, ImageOps
    R = json.load(open(src))
    traj = R["runs"][run_idx]
    start = next(e for e in traj["events"] if e["type"] == "start")
    size = tuple(start.get("size", (1280, 800)))
    all_steps = [e for e in traj["events"] if e["type"] == "step"]
    steps = all_steps[skip:]
    if clicks_only:  # the navigation decisions: clicks, no text-writing steps
        steps = [e for e in steps if e["action"] in ("click", "select")]
    models = agents(slots, vanilla)
    out_runs = {key: [] for _, key, _, _ in models}
    # warm-up on an image outside the trajectory (the first screenshot, inverted), enough rounds to reach
    # every vLLM API-server process (each one's first request is slow); the trajectory stays cold
    im = Image.open(io.BytesIO(base64.b64decode(steps[0]["image"].split(",", 1)[1]))).convert("RGB")
    b = io.BytesIO(); ImageOps.invert(im).save(b, "JPEG", quality=80)
    warm = "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()
    for _ in range(8):
        for model, _, _, _ in models:
            B.decide(model, warm, B.user_text(R["task"], []), size)
            B.write_text(warm, B.user_text(R["task"], []), "answer")
    done = [e["executed"] for e in all_steps[:skip] if e.get("executed")]
    for i, st in enumerate(steps):
        ut = B.user_text(R["task"], done)
        for model, key, _, _ in (models if i % 2 == 0 else models[::-1]):
            t = time.perf_counter()
            action, pt, args, text, raw = B.decide(model, st["image"], ut, size)  # slots: text steps timed too
            ms = (time.perf_counter() - t) * 1000
            out_runs[key].append(dict(i=i + 1, image=st["image"], action=action, point=pt, text=text, ms=round(ms)))
            print(f"  step {i + 1} {key:20s} {action:6s} {str(text or '')[:30]:30s} {ms:5.0f} ms", flush=True)
        if st.get("executed"):
            done.append(st["executed"])
    runs = []
    for _, key, name, sub in models:
        t, evs = 0.0, [{"type": "start"}]
        for d in out_runs[key]:  # the replay timeline: decisions back to back
            evs.append(dict(type="step", i=d["i"], image=d["image"], action=d["action"], point=d["point"], text=d["text"],
                            ms=d["ms"], t_shot=round(t, 3), t_decided=round(t + d["ms"] / 1000, 3),
                            t_acted=round(t + d["ms"] / 1000, 3)))
            t += d["ms"] / 1000
        evs.append(dict(type="end", reason="finished", t=round(t, 3), image=out_runs[key][-1]["image"]))
        runs.append(dict(model=key, name=name, sub=sub, events=evs))
    json.dump(dict(task=R["task"], url=R["url"], mode="decide", runs=runs, source=src,
                   recorded=time.strftime("%Y-%m-%d %H:%M")), open(out, "w"))
    print(f"wrote {out}", flush=True)


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Slots vs JSON</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap">
<style>
:root { --bg:#0e1116; --panel:#161b22; --ink:#e6edf3; --ink2:#9aa7b4; --ink3:#6b7785; --rule:#262d36;
        --fast:#3fb950; --fast-soft:rgba(63,185,80,.14); --slow:#e3934b; --slow-soft:rgba(227,147,75,.14); }
* { box-sizing:border-box; } html, body { margin:0; background:var(--bg); color:var(--ink);
  font:15px/1.45 "Instrument Sans",system-ui,sans-serif; }
.wrap { max-width:1600px; margin:0 auto; padding:26px 28px 22px; }
h1 { font-size:28px; margin:0; letter-spacing:-.01em; } .sub { color:var(--ink2); margin:4px 0 18px; font-size:15px; }
.task { font-family:"JetBrains Mono",monospace; color:var(--ink); background:var(--panel); border:1px solid var(--rule);
  padding:3px 9px; border-radius:6px; font-size:13.5px; }
.cols { display:grid; grid-template-columns:1fr 1fr; gap:22px; }
.col { background:var(--panel); border:1px solid var(--rule); border-radius:14px; padding:14px; position:relative; min-width:0; }
.head { display:flex; justify-content:space-between; align-items:flex-end; gap:12px; margin-bottom:10px; }
.name { font-size:20px; font-weight:700; } .who { color:var(--ink2); font-size:13px; }
.clock { font-family:"JetBrains Mono",monospace; font-size:34px; font-weight:600; font-variant-numeric:tabular-nums; }
.fast .name, .fast .clock { color:var(--fast); } .slow .name, .slow .clock { color:var(--slow); }
.screen { position:relative; line-height:0; border-radius:8px; overflow:hidden; border:1px solid var(--rule); background:#000; }
.screen img { width:100%; display:block; }
.mark { position:absolute; width:34px; height:34px; margin:-17px 0 0 -17px; border-radius:50%; border:3px solid #fff;
  box-shadow:0 0 0 3px rgba(0,0,0,.45); pointer-events:none; }
.fast .mark { background:rgba(63,185,80,.35); } .slow .mark { background:rgba(227,147,75,.35); }
.status { position:absolute; left:12px; top:12px; padding:6px 11px; border-radius:999px; font-size:14px; font-weight:600;
  backdrop-filter:blur(6px); background:rgba(14,17,22,.82); line-height:1.3; }
.fast .status { color:var(--fast); } .slow .status { color:var(--slow); }
.done { position:absolute; inset:0; display:flex; align-items:center; justify-content:center; background:rgba(14,17,22,.55);
  font-size:30px; font-weight:700; line-height:1.3; text-align:center; }
.done > div { background:#0e1116; border:1px solid var(--rule); border-radius:14px; padding:18px 28px;
  box-shadow:0 10px 40px rgba(0,0,0,.5); }
.fast .done { color:var(--fast); } .slow .done { color:var(--slow); }
.steps { margin-top:10px; display:flex; flex-direction:column; gap:4px; font-size:13.5px; min-height:150px; }
.st { display:flex; justify-content:space-between; gap:10px; padding:4px 8px; border-radius:6px; color:var(--ink2); }
.st b { color:var(--ink); font-weight:600; } .st .ms { font-family:"JetBrains Mono",monospace; font-variant-numeric:tabular-nums; }
.fast .st.on { background:var(--fast-soft); } .slow .st.on { background:var(--slow-soft); }
.bar { margin-top:16px; display:grid; grid-template-columns:repeat(3, 1fr); gap:14px; }
.kpi { background:var(--panel); border:1px solid var(--rule); border-radius:12px; padding:12px 14px; }
.kpi .v { font-family:"JetBrains Mono",monospace; font-size:24px; font-weight:600; font-variant-numeric:tabular-nums; }
.kpi .l { color:var(--ink2); font-size:13px; } .kpi .v .f { color:var(--fast); } .kpi .v .s { color:var(--slow); }
.foot { color:var(--ink3); font-size:12px; margin-top:10px; display:flex; justify-content:space-between; gap:12px; }
button { font:inherit; background:var(--panel); color:var(--ink); border:1px solid var(--rule); border-radius:8px; padding:5px 12px; cursor:pointer; }
</style></head><body><div class="wrap">
<h1 id="h1">Same model, same task. One reads its step, one writes it.</h1>
<div class="sub"><span id="lede">Gemma 4 31B on a live browser task:</span> <span class="task" id="task"></span></div>
<div class="cols" id="cols"></div>
<div class="bar" id="kpis"></div>
<div class="foot"><span id="note"></span><span><button id="replay">Replay</button></span></div>
</div>
<script>
const R = __DATA__;
const $ = id => document.getElementById(id);
$("task").textContent = R.task;
const DECIDE = R.mode === "decide";
if (DECIDE) { $("h1").textContent = "Same model, same screenshots. One reads its decision, one writes it.";
  $("lede").textContent = "Gemma 4 31B deciding each step of a browser task:"; }
const cls = ["fast", "slow"];
const runs = R.runs.map((r, k) => {
  const steps = r.events.filter(e => e.type === "step");
  const end = r.events.find(e => e.type === "end") || {};
  const tEnd = end.t ?? (steps.length ? (steps[steps.length - 1].t_acted ?? steps[steps.length - 1].t_decided) : 0);
  const col = document.createElement("div"); col.className = "col " + cls[k];
  col.innerHTML = `<div class="head"><div><div class="name">${r.name}</div><div class="who">${r.sub}</div></div>
    <div class="clock">0.0 s</div></div><div class="screen"><img alt=""><div class="mark" hidden></div>
    <div class="status"></div></div><div class="steps"></div>`;
  $("cols").appendChild(col);
  return { r, steps, end, tEnd, col, img: col.querySelector("img"), mark: col.querySelector(".mark"),
           status: col.querySelector(".status"), clock: col.querySelector(".clock"), list: col.querySelector(".steps"), shown: null };
});
const modelMs = r => r.steps.reduce((a, s) => a + s.ms, 0);
const fmt = s => s.toFixed(1) + " s";
const [A, B] = runs;
const perStep = r => modelMs(r) / Math.max(1, r.steps.length);
$("kpis").innerHTML =
  `<div class="kpi"><div class="v"><span class="f">${Math.round(perStep(A))} ms</span> vs <span class="s">${Math.round(perStep(B))} ms</span></div><div class="l">model time per step</div></div>` +
  `<div class="kpi"><div class="v">${(perStep(B) / perStep(A)).toFixed(1)}× faster</div><div class="l">deciding each step: reading slots instead of generating JSON</div></div>` +
  `<div class="kpi"><div class="v"><span class="f">${fmt(A.tEnd)}</span> vs <span class="s">${fmt(B.tEnd)}</span></div><div class="l">${DECIDE ? `all ${A.steps.length} decisions, back to back (no browser time)` : `whole task, browser time included (${A.steps.length} vs ${B.steps.length} steps)`}</div></div>`;
$("note").textContent = DECIDE
  ? `Each decision measured alone, cold, on one RTX PRO 6000 (vLLM, NVFP4), on the same screenshots and history; replayed in real time. Recorded ${R.recorded}.`
  : `Each run recorded alone on one RTX PRO 6000 (vLLM, NVFP4), replayed on its real timeline. Recorded ${R.recorded}.`;
const label = s => s.action + (s.text ? ` "${s.text.slice(0, 38)}"` : "");
function render(T) {
  for (const x of runs) {
    const t = Math.min(T, x.tEnd);
    x.clock.textContent = fmt(t);
    let k = -1;
    x.steps.forEach((s, i) => { if (s.t_shot <= T) k = i; });
    const s = x.steps[Math.max(0, k)];
    const finished = T >= x.tEnd;
    const src = finished && x.end.image ? x.end.image : s && s.image;
    if (src && x.shown !== src) { x.img.src = src; x.shown = src; }
    const deciding = s && k >= 0 && T < s.t_decided && !finished;
    x.status.hidden = finished;
    if (!s) x.status.textContent = "starting";
    else if (deciding) x.status.textContent = `step ${s.i} · deciding… ${Math.round((T - s.t_shot) * 1000)} ms`;
    else x.status.textContent = `step ${s.i} · ${label(s)} · ${s.ms} ms`;
    const showMark = s && s.point && !deciding && !finished && k >= 0;
    x.mark.hidden = !showMark;
    if (showMark) { x.mark.style.left = (s.point.x * 100) + "%"; x.mark.style.top = (s.point.y * 100) + "%"; }
    x.list.innerHTML = x.steps.filter((q, i) => i <= k).slice(-6).map(q => {
      const pending = q === s && deciding;  // not decided yet: neither the action nor its time is known
      return `<div class="st ${q === s && !finished ? "on" : ""}"><span><b>${q.i}.</b> ${pending ? "deciding…" : label(q)}</span>` +
             `<span class="ms">${pending ? "…" : q.ms + " ms"}</span></div>`; }).join("");
    let d = x.col.querySelector(".done");
    if (finished && !d) { d = document.createElement("div"); d.className = "done";
      d.innerHTML = `<div>${DECIDE ? `${x.steps.length} decisions` : x.end.reason === "finished" ? "Done" : "Not finished"} · ${fmt(x.tEnd)}<div style="font-size:17px;font-weight:500;margin-top:4px">${DECIDE ? "" : `${x.steps.length} steps · `}${Math.round(perStep(x))} ms per decision</div></div>`;
      x.col.querySelector(".screen").appendChild(d); }
    if (!finished && d) d.remove();
  }
}
let t0 = null;
function frame(now) {
  if (t0 === null) t0 = now;
  const T = (now - t0) / 1000;
  render(T);
  if (T < Math.max(A.tEnd, B.tEnd) + 3) requestAnimationFrame(frame);
  else window.__raceDone = true;
}
function start() { t0 = null; window.__raceDone = false; runs.forEach(x => { x.shown = null; const d = x.col.querySelector(".done"); if (d) d.remove(); });
  requestAnimationFrame(frame); }
$("replay").onclick = start;
const go = () => setTimeout(start, new URLSearchParams(location.search).has("video") ? 1200 : 400);
Promise.all([...document.images].map(i => i.decode ? i.decode().catch(() => {}) : 0)).then(go);
</script></body></html>"""


PAGE_MIN = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Slot read vs writing JSON</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Inter+Tight:wght@700;800&family=JetBrains+Mono:wght@400;500;600&display=swap">
<style>
/* the Ambient look of the blog figures: cream canvas, dark-brown ink, orange = slot read, dark brown = writes JSON */
:root { --bg:#fdf6f1; --panel:#fff; --ink:#1a110d; --ink2:#6b5850; --ink3:#9a877e; --rule:rgba(58,28,12,.11); --dot:rgba(58,28,12,.07);
  --fast:#ff5a14; --fast-soft:#ffe5d8; --slow:#3a2013; --ink-panel:#1c100b; --on-ink:#fbeee6;
  --shadow:0 1px 2px rgba(58,28,12,.05), 0 12px 32px -14px rgba(58,28,12,.18); }
* { box-sizing:border-box; } html, body { margin:0; color:var(--ink); font:15px/1.5 "Inter",system-ui,sans-serif;
  background:radial-gradient(var(--dot) 1.2px, transparent 1.2px) -12px -12px / 24px 24px, var(--bg); }
.wrap { max-width:1600px; margin:0 auto; padding:34px 40px; }
.top { display:flex; justify-content:space-between; align-items:flex-end; gap:30px; margin-bottom:22px; }
.eyebrow { font:600 12px "Inter",sans-serif; letter-spacing:.26em; text-transform:uppercase; color:var(--fast); margin-bottom:8px; }
.task { font:500 17px/1.45 "JetBrains Mono",monospace; color:var(--on-ink); background:var(--ink-panel); border-radius:12px; padding:12px 16px;
  max-width:980px; box-shadow:0 18px 36px -22px rgba(42,23,14,.6); }
.speed { text-align:right; opacity:0; transform:translateY(8px); transition:opacity .6s, transform .6s; white-space:nowrap; }
.speed.on { opacity:1; transform:none; }
.speed .x { font:800 64px/1 "Inter Tight",sans-serif; letter-spacing:-.04em; color:var(--fast); }
.speed .sub { font:500 15px "Inter",sans-serif; color:var(--ink2); margin-top:6px; }
.cols { display:grid; grid-template-columns:1fr 1fr; gap:26px; }
.col { background:var(--panel); border:1px solid var(--rule); border-radius:18px; padding:18px; box-shadow:var(--shadow); }
.head { display:flex; justify-content:space-between; align-items:baseline; margin-bottom:12px; }
.name { font:800 26px "Inter Tight",sans-serif; letter-spacing:-.025em; }
.clock { font:800 44px/1 "Inter Tight",sans-serif; letter-spacing:-.04em; font-variant-numeric:tabular-nums; }
.fast .name, .fast .clock { color:var(--fast); } .slow .name, .slow .clock { color:var(--slow); }
.screen { position:relative; line-height:0; border-radius:10px; overflow:hidden; border:1px solid var(--rule); }
.screen img { width:100%; display:block; }
.mark { position:absolute; width:40px; height:40px; margin:-20px 0 0 -20px; border-radius:50%; border:3px solid #fff; pointer-events:none;
  box-shadow:0 0 0 3px rgba(26,17,13,.35); }
.fast .mark { background:rgba(255,90,20,.45); } .slow .mark { background:rgba(58,32,19,.45); }
.status { position:absolute; left:14px; top:14px; padding:7px 13px; border-radius:999px; font:600 15px "JetBrains Mono",monospace; line-height:1.3;
  background:var(--ink-panel); color:var(--on-ink); box-shadow:0 8px 20px -10px rgba(42,23,14,.6); }
.fast .status b { color:#ff7a2e; } .slow .status b { color:#f6c177; }
.done { position:absolute; inset:0; display:flex; align-items:center; justify-content:center; background:rgba(253,246,241,.55); line-height:1.25; }
.done > div { background:var(--panel); border:1px solid var(--rule); border-radius:16px; padding:18px 30px; text-align:center; box-shadow:var(--shadow); }
.done .h { font:800 34px "Inter Tight",sans-serif; letter-spacing:-.03em; } .fast .done .h { color:var(--fast); } .slow .done .h { color:var(--slow); }
.done .s { font:500 15px "Inter",sans-serif; color:var(--ink2); margin-top:4px; }
</style></head><body><div class="wrap">
<div class="top">
  <div><div class="eyebrow">Task</div><div class="task" id="task"></div></div>
  <div class="speed" id="speed"><div class="x" id="sx"></div><div class="sub" id="ss"></div></div>
</div>
<div class="cols" id="cols"></div>
</div>
<script>
const R = __DATA__;
const $ = id => document.getElementById(id);
$("task").textContent = R.task;
const NAMES = ["Slot read", R.runs[1].name === "Thinks, then writes JSON" ? "Thinks, then writes JSON" : "Writes JSON"];
const runs = R.runs.map((r, k) => {
  const steps = r.events.filter(e => e.type === "step"), end = r.events.find(e => e.type === "end") || {};
  const tEnd = end.t ?? (steps.length ? (steps[steps.length - 1].t_acted ?? steps[steps.length - 1].t_decided) : 0);
  const col = document.createElement("div"); col.className = "col " + (k ? "slow" : "fast");
  col.innerHTML = `<div class="head"><div class="name">${NAMES[k]}</div><div class="clock">0.00 s</div></div>
    <div class="screen"><img alt=""><div class="mark" hidden></div><div class="status"></div></div>`;
  $("cols").appendChild(col);
  return { steps, end, tEnd, col, img: col.querySelector("img"), mark: col.querySelector(".mark"), status: col.querySelector(".status"),
           clock: col.querySelector(".clock"), shown: null };
});
const [A, B] = runs, dec = x => x.steps.reduce((a, s) => a + s.ms, 0) / Math.max(1, x.steps.length);
const decide = R.mode === "decide";
$("sx").textContent = (decide ? dec(B) / dec(A) : B.tEnd / A.tEnd).toFixed(1) + "× faster";
$("ss").textContent = decide ? `per decision · ${Math.round(dec(A))} ms vs ${Math.round(dec(B))} ms · same model, same screenshots`
                             : `whole task · ${A.tEnd.toFixed(1)} s vs ${B.tEnd.toFixed(1)} s · same model, same task`;
function render(T) {
  for (const x of runs) {
    const fin = T >= x.tEnd;
    x.clock.textContent = Math.min(T, x.tEnd).toFixed(2) + " s";
    let k = -1; x.steps.forEach((s, i) => { if (s.t_shot <= T) k = i; });
    const s = x.steps[Math.max(0, k)], src = fin && x.end.image ? x.end.image : s && s.image;
    if (src && x.shown !== src) { x.img.src = src; x.shown = src; }
    const deciding = s && k >= 0 && T < s.t_decided && !fin;
    x.status.hidden = fin;
    x.status.innerHTML = !s ? "starting" : deciding ? `step ${s.i} · <b>deciding… ${Math.round((T - s.t_shot) * 1000)} ms</b>`
      : `step ${s.i} · ${s.action}${s.text ? ` "${s.text.slice(0, 26)}"` : ""} · <b>${s.ms} ms</b>`;
    const show = s && s.point && !deciding && !fin && k >= 0;
    x.mark.hidden = !show; if (show) { x.mark.style.left = s.point.x * 100 + "%"; x.mark.style.top = s.point.y * 100 + "%"; }
    let d = x.col.querySelector(".done");
    if (fin && !d) { d = document.createElement("div"); d.className = "done";
      d.innerHTML = `<div><div class="h">${x.end.reason === "finished" ? "Done" : "Stopped"} · ${x.tEnd.toFixed(2)} s</div><div class="s">${x.steps.length} decisions · ${Math.round(dec(x))} ms each</div></div>`;
      x.col.querySelector(".screen").appendChild(d); }
    if (!fin && d) d.remove();
  }
  $("speed").classList.toggle("on", T >= Math.max(A.tEnd, B.tEnd));
}
let t0 = null;
function frame(now) { if (t0 === null) t0 = now; const T = (now - t0) / 1000; render(T);
  if (T < Math.max(A.tEnd, B.tEnd) + 3) requestAnimationFrame(frame); else window.__raceDone = true; }
function start() { t0 = null; window.__raceDone = false; runs.forEach(x => { x.shown = null; const d = x.col.querySelector(".done"); if (d) d.remove(); }); requestAnimationFrame(frame); }
addEventListener("click", start);  // click anywhere to replay
const go = () => setTimeout(start, new URLSearchParams(location.search).has("video") ? 1200 : 600);
Promise.all([...document.images].map(i => i.decode ? i.decode().catch(() => {}) : 0)).then(go);
</script></body></html>"""


def html(src, out, style="min"):
    """style "min": the blog look (task, the two panes, the speedup); "full": the earlier dark page with step lists."""
    R = json.load(open(src))
    open(out, "w").write((PAGE_MIN if style == "min" else PAGE).replace("__DATA__", json.dumps(R)))
    print(f"wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB)")


def video(page_path, out, size=(1600, 1000)):
    """Plays the replay once in headless Chromium, recording the viewport, then converts to MP4."""
    from playwright.sync_api import sync_playwright
    vdir = os.path.join(os.path.dirname(os.path.abspath(out)), "_race_video")
    os.makedirs(vdir, exist_ok=True)
    with sync_playwright() as p:
        b = p.chromium.launch()
        ctx = b.new_context(viewport={"width": size[0], "height": size[1]}, record_video_dir=vdir,
                            record_video_size={"width": size[0], "height": size[1]}, device_scale_factor=1)
        page = ctx.new_page()
        page.goto("file://" + os.path.abspath(page_path) + "?video=1")
        page.wait_for_function("window.__raceDone === true", timeout=300000, polling=500)
        webm = page.video.path()
        ctx.close(); b.close()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", webm, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "20", "-movflags", "+faststart", out], check=True)
    print(f"wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record"); r.add_argument("--task", required=True); r.add_argument("--url", required=True)
    r.add_argument("--slots", required=True, help="name=run_dir of the served slot adapter")
    r.add_argument("--vanilla", default="gemma4-native", choices=list(VANILLA))
    r.add_argument("--steps", type=int, default=12); r.add_argument("--out", default="results/race.json")
    r.add_argument("--pace", type=float, default=0.4)
    dd = sub.add_parser("decide"); dd.add_argument("src"); dd.add_argument("out")
    dd.add_argument("--slots", required=True); dd.add_argument("--run", type=int, default=0)
    dd.add_argument("--vanilla", default="gemma4-native", choices=list(VANILLA))
    dd.add_argument("--skip", type=int, default=0); dd.add_argument("--clicks-only", action="store_true")
    h = sub.add_parser("html"); h.add_argument("src"); h.add_argument("out")
    h.add_argument("--style", default="min", choices=["min", "full"], help="min: the blog look; full: with step lists")
    v = sub.add_parser("video"); v.add_argument("page"); v.add_argument("out")
    a = ap.parse_args()
    if a.cmd == "record":
        record(a.task, a.url, a.steps, a.out, a.pace, a.slots, a.vanilla)
    elif a.cmd == "decide":
        decide(a.src, a.run, a.out, a.slots, a.vanilla, a.skip, a.clicks_only)
    elif a.cmd == "html":
        html(a.src, a.out, a.style)
    else:
        video(a.page, a.out)
