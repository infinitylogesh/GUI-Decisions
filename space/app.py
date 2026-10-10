"""GUI-Decisions: computer-use decisions read from one forward pass (slot reading on Gemma 4 31B).

The adapter is loaded on the base model (not the merged model) so the base model can still write the text
a type / answer step needs, and run the vanilla comparison, with the adapter switched off.
"""
import os

import gradio as gr
import numpy as np
import spaces
import torch
from PIL import Image, ImageDraw

from slotread.client import agent_user
from slotread.data.aguvis import ACTIONS, KEYS, SDIR
from slotread.local import LocalSlotModel
from slotread.numbers import unwarp

BASE = os.environ.get("BASE", "google/gemma-4-31B-it")
ADAPTER = os.environ.get("ADAPTER", "infinitylogesh/GUI-Decisions-31B-LoRA")
SLOT, VANILLA = (255, 90, 20), (58, 32, 19)  # the Ambient theme: orange the slot read, dark brown the base model writing
MAX_PX = 3840 * 2160
WARM = set()  # functions that have run once in this process: the first call includes CUDA warm-up


def warm_note(name):
    first = name not in WARM
    WARM.add(name)
    return " The first request after a restart includes GPU warm-up; run it again for a warm time." if first else ""

m = LocalSlotModel.from_pretrained(BASE, adapter=ADAPTER, processor=ADAPTER, device_map="cpu")
m.model.to("cuda")


def fit(im):
    im = im.convert("RGB")
    if im.width * im.height > MAX_PX:
        s = (MAX_PX / (im.width * im.height)) ** 0.5
        im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
    return im


def overlay(im, px, py, color):
    """The slot distributions as a heatmap: P(x) times P(y), drawn as the colour's opacity."""
    heat = np.outer(py, px)
    heat = (heat / heat.max()) ** 0.5
    a = Image.fromarray((heat * 150).astype(np.uint8)).resize(im.size, Image.BILINEAR)
    layer = Image.new("RGBA", im.size, color + (0,))
    layer.putalpha(a)
    return Image.alpha_composite(im.convert("RGBA"), layer)


def mark(d, x, y, color, r=14):
    d.ellipse([x - r, y - r, x + r, y + r], outline=(255, 255, 255), width=6)
    d.ellipse([x - r, y - r, x + r, y + r], outline=color, width=3)
    d.line([x - 4, y, x + 4, y], fill=color, width=2)
    d.line([x, y - 4, x, y + 4], fill=color, width=2)


def probs(v):
    return torch.softmax(v.float(), -1).numpy()


def describe(a, d, text):
    if a in ("click", "select"):
        return f"{a}(x={d['x']:.3f}, y={d['y']:.3f})"
    if a == "swipe":
        return f"swipe(({d['x']:.3f}, {d['y']:.3f}) → ({d['x2']:.3f}, {d['y2']:.3f}))"
    if a == "press":
        return f"press(key={d['key']})"
    if a == "scroll":
        return f"scroll(direction={d['sdir']}, amount={d['samt']})"
    if text is not None:
        return f'{a}("{text}")'
    return f"{a}()"


@spaces.GPU(size="xlarge", duration=40)
def decide(image, instruction, previous, compare):
    if image is None or not instruction.strip():
        raise gr.Error("Add a screenshot and an instruction.")
    im = fit(image)
    user_prev = previous.strip() or "None"
    lp, ms = m.read("agent", im, agent_user(instruction, user_prev))
    d = m.tasks["agent"].decode(lp, m.arm)
    d.update(key=KEYS[int(lp["key"].argmax())], sdir=SDIR[int(lp["sdir"].argmax())], samt=int(lp["samt"].argmax()))
    a = d["action"]
    text = m.write_text(im, instruction, a, user_prev) if a in ("type", "answer", "open") else None

    out = im.convert("RGBA")
    if a in ("click", "select", "swipe"):
        out = overlay(out, probs(lp["x"]), probs(lp["y"]), SLOT)
    dr = ImageDraw.Draw(out)
    W, H = im.size
    if a in ("click", "select", "swipe"):
        mark(dr, d["x"] * W, d["y"] * H, SLOT)
    if a == "swipe":
        dr.line([d["x"] * W, d["y"] * H, d["x2"] * W, d["y2"] * H], fill=SLOT, width=4)
        mark(dr, d["x2"] * W, d["y2"] * H, SLOT, r=8)

    md = (f"### Slots: `{describe(a, d, text)}`\n\n"
          f"One forward pass: **{ms:.0f} ms** on this GPU (bf16, transformers"
          + (", plus the base model writing the text" if text is not None else "") + ")." + warm_note("decide"))
    if compare:
        va, vpt, vargs, vtext, raw, vms = m.vanilla_decide(im, instruction, user_prev)
        if vpt is not None:
            mark(dr, vpt["x"] * W, vpt["y"] * H, VANILLA)
        vd = dict(vargs, **(vpt or {"x": float("nan"), "y": float("nan")}))
        md += (f"\n\n### Base model writing JSON: `{describe(va, vd, vtext)}`\n\n"
               f"Generating its answer took **{vms:.0f} ms** (Gemma 4's native `\"point\": [y, x]` format):\n\n"
               f"```\n{raw.strip()[:400]}\n```\n\n"
               "Orange: the slot read, with its x × y distribution as the heatmap. Dark brown: the base model's point.")
    label = {k: float(p) for k, p in zip(ACTIONS, probs(lp["action"]))}
    return out.convert("RGB"), md, label


@spaces.GPU(size="xlarge", duration=20)
def box(image, description):
    if image is None or not description.strip():
        raise gr.Error("Add an image and describe the element or object.")
    im = fit(image)
    lp, ms = m.read("box", im, description)
    b = m.tasks["box"].decode(lp, m.arm)
    W, H = im.size
    out = overlay(im, probs(lp["cx"]), probs(lp["cy"]), SLOT)
    dr = ImageDraw.Draw(out)
    xy = [b[0] * W, b[1] * H, b[2] * W, b[3] * H]
    dr.rectangle(xy, outline=(255, 255, 255), width=6)
    dr.rectangle(xy, outline=SLOT, width=3)
    w = unwarp(float(np.argmax(probs(lp["w"])) + 0.5) / m.arm.bins, "log")
    md = (f"### Box: `[{', '.join(f'{v:.3f}' for v in b)}]`\n\n"
          f"Centre and log-scaled size, four slots read in one forward pass: **{ms:.0f} ms** on this GPU. "
          f"The heatmap is the centre's x × y distribution (most likely width {w:.1%} of the image)." + warm_note("box"))
    return out.convert("RGB"), md


HERE = os.path.dirname(os.path.abspath(__file__))
ex = lambda f: os.path.join(HERE, "examples", f)
rows = lambda f: [l.split("\t") for l in open(ex(f)).read().splitlines() if l.strip()]  # tab-separated examples
INTRO = open(os.path.join(HERE, "intro.html")).read()
CSS = open(os.path.join(HERE, "ambient.css")).read()
HOW = open(os.path.join(HERE, "how.md")).read()

with gr.Blocks(title="GUI-Decisions") as demo:
    gr.HTML(INTRO)
    gr.HTML('<p class="amb-note">Timings here are this Space\'s shared GPU running bf16 in plain transformers. '
            'Served on vLLM (NVFP4, one RTX PRO 6000) a step takes ~150 ms, against ~470–550 ms for the same model '
            'writing its native JSON.</p>')
    with gr.Tab("Computer use"):
        with gr.Row():
            with gr.Column():
                img = gr.Image(type="pil", label="Screenshot")
                ins = gr.Textbox(label="Instruction", placeholder="open the settings")
                prev = gr.Textbox(label="Previous actions (optional)", placeholder="1. click(...)", lines=2)
                cmp = gr.Checkbox(value=True, label="Also run the base model writing its answer as JSON (slower)")
                go = gr.Button("Decide", variant="primary")
            with gr.Column():
                out_img = gr.Image(label="Decision")
                gr.HTML('<div class="amb-legend"><span><i style="background:#ff5a14"></i>slot read · P(x) × P(y)</span>'
                        '<span><i style="background:#3a2013"></i>base model writing JSON</span></div>')
                out_md = gr.Markdown()
                out_lab = gr.Label(label="action slot", num_top_classes=4)
        go.click(decide, [img, ins, prev, cmp], [out_img, out_md, out_lab])
        gr.Examples([[ex(f), i, p.replace("\\n", "\n"), True] for f, i, p in rows("agent.tsv")], [img, ins, prev, cmp])
    with gr.Tab("Boxes"):
        with gr.Row():
            with gr.Column():
                bimg = gr.Image(type="pil", label="Image")
                bdesc = gr.Textbox(label="Element or object", placeholder="the red car on the left")
                bgo = gr.Button("Locate", variant="primary")
            with gr.Column():
                bout = gr.Image(label="Box")
                bmd = gr.Markdown()
        bgo.click(box, [bimg, bdesc], [bout, bmd])
        gr.Examples([[ex(f), t] for f, t in rows("box.tsv")], [bimg, bdesc])
    with gr.Tab("How it works"):
        gr.Markdown(HOW)

# the Ambient theme: Inter / Inter Tight / JetBrains Mono, cream canvas, dark-brown ink, orange accent (light only)
ORANGE = gr.themes.Color(c50="#fff4ee", c100="#ffe5d8", c200="#ffc9ad", c300="#ffa77d", c400="#ff7a2e", c500="#ff5a14",
                         c600="#e64a08", c700="#bf3c06", c800="#8f2e07", c900="#5e1f06", c950="#3a1304")
TAUPE = gr.themes.Color(c50="#fdf6f1", c100="#f8ece4", c200="#efe2d9", c300="#d9c6ba", c400="#9a877e", c500="#6b5850",
                       c600="#4f3d35", c700="#3a2013", c800="#2a170e", c900="#1c100b", c950="#1a110d")
LIGHT = dict(
    body_background_fill="#fdf6f1", body_text_color="#1a110d", body_text_color_subdued="#6b5850",
    background_fill_primary="#ffffff", background_fill_secondary="#fbf1ea",
    block_background_fill="#ffffff", block_border_color="rgba(58,28,12,0.11)", block_border_width="1px",
    block_label_background_fill="#fbf1ea", block_label_text_color="#6b5850", block_title_text_color="#6b5850",
    block_shadow="0 1px 2px rgba(58,28,12,0.05), 0 12px 32px -14px rgba(58,28,12,0.18)",
    border_color_primary="rgba(58,28,12,0.11)", border_color_accent="#ff5a14",
    input_background_fill="#fdf6f1", input_border_color="rgba(58,28,12,0.11)", input_border_color_focus="#ff7a2e",
    button_primary_background_fill="#ff5a14", button_primary_background_fill_hover="#ff7a2e",
    button_primary_text_color="#ffffff", button_primary_border_color="#ff5a14",
    button_secondary_background_fill="#ffffff", button_secondary_background_fill_hover="#fbf1ea",
    button_secondary_text_color="#1a110d", button_secondary_border_color="rgba(58,28,12,0.11)",
    color_accent="#ff5a14", color_accent_soft="#ffe5d8", link_text_color="#ff5a14", link_text_color_hover="#e64a08",
    checkbox_background_color_selected="#ff5a14", checkbox_border_color_selected="#ff5a14", slider_color="#ff5a14",
    loader_color="#ff5a14", code_background_fill="#f3e7df", table_even_background_fill="#ffffff",
    table_odd_background_fill="#fbf1ea", table_border_color="rgba(58,28,12,0.11)",
)
THEME = gr.themes.Base(primary_hue=ORANGE, secondary_hue=TAUPE, neutral_hue=TAUPE, radius_size=gr.themes.sizes.radius_lg,
                       font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
                       font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"])
THEME.set(block_radius="16px", **LIGHT, **{k + "_dark": v for k, v in LIGHT.items() if hasattr(THEME, k + "_dark")})
HEAD = ('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700'
        '&family=Inter+Tight:wght@700;800&family=JetBrains+Mono:wght@400;500;600&display=swap">')
LIGHT_ONLY = "() => { document.body.classList.remove('dark'); document.documentElement.classList.remove('dark'); }"

if __name__ == "__main__":
    demo.queue().launch(theme=THEME, css=CSS, head=HEAD, js=LIGHT_ONLY)
