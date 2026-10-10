"""Masks from slot reads, refined by SAM 2 (the blog's appendix): our geometry as SAM 2's prompt.

The adapter reads a mask (centre + 24 rays, one request with prompt log-probabilities) and, separately,
a box from the box task; SAM 2.1 then draws the mask from them:

  poly            our 24-ray polygon itself (the baseline)
  pt              SAM 2 from the centre point
  box / box+pt    SAM 2 from the polygon's box (and the point)
  box+pt/agree    box+pt, keeping the one of SAM 2's three masks closest to our polygon (picks parts)
  grow1.4+pt      the polygon's box grown 40% + the point
  boxtask>sam     SAM 2 from the box task's box
  boxtask+pt      the box task's box + the point;  boxtask1.2+pt: that box grown 20% (the best)
  oracles         gt_box, gt_pt: SAM 2 from the true box / the true centre

  python scripts/sam2_refine.py 1000 j-E3=runs/j_E3 [j-E=runs/j_E]           # RefCOCO val
  SEG_SET=coco17 python scripts/sam2_refine.py 500 j-E3=runs/j_E3            # COCO val2017 (outside RefCOCO)

SAM 2 runs on SAM_GPU (default 0, beside vLLM: it needs a few GB). Writes
results/sam2_refine_<set>_<n>.json with per-case IoUs and SAM 2 timings.
"""

import json, math, os, statistics, sys, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from slotread.client import SlotModel
from slotread.data import evalsets, refcoco
from slotread.geometry import box_iou, grow, mask_iou, poly_mask, radial_decode
from slotread.images import data_url, image_of
from slotread.tasks import SegTask

OUT = os.environ.get("SLOTREAD_RESULTS", "results")
SAM_ID = os.environ.get("SAM_ID", "facebook/sam2.1-hiera-large")
READOUTS = ("default", "expected")
PROMPTS = ("poly", "pt", "box", "box+pt", "box+pt/agree", "grow1.4+pt", "boxtask>sam", "boxtask+pt", "boxtask1.2+pt")
BOXES = ("boxtask", "polybox", "sam_box")


def polygon(p, W, H):
    """Our decoded mask: (pixel mask, centre in pixels, polygon box in pixels)."""
    diag = math.hypot(W, H)
    m, pts = radial_decode(p["cx"] * W, p["cy"] * H, [r * diag for r in p["r"]], W, H)
    xs, ys = [q[0] for q in pts], [q[1] for q in pts]
    box = [max(0, min(xs)), max(0, min(ys)), min(W - 1, max(xs)), min(H - 1, max(ys))]
    return m, (p["cx"] * W, p["cy"] * H), box


def mask_box(m, W, H):
    ys, xs = np.nonzero(m)
    return [xs.min() / W, ys.min() / H, (xs.max() + 1) / W, (ys.max() + 1) / H] if len(xs) else [0, 0, 0, 0]


def sam_predict(pred, point=None, box=None, multi=False):
    kw = dict(multimask_output=multi)
    if point is not None:
        kw.update(point_coords=np.array([point], dtype=np.float32), point_labels=np.array([1]))
    if box is not None:
        kw["box"] = np.array(box, dtype=np.float32)
    masks, scores, _ = pred.predict(**kw)
    return masks.astype(bool), scores


def main():
    n, arms = int(sys.argv[1]), [SlotModel(*a.split("=", 1)) for a in sys.argv[2:]]
    seg_set = os.environ.get("SEG_SET", "refcoco")
    cases = (evalsets.coco17(n) if seg_set == "coco17" else
             refcoco.examples(refcoco.rows("validation"), n, seed=0, one_sentence=True))
    print(f"{len(cases)} {seg_set} cases, adapters {', '.join(a.name for a in arms)}", flush=True)

    # 1. slot reads, all arms (vLLM)
    def read(ic):
        i, ex = ic
        url = data_url(image_of(ex), fmt="png", max_px=0)  # as the post measured
        out = {}
        for arm in arms:
            t = time.perf_counter()
            lp = arm.read_one("seg", url, arm.tasks["seg"].user(ex))
            ms = (time.perf_counter() - t) * 1000
            bt = arm.tasks["box"]  # the box task, trained on UI elements only, on the same object
            lpb = arm.read("box", url, bt.user({"instruction": ex["sentence"]}))
            out[arm.name] = dict(ms=ms, p={how: arm.tasks["seg"].decode(lp, arm.arm, how) for how in READOUTS},
                                 box={how: bt.decode(lpb, arm.arm, how) for how in READOUTS})
        return out

    t = time.time()
    with ThreadPoolExecutor(6) as ex_:
        reads = list(ex_.map(read, enumerate(cases)))
    print(f"reads done in {time.time() - t:.0f}s", flush=True)

    # 2. SAM 2 on each image once, then every prompt
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    dev = f"cuda:{os.environ.get('SAM_GPU', '0')}"
    pred = SAM2ImagePredictor.from_pretrained(SAM_ID, device=dev)
    rows, embed_ms, prompt_ms = [], [], []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for k, (ex, rd) in enumerate(zip(cases, reads)):
            W, H = ex["size"]
            gt = poly_mask(ex["polys"], W, H)
            iou = lambda m: mask_iou(m, gt)  # (iou, inter, union)
            torch.cuda.synchronize(dev); t = time.perf_counter()
            pred.set_image(np.array(image_of(ex)))
            torch.cuda.synchronize(dev); embed_ms.append((time.perf_counter() - t) * 1000)
            row = dict(id=k, ref_id=ex["ref_id"], arms={})
            bx = ex["bbox"]  # x1, y1, x2, y2 in pixels (as SegTask reads it)
            gcx, gcy, _ = SegTask(24).encoded(ex)
            row["oracle"] = {"gt_box": iou(sam_predict(pred, box=bx)[0][0]),
                             "gt_pt": iou((lambda o: o[0][int(o[1].argmax())])(sam_predict(pred, point=(gcx, gcy), multi=True)))}
            gb = [bx[0] / W, bx[1] / H, bx[2] / W, bx[3] / H]
            for name, r in rd.items():
                row["arms"][name] = dict(read_ms=r["ms"])
                for how, p in r["p"].items():
                    poly, c, b = polygon(p, W, H)
                    res = {"poly": iou(poly)}
                    bt = r["box"][how]
                    bm = sam_predict(pred, box=[bt[0] * W, bt[1] * H, bt[2] * W, bt[3] * H])[0][0]
                    res["boxtask>sam"] = iou(bm)
                    t = time.perf_counter()
                    ms, sc = sam_predict(pred, point=c, multi=True)
                    res["pt"] = iou(ms[int(sc.argmax())])
                    res["box"] = iou(sam_predict(pred, box=b)[0][0])
                    ms, sc = sam_predict(pred, point=c, box=b, multi=True)
                    res["box+pt"] = iou(ms[int(sc.argmax())])
                    res["box+pt/agree"] = iou(max(ms, key=lambda m: mask_iou(m, poly)[0]))
                    best = lambda box: (lambda o: o[0][int(o[1].argmax())])(sam_predict(pred, point=c, box=box, multi=True))
                    res["grow1.4+pt"] = iou(best(grow(b, 1.4, W, H)))
                    btp = [bt[0] * W, bt[1] * H, bt[2] * W, bt[3] * H]
                    res["boxtask+pt"] = iou(best(btp))
                    res["boxtask1.2+pt"] = iou(best(grow(btp, 1.2, W, H)))
                    # boxes for the object, all against the true box (normalized)
                    res["boxes"] = {"boxtask": box_iou(bt, gb), "polybox": box_iou([b[0] / W, b[1] / H, b[2] / W, b[3] / H], gb),
                                    "sam_box": box_iou(mask_box(ms[int(sc.argmax())], W, H), gb)}
                    res["box_pred"] = dict(boxtask=[round(v, 4) for v in bt], gt=[round(v, 4) for v in gb])
                    prompt_ms.append((time.perf_counter() - t) * 1000 / 3)
                    row["arms"][name][how] = res
            rows.append(row)
            if (k + 1) % 100 == 0:
                print(f"  {k + 1}/{len(cases)}", flush=True)

    os.makedirs(OUT, exist_ok=True)
    path = f"{OUT}/sam2_refine_{os.environ.get('SEG_SET', 'refcoco')}_{n}.json"
    json.dump(dict(n=len(rows), sam=SAM_ID, embed_ms=embed_ms, prompt_ms=prompt_ms, rows=rows), open(path, "w"),
              default=float)

    # 3. summary: mIoU (mean per case) and cIoU (pooled intersection / union)
    agg = lambda vals: dict(miou=statistics.mean(v[0] for v in vals),
                            ciou=sum(v[1] for v in vals) / max(1, sum(v[2] for v in vals)))
    summary = {f"oracle/{o}": agg([r["oracle"][o] for r in rows]) for o in ("gt_box", "gt_pt")}
    print(f"\n{len(rows)} cases | SAM 2 ({SAM_ID}) image embedding {statistics.median(embed_ms):.0f} ms median, "
          f"one prompt {statistics.median(prompt_ms):.1f} ms\n")
    print("| arm | readout | " + " | ".join(PROMPTS) + " |\n|---|---|" + "---|" * len(PROMPTS))
    for arm in arms:
        for how in READOUTS:
            cells = []
            for pr in PROMPTS:
                s = agg([r["arms"][arm.name][how][pr] for r in rows])
                summary[f"{arm.name}/{how}/{pr}"] = s
                cells.append(f"{s['miou']:.3f} ({s['ciou']:.3f})")
            print(f"| {arm.name} | {how} | " + " | ".join(cells) + " |")
    print("\nBoxes vs the true box, mean IoU (IoU >= 0.5): boxtask = the box task (UI-trained), polybox = our "
          "polygon's box, sam_box = the box of SAM 2's box+pt mask\n")
    print("| arm | readout | " + " | ".join(BOXES) + " |\n|---|---|" + "---|" * len(BOXES))
    for arm in arms:
        for how in READOUTS:
            cells = []
            for bk in BOXES:
                v = [r["arms"][arm.name][how]["boxes"][bk] for r in rows]
                s = dict(iou=statistics.mean(v), iou50=statistics.mean(float(x >= 0.5) for x in v))
                summary[f"{arm.name}/{how}/box/{bk}"] = s
                cells.append(f"{s['iou']:.3f} ({s['iou50']:.3f})")
            print(f"| {arm.name} | {how} | " + " | ".join(cells) + " |")
    for o in ("gt_box", "gt_pt"):
        print(f"\nSAM 2 oracle from {o}: mIoU {summary['oracle/' + o]['miou']:.3f} (cIoU {summary['oracle/' + o]['ciou']:.3f})")
    print("cells: mIoU (cIoU)")
    os.makedirs(OUT, exist_ok=True)
    json.dump(dict(n=len(rows), sam=SAM_ID, embed_ms=embed_ms, prompt_ms=prompt_ms, summary=summary, rows=rows),
              open(path, "w"), default=float)
    print(f"\nwrote {path}", flush=True)


if __name__ == "__main__":
    main()
